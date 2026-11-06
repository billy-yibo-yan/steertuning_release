"""Pretraining on D_wild. The settings are in configs/pretrain.yaml."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import numpy as np
import torch
from diffusers.optimization import get_cosine_schedule_with_warmup
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from steertuning.data.dataset import EndlessWindows, TextTokenizer, WindowDataset, collate
from steertuning.data.normalizer import ActionNormalizer, fit_normalizer
from steertuning.policy import checkpoint
from steertuning.policy.policy import PolicyConfig, SteertuningPolicy

log = logging.getLogger(__name__)

TRAINING_STATE_FILE = "training_state.pt"
RUN_CONFIG_FILE = "pretrain_config.yaml"
KEEP_CHECKPOINTS = 3


def _seed_generators(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed + 1)


def _loader(dataset: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    # The batches stay outside Accelerate: it would shard them along dim 0, while the collated cloud
    # concatenates points. Each process draws its own windows.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def _read_or_fit_normalizer(cfg: DictConfig, accelerator) -> ActionNormalizer:
    """The stats file named in the config, else a fit on the training split kept in the run folder."""
    if cfg.normalizer is not None:
        return ActionNormalizer.from_json(cfg.normalizer)
    path = Path(cfg.output_dir) / checkpoint.NORMALIZER_FILE
    with accelerator.main_process_first():
        if not path.is_file():
            fit_normalizer(cfg.dataset_root).to_json(path)
    return ActionNormalizer.from_json(path)


def _checkpoints(output_dir: Path) -> list[Path]:
    """The complete checkpoints of a run, oldest first."""
    paths = output_dir.glob("checkpoint_*[0-9]")
    return sorted(paths, key=lambda path: int(path.name.split("_")[1]))


def save_checkpoint(output_dir: Path, policy: SteertuningPolicy, optimizer, schedule, step: int) -> None:
    """Write a checkpoint and remove the oldest beyond KEEP_CHECKPOINTS.

    The weights use the pretrained directory format, so fine-tuning can start from any checkpoint.
    A save is written to a .partial folder and renamed when complete, so resuming never reads half of one.
    """
    partial = output_dir / f"checkpoint_{step}.partial"
    policy.save_pretrained(partial)
    state = {"optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(), "step": step}
    torch.save(state, partial / TRAINING_STATE_FILE)
    partial.rename(output_dir / f"checkpoint_{step}")
    for path in _checkpoints(output_dir)[:-KEEP_CHECKPOINTS]:
        shutil.rmtree(path)


def restore_checkpoint(path: Path, policy: SteertuningPolicy, optimizer, schedule) -> int:
    """Restore the model, optimizer and schedule. Returns the step."""
    checkpoint.load_weights(policy, path)
    state = torch.load(path / TRAINING_STATE_FILE, map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    schedule.load_state_dict(state["schedule"])
    return state["step"]


def _build_accelerator(cfg: DictConfig):
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    ddp = DistributedDataParallelKwargs(
        find_unused_parameters=False,
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
        static_graph=True,
        bucket_cap_mb=100,
    )
    return Accelerator(
        mixed_precision="no",  # autocast is applied around the forward pass in `pretrain`
        log_with="wandb" if cfg.wandb.enabled else None,
        kwargs_handlers=[ddp],
    )


def _build_loader(cfg: DictConfig, policy: SteertuningPolicy) -> DataLoader:
    """Endless windows of the training split."""
    tokenizer = TextTokenizer(policy.config.text_model, policy.config.text_revision)
    train_set = WindowDataset(cfg.dataset_root, policy.normalizer, tokenizer=tokenizer)
    return _loader(EndlessWindows(train_set), cfg.batch_size, cfg.num_workers)


def _build_optimizer(cfg: DictConfig, policy: SteertuningPolicy, accelerator) -> tuple:
    """The distributed model, AdamW on the trainable weights, and the warmup and cosine schedule."""
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=tuple(cfg.adam_betas),
        eps=cfg.adam_eps,
    )
    model, optimizer = accelerator.prepare(policy, optimizer)
    return model, optimizer, get_cosine_schedule_with_warmup(optimizer, cfg.warmup_steps, cfg.steps)


def _resume(cfg: DictConfig, policy: SteertuningPolicy, optimizer, schedule, accelerator) -> int:
    """Restore the newest checkpoint in the run folder, if any. Returns the step to continue from."""
    checkpoints = _checkpoints(Path(cfg.output_dir))
    if not checkpoints:
        return 0
    step = restore_checkpoint(checkpoints[-1], policy, optimizer, schedule)
    accelerator.print(f"Resuming from step {step}")
    return step


def pretrain(cfg: DictConfig) -> Path:
    """Train the policy and export it to <output_dir>/pretrained. Returns the export path."""
    from accelerate.utils import send_to_device

    accelerator = _build_accelerator(cfg)
    device = accelerator.device
    # Attention runs in fp32 under bf16 autocast; TF32 matmuls keep that fast.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    _seed_generators(cfg.seed + accelerator.process_index)
    if accelerator.is_main_process:
        processes, batch = accelerator.num_processes, cfg.batch_size
        log.info("Processes: %d, batch per process: %d, global batch: %d", processes, batch, processes * batch)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if accelerator.is_main_process:
        OmegaConf.save(cfg, output_dir / RUN_CONFIG_FILE, resolve=True)
    normalizer = _read_or_fit_normalizer(cfg, accelerator)
    policy = SteertuningPolicy(PolicyConfig(**OmegaConf.to_container(cfg.policy, resolve=True)))
    policy.normalizer = normalizer
    train_loader = _build_loader(cfg, policy)
    model, optimizer, schedule = _build_optimizer(cfg, policy, accelerator)
    start_step = _resume(cfg, policy, optimizer, schedule, accelerator)
    # The step joins the seed so that a resumed run draws new windows.
    _seed_generators(cfg.seed + accelerator.process_index + start_step)
    name = cfg.wandb.run_name or output_dir.resolve().name  # also the run id, so a resumed run continues
    accelerator.init_trackers(
        cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        init_kwargs={"wandb": {"name": name, "id": name, "resume": "allow"}},
    )
    batches = iter(train_loader)

    model.train()
    progress = tqdm(
        range(start_step + 1, cfg.steps + 1),
        initial=start_step,
        total=cfg.steps,
        disable=not accelerator.is_main_process,
    )
    for step in progress:
        batch = send_to_device(next(batches), device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.bf16_autocast):
            loss = model(batch)
        optimizer.zero_grad()
        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        schedule.step()

        progress.set_postfix(loss=f"{loss.item():.4f}", refresh=False)
        if step % cfg.log_every == 0:
            metrics = {
                "train/loss": loss.item(),
                "train/grad_norm": grad_norm.item(),
                "train/lr": schedule.get_last_lr()[0],
            }
            accelerator.log(metrics, step=step)
        if accelerator.is_main_process and (step % cfg.save_every == 0 or step == cfg.steps):
            save_checkpoint(output_dir, policy, optimizer, schedule, step)

    export = output_dir / "pretrained"
    if accelerator.is_main_process:
        policy.save_pretrained(export)
    accelerator.end_training()
    return export
