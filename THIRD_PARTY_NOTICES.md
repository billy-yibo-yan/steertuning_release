# Third-party notices

The code in this repository is under the MIT License (`LICENSE`). It uses the projects below, which have their own licenses. Models and packages install from their owners, and this repository does not contain them. This list is not legal advice.

| Project | Role | License |
|---|---|---|
| [Utonia](https://github.com/Pointcept/Utonia) | Frozen point-cloud encoder | Code Apache-2.0. **Weights CC-BY-NC 4.0** |
| [S2M2](https://github.com/junhong-3dv/s2m2) | Metric depth from stereo | **CC-BY-NC 4.0**, code and weights |
| [SAM3](https://github.com/facebookresearch/sam3) | Arm, hand and object masks, navigation latents | SAM License. Weights are gated: request access from the owner and log in with `hf auth login`. Do not redistribute them |
| [SigLIP](https://huggingface.co/google/siglip-base-patch16-224) | Frozen text encoder | Apache-2.0 |
| [Qwen3-VL-8B](https://github.com/QwenLM/Qwen3-VL) | Instructions and paraphrases | Apache-2.0 |
| Gemini 2.5 Flash, Gemini Robotics-ER | Object keypoints, the command planner | Google API terms, through your own `GEMINI_API_KEY` |
| [YOR](https://github.com/YOR-robot/YOR) | The robot's driver, a separate project, and the robot model in `assets/robot/` | MIT |
| [wuji-description](https://github.com/wuji-technology/wuji-description) | The hand models in `assets/robot/` | MIT |
| [wuji-retargeting](https://github.com/wuji-technology/wuji-retargeting) | Kinematic hand retargeting, and the example values of `retargeting` in `configs/deploy.yaml` | MIT |
| [dex-retargeting](https://github.com/dexsuite/dex-retargeting), from [AnyTeleop](https://yzqin.github.io/anyteleop/) | The retargeting method that wuji-retargeting builds on | MIT |
| [wujihandpy](https://github.com/wuji-technology/wujihandpy) | Hand driver, on the robot computer | MIT |
| [aria2mano](https://github.com/KevinyWu/aria2mano) | Aria streaming and offline loading | MIT |
| [Project Aria tools](https://github.com/facebookresearch/projectaria_tools) | Aria data | Apache-2.0 |
| [commlink](https://pypi.org/project/commlink/) | RPC between computers | MIT |
| [mink](https://github.com/kevinzakka/mink), [MuJoCo](https://github.com/google-deepmind/mujoco) | Cartesian IK, forward kinematics | Apache-2.0 |
| [Diffusers](https://github.com/huggingface/diffusers) | DDIM scheduler | Apache-2.0 |

**What this means for checkpoints.** The Utonia weights and S2M2 are non-commercial. A policy checkpoint needs the Utonia weights to run, and its training depth comes from S2M2. Treat checkpoints and adapters as non-commercial, under CC-BY-NC 4.0.

## Robot models

`assets/robot/` adapts the [YOR](https://github.com/YOR-robot/YOR) robot model and the Wuji hand models from [wuji-description](https://github.com/wuji-technology/wuji-description). The upstream projects hold the copyright on the meshes and kinematic descriptions. The following license covers these files.

MIT License

Copyright (c) 2026 NYU Generalizable Robotics and AI Lab (GRAIL)

Copyright (c) 2025 Wuji Technology

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Transformer and conditional temporal UNet

`src/steertuning/policy/transformer.py` adapts the diffusion transformer from [DiT-Policy](https://github.com/SudeepDasari/dit-policy/blob/ba22ee3ff455f0423fc9cfa5658a1a1cdddc6e73/data4robotics/models/diffusion.py). `src/steertuning/policy/unet.py` adapts the conditional temporal U-Net from [the same repository](https://github.com/SudeepDasari/dit-policy/blob/ba22ee3ff455f0423fc9cfa5658a1a1cdddc6e73/data4robotics/models/diffusion_unet.py), which takes it from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) by Chi et al. The following license covers the adapted code.

MIT License

Copyright (c) 2023 Sudeep Dasari

Copyright (c) Meta Platforms, Inc. and affiliates.

Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Point Transformer V3

`configs/policy/ptv3.yaml` runs [Point Transformer V3](https://github.com/Pointcept/PointTransformerV3) (MIT License, Copyright (c) 2023 Pointcept). It is an optional dependency, installed separately. This repository does not contain its code.
