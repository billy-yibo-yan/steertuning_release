"""Serve the Wuji hands over RPC. Runs on the robot computer and needs only wujihandpy, commlink and numpy.

    python scripts/wuji_server.py --left-serial <serial> --right-serial <serial>

Each hand gets its own process. A client writes `target_angles` and reads `current_angles`, both (5, 4)
in radians from thumb to pinky, knuckle to tip. On exit the fingers go to zero and the joints are disabled.
A hand that fails ends its process, so that a client stops getting answers.
"""

import argparse
import logging
import multiprocessing
import os
import signal
import sys
import time

import numpy as np

log = logging.getLogger("wuji_server")


class WujiServer:
    """Pushes the latest target to one hand at a fixed rate. Only this loop talks to the hand."""

    def __init__(self, serial_number: str) -> None:
        import wujihandpy

        self.hand = wujihandpy.Hand(serial_number=serial_number)
        self.hand.write_joint_enabled(True)
        self.controller = self.hand.realtime_controller(
            enable_upstream=False, filter=wujihandpy.filter.LowPass(cutoff_freq=5.0)
        )
        time.sleep(0.5)
        self.current_angles = np.asarray(self.hand.read_joint_actual_position())
        self.target_angles = self.current_angles.copy()

    def stop(self) -> None:
        try:
            self.controller.set_joint_target_position(np.zeros((5, 4)))
            time.sleep(1.0)
            self.controller.close()
        finally:
            self.hand.write_joint_enabled(False)

    def run(self, rate: float) -> None:
        period = 1.0 / rate
        while True:
            started = time.monotonic()
            self.controller.set_joint_target_position(self.target_angles)
            self.current_angles = self.hand.read_joint_actual_position()
            time.sleep(max(0.0, period - (time.monotonic() - started)))


def serve(serial_number: str, port: int, rate: float) -> None:
    from commlink import RPCServer

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for stop_signal in (signal.SIGINT, signal.SIGTERM):
        signal.signal(stop_signal, signal.default_int_handler)
    hand = WujiServer(serial_number)
    status = 0
    try:
        RPCServer(hand, port=port).start()
        print(f"Wuji hand {serial_number} on port {port}", flush=True)
        hand.run(rate)
    except KeyboardInterrupt:
        pass
    except Exception:
        log.exception("Wuji hand %s failed.", serial_number)
        status = 1
    # A second Ctrl+C must not cut the way to zero short.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    try:
        hand.stop()
    except Exception:
        log.exception("Wuji hand %s did not stop cleanly.", serial_number)
        status = 1
    # commlink's server thread is not a daemon and can block in a read, so a normal exit could never finish.
    os._exit(status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left-serial", required=True)
    parser.add_argument("--right-serial", required=True)
    parser.add_argument("--left-port", type=int, default=8008)
    parser.add_argument("--right-port", type=int, default=8007)
    parser.add_argument("--rate", type=float, default=100.0, help="commands per second")
    args = parser.parse_args()
    hands = [(args.left_serial, args.left_port), (args.right_serial, args.right_port)]
    processes = [multiprocessing.Process(target=serve, args=(*hand, args.rate)) for hand in hands]
    for process in processes:
        process.start()
    # Ctrl+C reaches the hands directly. This process waits for them to shut down.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, lambda *_: [process.terminate() for process in processes])
    for process in processes:
        process.join()
    if any(process.exitcode for process in processes):
        sys.exit(1)


if __name__ == "__main__":
    main()
