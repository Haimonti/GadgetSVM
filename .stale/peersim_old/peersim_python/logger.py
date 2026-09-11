"""Tiny stdout logger — a p2pfl-free stand-in for p2pfl's logger.

The PeerSim port must run without p2pfl installed, so it cannot import
`p2pfl.management.logger`. This mirrors the `logger.info(tag, msg)` call shape
used across the project.
"""

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)


class _Logger:
    def info(self, tag: str, msg: str) -> None:
        logging.info(f"{tag:>10} | {msg}")

    def warning(self, tag: str, msg: str) -> None:
        logging.warning(f"{tag:>10} | {msg}")


logger = _Logger()
