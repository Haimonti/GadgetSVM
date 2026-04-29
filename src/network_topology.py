from p2pfl.management.logger import logger
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.frameworks.pytorch.lightning_model import LightningModel
from p2pfl.node import Node

from config import CONFIG
from model import SVMSDCALightning


def connect_topology(nodes: list, topology: str, base_port: int) -> None:
    """Wire nodes together according to the chosen topology.

    topology options:
        "ring"  — each node connects to its immediate successor (default)
        "full"  — every node connects to every other node
        "star"  — nodes 1..N-1 all connect to node 0 only
        "mesh"  — each node connects to neighbours at distance 1 and 2
    """
    n     = len(nodes)
    added = set()

    def _link(i: int, j: int):
        key = (min(i, j), max(i, j))
        if key not in added:
            nodes[i].connect(f"127.0.0.1:{base_port + j}")
            added.add(key)

    if topology == "ring":
        for i in range(n):
            _link(i, (i + 1) % n)

    elif topology == "full":
        for i in range(n):
            for j in range(i + 1, n):
                _link(i, j)

    elif topology == "star":
        for i in range(1, n):
            _link(0, i)

    elif topology == "mesh":
        for i in range(n):
            _link(i, (i + 1) % n)
            _link(i, (i + 2) % n)

    else:
        raise ValueError(
            f"Unknown topology '{topology}'. "
            "Choose from: ring | full | star | mesh"
        )

    logger.info("topology", f"'{topology}' wired — {len(added)} edge(s).")


def setup_nodes(worker_data: list) -> tuple[list, list]:
    """Create, configure, and start all p2pfl nodes."""
    lightning_modules = []
    nodes = []

    for i, d in enumerate(worker_data):
        lm = SVMSDCALightning(
            X_csr       = d["X_csr"],
            y_np        = d["y"],
            lambda_reg  = CONFIG["LAMBDA"],
            t0_fraction = CONFIG["T0_FRACTION"],
        )
        lightning_modules.append(lm)

        addr = f"127.0.0.1:{CONFIG['BASE_PORT'] + i}"
        node = Node(
            model = LightningModel(lm),
            data  = P2PFLDataset(d["hf_dataset"], dataset_name=f"rcv1_worker_{i}"),
            addr  = addr,
        )
        nodes.append(node)
        logger.info("network", f"Node {i} created at {addr}")

    for node in nodes:
        node.start()

    logger.info("network", f"All {len(nodes)} nodes started")
    return lightning_modules, nodes
