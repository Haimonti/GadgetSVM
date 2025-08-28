import yaml
import time
import logging
from p2pfl.node import Node
from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.utils.utils import wait_convergence, wait_to_finish  # New imports



def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# --- Main script execution ---

# Load configuration from config.yaml
CFG = load_config("config.yaml")

nodes = []
# Use a top-level try/finally block to ensure all nodes are stopped on exit
try:
    # --- Step 1: Instantiate and Start Nodes ---
    # We'll create and start the nodes in the same loop to ensure they begin
    # to acquire their network addresses immediately.
    print("Instantiating and starting all nodes...")
    for i in range(CFG['num_nodes']):
        try:
            # FIXED: Explicitly pass the bind address to the node constructor.
            node = Node(
                model_build_fn(), P2PFLDataset.from_huggingface("p2pfl/MNIST"), addr="127.0.0.1"
            )
            node.start()
            nodes.append(node)
            print(f"Node {i} started with address: {node.addr}")
        except Exception as e:
            logging.error(f"Failed to start node {i}: {e}")
            # Clean up any already started nodes before exiting
            for n in nodes:
                n.stop()
            exit(1)

    # Wait for a moment to ensure all nodes have finished starting their servers
    # and their addresses are ready for connection. This helps prevent the
    # "Cannot add 0.0.0.0:xxxx" error.
    time.sleep(2)
    print("All nodes started. Proceeding with connections...")

    # --- Step 2: Connect Nodes Based on Topology ---
    # Now that all nodes are started and have a valid address, we can
    # connect them according to the specified graph topology.
    if CFG["graph"] == "full":
        print("Connecting nodes in a 'full' graph topology...")
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                try:
                    # No longer need to replace the address, as the node already
                    # has a valid connectable address
                    nodes[i].connect(nodes[j].addr)
                    print(f"Node {i} connected to Node {j} at {nodes[j].addr}")
                except Exception as e:
                    logging.error(f"Failed to connect Node {i} to Node {j}: {e}")

    elif CFG["graph"] == "ring":
        print("Connecting nodes in a 'ring' graph topology...")
        for i in range(len(nodes)):
            try:
                # No longer need to replace the address
                nodes[i].connect(nodes[(i + 1) % len(nodes)].addr)
                print(f"Node {i} connected to Node {(i + 1) % len(nodes)} at {nodes[(i + 1) % len(nodes)].addr}")
            except Exception as e:
                logging.error(f"Failed to connect Node {i} to Node {(i + 1) % len(nodes)}: {e}")

    else:
        raise ValueError(f"Unsupported graph type: {CFG['graph']}")

    # --- Step 3: Wait for Network Convergence ---
    # This is a key step from the library's example.
    # It waits until all nodes have received an update from at least n-1 nodes.
    print("Waiting for network convergence...")
    wait_convergence(nodes, len(nodes) - 1, only_direct=False, wait=60)

    # --- Step 4: Initiate Learning Process ---
    # Start the learning process from a single node (e.g., the first one).
    print("Initiating learning from the first node...")
    nodes[0].set_start_learning(rounds=CFG.get('rounds'), epochs=CFG.get('epochs'))

    # --- Step 5: Wait for Learning to Finish ---
    # This function waits until all nodes have finished their learning rounds
    # or until a timeout is reached. It replaces the infinite loop.
    print("Learning process started. Waiting for completion...")
    wait_to_finish(nodes, timeout=3600) # 1 hour timeout

    print("Learning process finished.")

except KeyboardInterrupt:
    print("\nLearning process stopped by user.")
except Exception as e:
    logging.error(f"An error occurred during the simulation: {e}")
finally:
    # Ensure all nodes are properly stopped on exit
    print("Stopping all nodes...")
    for node in nodes:
        node.stop()
