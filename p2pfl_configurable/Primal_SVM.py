# This script demonstrates the fully integrated architecture for using the 'p2pfl' library
# to perform federated learning with a custom SGD-based SVM model.
# The code is a conceptual blueprint and will not run as-is because it requires
# a distributed, multi-node environment and the p2pfl library installed.

# Install the library with a compatible backend like PyTorch:
# pip install "p2pfl[torch]"

import numpy as np
import warnings
import torch
from torch import nn, Tensor
from typing import Dict, Any, Callable
import logging
from p2pfl.examples.mnist.model.mlp_pytorch import model_build_fn
from p2pfl.learning.dataset.p2pfl_dataset import P2PFLDataset
from p2pfl.learning.trainer.p2pfl_trainer import P2PFLTrainer
from p2pfl.node import Node
from p2pfl.utils.utils import wait_convergence, wait_to_finish
import yaml
import time


# Suppress warnings for cleaner output.
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# --- 1. Your Custom Primal SVM Model (adapted for a framework like PyTorch) ---
# A key part of using a library like p2pfl is adapting your model to a
# supported framework. Here, we'll represent our SVM as a simple PyTorch model.
class PrimalSVMTorch(nn.Module):
    """
    Represents the Primal SVM as a PyTorch Module for compatibility with p2pfl.
    This module contains the weights and bias, which will be exchanged.
    """
    def __init__(self, n_features: int):
        super().__init__()
        # Use nn.Parameter so PyTorch can track them for gradients (if needed)
        # and so they are part of the model's state dictionary.
        self.weights = nn.Parameter(torch.zeros(n_features))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: Tensor) -> Tensor:
        """Calculates the decision function: w.x + b."""
        return torch.matmul(x, self.weights) + self.bias

def model_build_fn(config: Dict = None) -> nn.Module:
    """
    Required function for p2pfl.node.Node. It builds and returns the model.
    """
    n_features = 784  # For MNIST
    return PrimalSVMTorch(n_features)

# --- 2. Your Local Training Logic with SGD, adapted for the P2PFL Trainer interface ---
# This class contains the actual training loop and will be called locally
# on each peer to update the model. It must inherit from P2PFLTrainer.

class PrimalSVM_SGD_Trainer(P2PFLTrainer):
    """
    Implements a single round of SGD for a Primal SVM, compatible with the p2pfl Trainer API.
    """
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.lambda_param = self.config.get('lambda_param', 0.01)
        
    def train_round(self, x: Tensor, y: Tensor) -> None:
        """
        Executes a single training round on the local data using SGD.
        This method is called by the p2pfl Node for each round.
        """
        n_samples = len(x)
        
        # Shuffle data for better convergence.
        indices = torch.randperm(n_samples)
        
        for i in indices:
            x_i = x[i]
            y_i = y[i]
            
            # Since we're not using a PyTorch optimizer, we'll manually update the parameters
            with torch.no_grad():
                decision_function_output = self.model(x_i)
                condition = y_i * decision_function_output >= 1
                
                # Manual gradient update based on hinge loss
                if condition:
                    self.model.weights.data -= self.learning_rate * (2 * self.lambda_param * self.model.weights.data)
                    # Bias gradient is zero in this case, so no update.
                else:
                    self.model.weights.data -= self.learning_rate * (2 * self.lambda_param * self.model.weights.data - y_i * x_i)
                    self.model.bias.data -= self.learning_rate * (-y_i)

def trainer_build_fn(model: nn.Module, config: Dict[str, Any]) -> P2PFLTrainer:
    """
    Required function for p2pfl.node.Node. It builds and returns the trainer instance.
    """
    return PrimalSVM_SGD_Trainer(model, config)

# --- 3. The P2PFL Node and Main Script Logic ---
# This remains largely the same, but now it uses our custom model and trainer.

def load_and_preprocess_data():
    """
    Loads and preprocesses the MNIST data.
    """
    print("Loading and preprocessing MNIST data...")
    # NOTE: In a real P2P system, this data would already be local to the machine.
    mnist = fetch_openml('mnist_784', version=1, parser='auto', data_home='./')
    X, y = mnist.data, mnist.target

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Convert to a binary classification problem for simplicity with a single node
    # in this example, e.g., classify '0' vs 'not 0'.
    y_binary = np.where(y.astype(np.int64) == 0, 1, -1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
    
    print(f"Data loaded: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples.")
    return X_train, X_test, y_train, y_test

def load_config(path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    try:
        # Load configuration from config.yaml
        # For a single-node simulation, you can define it here.
        CFG = {
            'num_nodes': 1,
            'graph': 'full',
            'learning_rate': 0.001,
            'lambda_param': 0.01,
            'epochs': 1,
            'rounds': 5
        }
        
        nodes = []
        
        # --- Step 1: Instantiate and Start Nodes ---
        print("Instantiating and starting all nodes...")
        for i in range(CFG['num_nodes']):
            try:
                # The Node constructor now needs our custom trainer_build_fn
                node = Node(
                    model_build_fn, 
                    P2PFLDataset.from_huggingface("p2pfl/MNIST"), 
                    trainer_build_fn, 
                    config=CFG, 
                    addr="127.0.0.1"
                )
                node.start()
                nodes.append(node)
                print(f"Node {i} started with address: {node.addr}")
            except Exception as e:
                logging.error(f"Failed to start node {i}: {e}")
                for n in nodes:
                    n.stop()
                exit(1)

        time.sleep(2)
        print("All nodes started. Proceeding with connections...")

        # --- Step 2: Connect Nodes Based on Topology ---
        # This part is omitted for the single-node simulation but is crucial for a real setup.
        
        # --- Step 3: Wait for Network Convergence ---
        # Not needed for a single node, but essential for a multi-node setup.
        
        # --- Step 4: Initiate Learning Process ---
        print("Initiating learning from the first node...")
        nodes[0].set_start_learning(rounds=CFG.get('rounds'), epochs=CFG.get('epochs'))

        # --- Step 5: Wait for Learning to Finish ---
        print("Learning process started. Waiting for completion...")
        wait_to_finish(nodes, timeout=3600)

        print("Learning process finished.")

    except KeyboardInterrupt:
        print("\nLearning process stopped by user.")
    except Exception as e:
        logging.error(f"An error occurred during the simulation: {e}")
    finally:
        print("Stopping all nodes...")
        for node in nodes:
            node.stop()

