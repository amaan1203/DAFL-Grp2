import numpy as np
import random
import time
from typing import List, Dict, Tuple
from math import ceil
from decimal import Decimal

# Constants
FIELD_SIZE = 10 ** 5  # Field size for secret sharing

def measure_execution_time(func):
    """Decorator to measure execution time of a function."""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.4f} seconds")
        return result

    return wrapper


def polynom(x: int, coefficients: List[int]) -> int:
    """
    This generates a single point on the graph of given polynomial
    in x. The polynomial is given by the list of coefficients.
    """
    point = 0
    for coefficient_index, coefficient_value in enumerate(coefficients[::-1]):
        # print("coefficient_index:", coefficient_index, "coefficient_value:", coefficient_value)
        point += x ** coefficient_index * coefficient_value
    return point


def coeff(t : int, secret : int) -> List[int]:
    """
    Randomly generate a list of coefficients for a polynomial with
    degree of t - 1, whose constant is secret.
    """
    coeff = [random.randrange(0, FIELD_SIZE) for _ in range(t - 1)]
    coeff.append(secret)
    return coeff


def generate_shares(n : int, m : int, secret : int) -> List[Tuple[int, int]]:
    """
    Split given secret into n shares with minimum threshold
    of m shares to recover this secret, using SSS algorithm.
    """
    coefficients = coeff(m, secret)
    shares = []

    for i in range(1, n + 1):
        x = random.randrange(1, FIELD_SIZE)
        shares.append((x, polynom(x, coefficients)))

    return shares


def reconstruct_secret(shares : List[Tuple[int, int]]) -> int:
    """
    Combines individual shares (points on graph)
    using Lagrange's interpolation.

    shares is a list of points (x, y) belonging to a
    polynomial with a constant of our key.
    """
    sums = 0
    prod_arr = []

    for j, share_j in enumerate(shares):
        xj, yj = share_j
        prod = Decimal(1)

        for i, share_i in enumerate(shares):
            xi, _ = share_i
            if i != j:
                prod *= Decimal(Decimal(xi) / (xi - xj))

        prod *= yj
        sums += Decimal(prod)
    print(sums)
    return int(round(Decimal(sums), 0))


class Client:
    def _init_(self, client_id: int, num_clients: int):
        self.client_id = client_id
        self.num_clients = num_clients
        self.model_update = None
        self.my_shares = {}  # {recipient_id: {layer_name: {value_index: {owner_id, shares}}}}
        self.received_shares = {}  # {sender_id: {layer_name: {value_index: {owner_id, shares}}}}

    def generate_model_update(self):
        """Generate a mock model update with fixed values."""
        self.model_update = {
            "layer1": np.array([[1.0, 2.0], [3.0, 4.0]]) * (self.client_id + 1),
            "layer2": np.array([5.0, 6.0, 7.0]) * (self.client_id + 1)
        }
        print(f"Client {self.client_id} generated model update:")
        for layer_name, values in self.model_update.items():
            print(f"{layer_name}:\n{values}")
        print()

    def create_shares(self, threshold: int):
        """Create shares of the model update using the SSS algorithm."""
        # Initialize share structure
        for recipient_id in range(self.num_clients):
            self.my_shares[recipient_id] = {}

        # Process each layer of the model update
        for layer_name, layer_values in self.model_update.items():
            flat_values = layer_values.flatten()

            # Initialize layer structure for each recipient
            for recipient_id in range(self.num_clients):
                self.my_shares[recipient_id][layer_name] = {}

            # Create shares for each value
            for i, value in enumerate(flat_values):
                # Convert float to int for SSS
                int_value = int(value * 1000)  # Scale up to preserve precision

                # Create shares
                shares = generate_shares(self.num_clients, threshold, int_value)

                # Distribute shares to recipients
                for recipient_id in range(self.num_clients):
                    self.my_shares[recipient_id][layer_name][i] = {
                        'owner_id': self.client_id,
                        'share': shares[recipient_id]
                    }

    def distribute_shares(self, clients: List['Client']):
        """Distribute shares to other clients."""
        for recipient_id, shares in self.my_shares.items():
            if recipient_id != self.client_id:
                # Send to other client
                clients[recipient_id].receive_shares(self.client_id, shares)
            else:
                # Keep own shares
                self.received_shares[self.client_id] = shares

    def receive_shares(self, sender_id: int, shares: Dict):
        """Receive shares from another client."""
        self.received_shares[sender_id] = shares

    def forward_to_buffer(self, buffer: 'Buffer'):
        """Forward received shares to the buffer."""
        for sender_id, shares in self.received_shares.items():
            buffer.receive_shares(sender_id, shares)


class Buffer:
    def _init_(self, buffer_id: int, threshold: int):
        self.buffer_id = buffer_id
        self.threshold = threshold
        # Reorganized share structure
        self.shares = {}  # {owner_id: {sender_id: {layer_name: {value_index: share}}}}
        self.layer_shapes = {
            "layer1": (2, 2),
            "layer2": (3,)
        }

    def receive_shares(self, sender_id: int, shares: Dict):
        """Receive shares from a client with owner metadata."""
        for layer_name, layer_shares in shares.items():
            for value_index, share_data in layer_shares.items():
                owner_id = share_data['owner_id']
                share = share_data['share']

                # Ensure structure is initialized
                if owner_id not in self.shares:
                    self.shares[owner_id] = {}
                if sender_id not in self.shares[owner_id]:
                    self.shares[owner_id][sender_id] = {}
                if layer_name not in self.shares[owner_id][sender_id]:
                    self.shares[owner_id][sender_id][layer_name] = {}

                # Store the share
                self.shares[owner_id][sender_id][layer_name][value_index] = share

    def reconstruct_update(self, client_id: int, share_ids: List[int]):
        """Reconstruct a client's model update using shares from specified clients."""
        if client_id not in self.shares:
            print(f"Buffer {self.buffer_id}: No shares received for client {client_id}")
            return None

        if len(share_ids) < self.threshold:
            print(f"Buffer {self.buffer_id}: Not enough share IDs ({len(share_ids)})")
            return None

        # Check if all requested share IDs are available for this client
        for share_id in share_ids:
            if share_id not in self.shares[client_id]:
                print(f"Buffer {self.buffer_id}: Missing shares from client {share_id} for client {client_id}'s update")
                return None

        # Get layer names from the first share source
        first_sender = share_ids[0]
        layer_names = list(self.shares[client_id][first_sender].keys())

        reconstructed_update = {}

        # Process each layer
        for layer_name in layer_names:
            shape = self.layer_shapes[layer_name]
            flat_size = np.prod(shape)
            reconstructed_flat = np.zeros(flat_size)

            # Get indices from the first sender
            value_indices = list(self.shares[client_id][first_sender][layer_name].keys())

            # Reconstruct each value
            for i in value_indices:
                # Collect shares for this value from specified clients
                value_shares = []
                for sender_id in share_ids:
                    if (layer_name in self.shares[client_id][sender_id] and
                            i in self.shares[client_id][sender_id][layer_name]):
                        value_shares.append(self.shares[client_id][sender_id][layer_name][i])

                # Reconstruct the value
                try:
                    if len(value_shares) >= self.threshold:
                        reconstructed_value = reconstruct_secret(value_shares)
                        reconstructed_flat[int(i)] = reconstructed_value / 1000  # Reverse scaling
                    else:
                        print(
                            f"Not enough shares for value at index {i}: got {len(value_shares)}, need {self.threshold}")
                        reconstructed_flat[int(i)] = 0
                except Exception as e:
                    print(f"Error reconstructing value at index {i}: {e}")
                    reconstructed_flat[int(i)] = 0  # Set to default on error

            # Reshape to original dimensions
            reconstructed_update[layer_name] = reconstructed_flat.reshape(shape)

        return reconstructed_update


class SimulationSystem:
    def _init_(self, num_clients: int, threshold: int):
        self.num_clients = num_clients
        self.threshold = threshold
        self.clients = [Client(i, num_clients) for i in range(num_clients)]
        self.buffers = [Buffer(i, threshold) for i in range(num_clients)]

    @measure_execution_time
    def setup(self):
        """Set up the simulation."""
        print("Generating model updates...")
        for client in self.clients:
            client.generate_model_update()

        print("Creating shares...")
        for client in self.clients:
            client.create_shares(self.threshold)

        print("Distributing shares between clients...")
        for client in self.clients:
            client.distribute_shares(self.clients)

        print("Forwarding shares to buffers...")
        for i, client in enumerate(self.clients):
            client.forward_to_buffer(self.buffers[i])

    @measure_execution_time
    def reconstruct_updates(self):
        """Simulate reconstruction of model updates."""
        for client_id in range(self.num_clients):
            print(f"\n{'=' * 50}")
            print(f"Reconstructing model update for client {client_id}")
            print(f"{'=' * 50}")

            # Choose threshold number of random clients to use their shares
            share_ids = random.sample(range(self.num_clients), self.threshold)
            print(f"Using shares from clients: {share_ids}")

            # Try reconstruction in each buffer
            for buffer_id in range(self.num_clients):
                buffer = self.buffers[buffer_id]

                print(f"\nBuffer {buffer_id} attempting reconstruction:")
                reconstructed_update = buffer.reconstruct_update(client_id, share_ids)

                if reconstructed_update:
                    print("Reconstructed model update:")
                    for layer_name, values in reconstructed_update.items():
                        print(f"{layer_name}:\n{values}")

                    # Compare with original
                    print("\nOriginal model update:")
                    for layer_name, values in self.clients[client_id].model_update.items():
                        print(f"{layer_name}:\n{values}")

                    # Calculate error
                    print("\nReconstruction error:")
                    for layer_name in reconstructed_update:
                        original = self.clients[client_id].model_update[layer_name]
                        reconstructed = reconstructed_update[layer_name]
                        error = np.abs(original - reconstructed).mean()
                        print(f"{layer_name}: {error:.6f}")
                else:
                    print("Reconstruction failed.")


@measure_execution_time
def main():
    # Set random seed for reproducibility
    random.seed(42)

    # Parameters
    num_clients = 5
    threshold = 3

    # Run simulation
    print(f"Initializing system with {num_clients} clients and threshold {threshold}...")
    system = SimulationSystem(num_clients, threshold)

    print("\nSetting up the system...")
    system.setup()

    print("\nReconstructing model updates...")
    system.reconstruct_updates()


# Simple demonstration of the SSS algorithm
def demo_sss():
    # (3,5) sharing scheme
    t, n = 3, 5
    secret = 1234
    print(f'Original Secret: {secret}')

    # Phase I: Generation of shares
    shares = generate_shares(n, t, secret)
    print(f'Shares: {", ".join(str(share) for share in shares)}')

    # Phase II: Secret Reconstruction
    # Picking t shares randomly for reconstruction
    pool = random.sample(shares, t)
    print(f'Combining shares: {", ".join(str(share) for share in pool)}')
    print(f'Reconstructed secret: {reconstruct_secret(pool)}')


if __name__ == "_main_":
    # Uncomment to run the SSS demo
    demo_sss()

    # Run the federated learning simulation
    main()
