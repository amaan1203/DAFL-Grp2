import numpy as np
import random
import time
from typing import List, Dict, Tuple
from decimal import Decimal
from math import ceil


FIELD_SIZE = 10 ** 5  # Field size for secret sharing


def polynom(x, coefficients):
    """
    Generate a single point on the graph of a given polynomial.

    Args:
        x (int): The x-coordinate
        coefficients (List[int]): Coefficients of the polynomial

    Returns:
        int: The y-coordinate for the given x
    """
    point = 0
    for coefficient_index, coefficient_value in enumerate(coefficients[::-1]):
        point += x ** coefficient_index * coefficient_value
    return point


def coeff(t, secret):
    """
    Randomly generate coefficients for a polynomial.

    Args:
        t (int): Threshold for reconstruction
        secret (int): Secret value to be shared

    Returns:
        List[int]: Coefficients of the polynomial
    """
    coefficients = [random.randrange(0, FIELD_SIZE) for _ in range(t - 1)]
    coefficients.append(secret)
    return coefficients


def generate_shares(n, m, secret):
    """
    Split a secret into n shares with a minimum threshold of m shares.

    Args:
        n (int): Total number of shares
        m (int): Minimum shares required for reconstruction
        secret (int): Secret to be shared

    Returns:
        List[Tuple[int, int]]: Generated shares
    """
    coefficients = coeff(m, secret)
    shares = []

    for i in range(1, n + 1):
        x = random.randrange(1, FIELD_SIZE)
        shares.append((x, polynom(x, coefficients)))

    return shares


def reconstruct_secret(shares):
    """
    Reconstruct secret using Lagrange interpolation.

    Args:
        shares (List[Tuple[int, int]]): Shares to reconstruct secret

    Returns:
        int: Reconstructed secret
    """
    sums = 0
    for j, share_j in enumerate(shares):
        xj, yj = share_j
        prod = Decimal(1)

        for i, share_i in enumerate(shares):
            xi, _ = share_i
            if i != j:
                prod *= Decimal(Decimal(xi) / (xi - xj))

        prod *= yj
        sums += Decimal(prod)

    return int(round(Decimal(sums), 0))


class Buffer:
    def __init__(self, buffer_id: int, threshold: int):
        """
        Initialize a buffer for secret sharing.

        Args:
            buffer_id (int): Unique identifier for the buffer
            threshold (int): Minimum shares required for reconstruction
        """
        self.buffer_id = buffer_id
        self.threshold = threshold
        self.shares = {}  # Shares from different senders
        self.total_shares_count = 0
        self.layer_shapes = {"model_layer": (2, 2)}
        self.position_values = {}  # Store shares by position for reconstruction

    def add_share(self, sender_id: int, position_index: str, share_data: Dict):
        """
        Add a share to the buffer.

        Args:
            sender_id (int): ID of the sending client
            position_index (str): Position index of the share
            share_data (Dict): Share data
        """
        if position_index not in self.position_values:
            self.position_values[position_index] = []

        self.position_values[position_index].append({
            'sender_id': sender_id,
            'share': share_data['share'],
            'owner_id': share_data['owner_id']
        })

        self.total_shares_count += 1

    def is_ready_for_reconstruction(self):
        """
        Check if buffer has received enough shares for reconstruction.

        Returns:
            bool: True if ready for reconstruction, False otherwise
        """
        # Check if we have at least threshold number of shares for any position
        for position, shares in self.position_values.items():
            if len(shares) >= self.threshold:
                return True
        return False

    def reconstruct_update(self):
        """
        Reconstruct model update using all available shares.

        Returns:
            Dict or None: Reconstructed model update
        """
        reconstructed_values = {}

        # Group shares by owner_id
        shares_by_owner = {}

        for position, shares_list in self.position_values.items():
            for share_info in shares_list:
                owner_id = share_info['owner_id']

                if owner_id not in shares_by_owner:
                    shares_by_owner[owner_id] = {}

                if position not in shares_by_owner[owner_id]:
                    shares_by_owner[owner_id][position] = []

                shares_by_owner[owner_id][position].append(share_info['share'])

        # Reconstruct values for each owner
        for owner_id, positions in shares_by_owner.items():
            reconstructed_values[owner_id] = {}

            for position, shares in positions.items():
                if len(shares) >= self.threshold:
                    try:
                        reconstructed_value = reconstruct_secret(shares)
                        reconstructed_values[owner_id][position] = reconstructed_value / 1000  # Scale back down
                    except Exception as e:
                        print(f"Error reconstructing value for owner {owner_id} at position {position}: {e}")

        # Convert to numpy arrays
        final_reconstructions = {}
        for owner_id, positions in reconstructed_values.items():
            # Create empty array with the model shape
            shape = self.layer_shapes["model_layer"]
            model = np.zeros(shape)

            # Fill in the values
            for position, value in positions.items():
                row, col = map(int, position.split('_'))
                model[row, col] = value

            final_reconstructions[owner_id] = model

        return final_reconstructions


class Client:
    def __init__(self, client_id: int, num_clients: int, buffer_id: int):
        """
        Initialize a client in the secret sharing system.

        Args:
            client_id (int): Unique identifier for the client
            num_clients (int): Total number of clients
            buffer_id (int): ID of the buffer mapped to this client
        """
        self.client_id = client_id
        self.num_clients = num_clients
        self.buffer_id = buffer_id
        self.model_update = None
        # Initialize my_shares as a list of empty dictionaries for each client
        self.my_shares = [{} for _ in range(num_clients)]
        self.received_shares = {}  # Initialize as a dictionary instead of a list
        self.model_dimensions = None

    def generate_model_update(self):
        """Generate a mock model update with fixed values."""
        self.model_update = np.array([[100.0, 200.0] , [300.0 , 400.0]]) * (self.client_id + 1)
        self.model_dimensions = self.model_update.shape

        print(f"Client {self.client_id} generated model update:")
        print(f" printing model updates for client {self.client_id} -> {self.model_update}")

    def create_shares(self, threshold: int):
        """
        Create shares of the model update.

        Args:
            threshold (int): Minimum shares required for reconstruction

        Returns:
            List[Dict]: Shares to be distributed
        """
        # Loop through the 2D array properly
        for row_idx in range(self.model_update.shape[0]):
            for col_idx in range(self.model_update.shape[1]):
                value = self.model_update[row_idx, col_idx]
                int_value = int(value * 1000)  # Scale up to preserve precision
                shares = generate_shares(self.num_clients, threshold, int_value)
                print(
                    f"printing shares generated by client {self.client_id} for value {value} at position ({row_idx},{col_idx}) -> {shares}")

                # Create a unique index for the value based on its position
                position_index = f"{row_idx}_{col_idx}"

                for recipient_id in range(self.num_clients):
                    # Initialize this position if it doesn't exist
                    if position_index not in self.my_shares[recipient_id]:
                        self.my_shares[recipient_id][position_index] = {}

                    # Store the share with position information
                    self.my_shares[recipient_id][position_index] = {
                        'owner_id': self.client_id,
                        'share': shares[recipient_id],
                        'row': row_idx,
                        'column': col_idx,

                    }

        print(f"Shares have been generated for client {self.client_id} and stored in my_shares array")
        print(f"printing my_shares array of this client {self.client_id}: {self.my_shares}")

        print(f"adding in received shares for client {self.client_id}")

        #Add a copy of the values to the received_shares dictionary for this client
        # so they have their own shares as well
        self.received_shares[self.client_id] = {}

        # For each row and column in the model
        for row_idx in range(self.model_update.shape[0]):
            for col_idx in range(self.model_update.shape[1]):
                position_index = f"{row_idx}_{col_idx}"

                # Add the share for this position from my_shares to received_shares
                if position_index in self.my_shares[self.client_id]:
                    self.received_shares[self.client_id][position_index] = self.my_shares[self.client_id][
                        position_index]

        return self.my_shares

    def distribute_shares(self, clients: List['Client']):
        """
        Distribute shares to other clients and include own share.

        Args:
            clients (List[Client]): List of all clients
        """
        # For each recipient client
        for recipient_id in range(self.num_clients):
            # Skip self (we already have our own shares)
            if recipient_id == self.client_id:
                continue

            # Get the recipient client object
            recipient_client = clients[recipient_id]

            # Check if recipient already has shares from this client
            if self.client_id not in recipient_client.received_shares:
                # Initialize the entry for this client in recipient's received_shares
                recipient_client.received_shares[self.client_id] = {}

                # Copy all shares meant for this recipient
                for position_index, share_data in self.my_shares[recipient_id].items():
                    recipient_client.received_shares[self.client_id][position_index] = share_data

    def forward_to_buffer(self, buffers: List['Buffer'], threshold: int):
        """
        Forward received shares to the correct buffer based on owner_id.

        Args:
            buffers (List[Buffer]): List of all buffers
            threshold (int): Minimum shares required for reconstruction
        """
        # Iterate through all clients in received_shares
        for sender_id, positions in self.received_shares.items():
            # Iterate through all position indices for this sender
            for position_index, share_data in positions.items():
                # Get the owner_id of this share
                owner_id = share_data['owner_id']

                # Forward to the buffer with matching owner_id
                target_buffer = buffers[owner_id]

                # Add the share to the buffer
                target_buffer.add_share(sender_id, position_index, share_data)


class SimulationSystem:
    def __init__(self, num_clients: int, threshold: int):
        """
        Initialize the secret sharing simulation system.

        Args:
            num_clients (int): Total number of clients
            threshold (int): Minimum shares required for reconstruction
        """
        self.num_clients = num_clients
        self.threshold = threshold

        self.buffers = [Buffer(i, threshold) for i in range(num_clients)]
        self.clients = [Client(i, num_clients, i) for i in range(num_clients)]

    def setup(self):
        """Set up the simulation by generating and distributing shares."""
        print("Generating model updates...")
        for client in self.clients:
            client.generate_model_update()

        print("\nCreating shares...")
        for client in self.clients:
            client.create_shares(self.threshold)

        print("\nDistributing shares between clients...")
        for client in self.clients:
            client.distribute_shares(self.clients)

        print(f"\nNow printing the received shares of all the clients")
        for client in self.clients:
            sender_count = len(client.received_shares)
            print(f"Client {client.client_id} received shares from {sender_count} clients")

        print("\nForwarding shares to buffers...")
        # Forward shares from all clients to buffers
        selected_clients = random.sample(self.clients, self.threshold)
        for client in selected_clients:
            print(f"Client {client.client_id} forwarding shares to buffers")
            client.forward_to_buffer(self.buffers, self.threshold)

        print("\nBuffer contents after forwarding:")
        for buffer in self.buffers:
            position_count = len(buffer.position_values)
            total_shares = buffer.total_shares_count
            print(f"Buffer {buffer.buffer_id}: {position_count} positions, {total_shares} total shares")

    def reconstruct_updates(self):
        """Reconstruct model updates from the buffers."""
        print("\nReconstructing model updates...")
        for buffer in self.buffers:
            if buffer.is_ready_for_reconstruction():
                print(f"\nReconstructing update for buffer {buffer.buffer_id} (owner {buffer.buffer_id})...")
                reconstructed = buffer.reconstruct_update()
                if reconstructed:
                    for owner_id, model in reconstructed.items():
                        print(f"Reconstructed model for owner {owner_id}:")
                        print(model)

                        # Compare with original model for verification
                        original_model = self.clients[owner_id].model_update
                        print(f"Original model for owner {owner_id}:")
                        print(original_model)

                        # Check if reconstruction was accurate
                        is_close = np.allclose(model, original_model)
                        print(f"Reconstruction successful: {is_close}")
            else:
                print(f"\nBuffer {buffer.buffer_id} not ready for reconstruction")


def main():
    """Main function to run the secret sharing simulation."""
    # Set random seed for reproducibility
    random.seed(42)

    # Parameters
    num_clients = 6
    threshold = 3

    # Run simulation
    print(f"Initializing system with {num_clients} clients and threshold {threshold}...")
    system = SimulationSystem(num_clients, threshold)

    print("\nSetting up the system...")
    system.setup()

    print("\nReconstructing model updates...")
    system.reconstruct_updates()


if __name__ == "__main__":
    main()