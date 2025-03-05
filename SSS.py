import numpy as np
import random
import time
from typing import List, Dict, Tuple

def measure_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time: {end_time - start_time:.4f} seconds\n")
        return result
    return wrapper

def mod_inverse(a: int, m: int) -> int:
    a = a % m
    if a < 0:
        a += m
    return pow(a, m - 2, m)

def create_shares(secret: int, num_shares: int, threshold: int, prime: int) -> List[Tuple[int, int]]:
    coefficients = [secret] + [random.randint(1, prime - 1) for _ in range(threshold - 1)]
    shares = [(i, sum(coeff * (i ** idx) for idx, coeff in enumerate(coefficients)) % prime) for i in range(1, num_shares + 1)]
    return shares

def reconstruct_secret(shares: List[Tuple[int, int]], prime: int) -> int:
    secret = 0
    for i, (x_i, y_i) in enumerate(shares):
        numerator, denominator = 1, 1
        for j, (x_j, _) in enumerate(shares):
            if i != j:
                numerator = (numerator * -x_j) % prime
                denominator = (denominator * (x_i - x_j)) % prime
        secret = (secret + y_i * numerator * mod_inverse(denominator, prime)) % prime
    return secret

class Client:
    def _init_(self, client_id: int, num_clients: int):
        self.client_id = client_id
        self.num_clients = num_clients
        self.model_update = {}
        self.my_shares = {}
        self.received_shares = {}

    def generate_model_update(self):
        self.model_update = {
            "layer1": np.array([[1.0, 2.0], [3.0, 4.0]]) * (self.client_id + 1),
            "layer2": np.array([5.0, 6.0, 7.0]) * (self.client_id + 1)
        }
        print(f"Client {self.client_id} generated model update:")
        for layer_name, values in self.model_update.items():
            print(f"{layer_name}:\n{values}")
        print()

    def create_shares(self, threshold: int, prime: int):
        for recipient_id in range(self.num_clients):
            self.my_shares[recipient_id] = {}
        for layer_name, layer_values in self.model_update.items():
            flat_values = layer_values.flatten()
            for recipient_id in range(self.num_clients):
                self.my_shares[recipient_id][layer_name] = {}
            for i, value in enumerate(flat_values):
                int_value = int(value * 1000) % prime
                value_shares = create_shares(int_value, self.num_clients, threshold, prime)
                for recipient_id in range(self.num_clients):
                    self.my_shares[recipient_id][layer_name][i] = value_shares[recipient_id]
                print(f"Shares for value {value} (index {i}): {value_shares}\n")

    def distribute_shares(self, clients: List['Client']):
        for recipient_id, shares in self.my_shares.items():
            if recipient_id != self.client_id:
                clients[recipient_id].receive_shares(self.client_id, shares)
            else:
                self.received_shares[self.client_id] = shares
        print(f"Client {self.client_id} distributed shares\n")

    def receive_shares(self, sender_id: int, shares: Dict):
        self.received_shares[sender_id] = shares
        print(f"Client {self.client_id} received shares from Client {sender_id}\n")

    def forward_to_buffer(self, buffer: 'Buffer'):
        for sender_id, shares in self.received_shares.items():
            buffer.receive_shares(sender_id, shares)

class Buffer:
    def _init_(self, buffer_id: int, threshold: int, prime: int):
        self.buffer_id = buffer_id
        self.threshold = threshold
        self.prime = prime
        self.shares = {}
        self.layer_shapes = {"layer1": (2, 2), "layer2": (3,)}

    def receive_shares(self, client_id: int, shares: Dict):
        self.shares[client_id] = shares

    def reconstruct_update(self, client_id: int, share_ids: List[int]):
        reconstructed_update = {}
        for layer_name in self.shares[client_id].keys():
            shape = self.layer_shapes[layer_name]
            flat_size = np.prod(shape)
            reconstructed_flat = np.zeros(flat_size)
            for i in range(int(flat_size)):
                value_shares = [(share_id + 1, self.shares[share_id][layer_name][i][1]) for share_id in share_ids]
                try:
                    reconstructed_value = reconstruct_secret(value_shares, self.prime)
                    reconstructed_flat[i] = reconstructed_value / 1000
                    print(f"Reconstructed {i}: {reconstructed_flat[i]} using shares {value_shares}\n")
                except Exception as e:
                    print(f"Error reconstructing value at index {i}: {e}\n")
                    reconstructed_flat[i] = 0
            reconstructed_update[layer_name] = reconstructed_flat.reshape(shape)
        return reconstructed_update

class SimulationSystem:
    def _init_(self, num_clients: int, threshold: int):
        self.num_clients = num_clients
        self.threshold = threshold
        self.prime = 2 ** 31 - 1
        self.clients = [Client(i, num_clients) for i in range(num_clients)]
        self.buffers = [Buffer(i, threshold, self.prime) for i in range(num_clients)]

    @measure_execution_time
    def setup(self):
        for client in self.clients:
            client.generate_model_update()
            client.create_shares(self.threshold, self.prime)
            client.distribute_shares(self.clients)
        for i, client in enumerate(self.clients):
            client.forward_to_buffer(self.buffers[i])

    @measure_execution_time
    def reconstruct_updates(self):
        for client_id in range(self.num_clients):
            print(f"\nReconstructing model update for Client {client_id}")
            share_ids = random.sample(range(self.num_clients), self.threshold)
            print(f"Using shares from clients: {share_ids}\n")
            for buffer in self.buffers:
                print(f"Buffer {buffer.buffer_id} attempting reconstruction:\n")
                reconstructed_update = buffer.reconstruct_update(client_id, share_ids)
                print("Final reconstructed update:")
                for layer_name, values in reconstructed_update.items():
                    print(f"{layer_name}:\n{values}\n")

def main():
    random.seed(42)
    system = SimulationSystem(num_clients=5, threshold=3)
    system.setup()
    system.reconstruct_updates()

if __name__ == "_main_":
    main()