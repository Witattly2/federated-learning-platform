import random
import numpy as np
from typing import List, Dict, Any
from datetime import datetime

class Client:
    def __init__(self, client_id: int, data_size: int, model_params: Dict[str, Any]):
        self.client_id = client_id
        self.data_size = data_size
        self.model_params = model_params
        self.local_model = self._initialize_model()
        self.local_data = self._generate_local_data(data_size)
        print(f"Client {self.client_id} initialized with {self.data_size} data points.")

    def _initialize_model(self) -> Dict[str, np.ndarray]:
        # Simulate a simple model with weights and biases
        weights = np.random.rand(self.model_params["input_dim"], self.model_params["output_dim"])
        bias = np.random.rand(self.model_params["output_dim"])
        return {"weights": weights, "bias": bias}

    def _generate_local_data(self, size: int) -> np.ndarray:
        # Simulate local data (e.g., features and labels)
        return np.random.rand(size, self.model_params["input_dim"] + self.model_params["output_dim"])

    def train_local_model(self, global_model: Dict[str, np.ndarray]):
        # Simulate local training using global model as starting point
        print(f"Client {self.client_id}: Training local model on {self.data_size} data points...")
        self.local_model = global_model.copy() # Start with global model
        # Simulate updating weights and biases based on local data
        self.local_model["weights"] += np.random.rand(*self.local_model["weights"].shape) * 0.01
        self.local_model["bias"] += np.random.rand(*self.local_model["bias"].shape) * 0.01
        print(f"Client {self.client_id}: Local training complete.")

    def get_local_updates(self) -> Dict[str, np.ndarray]:
        # Return local model updates (e.g., gradients or updated weights)
        return self.local_model

class Server:
    def __init__(self, num_clients: int, model_params: Dict[str, Any]):
        self.num_clients = num_clients
        self.global_model = self._initialize_global_model(model_params)
        print(f"Federated Learning Server initialized with {self.num_clients} clients.")

    def _initialize_global_model(self, model_params: Dict[str, Any]) -> Dict[str, np.ndarray]:
        weights = np.random.rand(model_params["input_dim"], model_params["output_dim"])
        bias = np.random.rand(model_params["output_dim"])
        return {"weights": weights, "bias": bias}

    def aggregate_models(self, client_updates: List[Dict[str, np.ndarray]]):
        print("Server: Aggregating client models...")
        if not client_updates:
            print("No client updates to aggregate.")
            return

        # Simple averaging aggregation
        avg_weights = np.mean([update["weights"] for update in client_updates], axis=0)
        avg_bias = np.mean([update["bias"] for update in client_updates], axis=0)
        self.global_model = {"weights": avg_weights, "bias": avg_bias}
        print("Server: Aggregation complete. Global model updated.")

    def get_global_model(self) -> Dict[str, np.ndarray]:
        return self.global_model

def run_federated_learning(num_rounds: int, num_clients: int, model_params: Dict[str, Any]):
    server = Server(num_clients, model_params)
    clients = [Client(i, random.randint(50, 200), model_params) for i in range(num_clients)]

    for round_num in range(num_rounds):
        print(f"\n--- Federated Learning Round {round_num + 1}/{num_rounds} ---")
        global_model = server.get_global_model()
        
        client_updates = []
        for client in clients:
            client.train_local_model(global_model)
            client_updates.append(client.get_local_updates())
        
        server.aggregate_models(client_updates)

    print("\n--- Federated Learning Process Completed ---")
    print("Final Global Model Weights (first 5x5):")
    print(server.get_global_model()["weights"][:5,:5])

if __name__ == "__main__":
    model_parameters = {"input_dim": 10, "output_dim": 2}
    run_federated_learning(num_rounds=5, num_clients=3, model_params=model_parameters)
