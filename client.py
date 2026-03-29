import torch

class FederatedClient:
    def __init__(self, client_id):
        self.client_id = client_id

    def train_local(self, model, data):
        # Simulated local training
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        # ... training logic ...
        return model.state_dict()