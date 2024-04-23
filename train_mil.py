import torch
from mil import MILNet
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader
from utils import ImageNetValidationDatasetLoader, get_device


def train(milnet: MILNet, data_path: str, num_epochs: int, lr: float, weight_decay: float, batch_size: int, num_workers: int, cv_fold: int):
    dataset = ImageNetValidationDatasetLoader(data_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    
    optimizer = torch.optim.Adam(milnet.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = BCEWithLogitsLoss()
    
    device = get_device()
    criterion = criterion.to(device)
    
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(data_loader):
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            classes, out = milnet(inputs)
            out = out.reshape([128])
            loss = criterion(out, labels.float())
            loss.backward()
            optimizer.step()
            
            if i % 10 == 0:
                print(f"Epoch {epoch} - Iteration {i} - Loss {loss.item()}")
                exit(0)
    