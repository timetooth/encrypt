import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from unet.unet_model import UNet
from dataloader import MimicDataset
import torch.nn as nn
import pandas as pd
import os

EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 32

## Build the checkpoint flow
## Build a train, test and validation split

def get_dataloader(dataset_root = "./dataset/mimic-cxr-dataset", positions=[], batch_size=32, shuffle=True, num_workers=4):
    # positions list should only have one of ["PA","AP","LATERAL"]
    # return concatenation of all dfs
    all_dfs = []
    for pos in positions:
        csv_path = f"{dataset_root}/image_paths_{pos}.csv"
        if not os.path.exists(csv_path):
            raise Exception(f"CSV file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        all_dfs.append(df)
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    train_val_df, test_df = train_test_split(combined_df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42)
    


    # dataset = MimicDataset(combined_df, h=512, w=512)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

loss_fn = nn.MSELoss()
model = UNet(in_channels=1, out_channels=64)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

csv_path = "./dataset/mimic-cxr-dataset/image_paths_PA.csv"
dataset = MimicDataset(csv_path, h=512, w=512)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

losses = []

for epoch in range(EPOCHS):
    total_loss = 0.0
    for batch in dataloader:
        outputs = model(batch)
        loss = loss_fn(outputs, batch)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(dataloader):.4f}")
    losses.append(total_loss/len(dataloader))




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")