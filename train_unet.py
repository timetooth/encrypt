""""
uv run train_unet.py \
  --epochs 100 \
  --learning_rate 1e-4 \
  --dataset_root ./dataset/mimic-cxr-dataset \
  --checkpoint_path "./results/checkpoints/best_model.pth" \
  --results ./results \
  --positions AP \
  --batch_size 192 \
  --shuffle True \
  --num_workers 4 \
  --height 512 \
  --width 512 \
  --initial_channels 64
"""

import os
import argparse

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchvision.utils import save_image

from models.unet_model import UNet
from dataloader import MimicDataset


EPOCHS = 100
LEARNING_RATE = 1e-4
BATCH_SIZE = 192

torch.backends.cudnn.benchmark = True

def get_dataloader(dataset_root="./dataset/mimic-cxr-dataset",
                   positions=("AP",),
                   batch_size=32,
                   shuffle=True,
                   num_workers=4,
                   height=512,
                   width=512):
    # positions list should only have one of ["PA","AP","LATERAL"]
    all_dfs = []
    for pos in positions:
        train_csv_path = f"{dataset_root}/image_paths_{pos}_train.csv"
        if not os.path.exists(train_csv_path):
            raise ValueError(f"CSV file for position {pos} does not exist at path {train_csv_path}")
        df = pd.read_csv(train_csv_path)
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    train_df, val_df = train_test_split(combined_df, test_size=0.2, random_state=42)

    train_dataset = MimicDataset(train_df, h=height, w=width)
    val_dataset = MimicDataset(val_df, h=height, w=width)

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  pin_memory=True,
                                  prefetch_factor=2,
                                  persistent_workers=True,
                                  num_workers=num_workers)

    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True,
                                persistent_workers=True,
                                prefetch_factor=2
                                )

    return train_dataloader, val_dataloader


def get_args():
    parser = argparse.ArgumentParser(description="Train U-Net on MIMIC-CXR dataset")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE, help="Learning rate for optimizer")
    parser.add_argument("--dataset_root", type=str, default="./dataset/mimic-cxr-dataset",
                        help="Root directory of the dataset")
    parser.add_argument("--checkpoint_path", type=str, default="./results/checkpoints/last.pth",
                        help="Path to save model checkpoints (.pth)")
    parser.add_argument("--results", type=str, default="./results",
                        help="Folder to save training results and logs")
    parser.add_argument("--positions", nargs='+', default=["AP"],
                        help="List of image positions to include (e.g., AP, PA, LATERAL)")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training")
    parser.add_argument("--shuffle", type=bool, default=True, help="Whether to shuffle the dataset")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--height", type=int, default=512, help="Height of input images")
    parser.add_argument("--width", type=int, default=512, help="Width of input images")
    parser.add_argument("--initial_channels", type=int, default=64, help="Number of initial channels in U-Net")
    args = parser.parse_args()
    return args


def load_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    print(f"Loaded checkpoint {checkpoint_path} from epoch {start_epoch}, best_val_loss={best_val_loss}")
    return model, optimizer, start_epoch, best_val_loss


def compute_psnr(pred, target, eps=1e-8):
    # both in [0,1]
    mse = F.mse_loss(pred, target)
    psnr = 10.0 * torch.log10(1.0 / (mse + eps))
    return psnr.item()


def train(train_dataloader,
          val_dataloader,
          epochs,
          learning_rate,
          checkpoint_path,
          writer,
          results_dir,
          initial_channels=64
          ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ----- init model & optimizer -----
    model = UNet(in_channels=1, out_channels=initial_channels).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    start_epoch = 0
    best_val_loss = float('inf')

    if checkpoint_path and os.path.exists(checkpoint_path):
        model, optimizer, start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, checkpoint_path, device
        )

    if model is None or optimizer is None:
        raise ValueError("Model or optimizer not initialized properly.")

    criterion = nn.MSELoss()

    train_losses, val_losses, val_psnrs = [], [], []

    print(f"Starting training for {epochs} epochs...")
    global_step = 0

    for epoch in range(start_epoch, start_epoch + epochs):
        # --------- TRAIN ---------
        model.train()
        running_loss = 0.0

        for batch in train_dataloader:
            imgs = batch.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, imgs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

            # log train loss per step
            writer.add_scalar("Loss/train_step", loss.item(), global_step)
            global_step += 1

        epoch_train_loss = running_loss / len(train_dataloader.dataset)
        train_losses.append(epoch_train_loss)
        writer.add_scalar("Loss/train_epoch", epoch_train_loss, epoch)
        print(f"Epoch [{epoch+1}], Train Loss: {epoch_train_loss:.4f}")

        # --------- VALIDATION ---------
        model.eval()
        running_val_loss = 0.0
        running_psnr = 0.0

        # we’ll also capture one sample batch for images
        sample_imgs = None
        sample_recons = None

        with torch.no_grad():
            for i, val_batch in enumerate(val_dataloader):
                imgs = val_batch.to(device)
                val_outputs = model(imgs)
                val_loss = criterion(val_outputs, imgs)

                running_val_loss += val_loss.item() * imgs.size(0)

                # PSNR metric
                running_psnr += compute_psnr(val_outputs, imgs) * imgs.size(0)

                if sample_imgs is None:
                    sample_imgs = imgs[:4].detach().cpu()
                    sample_recons = val_outputs[:4].detach().cpu()

        epoch_val_loss = running_val_loss / len(val_dataloader.dataset)
        epoch_val_psnr = running_psnr / len(val_dataloader.dataset)
        val_losses.append(epoch_val_loss)
        val_psnrs.append(epoch_val_psnr)

        writer.add_scalar("Loss/val_epoch", epoch_val_loss, epoch)
        writer.add_scalar("PSNR/val_epoch", epoch_val_psnr, epoch)

        print(f"Epoch [{epoch+1}], Val Loss: {epoch_val_loss:.4f}, Val PSNR: {epoch_val_psnr:.2f} dB")

        # --------- IMAGE LOGGING ---------
        if sample_imgs is not None and sample_recons is not None:
            # Images for TensorBoard (normalize=False since already [0,1])
            writer.add_images("val/input", sample_imgs, epoch)
            writer.add_images("val/recon", sample_recons, epoch)

            # Save a PNG grid every N epochs
            if (epoch + 1) % 10 == 0 or epoch == start_epoch:
                img_dir = os.path.join(results_dir, "images")
                os.makedirs(img_dir, exist_ok=True)

                # Stack [input | recon] horizontally
                save_tensor = torch.cat([sample_imgs, sample_recons], dim=0)
                save_path = os.path.join(img_dir, f"epoch_{epoch+1:03d}_recon.png")
                save_image(save_tensor, save_path, nrow=4)
                print(f"Saved sample reconstructions to {save_path}")

        # --------- CHECKPOINTING ---------
        if checkpoint_path and (epoch + 1) % 10 == 0:
            # save "last"
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                },
                os.path.join(os.path.dirname(checkpoint_path), "last.pth")
                if os.path.dirname(checkpoint_path) else "last.pth"
            )

            # save best by val loss
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_loss": best_val_loss,
                    },
                    checkpoint_path,
                )
                print(f"Checkpoint saved at epoch {epoch+1} with validation loss {best_val_loss:.4f}")

    # --------- SAVE METRICS CSV ---------
    metrics_path = os.path.join(results_dir, "metrics.csv")
    df = pd.DataFrame({
        "epoch": list(range(start_epoch + 1, start_epoch + epochs + 1)),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "val_psnr": val_psnrs,
    })
    df.to_csv(metrics_path, index=False)
    print(f"Saved metrics CSV to {metrics_path}")


if __name__ == "__main__":
    args = get_args()

    os.makedirs(args.results, exist_ok=True)
    os.makedirs(os.path.join(args.results, "checkpoints"), exist_ok=True)
    # if no checkpoint path provided, put it under results
    if not args.checkpoint_path:
        args.checkpoint_path = os.path.join(args.results, "checkpoints/best_model.pth")

    train_dataloader, val_dataloader = get_dataloader(
        dataset_root=args.dataset_root,
        positions=args.positions,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=args.num_workers,
        height=args.height,
        width=args.width,
    )

    # TensorBoard logs under results/runs
    log_dir = os.path.join(args.results, "runs")
    writer = SummaryWriter(log_dir=log_dir)

    train(
        train_dataloader,
        val_dataloader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_path=args.checkpoint_path,
        writer=writer,
        results_dir=args.results,
        initial_channels=args.initial_channels
    )

    writer.close()
