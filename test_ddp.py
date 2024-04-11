import os

import pytorch_lightning as pl
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


class IrisDataModule(pl.LightningDataModule):
    """Data module for the Iris dataset."""

    def __init__(self):
        """Initializes the data module."""
        super().__init__()
        iris = load_iris()
        X_train, X_val, y_train, y_val = train_test_split(
            iris.data, iris.target, test_size=0.2, random_state=42)
        self.train_data = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train))
        self.val_data = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val))

    def train_dataloader(self):
        """Returns a DataLoader for the training data."""
        return DataLoader(self.train_data, batch_size=32, num_workers=16)

    def val_dataloader(self):
        """Returns a DataLoader for the validation data."""
        return DataLoader(self.val_data, batch_size=32, num_workers=16)


class LogisticRegression(pl.LightningModule):
    """Logistic Regression model."""

    def __init__(self):
        """Initializes the model."""
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, x):
        """Performs a forward pass through the model."""
        return F.softmax(self.linear(x), dim=1)

    def training_step(self, batch, batch_idx):
        """Performs a training step."""
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """Configures the optimizer."""
        return torch.optim.SGD(self.parameters(), lr=0.01)

    def validation_step(self, batch, batch_idx):
        """Performs a validation step."""
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss, sync_dist=True)


if __name__ == "__main__":
    """Main function to set up and train the model."""
    torch.set_float32_matmul_precision('high')
    model = LogisticRegression()
    data = IrisDataModule()

    print("num_devices: ", torch.cuda.device_count())
    print("num_nodes: ", int(os.environ.get('SLURM_JOB_NUM_NODES')))

    trainer = pl.Trainer(
        max_epochs=10,
        devices=torch.cuda.device_count(),
        accelerator='cuda',
        strategy="ddp",
        log_every_n_steps=1,
        num_nodes=int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
    )
    trainer.fit(model, data)
