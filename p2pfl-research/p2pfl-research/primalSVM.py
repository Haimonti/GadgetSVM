import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

class LinearSVM(L.LightningModule):
    def __init__(self, input_size=28*28, out_channels=10, lr_rate=0.01, C=1.0):
        super().__init__()
        self.fc = nn.Linear(input_size, out_channels)
        self.lr_rate = lr_rate
        self.C = C  # regularization constant

    def forward(self, x):
        # Flatten images
        batch_size, _, _ = x.size()
        x = x.view(batch_size, -1)
        return self.fc(x)

    def hinge_loss(self, outputs, labels):
        # One-vs-rest hinge loss
        # Convert labels to one-hot
        y_onehot = F.one_hot(labels, num_classes=outputs.size(1)).float()
        # Correct class scores
        correct_class_scores = (outputs * y_onehot).sum(dim=1, keepdim=True)
        # Margin loss
        margins = torch.clamp(outputs - correct_class_scores + 1.0, min=0)
        margins = margins * (1 - y_onehot)  # ignore correct class
        loss = margins.sum(dim=1).mean()
        # Regularization
        reg = 0.5 * torch.norm(self.fc.weight) ** 2
        return loss + self.C * reg

    def training_step(self, batch, batch_idx):
        x, y = batch["image"].float(), batch["label"]
        outputs = self(x)
        loss = self.hinge_loss(outputs, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch["image"].float(), batch["label"]
        logits = self(x)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("test_acc", acc, prog_bar=True)
        return acc

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr_rate)
