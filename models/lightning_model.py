import torch
import torch.nn as nn
from torch.optim import Adam
import pytorch_lightning as pl
from torchvision.models import get_model, ResNet18_Weights
from torchmetrics.classification import Accuracy

class LitClassificationModel(pl.LightningModule):
    """
    PyTorch Lightning module for image classification.
    Encapsulates the model, loss function, optimizer, and training logic.
    """
    def __init__(self, model_name, num_classes, learning_rate, class_weights=None, pretrained=True):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters()

        # Load a pretrained model
        if pretrained:
            weights = ResNet18_Weights.DEFAULT if model_name == "resnet18" else None
            self.model = get_model(model_name, weights=weights)
        else:
            self.model = get_model(model_name, weights=None)

        # Replace the classifier layer for our specific number of classes
        if 'resnet' in model_name:
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)
        # Add other model types here if needed (e.g., 'densenet')

        # Define the loss function (optionally with class weights)
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # Define metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        
        # Log loss and accuracy
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.train_acc(logits, y)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)

        # Log validation loss and accuracy
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        self.val_acc(logits, y)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer