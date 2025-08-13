import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

# --- Python Path Fix ---
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
PROJECT_ROOT = os.path.dirname(script_dir)
sys.path.insert(0, PROJECT_ROOT)

from data_prep.dataset import MedicalImageDataset
from data_prep.augmentations import get_train_transforms, get_val_transforms
from models.lightning_model import LitClassificationModel
from models.utils import calculate_class_weights

def train(config_path):
    """
    Main training function.
    """
    # 1. Load Configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print(f"Loaded configuration from {config_path}")

    # 2. Setup Data
    train_data_path = os.path.join(PROJECT_ROOT, config['data_params']['data_path'], 'train')
    val_data_path = os.path.join(PROJECT_ROOT, config['data_params']['data_path'], 'val')
    
    train_dataset = MedicalImageDataset(img_dir=train_data_path, transform=get_train_transforms())
    val_dataset = MedicalImageDataset(img_dir=val_data_path, transform=get_val_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data_params']['batch_size'],
        shuffle=True,
        num_workers=config['data_params']['num_workers']
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data_params']['batch_size'],
        shuffle=False,
        num_workers=config['data_params']['num_workers']
    )

    # 3. Calculate Class Weights (Sprint 2)
    class_weights = None
    if config['trainer_params']['use_weighted_loss']:
        class_weights = calculate_class_weights(train_dataset)
        device = torch.device(config['trainer_params']['accelerator'] if torch.cuda.is_available() else "cpu")
        class_weights = class_weights.to(device)


    # 4. Initialize Model
    model = LitClassificationModel(
        model_name=config['model_params']['name'],
        num_classes=config['model_params']['num_classes'],
        learning_rate=config['trainer_params']['learning_rate'],
        class_weights=class_weights,
        pretrained=config['model_params']['pretrained']
    )

    # 5. Setup Callbacks and Logger
    logger = TensorBoardLogger("lightning_logs", name="chest_xray_classifier")
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',      # Monitor validation accuracy
        dirpath='checkpoints/',
        filename='best-checkpoint-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1,           # Save the best model
        mode='max'              # 'max' because we want to maximize accuracy
    )
    
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=3, # Stop if val_loss does not improve for 3 epochs
        verbose=True,
        mode='min'
    )

    # 6. Initialize Trainer
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        max_epochs=config['trainer_params']['max_epochs'],
        accelerator=config['trainer_params']['accelerator'],
        devices="auto"
    )

    # 7. Start Training
    print("--- Starting Model Training ---")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print("--- Model Training Finished ---")


if __name__ == '__main__':
    # Default config path
    default_config = os.path.join(PROJECT_ROOT, 'configs', 'default.yaml')
    train(default_config)