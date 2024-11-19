import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import json

import time

from peft import LoraConfig, get_peft_model

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss

# dir_img = Path('./data/imgs/train/')
# dir_mask = Path('./data/masks/train_masks/')
dir_checkpoint_regular = Path('./checkpoints_reg/')
dir_checkpoint_lora = Path('./checkpoints_lora/')

dir_img = Path('./data/imgs/resized_images_gta/')
dir_mask = Path('./data/masks/resized_masks_gta/')

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return F_loss.mean()
        else:
            return F_loss


def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        checkpoint_dir: Path = Path('./checkpoints_reg/'),
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        using_lora = False
):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop([param for param in model.parameters() if param.requires_grad],
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    weight = torch.tensor([9, 1]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight) if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    record = "logs/training_record.txt"
    def tensor_to_serializable(obj):
        """
        Recursively converts PyTorch tensors in a nested structure into serializable types.
        Args:
            obj: Any Python object (dict, list, tuple, etc.)
        Returns:
            obj: The same object with tensors converted to lists or other serializable types.
        """
        if isinstance(obj, torch.Tensor):
            return obj.tolist()  # Convert tensors to lists
        elif isinstance(obj, dict):
            return {key: tensor_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [tensor_to_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(tensor_to_serializable(item) for item in obj)
        else:
            return obj  # Return as-is if not a tensor

    # 5. Begin training
    total_time = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        start_time = time.time()
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            
            for i, batch in enumerate(train_loader):
                # if i ==3:
                #     break
                
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                if i == 0:
                    print(torch.numel(true_masks[0] == 0), torch.numel(true_masks[0] == 255))
                    true_masks[0][true_masks[0] == 255] = 1
                    print(torch.numel(true_masks[0] == 0), torch.numel(true_masks[0] == 255), torch.numel(true_masks[0] == 1))
                    return

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        # loss += dice_loss(
                        #     F.softmax(masks_pred, dim=1).float(),
                        #     F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                        #     multiclass=True
                        # )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // 2)
                if division_step > 0:
                    if global_step % division_step == 0:
                        period_time = time.time() - start_time
                        total_time += period_time

                        histograms = {}
                        for tag, value in model.named_parameters():
                            if value.requires_grad:
                                tag = tag.replace('/', '.')
                                if not (torch.isinf(value) | torch.isnan(value)).any():
                                    histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                                if value.grad is not None and not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                    histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))

                        try:
                            info = {
                                "train_method": f"Using LoRA: {using_lora}",
                                "current_period": f"epoch: {epoch}, {global_step // division_step} division step",
                                "period_time": period_time,
                                "total_time": total_time,
                                "updated_parameter count": sum(p.numel() for p in model.parameters() if p.requires_grad),
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                # 'images': wandb.Image(images[0].cpu()),
                                # 'masks': {
                                #     'true': wandb.Image(true_masks[0].float().cpu()),
                                #     'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                # },
                                'step': global_step,
                                'epoch': epoch,
                                # **histograms
                            }
                            experiment.log(info)
                            with open(record, "rw+") as f:
                                prev = json.load(f)
                                if prev:
                                    prev.append(tensor_to_serializable(info))
                                    
                                json.dump(prev, f, indent=4)

                            

                        except:
                            print("Error in logging")

        if save_checkpoint:
            if using_lora:
                lora_save_path = os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}")
                # lora_save_path = checkpoint_dir
                model.save_pretrained(lora_save_path)
                print(f"Lora Model saved at: {lora_save_path}")
                continue

            else:
                Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                state_dict = model.state_dict()
                state_dict['mask_values'] = dataset.mask_values
                torch.save(state_dict, str(checkpoint_dir / 'checkpoint_epoch{}.pth'.format(epoch)))
                logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=2, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--using_lora', '-lora', type=str, default=False, help='Use LoRA or not use LoRa, that is the question?')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        # del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

        # print(model)
    if args.using_lora:
        modules = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                modules.append(name)
                # print(f"Found Conv2d module: {name}")
        
        # Define LoRA configuration
        lora_config = LoraConfig(
            r=16,  # Rank
            lora_alpha=32,
            target_modules= modules,  # Apply to Conv2d layers
            lora_dropout=0.1,
            bias="none",
            task_type="IMAGE_SEGMENTATION"
        )

        lora_model = get_peft_model(model, lora_config)

        for param in lora_model.parameters():
            param.requires_grad = False
        
        # Unfreeze LoRA parameters
        for name, param in lora_model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True

        lora_model.to(device=device)
        try:
            # start_time_A = time.time()
            train_model(
                model=lora_model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                device=device,
                img_scale=args.scale,
                val_percent=args.val / 100,
                amp=args.amp,
                checkpoint_dir = dir_checkpoint_lora,
                using_lora = args.using_lora
            )
            # end_time_A = time.time()
            # training_time_A = end_time_A - start_time_A
            # print(f"Training Time for LORA model: {training_time_A:.2f} seconds")
        except torch.cuda.OutOfMemoryError:
            logging.error('Detected OutOfMemoryError! '
                        'Enabling checkpointing to reduce memory usage, but this slows down training. '
                        'Consider enabling AMP (--amp) for fast and memory efficient training')
            torch.cuda.empty_cache()
            model.use_checkpointing()
            train_model(
                model=lora_model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                device=device,
                img_scale=args.scale,
                val_percent=args.val / 100,
                amp=args.amp
            )


    if not args.using_lora:
        model.to(device=device)
        try:
            start_time_A = time.time()
            train_model(
                model=model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                device=device,
                img_scale=args.scale,
                val_percent=args.val / 100,
                amp=args.amp,
                checkpoint_dir = dir_checkpoint_regular
            )
            end_time_A = time.time()
            training_time_A = end_time_A - start_time_A
            print(f"Training Time for reg model: {training_time_A:.2f} seconds")
        except torch.cuda.OutOfMemoryError:
            logging.error('Detected OutOfMemoryError! '
                        'Enabling checkpointing to reduce memory usage, but this slows down training. '
                        'Consider enabling AMP (--amp) for fast and memory efficient training')
            torch.cuda.empty_cache()
            model.use_checkpointing()
            train_model(
                model=model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                device=device,
                img_scale=args.scale,
                val_percent=args.val / 100,
                amp=args.amp
            )
