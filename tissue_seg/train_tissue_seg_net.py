import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt


from tissue_seg_dataloader import BasicDataset
from tissue_seg_dice import dice_loss
from tissue_seg_plot_masks import plot_img_and_mask
from tissue_seg_evaluate import evaluate
from tissue_seg_unet import UNet

dir_img = Path('/data/public/HULA/WSIs_renal_compartment_segmentations/Tissue_seg_imgs/')
dir_mask = Path('/data/public/HULA/WSIs_renal_compartment_segmentations/Tissue_seg_masks/')
# dir_checkpoint = Path('/data/public/HULA/WSIs_renal_compartment_segmentations/Tissue_seg_checkpoints/')
dir_checkpoint = Path('/data/hqvo3/HULA/WSIs_renal_compartment_segmentations/Tissue_seg_checkpoints/')


def train_net(net,
              device,
              epochs: int = 5,
              batch_size: int = 1,
              val_batch_size=1,
              learning_rate: float = 1e-5,
              val_percent: float = 0.1,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False):
    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    train_loader_args = dict(batch_size=batch_size, num_workers=16, pin_memory=True)
    val_loader_args = dict(batch_size=val_batch_size, num_workers=16, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **train_loader_args)
    val_loader = DataLoader(val_set, shuffle=True, drop_last=False, **val_loader_args)

    # # placeholders
    # psum = torch.tensor([0.0, 0.0, 0.0])
    # psum_sq = torch.tensor([0.0, 0.0, 0.0])
    # count = 0
    #
    # # loop through images
    # for inputs in tqdm(train_loader):
    #     psum += inputs['image'].sum(axis=[0, 2, 3])
    #     psum_sq += (inputs['image'] ** 2).sum(axis=[0, 2, 3])
    #     count += inputs['image'].shape[2] * inputs['image'].shape[2]
    #
    # # mean and std
    # total_mean = psum / count
    # total_var = (psum_sq / count) - (total_mean ** 2)
    # total_std = torch.sqrt(total_var)

    # get the class weights, we will add more weight to scarce classes
    class_sums = torch.zeros(args.classes)
    for inputs in tqdm(train_loader):
        uniques, unique_counts = torch.unique(inputs['mask'], return_counts=True)
        for index_id, id in enumerate(uniques):
            class_sums[id] += unique_counts[index_id]

    class_weights = torch.softmax(1 - (class_sums/torch.sum(class_sums)), dim=0)

    # total_mean = torch.tensor([0.485, 0.456, 0.406])
    # total_std = torch.tensor([0.229, 0.224, 0.225])

    total_mean = torch.tensor([0.5, 0.5, 0.5])
    total_std = torch.tensor([0.5, 0.5, 0.5])

    # save the mean and std
    np.save(str(dir_checkpoint) + '/train_norm.npy', np.array([total_mean.tolist(), total_std.tolist()]))

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
                                  val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Validation batch size: {val_batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=100)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss(weight=class_weights.cuda(), label_smoothing=0.1)
    global_step = 0

    # LOG BEST DICE SCORE
    val_dice_best = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                # perform train data augmentation
                for id, (train_img, train_mask) in enumerate(zip(images, true_masks)):

                    # horizontal flipping
                    if random.random() > 0.5:
                        train_img = F.hflip(train_img)
                        train_mask = F.hflip(train_mask)

                    # vertical flipping
                    if random.random() > 0.5:
                        train_img = F.vflip(train_img)
                        train_mask = F.vflip(train_mask)

                    # channel shuffling
                    if random.random() > 0.5:
                        channel_shuffle = torch.nn.ChannelShuffle(3)
                        train_img = channel_shuffle(train_img.unsqueeze(0))
                        train_img = train_img.squeeze(0)

                    # random cropping
                    if random.random() > 0.5:
                        x = random.randint(0, train_img.shape[1] // 2)
                        y = random.randint(0, train_img.shape[2] // 2)
                        width = random.randint(train_img.shape[1] // 4, train_img.shape[1] // 2)
                        height = random.randint(train_img.shape[2] // 4, train_img.shape[2] // 2)
                        ori_shape_img_x = train_img.shape[1]
                        ori_shape_img_y = train_img.shape[2]
                        train_img = F.resized_crop(train_img, x, y, height, width, [ori_shape_img_x, ori_shape_img_y])
                        # plt.imshow(  train_img.permute(1, 2, 0)  )
                        train_mask = F.resized_crop(train_mask.unsqueeze(0), x, y, height, width, [ori_shape_img_x, ori_shape_img_y]).squeeze(0)

                    # normalize the images
                    train_img = torchvision.transforms.Normalize(total_mean, total_std)(train_img)

                    # update values
                    images[id, :, :, :] = train_img
                    true_masks[id, :, :] = train_mask

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
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
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score, val_img, val_mask, val_pred = evaluate(net, val_loader, device, total_mean, total_std, val_dice_best, str(dir_checkpoint))
                        scheduler.step(val_score)

                        # save the best model
                        if val_score.item() >= val_dice_best:
                            val_dice_best = val_score.item()
                            logging.info('Validation Dice score: {} (improved)'.format(val_score))
                            if save_checkpoint:
                                Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
                                torch.save(net.state_dict(),
                                           str(dir_checkpoint / 'best_model.pth'))
                                logging.info(f'Top model saved!')
                        else:
                            logging.info('Validation Dice score: {}'.format(val_score))

                        rand_val = random.randint(0, val_img.shape[0] - 1)
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'validation Dice': val_score,
                            'images': wandb.Image(val_img[rand_val].cpu()),
                            'masks': {
                                'true': wandb.Image(val_mask[rand_val].argmax(dim=0).float().cpu()),
                                'pred': wandb.Image(torch.softmax(val_pred[rand_val], dim=0).argmax(dim=0).float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        # if save_checkpoint:
        #     Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
        #     torch.save(net.state_dict(), str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
        #     logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=6, help='Batch size')
    parser.add_argument('--val_batch-size', '-vb', dest='val_batch_size', metavar='VB', type=int, default=12, help='Validation batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=4, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  val_batch_size=args.val_batch_size,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  amp=args.amp)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise