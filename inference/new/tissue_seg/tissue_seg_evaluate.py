import torch
import torch.nn.functional as F
from tqdm import tqdm
import torchvision
from skimage import io
from skimage import color
import matplotlib.pyplot as plt
import numpy as np

from tissue_seg_dice import multiclass_dice_coeff, dice_coeff


def evaluate(net, dataloader, device, train_mean, train_std, val_dice_best, plot_path):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    image_saver = []
    mask_saver = []
    pred_saver = []
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true = batch['image'], batch['mask']
        image_saver.append(image.clone().cpu())
        mask_saver.append(mask_true.clone().cpu())

        # normalize the images
        for id, image_ele in enumerate(image):
            image_ele = torchvision.transforms.Normalize(train_mean, train_std)(image_ele)
            image[id, :, :, :] = image_ele

        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            mask_plot = mask_pred.clone()
            pred_saver.append(mask_pred.clone().cpu())

            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...],
                                                    reduce_batch_first=False)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        final_dice = dice_score
    else:
        final_dice = dice_score / num_val_batches

    # save the plots of the masks if the performance is better than the last
    if final_dice.item() >= val_dice_best:
        print('Top performance exceeded! Saving figures...')
        id = 0
        for im_batch, mask_batch, pred_batch in zip(image_saver, mask_saver, pred_saver):
            for im_sin, mask_sin, pred_sin in zip(im_batch, mask_batch, pred_batch):
                if id > 10:
                    continue
                fig = io.imshow(color.label2rgb(np.array(torch.softmax(pred_sin, dim=0).argmax(dim=0)), np.array(im_sin.permute(1, 2, 0) * 255).astype('uint8')))
                fig.axes.get_xaxis().set_visible(False)
                fig.axes.get_yaxis().set_visible(False)
                plt.savefig(plot_path + '/val_pred_best_plot_{}.png'.format(id), dpi=1000, bbox_inches='tight')
                plt.close()
                id += 1

    return final_dice, image, mask_true, mask_plot