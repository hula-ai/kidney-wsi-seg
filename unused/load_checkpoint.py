from mmdet.apis import init_detector

# Specify the path to model config and checkpoint file
config_file = '/home/cougarnet.uh.edu/srizvi7/Desktop/mmdetection/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_1x_coco_HULA_compartment.py'
checkpoint_file = '/data/syed/mmdet/run11_swin_multi_gpu_autoaug/epoch_2.pth'

# build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:5')
print(model)  # Successfully loaded
