import argparse
from tempfile import gettempdir

import torch
from clearml import Task, InputModel
from kornia import augmentation as K
from torch.utils.data import DataLoader
from tqdm import trange

from unet_training.forward_func import forward_bce
from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset, SavedSegGrazPedWriDataset
from custom_arcitecture.classic_u_net import UNet
from unet_training.hyper_params import hp_parser
from torchmetrics import MeanMetric
from pathlib import Path

pretrained_model_id = '0427c1de20c140c5bff7284c7a4ae614'
cl_model = InputModel(pretrained_model_id)

hp_parser.add_argument('--data_aug', default=False, action=argparse.BooleanOptionalAction,
                       help='whether to use data augmentation')
hp_parser.add_argument('--train_from_scratch', default=True, action=argparse.BooleanOptionalAction,
                       help='whether to train from scratch')
hp_parser.add_argument('--split500', type=bool, default=True,
                       help='whether to use the predefined 500 split instead of all available data')
hp_parser.add_argument('--pseudo_label', choices=['init', 'sam', 'nnunet'], help='pseudo label method')
hp = hp_parser.parse_args()

tags = []
if hp.data_aug:
    tags.append('data_aug')
if hp.lr_scheduler:
    tags.append('lr_scheduler')
task_name = 'Training' if hp.train_from_scratch else 'Fine-tuning'
task = Task.init(project_name='Kids Bone Checker/Bone segmentation',
                 task_name=task_name + f' with {hp.pseudo_label} pseudo labels',
                 auto_connect_frameworks=False, tags=tags)
task.set_input_model(cl_model.id)

# init pytorch
torch.manual_seed(hp.seed)
device = torch.device(f'cuda:{hp.gpu_id}' if torch.cuda.is_available() else 'cpu')

# data augmentation
data_aug_transform = K.container.AugmentationSequential(
    K.RandomAffine(degrees=25, translate=(0.2, 0.2), scale=(0.9, 1.1), p=0.5),
    # K.RandomGaussianNoise(std=0.1, p=0.1),
    # K.RandomGaussianBlur(kernel_size=(9, 9), sigma=(0.5, 1.0), p=0.1),
    # K.RandomBrightness((0.75, 1.25), p=0.15),
    # K.RandomContrast((0.75, 1.25), p=0.15),
    data_keys=["image", "mask"],
)

norm = K.Normalize(LightSegGrazPedWriDataset.IMG_MEAN, LightSegGrazPedWriDataset.IMG_STD)

# define data loaders
saved_seg_path = Path('data/seg_masks')
if hp.pseudo_label == 'nnunet':
    saved_seg_path = saved_seg_path.joinpath('SegGraz_nnunet_predictions.h5')
else:
    saved_seg_path = saved_seg_path.joinpath(f'{hp.pseudo_label}_{pretrained_model_id}_500.h5')
dl_kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
train_dl = DataLoader(SavedSegGrazPedWriDataset(str(saved_seg_path), use_500_split=hp.split500),
                      batch_size=hp.batch_size, shuffle=True, drop_last=True, **dl_kwargs)
val_dl = DataLoader(LightSegGrazPedWriDataset('val'), batch_size=hp.infer_batch_size, shuffle=False, drop_last=False,
                    **dl_kwargs)

# define model
if hp.train_from_scratch:
    model = UNet(1, train_dl.dataset.N_CLASSES).to(device)
else:
    model = UNet.load(cl_model.get_weights(), device).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=hp.lr_gamma)

loss_collector = MeanMetric().to(device)

fwd_kwargs = {'model': model, 'optimizer': optimizer, 'device': device, 'norm': norm, 'loss_collector': loss_collector,
              'data_aug': data_aug_transform if hp.data_aug else None,
              'bce_pos_weight': LightSegGrazPedWriDataset.POS_CLASS_WEIGHT.view(-1, 1, 1).expand(-1, 384, 224).to(
                  device)}

for epoch in trange(hp.epochs, desc='training'):
    forward_bce('train', train_dl, epoch, **fwd_kwargs)
    forward_bce('val', val_dl, epoch, **fwd_kwargs)

    if hp.lr_scheduler:
        if epoch >= hp.lr_patience:
            scheduler.step()

        # log learning rate
        task.get_logger().report_scalar(title='Learning rate', series='lr', value=scheduler.get_last_lr()[0],
                                        iteration=epoch)

# save model to ClearML
save_path = gettempdir() + '/bone_segmentator.pth'
model.save(save_path)
task.update_output_model(save_path, model_name='final_model')
task.close()
