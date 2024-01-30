import argparse
from tempfile import gettempdir

import torch
from clearml import Task
from kornia import augmentation as K
from torch.utils.data import DataLoader
from tqdm import trange

from unet_training.forward_func import forward_bce
from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset
from unet.classic_u_net import UNet
from unet_training.hyper_params import hp_parser
from torchmetrics import MeanMetric

hp_parser.add_argument('--data_aug', default=False, action=argparse.BooleanOptionalAction,
                       help='whether to use data augmentation')
hp = hp_parser.parse_args()

tags = []
if hp.data_aug:
    tags.append('data_aug')
if hp.lr_scheduler:
    tags.append('lr_scheduler')
task = Task.init(project_name='Kids Bone Checker/Bone segmentation', task_name=f'initial on training data',
                 auto_connect_frameworks=False, tags=tags)
# init pytorch
torch.manual_seed(hp.seed)
device = torch.device(f'cuda:{hp.gpu_id}' if torch.cuda.is_available() else 'cpu')

# data augmentation
data_aug_transform = K.container.AugmentationSequential(
    K.RandomAffine(degrees=25, translate=(0.2, 0.2), scale=(0.9, 1.1), p=0.5),
    #K.RandomGaussianNoise(std=0.1, p=0.1),
    #K.RandomGaussianBlur(kernel_size=(9, 9), sigma=(0.5, 1.0), p=0.1),
    #K.RandomBrightness((0.75, 1.25), p=0.15),
    #K.RandomContrast((0.75, 1.25), p=0.15),
    data_keys=["image", "mask"],
)

norm = K.Normalize(LightSegGrazPedWriDataset.IMG_MEAN, LightSegGrazPedWriDataset.IMG_STD)

# define data loaders
dl_kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}
train_dl = DataLoader(LightSegGrazPedWriDataset('train'), batch_size=hp.batch_size, shuffle=True, drop_last=True,
                      **dl_kwargs)
val_dl = DataLoader(LightSegGrazPedWriDataset('val'), batch_size=hp.infer_batch_size, shuffle=False, drop_last=False,
                    **dl_kwargs)

# define model
n_classes = train_dl.dataset.N_CLASSES
model = UNet(1, n_classes).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=hp.lr_gamma)

loss_collector = MeanMetric().to(device)

fwd_kwargs = {'model': model, 'optimizer': optimizer, 'device': device, 'norm': norm, 'loss_collector': loss_collector,
              'data_aug': data_aug_transform if hp.data_aug else None,
              'bce_pos_weight': train_dl.dataset.POS_CLASS_WEIGHT.view(-1, 1, 1).expand(-1, 384, 224).to(device)}

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
