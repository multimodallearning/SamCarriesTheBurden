import argparse
from tempfile import gettempdir

import torch
from clearml import Task
from kornia import augmentation as K
from torch.utils.data import DataLoader
from tqdm import trange

from unet_training.forward_func import forward_bce
from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset
from custom_arcitecture.classic_u_net import UNet
from unet_training.hyper_params import hp_parser
from torchmetrics import MeanMetric

hp_parser.add_argument('--data_aug', type=float, default=0, help='strength of affine data augmentation.')
hp_parser.add_argument('--architecture', default='unet', choices=['unet', 'lraspp_on_sam'],
                       help='which architecture to use')
hp_parser.add_argument('--lr_scheduler', default=True, action=argparse.BooleanOptionalAction,
                       help='whether to use lr scheduler')
hp = hp_parser.parse_args()

tags = []
if hp.data_aug > 0:
    tags.append('data_aug')
if hp.lr_scheduler:
    tags.append('lr_scheduler')
task = Task.init(project_name='Kids Bone Checker/Bone segmentation/hpo',
                 task_name=f'initial on training data {hp.data_aug} data aug strength; {hp.n_last_channel} dim',
                 auto_connect_frameworks=False, tags=tags)
# init pytorch
torch.manual_seed(hp.seed)
device = torch.device(f'cuda:{hp.gpu_id}' if torch.cuda.is_available() else 'cpu')

# define data loaders
dl_kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}
train_dl = DataLoader(LightSegGrazPedWriDataset('train'), batch_size=hp.batch_size, shuffle=True, drop_last=True,
                      **dl_kwargs)
val_dl = DataLoader(LightSegGrazPedWriDataset('val'), batch_size=hp.infer_batch_size, shuffle=False, drop_last=False,
                    **dl_kwargs)

# define model
n_classes = train_dl.dataset.N_CLASSES
if hp.architecture == 'unet':
    model = UNet(1, n_classes)
else:
    raise NotImplementedError('Unknown architecture')

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.epochs, eta_min=hp.lr / 100)

loss_collector = MeanMetric().to(device)

fwd_kwargs = {'model': model, 'optimizer': optimizer, 'device': device, 'loss_collector': loss_collector,
              'data_aug': hp.data_aug,
              'bce_pos_weight': train_dl.dataset.POS_CLASS_WEIGHT.view(-1, 1, 1).expand(-1, 384, 224).to(device)}

for epoch in trange(hp.epochs, desc='training'):
    forward_bce('train', train_dl, epoch, **fwd_kwargs)
    forward_bce('val', val_dl, epoch, **fwd_kwargs)

    if hp.lr_scheduler:
        scheduler.step()
        # log learning rate
        task.get_logger().report_scalar(title='Learning rate', series='lr', value=scheduler.get_last_lr()[0],
                                        iteration=epoch)

# save model to ClearML
save_path = gettempdir() + '/bone_segmentator.pth'
model.save(save_path)
task.update_output_model(save_path, model_name='final_model')
task.close()
