import argparse
from tempfile import gettempdir

import torch
from clearml import Task, InputModel
from kornia import augmentation as K
from torch.utils.data import DataLoader
from tqdm import trange

from unet_training.forward_func import forward_bce
from scripts.seg_grazpedwri_dataset import *
from unet.classic_u_net import UNet
from torchmetrics import MeanMetric
from unet_training.custom_collate import stack_two_ds_collate
import random

pretrained_model_id = '404bd577195044749a1658ecd76912f7'
cl_model = InputModel(pretrained_model_id)
hp_prev_task = Task.get_task(task_id=cl_model.task).get_parameters_as_dict(True)['Args']

hp_parser = argparse.ArgumentParser(description='training')
# settings
hp_parser.add_argument('--refinement', choices=['sam', 'self'], default='sam', help='refinement method')
hp_parser.add_argument('--gpu_id', type=int, help='gpu id to use')
hp_parser.add_argument('--seed', type=int, default=hp_prev_task['seed'], help='seed for reproducibility')

# hyperparameters
hp_parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
hp_parser.add_argument('--batch_size', type=int, default=hp_prev_task['batch_size'])
hp_parser.add_argument('--infer_batch_size', type=int, default=hp_prev_task['infer_batch_size'],
                       help='batch size during validation and testing')
hp_parser.add_argument('--weight_decay', type=float, default=hp_prev_task['weight_decay'],
                       help='weight decay used by optimizer')
hp_parser.add_argument('--epochs', type=int, default=3000, help='number of epochs for training')
hp_parser.add_argument('--data_aug', type=bool, default=hp_prev_task['data_aug'],
                       help='whether to use data augmentation')
hp = hp_parser.parse_args()

assert hp.batch_size % 2 == 0, 'batch size must be divisible by 2 due to stacking of datasets'

tags = [hp.refinement]
if hp.data_aug:
    tags.append('data_aug')
task = Task.init(project_name='Kids Bone Checker/SAM Refinement', task_name=f'{hp.refinement} refinement',
                 auto_connect_frameworks=False, tags=tags)
task.set_input_model(cl_model.id)
# init pytorch
torch.manual_seed(hp.seed)
random.seed(hp.seed)
device = torch.device(f'cuda:{hp.gpu_id}' if torch.cuda.is_available() else 'cpu')

# data augmentation
data_aug_transform = K.container.AugmentationSequential(
    K.RandomAffine(degrees=25, translate=(0.2, 0.2), scale=(0.7, 1.4), p=0.2),
    data_keys=["image", "mask"],
)

norm = K.Normalize(LightSegGrazPedWriDataset.IMG_MEAN, LightSegGrazPedWriDataset.IMG_STD)

# define data loaders
dl_kwargs = {'num_workers': 0, 'pin_memory': False} if torch.cuda.is_available() else {}
ds_gt = LightSegGrazPedWriDataset('train')
ds_pseudo = SavedSegGrazPedWriDataset(f'data/seg_masks/{hp.refinement}_{pretrained_model_id}.h5')
ds = CombinedSegGrazPedWriDataset(ds_gt, ds_pseudo)

# batch size is divided by 2 because of stacking of two datasets → full batch size is preserved
train_dl = DataLoader(ds, batch_size=hp.batch_size // 2, shuffle=True, drop_last=True, collate_fn=stack_two_ds_collate,
                      **dl_kwargs)
val_dl = DataLoader(LightSegGrazPedWriDataset('test'), batch_size=hp.infer_batch_size, shuffle=False, drop_last=False,
                    **dl_kwargs)

# define model
model = UNet.load(cl_model.get_weights(), device).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)

loss_collector = MeanMetric().to(device)

fwd_kwargs = {'model': model, 'optimizer': optimizer, 'device': device, 'norm': norm, 'loss_collector': loss_collector,
              'data_aug': data_aug_transform if hp.data_aug else None,
              'bce_pos_weight': ds_gt.POS_CLASS_WEIGHT.view(-1, 1, 1).expand(-1, 384, 224).to(device)}

for epoch in trange(hp.epochs, desc='training'):
    forward_bce('train', train_dl, epoch, **fwd_kwargs)
    forward_bce('test', val_dl, epoch, **fwd_kwargs)

# save model to ClearML
save_path = gettempdir() + '/bone_segmentator.pth'
model.save(save_path)
task.update_output_model(save_path, model_name='final_model')
