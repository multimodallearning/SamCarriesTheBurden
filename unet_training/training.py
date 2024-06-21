from tempfile import gettempdir

import torch
from clearml import Task
from torch.utils.data import DataLoader, RandomSampler
from torchmetrics import MeanMetric
from tqdm import trange

from custom_arcitecture.classic_u_net import UNet
from custom_arcitecture.lraspp import LRASPPOnSAM
from scripts.dental_dataset import DentalDataset
from unet_training.forward_func import forward_bce
from unet_training.hyper_params import hp_parser

hp_parser.add_argument('--architecture', default='unet', choices=['unet', 'lraspp_on_sam'],
                       help='which architecture to use')
hp_parser.add_argument('--data_sample_per_epoch', type=int, default=48,
                       help='number of samples per epoch. Used for bootstrapping.')
hp_parser.add_argument('--num_train_samples', type=int, default=-1,
                       help='number of training samples to use. -1 means all samples.')
hp = hp_parser.parse_args()

tags = ['instance_norm', 'bootstrap']
if hp.data_aug > 0:
    tags.append('data_aug')
if hp.lr_scheduler:
    tags.append('lr_scheduler')
if hp.architecture == 'lraspp_on_sam':
    tags.append('SAM')
task = Task.init(project_name='Kids Bone Checker/Bone segmentation/dental',
                 task_name=f'initial on {'all' if hp.num_train_samples == -1 else hp.num_train_samples} training data',
                 auto_connect_frameworks=False, tags=tags)
# init pytorch
torch.manual_seed(hp.seed)
device = torch.device(f'cuda:{hp.gpu_id}' if torch.cuda.is_available() else 'cpu')

# define data loaders
dl_kwargs = {'num_workers': 0, 'pin_memory': True} if torch.cuda.is_available() else {}
# bootstrap training set
ds_train = DentalDataset('train', number_training_samples=hp.num_train_samples if hp.num_train_samples != -1 else 'all')
train_dl = DataLoader(ds_train, batch_size=hp.batch_size, drop_last=True, **dl_kwargs,
                      sampler=RandomSampler(ds_train, replacement=True, num_samples=hp.data_sample_per_epoch))
val_dl = DataLoader(DentalDataset('val'), batch_size=hp.infer_batch_size, shuffle=False, drop_last=False,
                    **dl_kwargs)

# define model
n_classes = train_dl.dataset.N_CLASSES
if hp.architecture == 'unet':
    model = UNet(1, n_classes, n_last_channel=hp.n_last_channel)
elif hp.architecture == 'lraspp_on_sam':
    model = LRASPPOnSAM(n_classes=n_classes, n_last_channel=hp.n_last_channel)
else:
    raise NotImplementedError('Unknown architecture')

model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.epochs, eta_min=hp.lr / 100)

loss_collector = MeanMetric().to(device)

fwd_kwargs = {'model': model, 'optimizer': optimizer, 'device': device, 'loss_collector': loss_collector,
              'data_aug': hp.data_aug,
              'bce_pos_weight': train_dl.dataset.BCE_POS_WEIGHTS.to(device)}

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
