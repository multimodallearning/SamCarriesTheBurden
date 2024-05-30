import argparse
from pathlib import Path
from tempfile import gettempdir

import torch
from clearml import Task, InputModel
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from tqdm import trange

from custom_arcitecture.classic_u_net import UNet
from scripts.dental_dataset import DentalDataset, SavedDentalDataset
from unet_training.forward_func import forward_bce
from unet_training.hyper_params import hp_parser

hp_parser.add_argument('--train_from_scratch', default=True, action=argparse.BooleanOptionalAction,
                       help='whether to train from scratch')
hp_parser.add_argument('--split500', default=True, action=argparse.BooleanOptionalAction,
                       help='whether to use the predefined 500 split instead of all available data')
hp_parser.add_argument('--pseudo_label', choices=['raw', 'sam', 'nnunet'], help='pseudo label method')
hp_parser.add_argument('--prompt1st', type=str, nargs='*', default=None,
                       help='first prompts to use for SAM pseudo label')
hp_parser.add_argument('--prompt2nd', type=str, nargs='*', default=None,
                       help='second prompts to use for SAM pseudo label')
hp_parser.add_argument('--num_train_samples', type=int, default=43,
                       help='number of training samples initial model was trained on.')
hp = hp_parser.parse_args()

tags = []
if hp.data_aug > 0:
    tags.append('data_aug')
if hp.lr_scheduler:
    tags.append('lr_scheduler')
if not hp.train_from_scratch:
    tags.append('fine_tuning')

if hp.pseudo_label == 'sam':
    task_name = 'SAM ' + str.join('_', hp.prompt1st) + '_refine_' + str.join('_',
                                                                             hp.prompt2nd) + f'_num_train_{hp.num_train_samples}'
else:
    task_name = hp.pseudo_label + f'_num_train_{hp.num_train_samples}'

initial_trained_model_id = 'fff060f575994796936422b8c2819c5e'
cl_model = InputModel(initial_trained_model_id)

task = Task.init(project_name='Kids Bone Checker/Bone segmentation/dental pseudo label training',
                 task_name=task_name, auto_connect_frameworks=False, tags=tags)
task.set_input_model(cl_model.id)

# init pytorch
torch.manual_seed(hp.seed)
device = torch.device(f'cuda:{hp.gpu_id}' if torch.cuda.is_available() else 'cpu')

# define data loaders
saved_seg_path = Path('data/seg_masks')
if hp.pseudo_label == 'nnunet':
    saved_seg_path = saved_seg_path.joinpath('SegGraz_nnunet_predictions.h5')
elif hp.pseudo_label == 'raw':
    saved_seg_path /= initial_trained_model_id
    saved_seg_path /= 'raw_segmentations_450.h5'
elif hp.pseudo_label == 'sam':
    saved_seg_path /= initial_trained_model_id
    saved_seg_path /= f'sam_{str.join('_', hp.prompt1st) + '_refine_' + str.join('_', hp.prompt2nd)}_450.h5'
dl_kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
train_dl = DataLoader(SavedDentalDataset(str(saved_seg_path)), batch_size=hp.batch_size, shuffle=True, drop_last=True,
                      **dl_kwargs)
val_dl = DataLoader(DentalDataset('val'), batch_size=hp.infer_batch_size, shuffle=False, drop_last=False,
                    **dl_kwargs)

# define model
if hp.train_from_scratch:
    model = UNet(1, train_dl.dataset.N_CLASSES).to(device)
else:
    model = UNet.load(cl_model.get_weights(), device).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.epochs, eta_min=hp.lr / 100)

loss_collector = MeanMetric().to(device)

fwd_kwargs = {'model': model, 'optimizer': optimizer, 'device': device, 'loss_collector': loss_collector,
              'data_aug': hp.data_aug,
              'bce_pos_weight': DentalDataset.BCE_POS_WEIGHTS.to(device)}

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
