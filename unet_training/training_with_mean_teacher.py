from copy import deepcopy
from tempfile import gettempdir

import torch
from clearml import Task, InputModel
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from tqdm import trange

from custom_arcitecture.classic_u_net import UNet
from evaluation import clearml_model_id
from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset, MeanTeacherSegGrazPedWriDataset
from unet_training.custom_collate import create_mask_for_unlabeled_data
from unet_training.forward_func import forward_mean_teacher_bce
from unet_training.hyper_params import hp_parser

hp_parser.add_argument('--split500', type=bool, default=True,
                       help='whether to use the predefined 500 split instead of all available data')
hp_parser.add_argument('--num_train_samples', type=int, default=43,
                       help='number of training samples initial model was trained on.')
hp_parser.add_argument('--alpha', type=float, default=0.996, help='exponential moving average decay')
hp = hp_parser.parse_args()

tags = []
if hp.data_aug > 0:
    tags.append('data_aug')
if hp.lr_scheduler:
    tags.append('lr_scheduler')

student_model_id = clearml_model_id.unet_ids[hp.num_train_samples]
cl_model = InputModel(student_model_id)

task = Task.init(project_name='Kids Bone Checker/Bone segmentation/mean teacher',
                 task_name=f'Mean teacher {hp.alpha}', auto_connect_frameworks=False, tags=tags)
task.set_input_model(cl_model.id)

# init pytorch
torch.manual_seed(hp.seed)
device = torch.device(f'cuda:{hp.gpu_id}' if torch.cuda.is_available() else 'cpu')

# define data loaders
dl_kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}
train_dl = DataLoader(MeanTeacherSegGrazPedWriDataset(use_500_split=hp.split500, number_training_samples=hp.num_train_samples),
                      batch_size=hp.batch_size, shuffle=True, drop_last=True, **dl_kwargs, collate_fn=create_mask_for_unlabeled_data)
val_dl = DataLoader(LightSegGrazPedWriDataset('val'), batch_size=hp.infer_batch_size, shuffle=False, drop_last=False,
                    **dl_kwargs)

# define model
student_model = UNet.load(cl_model.get_weights(), device).to(device)
teacher_model = deepcopy(student_model).to(device)

optimizer = torch.optim.AdamW(student_model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.epochs, eta_min=hp.lr / 100)

loss_collectors = [MeanMetric().to(device), MeanMetric().to(device), MeanMetric().to(device)]

fwd_kwargs = {'student': student_model, 'teacher': teacher_model, 'optimizer': optimizer, 'device': device,
              'loss_collectors': loss_collectors, 'data_aug': hp.data_aug, 'alpha': hp.alpha,
              'bce_pos_weight': LightSegGrazPedWriDataset.POS_CLASS_WEIGHT.view(-1, 1, 1).expand(-1, 384, 224).to(
                  device)}

for epoch in trange(hp.epochs, desc='training'):
    forward_mean_teacher_bce('train', train_dl, epoch, **fwd_kwargs)
    forward_mean_teacher_bce('val', val_dl, epoch, **fwd_kwargs)

    if hp.lr_scheduler:
        scheduler.step()
        # log learning rate
        task.get_logger().report_scalar(title='Learning rate', series='lr', value=scheduler.get_last_lr()[0],
                                        iteration=epoch)

# save model to ClearML
save_path = gettempdir() + '/teacher.pth'
teacher_model.save(save_path)
task.update_output_model(save_path, model_name='teacher')
task.close()
