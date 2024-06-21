# Evaluates the UNet with different number of training samples.

import pandas as pd
import torch
from clearml import InputModel
from tqdm import tqdm

import clearml_model_id
from custom_arcitecture.classic_u_net import UNet
from custom_arcitecture.lraspp import LRASPPOnSAM
from scripts.dental_dataset import DentalDataset
from utils.dice_coefficient import multilabel_dice
from utils.seg_refinement import SAMSegRefiner, SegEnhance, RndWalkSegRefiner

# parameters
architecture = 'SAM_LRASPP'
refinement = 'raw'
print(f'Architecture: {architecture}, refinement: {refinement}')

ds = DentalDataset('test')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model_id = {
    'UNet': clearml_model_id.dental_models['unet_45_lbl'],
    'UNet_fully_supervised': clearml_model_id.dental_models['unet_all_lbl'],
    'UNet_pseudo_lbl_raw': clearml_model_id.dental_models['unet_raw_pseudo_lbl'],
    'UNet_pseudo_lbl_sam': clearml_model_id.dental_models['unet_sam_pseudo_lbl'],
    'SAM_LRASPP': clearml_model_id.dental_models['sam_lraspp'],
    'UNet_mean_teacher': clearml_model_id.dental_models['mean_teacher'],
    'UNet_mean_teacher_sam_selection': clearml_model_id.dental_models['mean_teacher_sam_selection'],
}[architecture]

if architecture != 'UNet' and refinement != 'raw':
    raise ValueError('Refinement should only used for initial UNet.')

refinement_func = {
    'raw': lambda mask, _: (mask > 0.5, None),
    'SAM': lambda mask, file: SegEnhance(
        SAMSegRefiner('SAM', device, [['box'], ['pos_points', 'neg_points']]), 'highest_probability', 'dilation',
        'square',
        8, device).enhance(mask, file),
    'MedSAM': lambda mask, file: SegEnhance(
        SAMSegRefiner('MedSAM', device, ['box']), 'highest_probability', 'dilation', 'square',
        8, device).enhance(mask, file),
    'RndWalk': lambda mask, file: SegEnhance(
        RndWalkSegRefiner(1, 19.86974677799124), 'highest_probability',
        'erosion', 'disk', 0, device).enhance(mask, file)
}

df = pd.DataFrame()
cl_model = InputModel(model_id)
if architecture.startswith('UNet'):
    model = UNet.load(cl_model.get_weights(), device).to(device).eval()
elif architecture.startswith('SAM'):
    model = LRASPPOnSAM.load(cl_model.get_weights(), device).to(device).eval()
else:
    raise ValueError('Unknown architecture.')

dsc = []
for img, y, file_stem in tqdm(ds, unit='img'):
    img, y = img.to(device), y.to(device)

    y = y.unsqueeze(0).bool()
    # forward
    with torch.inference_mode():
        x = (img - ds.IMG_MEAN) / ds.IMG_STD
        y_hat = model(x.unsqueeze(0)).squeeze(0)
        y_hat = torch.sigmoid(y_hat)
    y_hat = y_hat.clone()
    # refinement
    refined_sam_masks, _ = refinement_func[refinement](y_hat, file_stem)

    dsc.append(multilabel_dice(refined_sam_masks.unsqueeze(0).bool(), y))
    value_dict = {
        'Method': architecture + '_' + refinement if architecture == 'UNet' else architecture,
        'DSC mean': dsc[-1].nanmean().item(),
        'File stem': file_stem
    }
    value_dict.update({'DSC ' + lbl: dsc[-1][0][i].item() for i, lbl in enumerate(ds.CLASS_LABEL)})
    df = pd.concat([df, pd.DataFrame(value_dict, index=[len(df)])], ignore_index=True)

dsc = torch.cat(dsc)
print(f'{architecture} {refinement}: {dsc.nanmean().item()}')

df.to_csv(f'evaluation/csv_results/dental/{architecture}_{refinement}_dsc.csv', index=False)
