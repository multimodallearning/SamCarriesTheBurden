import torch
from clearml import InputModel
from tqdm import tqdm

from custom_arcitecture.classic_u_net import UNet
from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset
from utils.dice_coefficient import multilabel_dice
from utils.seg_refinement import SAMSegRefiner, SegEnhance, RndWalkSegRefiner
import pandas as pd
import clearml_model_id

# parameters
architecture = 'unet'
refinement = 'MedSAM'

ds = LightSegGrazPedWriDataset('test')
device = 'cuda:4' if torch.cuda.is_available() else 'cpu'

model_dict = {
    'unet': clearml_model_id.unet_ids
}[architecture]

refinement_func = {
    'raw': lambda mask, _: (mask > 0.5, None),
    'SAM': lambda mask, file: SegEnhance(
        SAMSegRefiner('SAM', device, [['box'], ['pos_points', 'neg_points']]), 'highest_probability', 'dilation', 'square',
        8, device).enhance(mask, file),
    'MedSAM': lambda mask, file: SegEnhance(
        SAMSegRefiner('MedSAM', device, ['box']), 'highest_probability', 'dilation', 'square',
        8, device).enhance(mask, file),
    'RndWalk': lambda mask, file: SegEnhance(
        RndWalkSegRefiner(1, 19.86974677799124), 'highest_probability',
        'erosion', 'disk', 0, device).enhance(mask, file)
}

df = pd.DataFrame(columns=['architecture', 'refinement', 'num_train', 'dsc', 'file_stem'])
for num_train in model_dict.keys():
    cl_model = InputModel(model_dict[num_train])
    model = UNet.load(cl_model.get_weights(), device).to(device, non_blocking=True).eval()

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
        df = pd.concat([df, pd.DataFrame({
            'architecture': architecture,
            'refinement': refinement,
            'num_train': num_train,
            'dsc': dsc[-1].nanmean().item(),
            'file_stem': file_stem
        }, index=[len(df)])], ignore_index=True)

    dsc = torch.cat(dsc)
    print(f'{architecture} {refinement} {num_train} samples: {dsc.nanmean().item()}')

df.to_csv(f'evaluation/csv_results/{architecture}_{refinement}_dsc.csv', index=False)
