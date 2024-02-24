from pathlib import Path

import optuna
from clearml import InputModel
from joblib import dump

from custom_arcitecture.classic_u_net import UNet
from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset
from utils.dice_coefficient import multilabel_dice
from utils.seg_refinement import RndWalkSegRefiner, SegEnhance
from utils.segmentation_preprocessing import *


@torch.inference_mode()
def objective(trial: optuna.Trial):
    rnd_refiner = RndWalkSegRefiner(
        trial.suggest_int('background_erosion_radius', 1, 16),
        trial.suggest_float('laplace_sigma', 1, 20),
    )
    seg_processor = SegEnhance(rnd_refiner,
                               "highest_probability",
                               'erosion',
                               'disk',
                               trial.suggest_int('radius', 0, 16),
                               device)

    dsc_unet = []
    dsc_rnd_walk = []
    for img, y, file_name in ds:
        img, y = img.to(device), y.to(device)
        y = y.unsqueeze(0).bool()
        # forward
        x = (img - ds.IMG_MEAN) / ds.IMG_STD
        y_hat = model(x.unsqueeze(0)).squeeze(0)
        y_hat = torch.sigmoid(y_hat)

        refined_masks = seg_processor.enhance(y_hat, file_name)

        unet_mask = y_hat > 0.5
        dsc_unet.append(multilabel_dice(unet_mask.unsqueeze(0), y))
        dsc_rnd_walk.append(multilabel_dice(refined_masks.unsqueeze(0), y))

    dsc_unet = torch.cat(dsc_unet, dim=0)
    dsc_rnd_walk = torch.cat(dsc_rnd_walk, dim=0)

    dsc_diff = dsc_rnd_walk - dsc_unet
    dsc_diff_score = dsc_diff.nanmean().item()

    return dsc_diff_score


device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
model_id = '2bd2f4be80b9446286416993ba6a87c1'  # initial training
cl_model = InputModel(model_id)
model = UNet.load(cl_model.get_weights(), device).eval()
model.to(device)
ds = LightSegGrazPedWriDataset('val')

study = optuna.create_study(direction="maximize", study_name=f"RndWalk refinement study for {model_id}")
study.set_user_attr("model_id", model_id)

study.optimize(objective, n_trials=200, catch=ValueError)
print(study.best_params, study.best_value)

save_path = Path(f"seg_processing/hpo_results/{model_id}")
save_path.mkdir(exist_ok=True, parents=True)
dump(study, save_path / "gpo_rnd_wlk_refine.pkl")
