import optuna
import torch
from clearml import InputModel

from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset
from seg_preprocessing.segmentation_preprocessing import opening_with_connected_component
from unet.classic_u_net import UNet
from utils.dice_coefficient import multilabel_dice
from joblib import dump

model_id = '0427c1de20c140c5bff7284c7a4ae614'  # initial training
device = 'cuda:4' if torch.cuda.is_available() else 'cpu'


def objective(trial: optuna.Trial):
    structering_element = trial.suggest_categorical("structuring_element", ["square", "disk", "diamond", "star"])
    radius = trial.suggest_int("radius", 0, 8)
    selection = trial.suggest_categorical("selection", ["largest", "highest_probability", None])
    trial.set_user_attr("num_iter", 250)

    dsc_init = []
    dsc_refined = []
    for img, y, file_name in ds:
        img, y = img.to(device), y.to(device)
        y = y.unsqueeze(0).bool()
        # forward
        with torch.inference_mode():
            x = (img - ds.IMG_MEAN) / ds.IMG_STD
            y_hat = model(x.unsqueeze(0)).squeeze(0)
            y_hat = torch.sigmoid(y_hat)

        y_refined = opening_with_connected_component(y_hat, structering_element, radius, trial.user_attrs["num_iter"],
                                                     selection)

        y_hat = y_hat > 0.5
        y_refined = y_refined > 0.5
        dsc_init.append(multilabel_dice(y_hat.unsqueeze(0), y))
        dsc_refined.append(multilabel_dice(y_refined.unsqueeze(0), y))

    dsc_init = torch.cat(dsc_init, dim=0)
    dsc_refined = torch.cat(dsc_refined, dim=0)

    dsc_diff = dsc_refined - dsc_init
    dsc_diff_score = dsc_diff.nanmean().item()

    return dsc_diff_score


cl_model = InputModel(model_id)
model = UNet.load(cl_model.get_weights(), 'cpu').eval()
model.to(device)
ds = LightSegGrazPedWriDataset('val')

search_space = {
    "structuring_element": ["square", "disk", "diamond", "star"],
    "radius": list(range(9)),
    "selection": ["largest", "highest_probability", None]
}
study = optuna.create_study(direction="maximize", study_name="HPO segmentation preprocessing",
                            sampler=optuna.samplers.GridSampler(search_space))
study.set_user_attr("clearml_model_id", model_id)

study.optimize(objective, n_trials=200)
print(study.best_params, study.best_value)

dump(study, f"seg_preprocessing/hpo_seg_proc_{model_id}.pkl")