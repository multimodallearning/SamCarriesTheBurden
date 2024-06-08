import optuna
from clearml import InputModel
from joblib import dump
from torch.cuda import amp
from pathlib import Path

from custom_arcitecture.classic_u_net import UNet
from scripts.dental_dataset import DentalDataset
from segment_anything.sam_mask_decoder_head import SAMMaskDecoderHead
from utils.dice_coefficient import multilabel_dice
from utils.seg_refinement import SAMSegRefiner, SegEnhance
from utils.segmentation_preprocessing import *


@torch.inference_mode()
def objective(trial: optuna.Trial):
    prompt_choices = ["box", "pos_points neg_points", "pos_points"]
    prompts2use1st = trial.suggest_categorical('prompts2use1st', prompt_choices).split()
    prompts2use2nd = trial.suggest_categorical('prompts2use2nd', prompt_choices + [None])

    if prompts2use2nd is None:
        prompts2use = prompts2use1st
    else:
        prompts2use = [prompts2use1st, prompts2use2nd.split()]

    sam_refiner = SAMSegRefiner('SAM', device, prompts2use)
    seg_processor = SegEnhance(sam_refiner,
                               "highest_probability",
                               trial.suggest_categorical('morph_op', ['erosion', 'dilation']),
                               trial.suggest_categorical('structuring_element', ['square', 'disk', 'diamond', 'star']),
                               trial.suggest_int('radius', 0, 16),
                               device)

    dsc_unet = []
    dsc_sam = []
    for img, y, file_name in ds:
        img, y = img.to(device), y.to(device)
        y = y.unsqueeze(0).bool()
        # forward
        x = (img - ds.IMG_MEAN) / ds.IMG_STD
        y_hat = model(x.unsqueeze(0)).squeeze(0)
        y_hat = torch.sigmoid(y_hat)

        with amp.autocast():
            refined_sam_masks, _ = seg_processor.enhance(y_hat, file_name)

        unet_mask = y_hat > 0.5
        dsc_unet.append(multilabel_dice(unet_mask.unsqueeze(0), y))
        dsc_sam.append(multilabel_dice(refined_sam_masks.unsqueeze(0), y))

    dsc_unet = torch.cat(dsc_unet, dim=0)
    dsc_sam = torch.cat(dsc_sam, dim=0)

    dsc_diff = dsc_sam - dsc_unet
    dsc_diff_score = dsc_diff.nanmean().item()

    return dsc_diff_score


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_id = 'fff060f575994796936422b8c2819c5e'
cl_model = InputModel(model_id)
model = UNet.load(cl_model.get_weights(), device).eval()
model.to(device)
ds = DentalDataset('val')

sam_checkpoint = "data/sam_vit_h_4b8939.pth"
sam_model_type = "vit_h"
img_embedding_h5 = "data/dental_sam_img_embedding.h5"
sam_predictor = SAMMaskDecoderHead(sam_checkpoint, sam_model_type, device, img_embedding_h5)

search_space = {
    'prompts2use1st': ["box", "pos_points neg_points", "pos_points"],
    'prompts2use2nd': ["box", "pos_points neg_points", "pos_points", None],
    'structuring_element': ['square', 'disk', 'diamond', 'star'],
    'radius': list(range(12)),
    'morph_op': ['erosion', 'dilation']
}
study = optuna.create_study(direction="maximize", study_name=f"SAM refinement study for {model_id}",
                            sampler=optuna.samplers.TPESampler())
study.set_user_attr("model_id", model_id)

study.optimize(objective, n_trials=200)
print(study.best_params, study.best_value)

save_path = Path(f"seg_processing/hpo_results/{model_id}")
save_path.mkdir(exist_ok=True, parents=True)
dump(study, save_path / "grid_search_sam_refine.pkl")
