import optuna
from clearml import InputModel
from joblib import dump
from torch.nn import functional as F
from torch.cuda import amp

from scripts.seg_grazpedwri_dataset import LightSegGrazPedWriDataset
from seg_preprocessing.segmentation_preprocessing import *
from segment_anything.sam_mask_decoder_head import SAMMaskDecoderHead
from segment_anything.utils.prompt_utils import PromptExtractor
from custom_arcitecture.classic_u_net import UNet
from utils.dice_coefficient import multilabel_dice


@torch.inference_mode()
def objective(trial: optuna.Trial):
    prompt_choices = ["box", "pos_points neg_points", "pos_points"]
    prompts2use1st = trial.suggest_categorical('prompts2use1st', prompt_choices).split()
    prompts2use2nd = trial.suggest_categorical('prompts2use2nd', prompt_choices + [None])
    prompts2use2nd = prompts2use2nd.split() if prompts2use2nd is not None else None

    struct = {
        'square': square,
        'disk': disk,
        'diamond': diamond,
        'star': star
    }[trial.suggest_categorical('structuring_element', ['square', 'disk', 'diamond', 'star'])]
    radius = trial.suggest_int('radius', 0, 8)
    if struct == square and radius == 0:
        radius = 1
    kernel = torch.from_numpy(struct(radius, dtype=int)).to(device)
    morph_op = {
        'erosion': erosion,
        'dilation': dilation
    }[trial.suggest_categorical('morph_op', ['erosion', 'dilation'])]

    dsc_unet = []
    dsc_sam = []
    for img, y, file_name in ds:
        img, y = img.to(device), y.to(device)
        y = y.unsqueeze(0).bool()
        # forward
        x = (img - ds.IMG_MEAN) / ds.IMG_STD
        y_hat = model(x.unsqueeze(0)).squeeze(0)
        y_hat = torch.sigmoid(y_hat)

        # preprocessing
        preprocess_mask = y_hat.clone()
        preprocess_mask = remove_all_but_one_connected_component(preprocess_mask, 'highest_probability',
                                                                 num_iter=trial.study.user_attrs['num_iter'])
        preprocess_mask = preprocess_mask > 0.5
        with amp.autocast():
            preprocess_mask = morph_op(preprocess_mask.unsqueeze(0).float(), kernel.float(),
                                   engine='convolution').squeeze().bool()

        prompt_extractor = PromptExtractor(preprocess_mask)
        prompts = prompt_extractor.extract()

        refined_sam_masks = torch.zeros_like(y_hat, dtype=bool)
        for prompt in prompts:
            mask, mask_score, mask_prev_iter = sam_predictor.predict_mask(file_name, prompt, prompts2use1st)
            if prompts2use2nd is not None:
                mask, mask_score, _ = sam_predictor.predict_mask(file_name, prompt, prompts2use2nd, mask_prev_iter)

            mask = F.interpolate(mask.float(), size=y_hat.shape[-2:], mode='nearest-exact')
            refined_sam_masks[prompt.class_idx] = mask.squeeze()

        unet_mask = y_hat > 0.5
        dsc_unet.append(multilabel_dice(unet_mask.unsqueeze(0), y))
        dsc_sam.append(multilabel_dice(refined_sam_masks.unsqueeze(0), y))

    dsc_unet = torch.cat(dsc_unet, dim=0)
    dsc_sam = torch.cat(dsc_sam, dim=0)

    dsc_diff = dsc_sam - dsc_unet
    dsc_diff_score = dsc_diff.nanmean().item()

    return dsc_diff_score


device = 'cuda:4' if torch.cuda.is_available() else 'cpu'
model_id = '0427c1de20c140c5bff7284c7a4ae614'  # initial training
cl_model = InputModel(model_id)
model = UNet.load(cl_model.get_weights(), device).eval()
model.to(device)
ds = LightSegGrazPedWriDataset('val')

sam_checkpoint = "data/sam_vit_h_4b8939.pth"
sam_model_type = "vit_h"
img_embedding_h5 = "data/graz_sam_img_embedding.h5"
sam_predictor = SAMMaskDecoderHead(sam_checkpoint, sam_model_type, device, img_embedding_h5)

search_space = {
    'prompts2use1st': ["box", "pos_points neg_points", "pos_points"],
    'prompts2use2nd': ["box", "pos_points neg_points", "pos_points", None],
    'structuring_element': ['square', 'disk', 'diamond', 'star'],
    'radius': list(range(9)),
    'morph_op': ['erosion', 'dilation']
}
study = optuna.create_study(direction="maximize", study_name=f"SAM refinement study for {model_id}",
                            sampler=optuna.samplers.GridSampler(search_space))
study.set_user_attr("model_id", model_id)
study.set_user_attr("num_iter", 250)

study.optimize(objective, n_trials=1000)
print(study.best_params, study.best_value)

dump(study, f"seg_preprocessing/hpo_sam_refinement_{model_id}.pkl")
