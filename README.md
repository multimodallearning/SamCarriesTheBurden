# Can SAM Teach How to Segment Medical Radiographs?

This repository contains code from the [official SAM repository](https://github.com/facebookresearch/segment-anything). Please give them credit for sharing their amazing work.

This recipie will explain how to reproduce the results exemplary for the GrazPedWri Dataset.
For the other dataset, please refer to the branch 'addional_ds' and for the implementation of the comparison methods, please refer to the branch 'master'.
We quite heavily rely on Clear-ML to manage and store our different models.
While it is possible, to rewrite the code to not use Clear-ML and store the models locally, we recommend to use Clear-ML since their free-plan is sufficient to reproduce our results.

## Environment

Please use the provided yaml (environment.yml) file to create the environment.

```bash
conda env create -f environment.yml
```

## GrazPedWriDataset
Please download the dataset using the provided link in the original [paper](https://www.nature.com/articles/s41597-022-01328-z) and preprocess it with their provided notebooks to obtain the 8-bit images. After this, please use our provided preprocessing script `scripts/copy_and_process_graz_imgs` to create the homogeneous dataset (all images flipped to left).
Our human experts segmentation mask of 64 representative images were annotated in CVAT and are stored in the 'annotations_*.xml' files.
The decoding is done by our custom `utils.cvat_parser` used by our PyTorch dataset implementation `scripts.se_gratpedwri_dataset.py`.

## Proposed pipeline
To train our initial U-Net $f_\theta$, run 'python -m unet_training.training --gpu_id 0' leaving all the hyperparameters on default.
Next, we predict our initial, unrefined segmentation masks for our unlabelled data by using 'scripts.save_segmentations', where you have to adjust the model id in line 19 to your clear-ml model id (in the clearml experiments: 'artefacts/output models/bone_segmentator').

To use SAM to refine the segmentation masks, we have to set up two things first:
+ Download the model checkpoint for the ViT-H from the [official repository](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints) and place it in the 'data' folder.
+ precompute SAM's image embedding to speed up the refinement process (believe me, you will run SAM on the same image at one point and you will be happy to have the embeddings precomputed). To do so, run 'scripts.save_refined_segmentations. Please use the same model id as before to load the correct initial segmentations.

As a last step, we can train the final U-Net $f_\varphi$ on the refined segmentations. To do so, run 'python -m unet_training.training_on_pseudo_labels --gpu_id 2 --pseudo_label sam --prompt1st box --prompt2nd pos_points neg_points --num_train_samples 43' leaving all others hyperparameters on default.
To evaluate the final model, you can reuse the 'scripts.save_segmentations' script and adjust the model id to the final model id.
