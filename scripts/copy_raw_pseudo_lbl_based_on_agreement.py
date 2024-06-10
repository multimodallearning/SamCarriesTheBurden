import h5py
from tqdm import tqdm

model_id = '3bdbce9d81b24aa796c389cd2e188313'
agreement_threshold = 0.65

# load h5 file
src = h5py.File(f'data/seg_masks/{model_id}/raw_segmentations_450.h5', 'r')
dst = h5py.File(
    f'data/seg_masks/{model_id}/selected_pseudo_labels_450_dsc_{str(agreement_threshold).replace('.', '')}.h5', 'w')
dst.attrs['labels'] = src.attrs['labels']
dst.attrs['clearml_model_id'] = model_id

ds = src['segmentation_mask']
for img_name in tqdm(ds.keys()):
    dsc = ds[img_name].attrs['SAM_agreement_dsc']
    if dsc >= agreement_threshold:
        dst['segmentation_mask/' + img_name] = ds[img_name][()]

src.close()
dst.close()
