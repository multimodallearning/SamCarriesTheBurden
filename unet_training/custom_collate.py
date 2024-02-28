import torch


def stack_two_ds_collate(batch):
    x1, y1, files_name1 = zip(*[b['gt'] for b in batch])
    x2, y2, files_name2 = zip(*[b['pseudo_lbl'] for b in batch])

    x = torch.stack(x1 + x2, dim=0)
    y = torch.stack(y1 + y2, dim=0)
    files_name = list(files_name1 + files_name2)

    return x, y, files_name

def create_mask_for_unlabeled_data(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    filenames = [b[2] for b in batch]

    gt_available_mask = torch.tensor([b[1] is not None for b in batch], dtype=torch.bool)
    if any(gt_available_mask):
        y = torch.stack([b[1] for b in batch if b[1] is not None], dim=0)
    else:
        y = None
    return x, y, filenames, gt_available_mask


if __name__ == '__main__':
    from scripts.seg_grazpedwri_dataset import *
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt

    ds = MeanTeacherSegGrazPedWriDataset(use_500_split=True)

    dl = DataLoader(ds, batch_size=4, shuffle=True, drop_last=True, collate_fn=create_mask_for_unlabeled_data)
    x, y, files_name, gt_available_mask = next(iter(dl))
    for img, seg, file in zip(x, y, files_name):
        fig, axs = plt.subplots(1, 2)
        axs[0].set_title(file)
        axs[0].imshow(img.squeeze().numpy(), cmap='gray')
        seg_mask = seg.any(dim=0).float()
        axs[1].imshow(img.squeeze().numpy(), cmap='gray')
        axs[1].imshow(seg.argmax(0).numpy(), alpha=seg_mask.numpy())
        plt.show()

    for i in range(ds.N_CLASSES):
        fig, axs = plt.subplots(2, 4)
        fig.suptitle(ds.BONE_LABEL[i])
        axs = axs.flatten()
        for ax, (img, seg, file) in zip(axs, zip(x, y, files_name)):
            ax.imshow(img.squeeze().numpy(), cmap='gray')
            seg_mask = seg[i].float()
            ax.imshow(seg_mask.numpy(), alpha=seg_mask.numpy())
        plt.show()
