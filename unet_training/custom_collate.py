import torch


def stack_two_ds_collate(batch):
    x1, y1, files_name1 = zip(*[b['gt'] for b in batch])
    x2, y2, files_name2 = zip(*[b['pseudo_lbl'] for b in batch])

    x = torch.stack(x1 + x2, dim=0)
    y = torch.stack(y1 + y2, dim=0)
    files_name = list(files_name1 + files_name2)

    return x, y, files_name


if __name__ == '__main__':
    from scripts.seg_grazpedwri_dataset import *
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt

    ds1 = LightSegGrazPedWriDataset('train')
    ds2 = SavedSegGrazPedWriDataset('data/seg_masks/self_404bd577195044749a1658ecd76912f7.h5')
    ds = CombinedSegGrazPedWriDataset(ds1, ds2)

    dl = DataLoader(ds, batch_size=4, shuffle=True, drop_last=True, collate_fn=stack_two_ds_collate)
    x, y, files_name = next(iter(dl))
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
