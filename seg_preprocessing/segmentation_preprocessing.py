import torch
from kornia.contrib import connected_components
from kornia.morphology import erosion, dilation
from skimage.morphology import disk, square, diamond, star


def remove_all_but_one_connected_component(prob_mask: torch.Tensor, selection: str,
                                           num_iter: int) -> torch.Tensor:
    """
    Removes all but the one connected component from a probability mask.
    Args:
        prob_mask: segmentation mask in range [0, 1] of shape (C, H, W)
        selection: method to select the connected component. Options: 'largest', 'highest_probability'
        num_iter: number of iterations for connected components labeling. Should be set to the longest size of the
                    biggest connected component.

    Returns: refined_mask: refined probability mask of shape (C, H, W)
    """

    assert prob_mask.ndim == 3, "segmentation_mask should be 3D tensor of shape (C, H, W)"
    assert prob_mask.dtype == torch.float, "segmentation_mask should be float tensor"

    bin_mask = (prob_mask > 0.5).float()
    lbl = connected_components(bin_mask.unsqueeze(1), num_iterations=num_iter).int()  # (C, 1, H, W)
    refined_mask = torch.zeros_like(prob_mask)
    for class_idx, component_map in enumerate(lbl):
        # segmentation mask is empty
        if not component_map.any():
            continue

        components = component_map.unique()  # (num_components)
        # remove background component
        components = components[components != 0]  # (num_components - 1)
        # selection
        if selection == 'largest':
            # select the largest connected component
            component_areas = (component_map == components.view(-1, 1, 1)).sum((1, 2))  # (num_components)
            winner_idx = components[component_areas.argmax()]
        elif selection == 'highest_probability':
            # select the connected component with the highest average probability
            prob_map = prob_mask[class_idx].unsqueeze(0)  # (1, H, W)
            component_bool_idx = component_map == components.view(-1, 1, 1)  # (num_components, H, W)
            component_areas = component_bool_idx.sum((1, 2))  # (num_components)
            component_prob = prob_map * component_bool_idx.float()  # (num_components, H, W)
            component_prob = component_prob.sum((1, 2)) / component_areas  # (num_components)
            winner_idx = components[component_prob.argmax()]
        else:
            raise NotImplementedError(f"Invalid selection: {selection}")

        refined_mask[class_idx] = component_map == winner_idx
    refined_mask *= prob_mask
    return refined_mask


def erode_mask_with_disc_struct(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """
    Applies erosion to a binary mask using a disk structuring element.
    Args:
        mask: binary mask of shape (C, H, W)
        radius: radius of the disk structuring element

    Returns: eroded_mask: eroded mask of shape (H, W)
    """

    assert mask.ndim == 3, "mask should be 3D tensor of shape (C, H, W)"
    assert radius > 0, "radius should be greater than 0"

    kernel = torch.from_numpy(disk(radius, dtype=int))
    eroded_mask = erosion(mask.unsqueeze(0).float(), kernel).squeeze().bool()

    return eroded_mask


def opening_with_connected_component(prob_mask: torch.Tensor, structuring_element: str, radius: int, num_iter: int,
                                     selection: str) -> torch.Tensor:
    """
    Applies opening to a probability mask using a structuring element and then removes all but the one connected
    component, which is selected based on the selection method.
    Args:
        prob_mask: segmentation mask in range [0, 1] of shape (C, H, W)
        structuring_element: method to select the structuring element. Options: 'square', 'disk', 'diamond', 'star'
        radius: radius of the structuring element. Choose 0 for identity mapping.
        num_iter: number of iterations for connected components labeling. Should be set to the longest size of the
                    biggest connected component.
        selection: selection method for the connected component. Options: 'largest', 'highest_probability'.

    Returns: processed_mask: processed probability mask of shape (C, H, W)

    """
    assert prob_mask.ndim == 3, "prob_mask should be 3D tensor of shape (C, H, W)"

    try:
        struct = {
            'square': square,
            'disk': disk,
            'diamond': diamond,
            'star': star
        }[structuring_element]
    except KeyError:
        raise NotImplementedError(f"Invalid structuring element: {structuring_element}")

    # handling identity for square structuring element
    if radius == 0 and structuring_element == 'square':
        radius = 1

    binary_mask = prob_mask > 0.5
    kernel = torch.from_numpy(struct(radius, dtype=int)).to(prob_mask)
    # Erosion
    eroded_mask = erosion(binary_mask.unsqueeze(0).float(), kernel, engine='convolution').squeeze(0)
    # Connected component
    if selection is not None:
        prob_mask = prob_mask * eroded_mask
        eroded_mask = remove_all_but_one_connected_component(prob_mask, selection, num_iter)
        eroded_mask = eroded_mask > 0.5
    # Dilation
    opened_mask = dilation(eroded_mask.unsqueeze(0).float(), kernel, engine='convolution').squeeze(0)
    refined_mask = opened_mask * prob_mask

    return refined_mask


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    # Example usage
    init = torch.zeros(2, 256, 256)
    init[0, 100:150, 100:150] = 0.6
    init[0, 200:220, 200:220] = 0.9
    init[1, 100:150, 100:150] = 0.8
    init[1, 200:220, 200:220] = 0.6
    processed = opening_with_connected_component(init, structuring_element='star', radius=10, num_iter=256,
                                                 selection=None)

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(init[0].numpy())
    axs[1].imshow(processed[0].numpy())

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(init[1].numpy())
    axs[1].imshow(processed[1].numpy())

    plt.show()
