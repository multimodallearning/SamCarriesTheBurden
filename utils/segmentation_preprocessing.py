import torch
from kornia.contrib import connected_components


def remove_all_but_largest_connected_component(segmentation_mask: torch.Tensor, num_iter: int) -> torch.Tensor:
    """
    Removes all but the largest connected component from a binary mask.
    Args:
        segmentation_mask: segmentation mask of shape (C, H, W)
        num_iter: number of iterations for connected components labeling. Should be set to the longest size of the
                    biggest connected component.

    Returns: refined_mask: refined mask of shape (C, H, W)
    """

    assert segmentation_mask.ndim == 3, "segmentation_mask should be 3D tensor of shape (C, H, W)"

    lbl = connected_components(segmentation_mask.unsqueeze(1).float(), num_iterations=num_iter)  # (C, 1, H, W)
    refined_mask = torch.zeros_like(segmentation_mask, dtype=torch.bool)
    for class_idx, component_map in enumerate(lbl):
        # segmentation mask is empty
        if not component_map.any():
            continue

        # supress all but the largest connected component
        components = component_map.unique()  # (num_components)
        # remove background component
        components = components[components != 0]
        component_areas = (component_map == components.view(-1, 1, 1)).sum((1, 2))  # (num_components)
        largest_component_idx = components[component_areas.argmax()]

        refined_mask[class_idx] = component_map == largest_component_idx

    return refined_mask