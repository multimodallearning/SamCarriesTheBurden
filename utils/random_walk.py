import torch
from scipy.sparse import csr_matrix
import pyamg
from torch.nn import functional as F
from utils import segmentation_preprocessing


def laplace_matrix(img):
    if not img.dtype is torch.float:
        raise TypeError('an image of type float is expected.')
    # given: parameters and image dimensions
    sigma = 10#7.5
    lambda_ = 1
    H, W = img.size()
    # given: create 1D index vector
    ind = torch.arange(H * W).view(H, W)
    # given: select left->right neighbours
    ii = torch.cat((ind[:, 1:].reshape(-1, 1), ind[:, :-1].reshape(-1, 1)), 1)
    val = torch.exp(-(img.take(ii[:, 0]) - img.take(ii[:, 1])) ** 2 / sigma ** 2)

    # given: create first part of neigbourhood matrix (similar to setFromTriplets in Eigen)
    A = torch.sparse.FloatTensor(ii.t(), val, torch.Size([H * W, H * W]))
    # given: select up->down neighbours
    ii = torch.cat((ind[1:, :].reshape(-1, 1), ind[:-1, :].reshape(-1, 1)), 1)
    val = torch.exp(-(img.take(ii[:, 0]) - img.take(ii[:, 1])) ** 2 / sigma ** 2)

    # given: create second part of neigbourhood matrix (similar to setFromTriplets in Eigen)
    A = A + torch.sparse.FloatTensor(ii.t(), val, torch.Size([H * W, H * W]))
    # given: make symmetric (add down->up and right->left)
    A = A + A.t()
    # given: compute degree matrix (diagonal sum)
    D = torch.sparse.sum(A, 0).to_dense()
    # given: put D and A together
    L = torch.sparse.FloatTensor(torch.cat((ind.view(1, -1), ind.view(1, -1)), 0), .00001 + lambda_ * D,
                                 torch.Size([H * W, H * W]))
    L += (A * (-lambda_))
    return L


# provided function that calls the sparse LSE solver using multi-grid
def sparseMultiGrid(A, b):  # A sparse torch matrix, b dense vector
    A_ind = A._indices().cpu().data.numpy()
    A_val = A._values().cpu().data.numpy()
    n1, n2 = A.size()
    SC = csr_matrix((A_val, (A_ind[0, :], A_ind[1, :])), shape=(n1, n2))
    ml = pyamg.ruge_stuben_solver(SC, max_levels=6)  # construct the multigrid hierarchy
    # print(ml)                                           # print hierarchy information
    b_ = b.cpu().data.numpy()
    x = b_ * 0
    for i in range(x.shape[1]):
        x[:, i] = ml.solve(b_[:, i], tol=1e-3)
    return torch.from_numpy(x)  # .view(-1,1)


# provided functions that removes/selects rows and/or columns from sparse matrices
def sparse_rows(S, slice):
    # sparse slicing
    S_ind = S._indices()
    S_val = S._values()
    # create auxiliary index vector
    slice_ind = -torch.ones(S.size(0)).long()
    slice_ind[slice] = torch.arange(slice.size(0))
    # def sparse_rows(matrix,indices):
    # redefine row indices of new sparse matrix
    inv_ind = slice_ind[S_ind[0, :]]
    mask = (inv_ind > -1)
    N_ind = torch.stack((inv_ind[mask], S_ind[1, mask]), 0)
    N_val = S_val[mask]
    S = torch.sparse.FloatTensor(N_ind, N_val, (slice.size(0), S.size(1)))
    return S


def sparse_cols(S, slice):
    # sparse slicing
    S_ind = S._indices()
    S_val = S._values()
    # create auxiliary index vector
    slice_ind = -torch.ones(S.size(1)).long()
    slice_ind[slice] = torch.arange(slice.size(0))
    # def sparse_rows(matrix,indices):
    # redefine row indices of new sparse matrix
    inv_ind = slice_ind[S_ind[1, :]]
    mask = (inv_ind > -1)
    N_ind = torch.stack((S_ind[0, mask], inv_ind[mask]), 0)
    N_val = S_val[mask]
    S = torch.sparse.FloatTensor(N_ind, N_val, (S.size(0), slice.size(0)))
    return S


@torch.inference_mode()
def random_walk(img: torch.Tensor, initial_segmentation: torch.Tensor):
    assert img.ndim == 2, 'img should be 2D'
    assert img.dtype == torch.uint8, 'img should be in range [0, 255]'
    H, W = img.shape
    assert initial_segmentation.ndim == 3
    assert initial_segmentation.shape[1] == H and initial_segmentation.shape[2] == W

    # add a background class to the initial segmentation
    background = torch.logical_not(initial_segmentation.any(0))
    background = segmentation_preprocessing.erode_mask_with_disc_struct(background.unsqueeze(0), radius=12).squeeze()
    initial_segmentation = torch.cat([background.unsqueeze(0), initial_segmentation], dim=0)

    linear_idx = torch.arange(H * W).view(H, W)
    idx_mask = initial_segmentation.any(0)
    seeded = linear_idx[idx_mask]
    unseeded = linear_idx[~idx_mask]

    L = laplace_matrix(img.float())
    L_u = sparse_rows(sparse_cols(L, unseeded), unseeded)
    B = sparse_rows(sparse_cols(L, unseeded), seeded)

    u_s = initial_segmentation[:, idx_mask].t()

    b = torch.mm(-B.t(), u_s.float())
    u_u = sparseMultiGrid(L_u, b)

    # combine prediction (u_u) with the labels for the known pixels (u_s)
    p_hat = torch.zeros(H * W, u_s.shape[-1])
    p_hat[seeded] = u_s.float()
    p_hat[unseeded] = u_u

    p_hat = p_hat.view(H, W, -1).permute(2, 0, 1)
    # remove the background class
    p_hat = p_hat[1:]

    return p_hat
