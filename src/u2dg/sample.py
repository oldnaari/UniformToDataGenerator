import torch
import numpy as np
from typing import Literal


def transform(grid: np.ndarray[[Literal["H"], Literal["W"], Literal[2]], np.dtype[np.float32]],
              queries: np.ndarray[[Literal["N"], Literal[2]], np.dtype[np.float32]]) -> (
        np.ndarray[[Literal["N"], Literal[2]], np.dtype[np.float32]]):
    """
    :param grid:
    Is an array of shape [H, W, 2] of floats [-1, 1]
    that describes where any points (x, y) with coordinates in (-1, 1) should be mapped
    :param queries:
    If the queries is sampled from uniform distribution (-1, 1) the resulting
    point cloud distribution should resemble the image it was trained for
    :return:
    """

    grid = np.moveaxis(grid, -1, 0)[None]
    queries = torch.tensor(queries, dtype=torch.float32)
    grid = torch.tensor(grid, dtype=torch.float32)

    points_new = torch.nn.functional.grid_sample(grid, queries[None, None])[0, :, 0, :].T

    return points_new.cpu().numpy()
