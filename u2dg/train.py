import torch
import argparse
import pathlib
import tqdm.auto as tqdm
import numpy as np
import imageio

__all__ = ["train"]


def dx(t: torch.Tensor) -> torch.Tensor:
    dt = 2.0 / t.shape[-1]
    diff = (t[..., :, 1:] - t[..., :, :-1]) / dt
    return (diff[..., 1:, :] + diff[..., :-1, :]) / 2


def dy(t: torch.Tensor) -> torch.Tensor:
    dt = 2.0 / t.shape[-2]
    diff = (t[..., 1:, :] - t[..., :-1, :]) / dt
    return (diff[..., :, 1:] + diff[..., :, :-1]) / 2


def estimate_density(init_density: torch.Tensor,
                     x: torch.Tensor,
                     y: torch.Tensor) -> torch.Tensor:
    eps = 1e-9
    u = (x[:, 1:] + x[:, :-1]) / 2
    v = (y[1:, :] + y[:-1, :]) / 2
    u = (u[1:, :] + u[:-1, :]) / 2
    v = (v[:, 1:] + v[:, :-1]) / 2
    uv = torch.stack([u, v], dim=-1)
    density_warped = torch.nn.functional.grid_sample(
        init_density[None, None].float(), uv[None])
    density_estimated = density_warped * torch.maximum(
        torch.abs(dx(x) * dy(y) - dx(y) * dy(x)), torch.tensor(eps)
    )
    return density_estimated / torch.sum(density_estimated)


def compute_xy(x_inc, y_inc):
    size = x_inc.shape[1]
    x = torch.cat([torch.zeros((1, size)), x_inc, torch.zeros((1, size))], dim=0)
    y = torch.cat([torch.zeros((size, 1)), y_inc, torch.zeros((size, 1))], dim=1)
    x = torch.cumsum(torch.nn.functional.softmax(x, dim=1), dim=1)
    y = torch.cumsum(torch.nn.functional.softmax(y, dim=0), dim=0)
    x = torch.cat([torch.zeros((size + 1, 1)), x], dim=1)
    y = torch.cat([torch.zeros((1, size + 1)), y], dim=0)
    x = x * 2 - 1
    y = y * 2 - 1
    return x, y


def convert_image_to_density(img: np.ndarray) -> np.ndarray:

    if len(img.shape) == 3:
        print("Converting image to grayscale")
        density = np.mean(img[..., :3], axis=-1)
    else:
        density = img

    density = density / np.sum(density)
    return density


def train(init_density: np.ndarray, n_iter=10000) -> np.ndarray:
    """
    :param init_density:
    A square array of point density values of shape [H, W=H]
    :param n_iter:
    Number of training iterations
    :return:
    """
    size = init_density.shape[0]
    x_inc = torch.zeros((size - 1, size), requires_grad=True)
    y_inc = torch.zeros((size, size - 1), requires_grad=True)
    init_density = torch.tensor(init_density)

    optimizer = torch.optim.Rprop([x_inc, y_inc], lr=0.001)

    if n_iter <= 0:
        raise ValueError("n_iter must be > 0")

    for _ in tqdm.trange(n_iter):
        x, y = compute_xy(x_inc, y_inc)
        optimizer.zero_grad()
        pred_density = estimate_density(init_density, x, y)
        loss = torch.std(pred_density)
        loss.backward()
        optimizer.step()

    return torch.stack([x, y], dim=-1).detach().cpu().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-image",
                        type=pathlib.Path,
                        required=True,
                        help="the image to use for density distribution")
    parser.add_argument("--output",
                        type=pathlib.Path,
                        default=None,
                        help="the image to use for density distribution")
    parser.add_argument("--color-scheme",
                        choices=["white-on-black", "black-on-white"],)
    args = parser.parse_args()

    if args.output is None:
        args.output = args.source_image.with_suffix(".npy")
    image = imageio.v3.imread(args.source_image)

    if image.shape[0] != image.shape[1]:
        raise ValueError("Image must be square")
    image = image / 255
    if args.color_scheme == "black-on-white":
        image = 1.0 - image

    original_density = convert_image_to_density(image)
    coordinate_map = train(original_density)

    np.save(args.output, coordinate_map)

