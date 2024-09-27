from .sample import transform
import numpy as np


def generate_preview():
    uvs = np.points
    coords = torch.stack([x, y], axis=0)[None]
    points_new = torch.nn.functional.grid_sample(coords, points[None, None])[0, :, 0, :].T

    render_resolution = 100
    render_pigment = 0.0005
    canvas = np.ones((render_resolution, render_resolution, 3))
    draw_points((points_new.data.numpy() * 0.5 + 0.5) * render_resolution,
                -np.ones([sample_size, 3]) * render_pigment,
                canvas)
    return to_pil_image(np.uint8(np.clip(canvas, 0.0, 1.0) * 255))
