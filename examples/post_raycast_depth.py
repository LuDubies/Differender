import torch
import torch.nn.functional as F
import math
from itertools import count
from torchvtk.datasets import TorchDataset
from torchvtk.rendering import plot_comp_render_tf
from torchvtk.utils import pool_map, make_4d
import matplotlib.pyplot as plt

import taichi as ti
import taichi_glsl as tl

from differender.utils import get_tf, in_circles
from differender.volume_raycaster import Raycaster, Compositing

from torchvision.utils import save_image


if __name__ == '__main__':
    vol_ds = TorchDataset('C:/Users/luca/Repos/Differender/vtk_dat/')
    vol = vol_ds[1]['vol'].float()
    tf = get_tf('tf1', 128)

    raycaster = Raycaster(vol.shape[-3:], (128, 128), 128, jitter=False, sampling_rate=1.0, max_samples=2048, compositing=Compositing.FirstHitDepth, ti_kwargs={'device_memory_GB': 4.0})

    vol = vol.to('cuda').requires_grad_(True)
    tf = tf.to('cuda').requires_grad_(True)
    lf = in_circles(1.7 * math.pi).float().to('cuda')

    print(vol.shape, raycaster.volume_shape, tf.shape, lf)

    im = raycaster(vol[None], tf[None], lf[None])
    save_image(im, 'std_render.png')

    save_image(torch.rot90(raycaster.vr.depth.to_torch(device=vol.device), 1, [0, 1]), 'depth_render.png')
    print(f"Loss is: {raycaster.vr.loss}")
    # get compare image
    im = raycaster.raycast_nondiff(vol[None], tf[None], lf[None], sampling_rate=16.0)

    save_image(im, 'nondiff_render.png')