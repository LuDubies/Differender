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
    sr = 1.0

    raycaster = Raycaster(vol.shape[-3:], (128, 128), 128, jitter=False, sampling_rate=sr, max_samples=2048,
     ti_kwargs={'device_memory_GB': 4.0,'debug': True, 'excepthook': True})

    vol = vol.to('cuda').requires_grad_(True)
    tf = tf.to('cuda').requires_grad_(True)
    lf = in_circles(1.7 * math.pi).float().to('cuda')

    print(vol.shape, raycaster.volume_shape, tf.shape, lf)
    
    # use raycaster.determine_batched to get the dimensions right
    batched, bs, vol_in, tf_in, lf_in = raycaster._determine_batch(vol, tf, lf)
    print(f"Batched: {batched}, VolShape: {vol_in.shape}")
    
    vr = raycaster.vr

    ''' do the shit from autograd function here'''
    vr.clear_grad()
    vr.set_cam_pos(lf_in)
    vr.set_volume(vol_in)
    vr.set_tf_tex(tf_in)
    vr.clear_framebuffer()
    vr.compute_rays()
    vr.compute_intersections(sr, 0)

    vr.raycast(sr)
    vr.get_depth_image()

    vr.visualize_ray()
