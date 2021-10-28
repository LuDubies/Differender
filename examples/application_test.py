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
    
    print("Calculating loss gradient:")
    vr = raycaster.vr

    ''' do the shit from autograd function here'''
    vr.clear_grad()
    vr.set_cam_pos(lf_in)
    vr.set_volume(vol_in)
    vr.set_tf_tex(tf_in)
    vr.clear_framebuffer()
    vr.compute_rays()
    vr.compute_intersections(sr, 0)

    ''' Tape syntax'''
    with ti.Tape(vr.loss):
        vr.raycast(sr)
        vr.get_final_image()
        vr.compute_loss()

    print(f"Calculated loss is: {vr.loss}")

    dtape_np = raycaster.vr.depth_tape.to_numpy()
    print(f"depthtape:  Shape: {dtape_np.shape}, Max: {dtape_np.max()}, Min: {dtape_np.min()}, Mean: {dtape_np.mean()}, Sum: {dtape_np.sum()}")

    # image_tensor = torch.rot90(vr.output_rgba.to_torch(device=vol.device), 1, [0, 1])
    print(vr.output_rgba.shape)
    image_tensor = vr.output_rgba.to_torch(device=vol.device)
    save_image(image_tensor, 'tape_test_image.png')



    """
    depth_np = raycaster.vr.depth.to_numpy()
    dg_np = raycaster.vr.depth.grad.to_numpy()
    print(f"depth_field:  Shape: {depth_np.shape}, Max: {depth_np.max()}, Min: {depth_np.min()}, Mean: {depth_np.mean()}, Sum: {depth_np.sum()}")
    print(f"depth_grad_field:  Shape: {dg_np.shape}, Max: {dg_np.max()}, Min: {dg_np.min()}, Mean: {dg_np.mean()}, Sum: {dg_np.sum()}")

    dtape_grad_np = raycaster.vr.depth_tape.grad.to_numpy()
    print(f"depthtape_grad:  Shape: {dtape_grad_np.shape}, Max: {dtape_grad_np.max()}, Min: {dtape_grad_np.min()}, Mean: {dtape_grad_np.mean()}, Sum: {dtape_grad_np.sum()}")

    rendertape_grad_np = raycaster.vr.render_tape.grad.to_numpy()
    print(f"rendertape_grad:  Shape: {rendertape_grad_np.shape}, Max: {rendertape_grad_np.max()}, Min: {rendertape_grad_np.min()}, Mean: {rendertape_grad_np.mean()}, Sum: {rendertape_grad_np.sum()}")

    tf_grad_np = raycaster.vr.tf_tex.grad.to_numpy()
    print(f"tf_grad:  Shape: {tf_grad_np.shape}, Max: {tf_grad_np.max()}, Min: {tf_grad_np.min()}, Mean: {tf_grad_np.mean()}, Sum: {tf_grad_np.sum()}")

    
    print("\nTesting grad application:")
    step = 0
    donezo = False
    while not donezo:
        raycaster.vr.clear_loss()
        with ti.Tape(raycaster.vr.loss):        
            raycaster.vr.compute_loss()
        raycaster.vr.mach_sachn()
        print(f"Step {step}: loss is {raycaster.vr.loss[None]}")

        if step % 200 == 0:
            image_tensor = torch.rot90(raycaster.vr.depth.to_torch(device=vol.device), 1, [0, 1])
            save_image(image_tensor, f"optimizing/step_{step}.png")
        
        if raycaster.vr.loss[None] < 0.0005:
            donezo = True
        
        step+=1
    """

