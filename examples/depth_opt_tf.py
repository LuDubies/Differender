import math
import torch
import torch.nn.functional as F
from torchvtk.datasets import TorchDataset

import numpy as np

from differender.utils import get_tf, in_circles
from differender.volume_raycaster import Raycaster

from torchvision.utils import save_image
import matplotlib.pyplot as plt


def print_field_info(field, name):
        print(f"{name}:  Shape: {field.shape}, Max: {field.max()}, Min: {field.min()}, Mean: {field.mean()}, Sum: {field.sum()}")


if __name__ == '__main__':

    ITERATIONS = 100



    vol_ds = TorchDataset('C:/Users/luca/Repos/Differender/vtk_dat/')
    vol = vol_ds[1]['vol'].float()
    sr = 16.0

    raycaster = Raycaster(vol.shape[-3:], (128, 128), 128, jitter=False, sampling_rate=1.0, max_samples=2048,
                            ti_kwargs={'device_memory_GB': 4.0,'debug': True, 'excepthook': True}, far=5.0)

    vol = vol.to('cuda')
    tf_gt = get_tf('tf1', 128).to('cuda')
    tf = get_tf('tf1_changed', 128).to('cuda').requires_grad_(True)
    lf = in_circles(1.7 * math.pi).float().to('cuda')

    print(vol.shape, raycaster.volume_shape, tf.shape, lf)
    vr = raycaster.vr

    opt = torch.optim.Adam([tf])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=ITERATIONS)


    for i in range(ITERATIONS):
        with torch.no_grad():
            im_gt = raycaster.raycast_nondiff(vol.detach(), tf_gt.detach(), lf.detach(), sampling_rate=sr)
        opt.zero_grad()
        res = raycaster(vol, tf, lf)
        mse_loss = F.mse_loss(res, im_gt)

        print(f"Step {i:03d}: MSE-LOSS: {mse_loss.detach().item():.5f}")

        opt.step()
        sched.step()

        with torch.no_grad():
            tf.clamp_(0.0, 1.0)
