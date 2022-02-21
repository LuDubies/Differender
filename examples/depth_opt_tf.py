import math
import torch
import torch.nn.functional as F
from torchvtk.datasets import TorchDataset

import numpy as np

from differender.utils import get_tf, in_circles
from differender.volume_raycaster import Raycaster

from torchvision.utils import save_image
import matplotlib.pyplot as plt


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

    vr = raycaster.vr

    opt = torch.optim.Adam([tf])


    for i in range(ITERATIONS):
        with torch.no_grad():
            im_gt = raycaster.raycast_nondiff(vol.detach(), tf_gt.detach(), lf.detach(), sampling_rate=sr)
            depth_gt = im_gt.squeeze()[4]
            vr.set_gtd(depth_gt)
        
        opt.zero_grad()
        res = raycaster(vol, tf, lf)
        depth_res = res.squeeze()[4]
        mse_loss = F.mse_loss(depth_res, depth_gt)
        mse_loss.backward()

        print(f"Step {i:03d}: MSE-LOSS: {mse_loss.detach().item():.6e}")

        opt.step()

        with torch.no_grad():
            tf.clamp_(0.0, 1.0)

        # create a control image for gt and raycasting
        if i == 0:
            with torch.no_grad():
                control1 = torch.unsqueeze(depth_gt.detach(), 0).expand(3, 128, 128).permute(1, 2, 0).cpu().numpy()
                control2 = torch.unsqueeze(depth_res.detach(), 0).expand(3, 128, 128).permute(1, 2, 0).cpu().numpy()
                fig, axs = plt.subplots(1, 2)
                axs = axs.flat
                axs[0].imshow(control1)
                axs[1].imshow(control2)
                fig.savefig('control.png', bbox_inches='tight')
