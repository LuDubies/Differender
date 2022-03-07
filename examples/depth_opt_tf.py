import math
from random import randint
import torch
import torch.nn.functional as F
from torchvtk.datasets import TorchDataset

import numpy as np

from differender.utils import get_tf, in_circles
from differender.volume_raycaster import Raycaster

from torchvision.utils import save_image
import matplotlib.pyplot as plt


if __name__ == '__main__':

    ITERATIONS = 401


    vol_ds = TorchDataset('C:/Users/luca/Repos/Differender/vtk_dat/')
    vol = vol_ds[1]['vol'].float()
    sr = 16.0

    raycaster = Raycaster(vol.shape[-3:], (128, 128), 128, jitter=False, sampling_rate=1.0, max_samples=2048,
                            ti_kwargs={'device_memory_GB': 4.0,'debug': True, 'excepthook': True}, far=5.0)

    vol = vol.to('cuda')
    tf_gt = get_tf('bones', 128).to('cuda')
    tf = get_tf('gray', 128).to('cuda').requires_grad_(True)
    lf = in_circles(1.7 * math.pi).float().to('cuda')

    vr = raycaster.vr

    opt = torch.optim.AdamW([tf], weight_decay=1e-3)

    #torch.autograd.set_detect_anomaly(True)
    
    with torch.no_grad():
            im_gt = raycaster.raycast_nondiff(vol.detach(), tf_gt.detach(), lf.detach(), sampling_rate=sr)
            depth_gt = im_gt.squeeze()[4]
            vr.set_gtd(torch.flip(depth_gt, (0,)))
            vr.visualize_tf('tf_original.png')

    for i in range(ITERATIONS):
          
        opt.zero_grad()
        res = raycaster(vol, tf, lf)
        depth_res = res.squeeze()[4]
        mse_loss = F.mse_loss(depth_res, depth_gt)
        
        print(f"Step {i:03d}: MSE-LOSS: {mse_loss.detach().item():.6e}")
        mse_loss.backward()


        # get tf graphic and ray visulaisation
        if i % 10 == 0:
            raycaster.vr.visualize_tf(f"tf_step_{i}.png")

            randi = randint(30, 90)
            randj = randint(30, 90)

            raycaster.vr.visualize_ray('a', randi, randj, f"raysample_step_{i}_at_{randi}_{randj}.png")

        opt.step()

        with torch.no_grad():
            tf.clamp_(0.0, 1.0)

        # create a control image for gt and raycasting
        if i % 50 == 0:
            with torch.no_grad():
                control1 = torch.unsqueeze(depth_gt.detach(), 0).expand(3, 128, 128).permute(1, 2, 0).cpu().numpy()
                control2 = torch.unsqueeze(depth_res.detach(), 0).expand(3, 128, 128).permute(1, 2, 0).cpu().numpy()
                fig, axs = plt.subplots(1, 2)
                axs = axs.flat
                axs[0].imshow(control1)
                axs[1].imshow(control2)
                fig.savefig(f'control_step_{i}.png', bbox_inches='tight')
