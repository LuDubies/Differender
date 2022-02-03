from PIL import Image
import numpy as np
import math
from torchvtk.datasets import TorchDataset
import matplotlib.pyplot as plt
from differender.utils import get_tf, in_circles
from differender.volume_raycaster import Raycaster, Mode

from torchvision.utils import save_image


if __name__ == '__main__':
    vol_ds = TorchDataset('C:/Users/lucad/Uni/Differender/vtk_dat/')
    vol = vol_ds[0]['vol'].float()
    tf = get_tf('tf1', 128)
    sr = 16.0
    pixels = 800

    raycaster = Raycaster(vol.shape[-3:], (pixels, pixels), 128, jitter=False, max_samples=1, sampling_rate=sr,
                                ti_kwargs={'device_memory_GB': 2.0, 'debug': True}, far=5.0)


    vol = vol.to('cuda').requires_grad_(True)
    tf = tf.to('cuda').requires_grad_(True)
    lf = in_circles(1.7 * math.pi).float().to('cuda')

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))

    for i, mode in enumerate(Mode):

        print(f"Rendering {mode.name} Image...")
        im = raycaster.raycast_nondiff(vol[None], tf[None], lf[None], sampling_rate=sr, mode=mode)

        axs[(i+1)//3][(i+1) % 3].imshow(im.squeeze()[[4, 4, 4]].permute(1, 2, 0).cpu().numpy())
        axs[(i+1)//3][(i+1) % 3].set_title(f"{mode.name}")

        if i == 0:
            axs[0][0].imshow(im.squeeze()[:3].permute(1, 2, 0).cpu().numpy())
            axs[0][0].set_title(f"Standard")
            axs[1][2].imshow(im.squeeze()[:3].permute(1, 2, 0).cpu().numpy())
            axs[1][2].set_title(f"Standard")

    fig.show()

