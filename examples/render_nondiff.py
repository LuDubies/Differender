import torch
import torch.nn.functional as F
import math
from itertools import count
from torchvtk.datasets import TorchDataset
from torchvtk.rendering import plot_comp_render_tf
from torchvtk.utils import pool_map, make_4d
import matplotlib.pyplot as plt

from differender.utils import get_tf, in_circles
from differender.volume_raycaster import Raycaster, Mode

from torchvision.utils import save_image


if __name__ == '__main__':
    vol_ds = TorchDataset('C:/Users/luca/Repos/Differender/vtk_dat/')
    vol = vol_ds[0]['vol'].float()
    tf = get_tf('tf1', 128)
    sr = 16.0
    pixels = 800

    raycaster = Raycaster(vol.shape[-3:], (pixels, pixels), 128, jitter=False, max_samples=1, sampling_rate=sr,
                                ti_kwargs={'device_memory_GB': 4.0,'debug': True, 'excepthook': True}, far=5.0, mode=Mode.MaxOpacity)


    vol = vol.to('cuda').requires_grad_(True)
    tf = tf.to('cuda').requires_grad_(True)
    lf = in_circles(1.7 * math.pi).float().to('cuda')

    for mode in Mode:
        raycaster.vr.set_mode(mode)
        print(raycaster.vr.get_mode())

        print(f"Rendering {mode.name} Image...")
        im = raycaster.raycast_nondiff(vol[None], tf[None], lf[None], sampling_rate=sr)

        save_image(im, mode.name + '_render.png')
