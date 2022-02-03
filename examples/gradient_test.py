
import math
from torchvtk.datasets import TorchDataset

import numpy as np

from differender.utils import get_tf, in_circles
from differender.volume_raycaster import Raycaster

from torchvision.utils import save_image
import matplotlib.pyplot as plt


if __name__ == '__main__':
    vol_ds = TorchDataset('C:/Users/luca/Repos/Differender/vtk_dat/')
    vol = vol_ds[1]['vol'].float()
    tf = get_tf('tf1', 128)
    sr = 16.0

    GTD = 0.6

    raycaster = Raycaster(vol.shape[-3:], (128, 128), 128, jitter=False, sampling_rate=1.0, max_samples=2048,
                            ti_kwargs={'device_memory_GB': 4.0,'debug': True, 'excepthook': True}, far=5.0)

    vol = vol.to('cuda').requires_grad_(True)
    tf = tf.to('cuda').requires_grad_(True)
    lf = in_circles(1.7 * math.pi).float().to('cuda')

    print(vol.shape, raycaster.volume_shape, tf.shape, lf)
    
    # use raycaster.determine_batched to get the dimensions right
    batched, bs, vol_in, tf_in, lf_in = raycaster._determine_batch(vol, tf, lf)
    print(f"Batched: {batched}, VolShape: {vol_in.shape}")

    vr = raycaster.vr

    """
    GET GROUND TRUTH FROM NONDIFF CASTING
    """
    im = raycaster.raycast_nondiff(vol[None], tf[None], lf[None], sampling_rate=sr)
    depth_gt = im.squeeze()[[4, 4, 4]].permute(1, 2, 0).cpu().numpy()
    fig_gt, ax_gt = plt.subplots()
    ax_gt.imshow(depth_gt)
    fig_gt.savefig('gt_depth.png', bbox_inches='tight')

    vr.clear_framebuffer()
    vr.clear_grad()
    
    im2 = raycaster(vol, tf, lf)
    depth_diff = im.squeeze()[[4, 4, 4]].permute(1, 2, 0).cpu().numpy()
    fig_diff, ax_diff = plt.subplots()
    ax_diff.imshow(depth_diff)
    fig_diff.savefig('control.png', bbox_inches='tight')

    quit()
    ''' 
    FORWARD PASS
    '''
    vr.clear_grad()
    vr.set_cam_pos(lf_in)
    vr.set_volume(vol_in)
    vr.set_tf_tex(tf_in)
    vr.set_gtd(gtd=np.full((128, 128), GTD, dtype=np.float32))
    vr.clear_framebuffer()
    vr.compute_rays()
    vr.compute_intersections(sr, 0)

    vr.raycast(sr)
    vr.get_final_image()
    vr.compute_loss()

    def print_field_info(field, name):
        print(f"{name}:  Shape: {field.shape}, Max: {field.max()}, Min: {field.min()}, Mean: {field.mean()}, Sum: {field.sum()}")
    
    print(f"Calculated loss is: {vr.loss}")

    '''
    MANUAL LOSS FROM GTD
    '''
    vr.compute_loss.grad()
    rtape_grad = vr.render_tape.grad.to_numpy() 
    print_field_info(rtape_grad, "render_tape_gradient")

    vr.loss_grad()
    rtape_grad = vr.render_tape.grad.to_numpy()
    print_field_info(rtape_grad, "manual_grads")

    vr.raycast.grad(sr)

    vr.visualize_ray()

    '''
    SHOW RESULTS
    '''
    tf_grad = vr.tf_tex.grad.to_numpy()
    '''
    print_field_info(tf_grad[:, 0], "tf_gradient_r")
    print_field_info(tf_grad[:, 1], "tf_gradient_g")
    print_field_info(tf_grad[:, 2], "tf_gradient_b")
    '''
    print_field_info(tf_grad[:, 3], "tf_gradient_a")
    fig, ax = plt.subplots()
    ax.plot(tf_grad[:, 3], 'k-')
    fig.savefig('tf_grad.png', bbox_inches='tight')

    print(f"tf at 0: {tf_grad[0, 3]}")
    print(f"tf at 1: {tf_grad[1, 3]}")
    print(f"tf at 2: {tf_grad[2, 3]}")
    print(f"tf at 3: {tf_grad[3, 3]}")
    print(f"tf at 4: {tf_grad[4, 3]}")
    print(f"tf at 5: {tf_grad[5, 3]}")
    print(f"tf at 6: {tf_grad[6, 3]}")
    print(f"tf at 7: {tf_grad[7, 3]}")




    ''' 
    print_field_info(tf_grad[0][10:], "tf_gradient_r")
    print_field_info(tf_grad[1][10:], "tf_gradient_g")
    print_field_info(tf_grad[2][10:], "tf_gradient_b")
    print_field_info(tf_grad[3][10:], "tf_gradient_a")
    '''

    vr.visualize_tf(f'tf_before.png')
    vr.apply_tf_grad()
    vr.visualize_tf(f'tf_after(gtd{GTD}).png')
    
