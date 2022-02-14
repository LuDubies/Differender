
import math
from torch import margin_ranking_loss
from torchvtk.datasets import TorchDataset

import numpy as np

from differender.utils import get_tf, in_circles
from differender.volume_raycaster import Raycaster

from torchvision.utils import save_image
import matplotlib.pyplot as plt


def print_field_info(field, name):
        print(f"{name}:  Shape: {field.shape}, Max: {field.max()}, Min: {field.min()}, Mean: {field.mean()}, Sum: {field.sum()}")


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
    vr = raycaster.vr

    """
    GET GROUND TRUTH FROM NONDIFF CASTING
    """
    im = raycaster.raycast_nondiff(vol[None], tf[None], lf[None], sampling_rate=sr)
    depth_gt = im.squeeze()[[4, 4, 4]].permute(1, 2, 0).cpu().numpy()
    fig, axs = plt.subplots(1, 3, figsize=(30, 95))
    axs = axs.flat
    axs[0].imshow(depth_gt)
    axs[0].set_title('Nondiff Raycast')

    """
    RAYCAST WITH UNCHANGED TF
    """
    #  control image from raycasting with same tf
    im2 = raycaster(vol, tf, lf)
    depth_control = im2.squeeze()[[4, 4, 4]].permute(1, 2, 0).cpu().detach().numpy()
    axs[1].imshow(depth_control)
    axs[1].set_title('Raycast')

    gt = depth_gt[:, :, 0]
    control = depth_control[:, :, 0]

    """
    MANUAL TEST LOSS
    """
    def get_loss(truth, sample):
        return (np.subtract(truth, sample)**2).mean()
    
    print(f"{'Manual-Loss is:':<20} {get_loss(gt, control):<20}")



    """
    SET GROUND TRUTH DEPTH
    """    
    single_channel_depth = gt.transpose()
    vr.set_gtd(single_channel_depth)
    #vr.set_gtd(np.zeros((128, 128)))

    """
    CHANGE TF
    """
    tf_changed = get_tf('tf1_changed', 128)
    tf2 = tf_changed.to('cuda').requires_grad_(True)
    im3 = raycaster(vol, tf2, lf)
    depth_changed = im3.squeeze()[[4, 4, 4]].permute(1, 2, 0).cpu().detach().numpy()
    axs[2].imshow(depth_changed)
    axs[2].set_title('Manipulated TF')
    fig.savefig('depths.png', bbox_inches='tight')
    
    test = depth_changed[:, :, 0]
    loss = get_loss(gt, test)
    print(f"{'TF-Loss is:':<20} {loss:<20}")

    vr.set_loss = loss

    vr.loss_grad()

    
    
    """
    VISUALIZE SOME DATA
    """
    vr.visualize_ray('a', 60, 60, filename='testray.png')
    rtape_grad = vr.render_tape.grad.to_numpy()
    print_field_info(rtape_grad, "RenderTape Gradient Field")



    ITERATIONS = 100


    vr.raycast.grad(sr)
    tf_grad = vr.tf_tex.grad.to_numpy()

    print_field_info(tf_grad[:, 0], "tf_gradient_r")
    print_field_info(tf_grad[:, 1], "tf_gradient_g")
    print_field_info(tf_grad[:, 2], "tf_gradient_b")
    
    print_field_info(tf_grad[:, 3], "tf_gradient_a")

    quit()


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
    
