import torch
import numpy as np
from torchvtk.utils import tex_from_pts, TFGenerator


def torch_to_ti(tf_pts, tf_res):
    return tex_from_pts(tf_pts, tf_res).permute(1, 0).contiguous().numpy()


def get_tf(id, res):
    if id == 'tf1':
        return torch_to_ti(torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                         [0.0840, 0.8510, 0.7230, 0.4672, 0.0000],
                                         [0.0850, 0.8510, 0.7230, 0.4672, 0.0831],
                                         [0.1844, 0.8510, 0.7230, 0.4672, 0.0801],
                                         [0.1890, 0.8510, 0.7230, 0.4672, 0.0000],
                                         [0.2444, 0.8667, 0.5166, 0.6566, 0.0000],
                                         [0.2528, 0.7176, 0.0675, 0.3276, 0.0782],
                                         [0.2621, 0.8667, 0.5166, 0.6566, 0.0000],
                                         [0.3407, 0.9843, 0.9843, 0.9843, 0.0000],
                                         [0.3601, 0.9843, 0.9843, 0.9843, 0.3904],
                                         [0.4475, 0.9843, 0.9843, 0.9843, 0.3917],
                                         [0.4655, 0.9843, 0.9843, 0.9843, 0.0000],
                                         [1.0000, 0.0000, 0.0000, 0.0000, 0.0000]]), res)
    elif id == 'tf2':
        return torch_to_ti(torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                         [0.0178, 0.5333, 0.3597, 0.1861, 0.0000],
                                         [0.0206, 0.5333, 0.3597, 0.1861, 0.1834],
                                         [0.0361, 0.5333, 0.3597, 0.1861, 0.1804],
                                         [0.0388, 0.5333, 0.3597, 0.1861, 0.0000],
                                         [0.2224, 0.6902, 0.0839, 0.1951, 0.0000],
                                         [0.2274, 0.6902, 0.0839, 0.1951, 0.0880],
                                         [0.2479, 0.6902, 0.0839, 0.1951, 0.0831],
                                         [0.2515, 0.6902, 0.0839, 0.1951, 0.0000],
                                         [0.2857, 0.9843, 0.9843, 0.9843, 0.0000],
                                         [0.3042, 0.9843, 0.9843, 0.9843, 0.8240],
                                         [0.4540, 0.9843, 0.9843, 0.9843, 0.8172],
                                         [0.4916, 0.9843, 0.9843, 0.9843, 0.0000],
                                         [1.0000, 0.0000, 0.0000, 0.0000, 0.0000]]), res)
    elif id == 'tf3':
        return torch_to_ti(torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                         [0.0279, 0.5991, 0.6235, 0.1345, 0.0000],
                                         [0.0477, 0.5991, 0.6235, 0.1345, 0.1736],
                                         [0.1090, 0.5991, 0.6235, 0.1345, 0.1779],
                                         [0.1304, 0.5991, 0.6235, 0.1345, 0.0000],
                                         [0.3654, 0.9843, 0.9843, 0.9843, 0.0000],
                                         [0.3991, 0.9843, 0.9843, 0.9843, 0.3912],
                                         [0.7440, 0.9843, 0.9843, 0.9843, 0.3893],
                                         [0.7850, 0.9843, 0.9843, 0.9843, 0.0000],
                                         [1.0000, 0.0000, 0.0000, 0.0000, 0.0000]]), res)
    elif id == 'tf4':
        return torch_to_ti(torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                         [0.0916, 0.5059, 0.1627, 0.1627, 0.0000],
                                         [0.1204, 0.5059, 0.1627, 0.1627, 0.1932],
                                         [0.1865, 0.5059, 0.1627, 0.1627, 0.1956],
                                         [0.2120, 0.5059, 0.1627, 0.1627, 0.0000],
                                         [0.4841, 0.9176, 0.9176, 0.9176, 0.0000],
                                         [0.5195, 0.9176, 0.9176, 0.9176, 0.6406],
                                         [0.6609, 0.9176, 0.9176, 0.9176, 0.6362],
                                         [0.6968, 0.9176, 0.9176, 0.9176, 0.0000],
                                         [1.0000, 0.0000, 0.0000, 0.0000, 0.0000]]), res)
    elif id == 'tf5':
        return torch_to_ti(torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                                         [0.1300, 0.5000, 0.5000, 0.5000, 0.0000],
                                         [0.1350, 0.5000, 0.5000, 0.5000, 0.7500],
                                         [0.1600, 0.5000, 0.5000, 0.5000, 0.7500],
                                         [0.1700, 0.5000, 0.5000, 0.5000, 0.0000],
                                         [1.0000, 0.0000, 0.0000, 0.0000, 0.0000]]), res)
    elif id == 'black':
        return np.zeros((res, 4)) + 1e-2
    elif id == 'gray':
        temp = np.ones((res, 4)) * 0.5
        temp[:, 3] = 0.02
        return temp
    elif id == 'rand':
        return np.random.random((res, 4))
    elif id == 'generate':
        tfgen = TFGenerator(peakgen_kwargs={'max_num_peaks': 2})
        tf_ref = tex_from_pts(tfgen.generate(), res)
        return tf_ref.permute(1, 0).contiguous().numpy()
    else:
        raise Exception(f'Invalid Transfer function identifier given ({id}).')
