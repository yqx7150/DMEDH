import datetime
import os
from random import betavariate
import sys
from PIL import Image
import pandas as pd
sys.path.append('..')
import copy

import functools
import matplotlib.pyplot as plt
import torch
import numpy as np
import abc
from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate, io
from mpepi_tool import *
import sde_lib
from models import utils as mutils
from skimage.metrics import peak_signal_noise_ratio as compare_psnr, structural_similarity as compare_ssim, \
    mean_squared_error as compare_mse
import odl
import glob
import pydicom
import cv2
from cv2 import imwrite, resize
from func_test import WriteInfo
from scipy.io import loadmat, savemat
from radon_utils import (create_sinogram, bp, filter_op,
                         fbp, reade_ima, write_img, sinogram_2c_to_img,
                         padding_img, unpadding_img, indicate)
from time import sleep
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, \
    mean_squared_error as mse
from cv2 import imwrite, resize
# import openpyxl

_CORRECTORS = {}
_PREDICTORS = {}

#网络内容

def set_predict(num):
    if num == 0:
        return 'None'
    elif num == 1:
        return 'EulerMaruyamaPredictor'
    elif num == 2:
        return 'ReverseDiffusionPredictor'


def set_correct(num):
    if num == 0:
        return 'None'
    elif num == 1:
        return 'LangevinCorrector'
    elif num == 2:
        return 'AnnealedLangevinDynamics'


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(f'Already registered model with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]

def get_corrector(name):
    return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
    """Create a sampling function.

  Args:
    config: A `ml_collections.ConfigDict` object that contains all configuration information.
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers representing the expected shape of a single sample.
    inverse_scaler: The inverse data normalizer function.
    eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

  Returns:
    A function that takes random states and a replicated training state and outputs samples with the
      trailing dimensions matching `shape`.
  """

    sampler_name = config.sampling.method  # pc
    # Probability flow ODE sampling with black-box ODE solvers
    if sampler_name.lower() == 'ode':
        sampling_fn = get_ode_sampler(sde=sde,
                                      shape=shape,
                                      inverse_scaler=inverse_scaler,
                                      denoise=config.sampling.noise_removal,
                                      eps=eps,
                                      device=config.device)
    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    elif sampler_name.lower() == 'pc':
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_sampler(sde=sde,
                                     shape=shape,
                                     predictor=predictor,
                                     corrector=corrector,
                                     inverse_scaler=inverse_scaler,
                                     snr=config.sampling.snr,
                                     n_steps=config.sampling.n_steps_each,
                                     probability_flow=config.sampling.probability_flow,
                                     continuous=config.training.continuous,
                                     denoise=config.sampling.noise_removal,
                                     eps=eps,
                                     device=config.device)
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the predictor.

    Args:
      x: A PyTorch tensor representing the current state
      t: A Pytorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the corrector.

    Args:
      x: A PyTorch tensor representing the current state
      t: A PyTorch tensor representing the current time step.

    Returns:
      x: A PyTorch tensor of the next state.
      x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
    """
        pass

@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean

# ===================================================================== ReverseDiffusionPredictor
@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        f, G = self.rsde.discretize(x, t)
        z = torch.randn_like(x)
        x_mean = x - f
        x = x_mean + G[:, None, None, None] * z
        return x, x_mean


# =====================================================================

@register_predictor(name='ancestral_sampling')
class AncestralSamplingPredictor(Predictor):
    """The ancestral sampling predictor. Currently only supports VE/VP SDEs."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)
        if not isinstance(sde, sde_lib.VPSDE) and not isinstance(sde, sde_lib.VESDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")
        assert not probability_flow, "Probability flow not supported by ancestral sampling"

    def vesde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        sigma = sde.discrete_sigmas[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t), sde.discrete_sigmas.to(t.device)[timestep - 1])
        score = self.score_fn(x, t)
        x_mean = x + score * (sigma ** 2 - adjacent_sigma ** 2)[:, None, None, None]
        std = torch.sqrt((adjacent_sigma ** 2 * (sigma ** 2 - adjacent_sigma ** 2)) / (sigma ** 2))
        noise = torch.randn_like(x)
        x = x_mean + std[:, None, None, None] * noise
        return x, x_mean

    def vpsde_update_fn(self, x, t):
        sde = self.sde
        timestep = (t * (sde.N - 1) / sde.T).long()
        beta = sde.discrete_betas.to(t.device)[timestep]
        score = self.score_fn(x, t)
        x_mean = (x + beta[:, None, None, None] * score) / torch.sqrt(1. - beta)[:, None, None, None]
        noise = torch.randn_like(x)
        x = x_mean + torch.sqrt(beta)[:, None, None, None] * noise
        return x, x_mean

    def update_fn(self, x, t):
        if isinstance(self.sde, sde_lib.VESDE):
            return self.vesde_update_fn(x, t)
        elif isinstance(self.sde, sde_lib.VPSDE):
            return self.vpsde_update_fn(x, t)


@register_predictor(name='none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t):
        return x, x

# ================================================================================================== LangevinCorrector
@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean


# ==================================================================================================

@register_corrector(name='ald')
class AnnealedLangevinDynamics(Corrector):
    """The original annealed Langevin dynamics predictor in NCSN/NCSNv2.

  We include this corrector only for completeness. It was not directly used in our paper.
  """

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE):
            raise NotImplementedError(f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        std = self.sde.marginal_prob(x, t)[1]

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            step_size = (target_snr * std) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None, None]

        return x, x_mean

@register_corrector(name='none')
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, score_fn, snr, n_steps):
        pass

    def update_fn(self, x, t):
        return x, x


# ========================================================================================================

def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow, continuous):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t)

def shared_corrector_update_fn(x, t, sde, model, corrector, continuous, snr, n_steps):
    """A wrapper tha configures and returns the update function of correctors."""
    score_fn = mutils.get_score_fn(sde, model, train=False, continuous=continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t)


# ========================================================================================================


#获得P-C的采样
def get_pc_sampler(sde, predictor, corrector, inverse_scaler, snr,
                   n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda', zl_arg=-1):
    """Create a Predictor-Corrector (PC) sampler.

  Args:
    sde: An `sde_lib.SDE` object representing the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
    corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
    inverse_scaler: The inverse data normalizer.
    snr: A `float` number. The signal-to-noise ratio for configuring correctors.
    n_steps: An integer. The number of corrector steps per predictor update.
    probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
    continuous: `True` indicates that the score model was continuously trained.
    denoise: If `True`, add one-step denoising to the final samples.
    eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            continuous=continuous,
                                            snr=snr,
                                            n_steps=n_steps)

    def pc_sampler(img_model, check_num, predict, correct):

        # ***********************************picNO
        # params to test
        testNUM = 1
        padding = 1000
        N = 512
        #用于终端指令输入的方式
        useNet = zl_arg.useNet  
        planes = zl_arg.planes#5 3 2
        
        # planes = planes_num
        for picNO in range(0, testNUM):

            with torch.no_grad():
                #数据类型转换，用于之后的网络
                def toNumpy(tensor):
                    return np.squeeze(tensor.cpu().numpy())
                #计算两张图的PSW
                def countPSM(aa, bb):
                    aa = np.squeeze(aa)
                    bb = np.squeeze(bb)

                    # 归一化
                    maxvalue1 = np.max(aa)
                    minvalue1 = np.min(aa)

                    aa = (aa - minvalue1) / (maxvalue1 - minvalue1)
                    maxvalue1 = np.max(bb)
                    minvalue1 = np.min(bb)

                    bb = (bb - minvalue1) / (maxvalue1 - minvalue1)

                    psnr0 = psnr(aa, bb, data_range=1)
                    ssim0 = ssim(aa, bb, gaussian_weights=True, use_sample_covariance=False, data_range=1.0)
                    ssim0 = ssim(aa, bb, data_range=1.0)
                    mse0 = mse(aa, bb)
                    psnr0 = round(psnr0, 2)
                    ssim0 = round(ssim0, 4)
                    mse0 = round(mse0, 6)

                    return psnr0, ssim0, mse0


                def savePng(path, img):
                    ppp = os.path.split(path)
                    if not os.path.isdir(ppp[0]):
                        os.makedirs(ppp[0])

                    plt.imsave(path,img,cmap='gray')

                def getInsideIndex(pad=padding, size=N):
                    return int((pad - size) / 2)

                def addPaddingAmp(amp, pad=padding, size=N):
                    # amp is 512*512
                    temp = np.ones((pad, pad), np.float32)

                    # if size > 512:
                    #     raise Exception("hey man, your cut is too big!")

                    # inIn = getInsideIndex(512,size)
                    # amp_cut = amp[inIn:inIn + size, inIn:inIn + size]

                    inIn = getInsideIndex(pad, size)
                    temp[inIn:inIn + size, inIn:inIn + size] = amp
                    return temp

                def addPaddingPhase(phase, pad=padding, size=N):
                    temp = np.zeros((pad, pad), np.float32)

                    if size > 512:
                        raise Exception("hey man, your cut is too big!")

                    inIn = getInsideIndex(512, size)
                    phase_cut = phase[inIn:inIn + size, inIn:inIn + size]

                    inIn = getInsideIndex(pad, size)
                    temp[inIn:inIn + size, inIn:inIn + size] = phase_cut
                    return temp

                def getFormatPSM(psnr2, ssim2, mse2):
                    return f"{('%.2f' % psnr2)}/{('%.4f' % ssim2)}/{('%.4f' % mse2)}"


                def downsample(image_data, s):
                    # 获取原始图像的尺寸
                    M, N = image_data.shape

                    # 计算下采样后的图像尺寸
                    new_M = M // s
                    new_N = N // s

                    # 初始化下采样后的图像数据数组
                    subsampled_image = np.zeros((new_M, new_N), dtype=image_data.dtype)

                    # 对每个s*s的窗口进行遍历
                    for i in range(0, M, s):
                        for j in range(0, N, s):
                            # 计算窗口内像素的均值
                            window = image_data[i:i + s, j:j + s]
                            mean_value = np.mean(window)

                            # 将均值赋给下采样图像的对应位置
                            subsampled_image[i // s, j // s] = mean_value

                    return subsampled_image

                

              
                #———————————————————————————————————————————————————————————————————————————————————————————————————————
                #origin params
                N1 = 400  #
                N = 600
                N2 = 600
                wavelength = 532 * (pow(10, -9))#532
                pixel = 19 * (pow(10, -6))  # optics 600  3000->600 3.8 *5 = 19
                area = N * pixel
                dataPath = "optics_data"
                savePath = 'optics_s_3'  #保存图片位置，需要自己设置

                title = 'NCSNPP' if useNet else 'MPEPI'
                title = f'{title}'
                
                if planes == 5:
                    z0 = 0.323# Plane5
                    z1 = 0.311# Plane4
                    z2 = 0.299# Plane3
                    z3 = 0.287 # Plane2
                    z4 = 0.275 # Plane1
                elif planes == 3:
                    z0 = 0.308 # Plane5
                    z2 = 0.284 # Plane3
                    z4 = 0.260 # Plane1
                elif planes == 2:
                    z0 = 0.323 # Plane5
                    z4 = 0.275 # Plane1
                
                # 算子定义
                if planes == 5:
                    prop0 = Propagator_function(N, wavelength, area, z0)
                    prop1 = Propagator_function(N, wavelength, area, z1)
                    prop2 = Propagator_function(N, wavelength, area, z2)
                    prop3 = Propagator_function(N, wavelength, area, z3)
                    prop4 = Propagator_function(N, wavelength, area, z4)
                elif planes == 3:
                    prop0 = Propagator_function(N, wavelength, area, z0)
                    prop2 = Propagator_function(N, wavelength, area, z2)
                    prop4 = Propagator_function(N, wavelength, area, z4)
                elif planes == 2:
                    prop0 = Propagator_function(N, wavelength, area, z0)
                    prop4 = Propagator_function(N, wavelength, area, z4)

                area1 = N1 * pixel
                if planes == 5:
                    prop00 = Propagator_function(N1, wavelength, area1, z0)
                    prop11 = Propagator_function(N1, wavelength, area1, z1)
                    prop22 = Propagator_function(N1, wavelength, area1, z2)
                    prop33 = Propagator_function(N1, wavelength, area1, z3)
                    prop44 = Propagator_function(N1, wavelength, area1, z4)
                elif planes == 3:
                    prop00 = Propagator_function(N1, wavelength, area1, z0)
                    prop22 = Propagator_function(N1, wavelength, area1, z2)
                    prop44 = Propagator_function(N1, wavelength, area1, z4)
                elif planes == 2:
                    prop00 = Propagator_function(N1, wavelength, area1, z0)
                    prop44 = Propagator_function(N1, wavelength, area1, z4)

                area2 = N2 * pixel
                if planes == 5:
                    prop000 = Propagator_function(N2, wavelength, area2, z0)
                    prop111 = Propagator_function(N2, wavelength, area2, z1)
                    prop222 = Propagator_function(N2, wavelength, area2, z2)
                    prop333 = Propagator_function(N2, wavelength, area2, z3)
                    prop444 = Propagator_function(N2, wavelength, area2, z4)
                elif planes == 3:
                    prop000 = Propagator_function(N2, wavelength, area2, z0)
                    prop222 = Propagator_function(N2, wavelength, area2, z2)
                    prop444 = Propagator_function(N2, wavelength, area2, z4)
                elif planes == 2:
                    prop000 = Propagator_function(N2, wavelength, area2, z0)
                    prop444 = Propagator_function(N2, wavelength, area2, z4)


            

                hologram_bkg = cv2.imread(os.path.join(dataPath, '72_bkg.png'),cv2.IMREAD_GRAYSCALE)
                hologram4 = cv2.imread(os.path.join(dataPath, '3_p1.png'),cv2.IMREAD_GRAYSCALE)
                hologram3 = cv2.imread(os.path.join(dataPath, '3_p2.png'), cv2.IMREAD_GRAYSCALE)
                hologram2 = cv2.imread(os.path.join(dataPath, '3_p3.png'), cv2.IMREAD_GRAYSCALE)
                hologram1 = cv2.imread(os.path.join(dataPath, '3_p4.png'), cv2.IMREAD_GRAYSCALE)
                hologram0 = cv2.imread(os.path.join(dataPath, '3_p5.png'), cv2.IMREAD_GRAYSCALE)

                hologram_bkg = hologram_bkg.astype(np.float32)/255.0
                hologram4 = hologram4.astype(np.float32)/255.0
                hologram3 = hologram3.astype(np.float32) / 255.0
                hologram2 = hologram2.astype(np.float32) / 255.0
                hologram1 = hologram1.astype(np.float32) / 255.0
                hologram0 = hologram0.astype(np.float32) / 255.0
                # 调用函数进行下采样
                hologram_bkg = downsample(hologram_bkg,5 )
                hologram4 = downsample(hologram4, 5)
                hologram3 = downsample(hologram3, 5)
                hologram2 = downsample(hologram2, 5)
                hologram1 = downsample(hologram1, 5)
                hologram0 = downsample(hologram0, 5)



                hologram_bkg = np.array(hologram_bkg).astype(float)
                hologram4 = np.array(hologram4).astype(float)
                hologram3 = np.array(hologram3).astype(float)
                hologram2 = np.array(hologram2).astype(float)
                hologram1 = np.array(hologram1).astype(float)
                hologram0 = np.array(hologram0).astype(float)


                hologram4 =  hologram4/ hologram_bkg
                hologram3 =  hologram3/ hologram_bkg
                hologram2 =  hologram2/ hologram_bkg
                hologram1 =  hologram1/ hologram_bkg
                hologram0 =  hologram0/ hologram_bkg

                hologram0[np.isinf(hologram0) | np.isnan(hologram0)] = 0
                hologram1[np.isinf(hologram1) | np.isnan(hologram1)] = 0
                hologram2[np.isinf(hologram2) | np.isnan(hologram2)] = 0
                hologram3[np.isinf(hologram3) | np.isnan(hologram3)] = 0
                hologram4[np.isinf(hologram4) | np.isnan(hologram4)] = 0

                #图片裁剪
                if planes == 5:
                    hologram_cropped0 = crops(hologram0, N1, N)#0-4
                    hologram_cropped1 = crops(hologram1, N1, N)
                    hologram_cropped2 = crops(hologram2, N1, N)
                    hologram_cropped3 = crops(hologram3, N1, N)
                    hologram_cropped4 = crops(hologram4, N1, N)
                elif planes == 3:
                    hologram_cropped0 = crops(hologram0, N1, N)
                    hologram_cropped2 = crops(hologram2, N1, N)
                    hologram_cropped4 = crops(hologram4, N1, N)
                elif planes == 2:
                    hologram_cropped0 = crops(hologram0, N1, N)
                    hologram_cropped4 = crops(hologram4, N1, N)

#-----------------------------------------------------------------------------------------------
                if planes == 5:
                    recons0 = IFT(FT(np.sqrt(hologram_cropped0)) * np.conj(prop00))
                    recons1 = IFT(FT(np.sqrt(hologram_cropped1)) * np.conj(prop11))
                    recons2 = IFT(FT(np.sqrt(hologram_cropped2)) * np.conj(prop22))
                    recons3 = IFT(FT(np.sqrt(hologram_cropped3)) * np.conj(prop33))
                    recons4 = IFT(FT(np.sqrt(hologram_cropped4)) * np.conj(prop44))
                elif planes == 3:
                    recons0 = IFT(FT(np.sqrt(hologram_cropped0)) * np.conj(prop00))
                    recons2 = IFT(FT(np.sqrt(hologram_cropped2)) * np.conj(prop22))
                    recons4 = IFT(FT(np.sqrt(hologram_cropped4)) * np.conj(prop44))
                elif planes == 2:
                    recons0 = IFT(FT(np.sqrt(hologram_cropped0)) * np.conj(prop00))
                    recons4 = IFT(FT(np.sqrt(hologram_cropped4)) * np.conj(prop44))


                # 设置每个plant的保真部分
                if planes == 5:
                    measured1 = np.sqrt(hologram_cropped1)
                    measured2 = np.sqrt(hologram_cropped2)
                    measured3 = np.sqrt(hologram_cropped3)
                    measured4 = np.sqrt(hologram_cropped4)
                    measured0 = np.sqrt(hologram_cropped0)
                elif planes == 3:
                    measured0 = np.sqrt(hologram_cropped0)
                    measured2 = np.sqrt(hologram_cropped2)
                    measured4 = np.sqrt(hologram_cropped4)
                elif planes == 2:
                    measured0 = np.sqrt(hologram_cropped0)
                    measured4 = np.sqrt(hologram_cropped4)

                if planes == 5:
                    amplitude0 = np.ones((N2, N2))
                    phase0 = np.zeros((N2, N2))
                    amplitude1 = np.ones((N2, N2))
                    phase1 = np.zeros((N2, N2))
                    amplitude2 = np.ones((N2, N2))
                    phase2 = np.zeros((N2, N2))
                    amplitude3 = np.ones((N2, N2))
                    phase3 = np.zeros((N2, N2))
                    amplitude4 = np.ones((N2, N2))
                    phase4 = np.zeros((N2, N2))
                elif planes == 3:
                    amplitude0 = np.ones((N2, N2))
                    phase0 = np.zeros((N2, N2))
                    amplitude2 = np.ones((N2, N2))
                    phase2 = np.zeros((N2, N2))
                    amplitude4 = np.ones((N2, N2))
                    phase4 = np.zeros((N2, N2))
                elif planes == 2:
                    amplitude0 = np.ones((N2, N2))
                    phase0 = np.zeros((N2, N2))
                    amplitude4 = np.ones((N2, N2))
                    phase4 = np.zeros((N2, N2))

                if planes == 5:
                    field_final0 = amplitude0 * np.exp(1j * phase0)
                    field_final1 = amplitude1 * np.exp(1j * phase1)
                    field_final2 = amplitude2 * np.exp(1j * phase2)
                    field_final3 = amplitude3 * np.exp(1j * phase3)
                    field_final4 = amplitude4 * np.exp(1j * phase4)


                M1 = (N2 - N1) // 2
                M2 = (N2 + N1) // 2

                # 扩散模型需要的噪声量
                timesteps = torch.linspace(sde.T, eps, sde.N, device=device)
                # 迭代次数
                if useNet:
                    startStep = 1600
                    endStep = 1630
                else:
                    startStep = 1600
                    endStep = 1700
                # 开始迭代
                for kk in range(startStep, endStep):
                    kk = kk - startStep +1
                    print(f"Iteration: {kk}")
                    # 控制扩散模型噪音大小
                    # timesteps:[1,0.999,0.998,.....,0.111,0.110,....,0.001]
                    t = timesteps[1200]#超参数
                    step = kk
                    vec_t = torch.ones(1, device=t.device) * t
                    
                    if planes == 5:
                        # Plane5
                        for ii in range(M1, M2):
                            for jj in range(M1, M2):
                                amplitude0[ii, jj] = measured0[ii - M1, jj - M1]

                        # 全息图反向转播变成稀疏的复数图并使用了prop000进行外推
                        field_final0 = amplitude0 * np.exp(1j * phase0)
                        t0 = IFT((FT(field_final0)) * np.conj(prop000))

                        # 获取稀疏复数图的取反的振幅和原始相位
                        object0 = 1 - np.abs(t0)
                        ph0 = np.angle(t0)

                        # 对像素强度（幅值大小）小于0的点的幅值和相位都设置为0
                        for ii in range(N2):
                            for jj in range(N2):
                                if object0[ii, jj] < 0:
                                    object0[ii, jj] = 0
                                    ph0[ii, jj] = 0

                        # # 构建support约束进行局部取样（只取中间得到的部分）转为复数图(设置为28*28)
                        support = supportPro0(N2)
                        support = support.astype(int)
                        object0 = object0 * support
                        object0 = 1 - object0
                        #相位约束

                        t0 = object0 * np.exp(1j * ph0)#



                        # 平面5的复数图使用平面4的prop111实现更新至平面4的全息图，并获取平面4的初始幅值和相位信息
                        field_final_updated1 = IFT((FT(t0)) * (prop111))
                        amplitude1 = abs(field_final_updated1)
                        phase1 = np.angle(field_final_updated1)#wenti


                        # Plane4
                        for ii in range(M1, M2):
                            for jj in range(M1, M2):
                                amplitude1[ii, jj] = measured1[ii - M1, jj - M1]


                        field_final1 = amplitude1 * np.exp(1j * phase1)
                        t1 = IFT((FT(field_final1)) * np.conj(prop111))


                        object1 = 1 - np.abs(t1)
                        ph1 = np.angle(t1)#wenti
                        for ii in range(N2):
                            for jj in range(N2):
                                if object1[ii, jj] < 0:
                                    object1[ii, jj] = 0
                                    ph1[ii, jj] = 0

                        object1 = object1 * support
                        object1 = 1 - object1



                        t1 = object1 * np.exp(1j * ph1)
                        field_final_updated2 = IFT((FT(t1)) * (prop222))
                        amplitude2 = np.abs(field_final_updated2)
                        phase2 = np.angle(field_final_updated2)

                        # Plane3
                        for ii in range(M1, M2):
                            for jj in range(M1, M2):
                                amplitude2[ii, jj] = measured2[ii - M1, jj - M1]

                        field_final2 = amplitude2 * np.exp(1j * phase2)
                        t2 = IFT((FT(field_final2)) * np.conj(prop222))
                        object2 = 1 - np.abs(t2)
                        ph2 = np.angle(t2)

                        for ii in range(N2):
                            for jj in range(N2):
                                if object2[ii, jj] < 0:
                                    object2[ii, jj] = 0
                                    ph2[ii, jj] = 0

                        object2 = object2 * support
                        object2 = 1 - object2

                        t2 = object2 * np.exp(1j * ph2)

                        field_final_updated3 = IFT((FT(t2)) * (prop333))
                        amplitude3 = abs(field_final_updated3)
                        phase3 = np.angle(field_final_updated3)

                        # Plane2
                        for ii in range(M1, M2):
                            for jj in range(M1, M2):
                                amplitude3[ii, jj] = measured3[ii - M1, jj - M1]

                        field_final3 = amplitude3 * np.exp(1j * phase3)
                        t3 = IFT((FT(field_final3)) * np.conj(prop333))
                        object3 = 1 - np.abs(t3)
                        ph3 = np.angle(t3)

                        for ii in range(N2):
                            for jj in range(N2):
                                if object3[ii, jj] < 0:
                                    object3[ii, jj] = 0
                                    ph3[ii, jj] = 0

                        object3 = object3 * support
                        object3 = 1 - object3

                        t3 = object3 * np.exp(1j * ph3)
                        field_final_updated4 = IFT((FT(t3)) * (prop444))
                        amplitude4 = abs(field_final_updated4)
                        phase4 = np.angle(field_final_updated4)

                        # Plane1
                        for ii in range(M1, M2):
                            for jj in range(M1, M2):
                                amplitude4[ii, jj] = measured4[ii - M1, jj - M1]

                        field_final4 = amplitude4 * np.exp(1j * phase4)
                        t4 = IFT((FT(field_final4)) * np.conj(prop444))

                        object4 = 1 - np.abs(t4)
                        ph4 = np.angle(t4)
                        for ii in range(N2):
                            for jj in range(N2):
                                if object4[ii, jj] < 0:
                                    object4[ii, jj] = 0
                                    ph4[ii, jj] = 0
                        object4 = object4 * support
                        object4 = 1 - object4
                        
                        if useNet:
                            for iii in range(0, 40):
                                inIn = getInsideIndex(600, 512)  # 用于NET
                                amp_phase = np.concatenate(
                                    [object4[None, inIn:inIn + 512, inIn:inIn + 512],
                                        ph4[None, inIn:inIn + 512, inIn:inIn + 512]],
                                    axis=0)
                                amp_phase = np.float32(amp_phase)
                                amp_phase = torch.from_numpy(amp_phase[None, :, :, :]).cuda()
                                amp_phase1, amp_phase = predictor_update_fn(amp_phase, vec_t, model=img_model)
                                # amp_phase1, amp_phase = corrector_update_fn(amp_phase, vec_t, model=img_model)
                                amp_phase = toNumpy(amp_phase)

                                object4[inIn:inIn + 512, inIn:inIn + 512] = amp_phase[0]
                                ph4[inIn:inIn + 512, inIn:inIn + 512] = amp_phase[1]

                        t4 = object4 * np.exp(1j * ph4)

                        # 最终1平面的复数图使用5平面的prop000得到5平面的全息图以及幅值、相位
                        field_final_updated0 = IFT((FT(t4)) * (prop000))

                        savePng(
                        f'./{savePath}/pic/{title}_planes{planes}/{picNO}/{kk}/t4_abs.png',
                        np.abs(t4))
                        savePng(
                        f'./{savePath}/pic/{title}_planes{planes}/{picNO}/{kk}/t4_angle.png',
                        np.angle(t4))
                        amplitude0 = abs(field_final_updated0)
                        phase0 = np.angle(field_final_updated0)
                    elif planes == 3:
                        # Plane5
                        # 数据保真(之后一轮迭代后又会产生amplitude0和phase0)，幅值部分进行叠加更新
                        for ii in range(M1, M2):
                            for jj in range(M1, M2):
                                amplitude0[ii, jj] = measured0[ii - M1, jj - M1]
                        # 全息图反向转播变成稀疏的复数图并使用了prop000进行外推
                        field_final0 = amplitude0 * np.exp(1j * phase0)
                        t0 = IFT((FT(field_final0)) * np.conj(prop000))

                        # 获取稀疏复数图的取反的振幅和原始相位
                        object0 = 1 - np.abs(t0)
                        ph0 = np.angle(t0)

                        # 对像素强度（幅值大小）小于0的点的幅值和相位都设置为0
                        for ii in range(N2):
                            for jj in range(N2):
                                if object0[ii, jj] < 0:
                                    object0[ii, jj] = 0
                                    ph0[ii, jj] = 0

                        # 构建support约束进行局部取样（只取中间得到的部分）转为复数图(设置为28*28)
                        support = supportPro0(N2)
                        support = support.astype(int)
                        object0 = object0 * support
                        object0 = 1 - object0

                        t0 = object0 * np.exp(1j * ph0)


                        # 平面5的复数图使用平面4的prop111实现更新至平面4的全息图，并获取平面4的初始幅值和相位信息
                        field_final_updated2 = IFT((FT(t0)) * (prop222))
                        amplitude2 = np.abs(field_final_updated2)
                        phase2 = np.angle(field_final_updated2)

                        # Plane3
                        for ii in range(M1, M2):
                            for jj in range(M1, M2):
                                amplitude2[ii, jj] = measured2[ii - M1, jj - M1]

                        field_final2 = amplitude2 * np.exp(1j * phase2)
                        t2 = IFT((FT(field_final2)) * np.conj(prop222))

                        object2 = 1 - np.abs(t2)
                        ph2 = np.angle(t2)

                        for ii in range(N2):
                            for jj in range(N2):
                                if object2[ii, jj] < 0:
                                    object2[ii, jj] = 0
                                    ph2[ii, jj] = 0

                        object2 = object2 * support
                        object2 = 1 - object2

                        t2 = object2 * np.exp(1j * ph2)

                        field_final_updated4 = IFT((FT(t2)) * (prop444))
                        amplitude4 = abs(field_final_updated4)
                        phase4 = np.angle(field_final_updated4)

                        # Plane1
                        for ii in range(M1, M2):
                            for jj in range(M1, M2):
                                amplitude4[ii, jj] = measured4[ii - M1, jj - M1]

                        field_final4 = amplitude4 * np.exp(1j * phase4)
                        t4 = IFT((FT(field_final4)) * np.conj(prop444))

                        object4 = 1 - np.abs(t4)
                        ph4 = np.angle(t4)
                        for ii in range(N2):
                            for jj in range(N2):
                                if object4[ii, jj] < 0:
                                    object4[ii, jj] = 0
                                    ph4[ii, jj] = 0
                        object4 = object4 * support
                        object4 = 1 - object4

                       
                        
                        if useNet:  
                            for iii in range(0, 40):
                                inIn = getInsideIndex(600, 512)  # 用于NET
                                amp_phase = np.concatenate(
                                    [object4[None, inIn:inIn + 512, inIn:inIn + 512],
                                     ph4[None, inIn:inIn + 512, inIn:inIn + 512]],
                                    axis=0)
                                amp_phase = np.float32(amp_phase)
                                amp_phase = torch.from_numpy(amp_phase[None, :, :, :]).cuda()
                                amp_phase1, amp_phase = predictor_update_fn(amp_phase, vec_t, model=img_model)
                                # amp_phase1, amp_phase = corrector_update_fn(amp_phase, vec_t, model=img_model)
                                amp_phase = toNumpy(amp_phase)

                                object4[inIn:inIn + 512, inIn:inIn + 512] = amp_phase[0]
                                ph4[inIn:inIn + 512, inIn:inIn + 512] = amp_phase[1]
                        
                        t4 = object4 * np.exp(1j * ph4)
                        
                        savePng(
                            f'./{savePath}/pic/{title}_planes{planes}/{picNO}/{kk}/t4_abs.png',
                            np.abs(t4))
                        savePng(
                            f'./{savePath}/pic/{title}_planes{planes}/{picNO}/{kk}/t4_angle.png',
                            np.angle(t4))


                        # 最终1平面的复数图使用5平面的prop000得到5平面的全息图以及幅值、相位
                        field_final_updated0 = IFT((FT(t4)) * (prop000))

                        amplitude0 = abs(field_final_updated0)
                        phase0 = np.angle(field_final_updated0)
                    elif planes == 2:
                        # Plane5
                        # 数据保真(之后一轮迭代后又会产生amplitude0和phase0)，幅值部分进行叠加更新
                        for ii in range(M1, M2):
                            for jj in range(M1, M2):
                                amplitude0[ii, jj] = measured0[ii - M1, jj - M1]
                        # 全息图反向转播变成稀疏的复数图并使用了prop000进行外推
                        field_final0 = amplitude0 * np.exp(1j * phase0)
                        t0 = IFT((FT(field_final0)) * np.conj(prop000))

                        # 获取稀疏复数图的取反的振幅和原始相位
                        object0 = 1 - np.abs(t0)
                        ph0 = np.angle(t0)

                        # 对像素强度（幅值大小）小于0的点的幅值和相位都设置为0
                        for ii in range(N2):
                            for jj in range(N2):
                                if object0[ii, jj] < 0:
                                    object0[ii, jj] = 0
                                    ph0[ii, jj] = 0

                        # 构建support约束进行局部取样（只取中间得到的部分）转为复数图(设置为28*28)
                        support = supportPro0(N2)
                        support = support.astype(int)
                        object0 = object0 * support
                        object0 = 1 - object0

                        t0 = object0 * np.exp(1j * ph0)

                        # 平面5的复数图使用平面4的prop111实现更新至平面4的全息图，并获取平面4的初始幅值和相位信息
                        field_final_updated4 = IFT((FT(t0)) * (prop444))
                        amplitude4 = abs(field_final_updated4)
                        phase4 = np.angle(field_final_updated4)

                        # Plane1
                        for ii in range(M1, M2):
                            for jj in range(M1, M2):
                                amplitude4[ii, jj] = measured4[ii - M1, jj - M1]

                        field_final4 = amplitude4 * np.exp(1j * phase4)
                        t4 = IFT((FT(field_final4)) * np.conj(prop444))

                        object4 = 1 - np.abs(t4)
                        ph4 = np.angle(t4)
                        for ii in range(N2):
                            for jj in range(N2):
                                if object4[ii, jj] < 0:
                                    object4[ii, jj] = 0
                                    ph4[ii, jj] = 0
                        object4 = object4 * support
                        object4 = 1 - object4

                        
                       
                        if useNet:  #
                            for iii in range(0, 40):
                                inIn = getInsideIndex(600, 512)  # 用于NET
                                amp_phase = np.concatenate(
                                    [object4[None, inIn:inIn + 512, inIn:inIn + 512],
                                     ph4[None, inIn:inIn + 512, inIn:inIn + 512]],
                                    axis=0)
                                amp_phase = np.float32(amp_phase)
                                amp_phase = torch.from_numpy(amp_phase[None, :, :, :]).cuda()
                                amp_phase1, amp_phase = predictor_update_fn(amp_phase, vec_t, model=img_model)
                                # amp_phase1, amp_phase = corrector_update_fn(amp_phase, vec_t, model=img_model)
                                amp_phase = toNumpy(amp_phase)

                                object4[inIn:inIn + 512, inIn:inIn + 512] = amp_phase[0]
                                ph4[inIn:inIn + 512, inIn:inIn + 512] = amp_phase[1]
                       
                        t4 = object4 * np.exp(1j * ph4)
                        savePng(
                            f'./{savePath}/pic/{title}_size_planes{planes}/{picNO}/{kk}/t4_abs.png',
                            np.abs(t4))
                        savePng(
                            f'./{savePath}/pic/{title}_size_planes{planes}/{picNO}/{kk}/t4_angle.png',
                            np.angle(t4))
                        # 最终1平面的复数图使用5平面的prop000得到5平面的全息图以及幅值、相位
                        field_final_updated0 = IFT((FT(t4)) * (prop000))
                        amplitude0 = abs(field_final_updated0)
                        phase0 = np.angle(field_final_updated0)
                    print(f"kk {kk}次")

    return pc_sampler

def get_ode_sampler(sde, shape, inverse_scaler,
                    denoise=False, rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
    """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

    def denoise_update_fn(model, x):
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        # Reverse diffusion predictor for denoising
        predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor_obj.update_fn(x, vec_eps)
        return x

    def drift_fn(model, x, t):
        """Get the drift function of the reverse-time SDE."""
        score_fn = get_score_fn(sde, model, train=False, continuous=True)
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def ode_sampler(model, z=None):
        """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
        with torch.no_grad():
            # Initial sample
            if z is None:
                # If not represent, sample the latent code from the prior distibution of the SDE.
                x = sde.prior_sampling(shape).to(device)
            else:
                x = z

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = drift_fn(model, x, vec_t)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                           rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(model, x)

            x = inverse_scaler(x)
            return x, nfe

    return ode_sampler

