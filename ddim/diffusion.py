import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data
import torchvision.utils as tvu

from ddim.denoising import compute_alpha
from ddim.model import Denoiser


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, config, device=None):
        self.config = config
        if device is None:
            device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()
        
        #TODO: get pretrained denoiser here
        self.trg_denoiser = Denoiser(config)
        pretrained_weight =  torch.load(config.variable.pretrained)
        self.trg_denoiser.load_state_dict(pretrained_weight)
        self.trg_denoiser.to(device)
        self.trg_denoiser.eval()

        self.timesteps = config.variable.timesteps
        skip = self.num_timesteps // (self.timesteps + 1)
        self.trg_step = torch.arange(0, self.num_timesteps, skip, device=self.device)

        self.img_channels = config.model.in_channels
        self.img_size = config.data.image_size

    
    def antithetic_timestep_sample(self, N):
        S = (len(self.trg_step) - 1)
        return ((torch.arange(N) + (S * torch.rand(1)).floor().long()) % S).to(self.device)
        # t = torch.arange(N).to(self.device) + self.timesteps * torch.rand(1).to(self.device)
        # t = torch.randint(
        #     low=0, high=self.timesteps, size=(N // 2 + 1,)
        # ).to(self.device)
        # t = torch.cat([t, self.timesteps - t - 1], dim=0)[:N] + 1
        # return t

    def sample_noised(self, x, e, t):
        step = self.trg_step.index_select(0, t)
        a = compute_alpha(self.betas, step)
        return x * a.sqrt() + e * (1 - a).sqrt(), a

    def denoise_step(self, noised, t, eps=None, **kwargs):
        assert (t.max() < len(self.trg_step)).all().item() and (t.min() >=0).all().item()
        step = self.trg_step.index_select(0, t)
        next_step = self.trg_step.index_select(0, (t-1).clip(min=0))
        at = compute_alpha(self.betas, step)
        at_next = compute_alpha(self.betas, next_step)
        # eps is distribution use sample and return distribution
        if eps is None:
            eps = self.trg_denoiser(noised, step)

        x0_t = (noised - eps * (1 - at).sqrt()) / at.sqrt()
        c1 = (
            kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
        )
        c2 = ((1 - at_next) - c1 ** 2).sqrt()
        xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(noised) + c2 * eps
        
        return xt_next, x0_t

    def sample(self, N, model=None, temperature=1.0):
        with torch.no_grad():
            if model is None:
                # Implement this case also
                model = self.trg_denoiser

            xt = torch.randn(
                (N, self.img_channels, self.img_size, self.img_size), device=self.device
            )

            xs_list = [xt]
            x0_list = [xt]
            tidx_list = reversed(list(range(0, len(self.trg_step))))

            for ti in tidx_list:
                tidx = ti * torch.ones(len(xt), device=self.device, dtype=torch.long)
                timestep = self.trg_step.index_select(0, tidx)
                # next_timestep = self.trg_step.index_select(0, tidx)
                # _, cond_x = self.denoise_step(xt, tidx-1)
                logits = model.inference(xt, timestep, temperature)

                a = compute_alpha(self.betas, timestep)
                pred_x0 = model.decoder_output(logits, xt, a).sample()
                pred_eps = (xt - pred_x0.sample() * a.sqrt()) / (1 - a).sqrt()

                xs, x0_t = self.denoise_step(xt, tidx, eps=pred_eps)
                xs_list.append(xs.clone())
                x0_list.append(x0_t)
                xt = xs

        # return xs_list, x0_list
        return xs_list[-1]

    def test(self):
        pass