# ---------------------------------------------------------------
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for NVAE. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
from copy import deepcopy
import torch
import torch.nn as nn
import numpy as np
import os
import yaml

import torch.distributed as dist
from torch.multiprocessing import Process
from torch.cuda.amp import autocast, GradScaler
from ddim.diffusion import Diffusion

from diffusion_model import AutoEncoder
from thirdparty.adamax import Adamax
import utils
import datasets
from torchvision.utils import save_image, make_grid

from fid.fid_score import compute_statistics_of_generator, load_statistics, calculate_frechet_distance
from fid.inception import InceptionV3


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def main(args):
    # ensures that weight initializations are all the same
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # torch.autograd.set_detect_anomaly(True)

    logging = utils.Logger(args.global_rank, args.save)
    writer = utils.Writer(args.global_rank, args.save)

    # Get data loaders.
    train_queue, valid_queue, num_classes = datasets.get_loaders(args)
    args.num_total_iter = len(train_queue) * args.epochs
    warmup_iters = len(train_queue) * args.warmup_epochs
    swa_start = len(train_queue) * (args.epochs - 1)

    arch_instance = utils.get_arch_cells(args.arch_instance)

    # TODO: Add denoiser for diffusion sampling 
    with open(args.ddim, "r") as f:
        ddim_config = yaml.safe_load(f)
    ddim_config = dict2namespace(ddim_config)
    denoiser = Diffusion(ddim_config)

    enc_config = deepcopy(ddim_config)
    # enc_config.model.in_channels=6
    enc_config.model.ch = 32
    enc_config.model.num_res_blocks = 3
    enc_config.model.ch_mult = [1, 2, 4, 8]

    # TODO: change autoencoer to Denoising Variational Autoencoder
    model = AutoEncoder(args, enc_config, writer, arch_instance)
    model = model.cuda()

    logging.info('args = %s', args)
    logging.info('param size = %fM ', utils.count_parameters_in_M(model))
    logging.info('groups per scale: %s, total_groups: %d', model.groups_per_scale, sum(model.groups_per_scale))

    if args.fast_adamax:
        # Fast adamax has the same functionality as torch.optim.Adamax, except it is faster.
        cnn_optimizer = Adamax(model.parameters(), args.learning_rate,
                               weight_decay=args.weight_decay, eps=1e-3)
    else:
        cnn_optimizer = torch.optim.Adamax(model.parameters(), args.learning_rate,
                                           weight_decay=args.weight_decay, eps=1e-3)

    cnn_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        cnn_optimizer, float(args.epochs - args.warmup_epochs - 1), eta_min=args.learning_rate_min)
    grad_scalar = GradScaler(2**10)

    num_output = utils.num_output(args.dataset)
    bpd_coeff = 1. / np.log(2.) / num_output

    # if load
    checkpoint_file = os.path.join(args.save, 'checkpoint.pt')
    if args.cont_training:
        logging.info('loading the model.')
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        init_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        model = model.cuda()
        cnn_optimizer.load_state_dict(checkpoint['optimizer'])
        grad_scalar.load_state_dict(checkpoint['grad_scalar'])
        cnn_scheduler.load_state_dict(checkpoint['scheduler'])
        global_step = checkpoint['global_step']
    else:
        global_step, init_epoch = 0, 0

    sample_latent(train_queue, model, denoiser, cnn_optimizer, grad_scalar, global_step, warmup_iters, writer, logging)

def sample_latent(train_queue, model, denoiser, cnn_optimizer, grad_scalar, global_step, warmup_iters, writer, logging):
    alpha_i = utils.kl_balancer_coeff(num_scales=model.num_latent_scales,
                                      groups_per_scale=model.groups_per_scale, fun='square')
    nelbo = utils.AvgrageMeter()
    model.train()
    for step, x in enumerate(train_queue):
        x = x[0] if len(x) > 1 else x
        x = x.cuda()

        # change bit length
        x = utils.pre_process(x, args.num_x_bits, normalize=True)

        # TODO: sample prev_x, next_x, trg_x, timestep
        tidx = denoiser.antithetic_timestep_sample(len(x))
        timestep =  denoiser.trg_step.index_select(0, tidx)
        prev_x, a_prev = denoiser.sample_noised(x, torch.randn_like(x), tidx+1)
        next_x, a_next = denoiser.sample_noised(x, torch.randn_like(x), tidx)
        # _, cond_x = denoiser.denoise_step(prev_x, tidx)
        _, trg_x = denoiser.denoise_step(next_x, tidx)
        # trg_eps = (prev_x - trg_x * a_prev.sqrt()) / (1 - a_prev).sqrt()

        # warm-up lr
        if global_step < warmup_iters:
            lr = args.learning_rate * float(global_step) / warmup_iters
            for param_group in cnn_optimizer.param_groups:
                param_group['lr'] = lr

        # sync parameters, it may not be necessary
        if step % 100 == 0:
            utils.average_params(model.parameters(), args.distributed)

        cnn_optimizer.zero_grad()
        with autocast():
            logits, log_q, log_p, kl_all, kl_diag = model(trg_x, prev_x, timestep)

            # TODO: assert pred_eps is NormalDecoder

            pred_x0 = model.decoder_output(logits, prev_x, a_prev)
            kl_coeff = utils.kl_coeff(global_step, args.kl_anneal_portion * args.num_total_iter,
                                      args.kl_const_portion * args.num_total_iter, args.kl_const_coeff)

            recon_loss = utils.reconstruction_loss(pred_x0, trg_x, crop=model.crop_output)
            balanced_kl, kl_coeffs, kl_vals = utils.kl_balancer(kl_all, kl_coeff, kl_balance=True, alpha_i=alpha_i)

            nelbo_batch = recon_loss + balanced_kl
            loss = torch.mean(nelbo_batch)
            norm_loss = model.spectral_norm_parallel()
            bn_loss = model.batchnorm_loss()
            # get spectral regularization coefficient (lambda)
            if args.weight_decay_norm_anneal:
                assert args.weight_decay_norm_init > 0 and args.weight_decay_norm > 0, 'init and final wdn should be positive.'
                wdn_coeff = (1. - kl_coeff) * np.log(args.weight_decay_norm_init) + kl_coeff * np.log(args.weight_decay_norm)
                wdn_coeff = np.exp(wdn_coeff)
            else:
                wdn_coeff = args.weight_decay_norm

            loss += norm_loss * wdn_coeff + bn_loss * wdn_coeff

        grad_scalar.scale(loss).backward()
        utils.average_gradients(model.parameters(), args.distributed)
        grad_scalar.step(cnn_optimizer)
        grad_scalar.update()
        nelbo.update(loss.data, 1)

        if ((global_step + 1) % 100 == 0) or (global_step == 0):
            if ((global_step + 1) % 100 == 0) or (global_step == 0):  # reduced frequency
                n = int(np.floor(np.sqrt(x.size(0))))
                x_img = trg_x[:n*n]
                # output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.sample()
                # TODO: make denoised state with pred eps
                # pred_eps = (prev_x - pred_x0.sample() * a_prev.sqrt()) / (1 - a_prev).sqrt()
                # _, output_img = denoiser.denoise_step(prev_x, tidx, eps=pred_eps)
                # output_img = (prev_x - (1 - a_prev).sqrt() * pred_eps.sample()) / a_prev.sqrt()
                output_img = pred_x0.sample()

                output_img = output_img[:n*n]
                prev_tiled = utils.tile_image(prev_x[:n*n], n)
                # cond_xtiled = utils.tile_image(cond_x[:n*n], n)
                x_tiled = utils.tile_image(x_img, n)
                output_tiled = utils.tile_image(output_img, n)
                in_out_tiled = ((torch.cat((prev_tiled, x_tiled, output_tiled), dim=2) + 1) * 0.5).clip(min=0, max=1)
                writer.add_image('reconstruction', in_out_tiled, global_step)

            # norm
            writer.add_scalar('train/norm_loss', norm_loss, global_step)
            writer.add_scalar('train/bn_loss', bn_loss, global_step)
            writer.add_scalar('train/norm_coeff', wdn_coeff, global_step)

            utils.average_tensor(nelbo.avg, args.distributed)
            logging.info('train %d %f', global_step, nelbo.avg)
            writer.add_scalar('train/nelbo_avg', nelbo.avg, global_step)
            writer.add_scalar('train/lr', cnn_optimizer.state_dict()[
                              'param_groups'][0]['lr'], global_step)
            writer.add_scalar('train/nelbo_iter', loss, global_step)
            writer.add_scalar('train/kl_iter', torch.mean(sum(kl_all)), global_step)
            writer.add_scalar('train/recon_iter', torch.mean(utils.reconstruction_loss(pred_x0, trg_x, crop=model.crop_output)), global_step)
            writer.add_scalar('kl_coeff/coeff', kl_coeff, global_step)
            total_active = 0
            for i, kl_diag_i in enumerate(kl_diag):
                utils.average_tensor(kl_diag_i, args.distributed)
                num_active = torch.sum(kl_diag_i > 0.1).detach()
                total_active += num_active

                # kl_ceoff
                writer.add_scalar('kl/active_%d' % i, num_active, global_step)
                writer.add_scalar('kl_coeff/layer_%d' % i, kl_coeffs[i], global_step)
                writer.add_scalar('kl_vals/layer_%d' % i, kl_vals[i], global_step)
            writer.add_scalar('kl/total_active', total_active, global_step)

        global_step += 1
        # break

    utils.average_tensor(nelbo.avg, args.distributed)
    return nelbo.avg, global_step


def create_generator_vae(model, batch_size, num_total_samples):
    num_iters = int(np.ceil(num_total_samples / batch_size))
    for i in range(num_iters):
        with torch.no_grad():
            logits = model.sample(batch_size, 1.0)
            output = model.decoder_output(logits)
            output_img = output.mean if isinstance(output, torch.distributions.bernoulli.Bernoulli) else output.mean()
        yield output_img.float()


def test_vae_fid(model, args, total_fid_samples):
    dims = 2048
    device = 'cuda'
    num_gpus = args.num_process_per_node * args.num_proc_node
    num_sample_per_gpu = int(np.ceil(total_fid_samples / num_gpus))

    g = create_generator_vae(model, args.batch_size, num_sample_per_gpu)
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], model_dir=args.fid_dir).to(device)
    m, s = compute_statistics_of_generator(g, model, args.batch_size, dims, device, max_samples=num_sample_per_gpu)

    # share m and s
    m = torch.from_numpy(m).cuda()
    s = torch.from_numpy(s).cuda()
    # take average across gpus
    utils.average_tensor(m, args.distributed)
    utils.average_tensor(s, args.distributed)

    # convert m, s
    m = m.cpu().numpy()
    s = s.cpu().numpy()

    # load precomputed m, s
    path = os.path.join(args.fid_dir, args.dataset + '.npz')
    m0, s0 = load_statistics(path)

    fid = calculate_frechet_distance(m0, s0, m, s)
    return fid


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6020'
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(args)
    cleanup()


def cleanup():
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('encoder decoder examiner')
    # experimental results
    parser.add_argument('--root', type=str, default='/tmp/nasvae/expr',
                        help='location of the results')
    parser.add_argument('--ddim', type=str, default='ddim/configs/celeba.yml',
                        help='location of ddim config')
    parser.add_argument('--save', type=str, default='exp',
                        help='id used for storing intermediate results')
    # data
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['cifar10', 'mnist', 'omniglot', 'celeba_64', 'celeba_256',
                                 'imagenet_32', 'ffhq', 'lsun_bedroom_128', 'stacked_mnist',
                                 'lsun_church_128', 'lsun_church_64'],
                        help='which dataset to use')
    parser.add_argument('--data', type=str, default='/tmp/nasvae/data',
                        help='location of the data corpus')
    # optimization
    parser.add_argument('--batch_size', type=int, default=200,
                        help='batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                        help='init learning rate')
    parser.add_argument('--learning_rate_min', type=float, default=1e-4,
                        help='min learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-4,
                        help='weight decay')
    parser.add_argument('--weight_decay_norm', type=float, default=0.,
                        help='The lambda parameter for spectral regularization.')
    parser.add_argument('--weight_decay_norm_init', type=float, default=10.,
                        help='The initial lambda parameter')
    parser.add_argument('--weight_decay_norm_anneal', action='store_true', default=False,
                        help='This flag enables annealing the lambda coefficient from '
                             '--weight_decay_norm_init to --weight_decay_norm.')
    parser.add_argument('--epochs', type=int, default=200,
                        help='num of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='num of training epochs in which lr is warmed up')
    parser.add_argument('--fast_adamax', action='store_true', default=False,
                        help='This flag enables using our optimized adamax.')
    parser.add_argument('--arch_instance', type=str, default='res_mbconv',
                        help='path to the architecture instance')
    # KL annealing
    parser.add_argument('--kl_anneal_portion', type=float, default=0.3,
                        help='The portions epochs that KL is annealed')
    parser.add_argument('--kl_const_portion', type=float, default=0.0001,
                        help='The portions epochs that KL is constant at kl_const_coeff')
    parser.add_argument('--kl_const_coeff', type=float, default=0.0001,
                        help='The constant value used for min KL coeff')
    # Flow params
    parser.add_argument('--num_nf', type=int, default=0,
                        help='The number of normalizing flow cells per groups. Set this to zero to disable flows.')
    parser.add_argument('--num_x_bits', type=int, default=8,
                        help='The number of bits used for representing data for colored images.')
    # latent variables
    parser.add_argument('--num_latent_scales', type=int, default=1,
                        help='the number of latent scales')
    parser.add_argument('--num_groups_per_scale', type=int, default=10,
                        help='number of groups of latent variables per scale')
    parser.add_argument('--num_latent_per_group', type=int, default=20,
                        help='number of channels in latent variables per group')
    parser.add_argument('--ada_groups', action='store_true', default=False,
                        help='Settings this to true will set different number of groups per scale.')
    parser.add_argument('--min_groups_per_scale', type=int, default=1,
                        help='the minimum number of groups per scale.')
    # encoder parameters
    parser.add_argument('--num_channels_enc', type=int, default=32,
                        help='number of channels in encoder')
    parser.add_argument('--num_preprocess_blocks', type=int, default=2,
                        help='number of preprocessing blocks')
    parser.add_argument('--num_preprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_enc', type=int, default=1,
                        help='number of cell for each conditional in encoder')
    # decoder parameters
    parser.add_argument('--num_channels_dec', type=int, default=32,
                        help='number of channels in decoder')
    parser.add_argument('--num_postprocess_blocks', type=int, default=2,
                        help='number of postprocessing blocks')
    parser.add_argument('--num_postprocess_cells', type=int, default=3,
                        help='number of cells per block')
    parser.add_argument('--num_cell_per_cond_dec', type=int, default=1,
                        help='number of cell for each conditional in decoder')
    parser.add_argument('--num_mixture_dec', type=int, default=1,
                        help='number of mixture components in decoder. set to 1 for Normal decoder.')

    parser.add_argument('--temb_dim', type=int, default=128,
                        help='number of channels in decoder')   
    # NAS
    parser.add_argument('--use_se', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--res_dist', action='store_true', default=False,
                        help='This flag enables squeeze and excitation.')
    parser.add_argument('--cont_training', action='store_true', default=False,
                        help='This flag enables training from an existing checkpoint.')
    # DDP.
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--global_rank', type=int, default=0,
                        help='rank of process among all the processes')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed used for initialization')
    args = parser.parse_args()
    args.save = args.root + '/eval-' + args.save
    utils.create_exp_dir(args.save)

    size = args.num_process_per_node

    if size > 1:
        args.distributed = True
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, main, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        # for debugging
        print('starting in debug mode')
        args.distributed = True
        init_processes(0, size, main, args)


