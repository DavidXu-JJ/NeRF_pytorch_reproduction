import os
import time

import imageio
import numpy as np
import torch
from load_llff import load_llff_data
from run_nerf_helpers import *

from tqdm import tqdm, trange

np.random.seed(0)
DEBUG = False

lrate_decay = 250
lrate = 5e-4

batch_chunk = 1024*32
# i_weights = 10000
# i_video = 50000
# i_testset = 50000
i_weights = 100
i_video = 200
i_testset = 200
i_print = 100

basedir = './logs'
expname = 'fern_test'

def batchify(fn, chunk):
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0],chunk)],0)
    return ret

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    # [N_rays, N_samples, 3] -> [N_rays * N_samples, 3]
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    # [N_rays * N_samples, 4 = rgb + density]
    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def batchify_rays(rays_flat, chunk=batch_chunk, **kwargs):
    """Render rays in smaller minibatches to avoid OOM.
    """
    all_ret = {}
    for i in range(0, rays_flat.shape[0], chunk):
        ret = render_rays(rays_flat[i:i+chunk], **kwargs)
        for k in ret:
            if k not in all_ret:
                all_ret[k] = []
            all_ret[k].append(ret[k])

    all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
    return all_ret

def create_nerf():
    embed_fn, input_ch = get_embedder(10, 0)

    input_ch_views = 0
    embeddirs_fn = None

    embeddirs_fn, input_ch_views = get_embedder(4,0)

    output_ch = 4
    skips = [4]
    model = NeRF(D = 8,
                 W = 256,
                 input_ch = input_ch,
                 output_ch = output_ch,
                 skips = skips,
                 input_ch_views = input_ch_views,
                 use_viewdirs = True).to(device)
    grad_vars = list(model.parameters())

    model_fine =  NeRF(D = 8,
                       W = 256,
                       input_ch = input_ch,
                       output_ch = output_ch,
                       skips = skips,
                       input_ch_views = input_ch_views,
                       use_viewdirs = True).to(device)
    grad_vars += list(model_fine.parameters())


    network_query_fn = lambda inputs, viewdirs, network_fn : \
                       run_network(inputs,viewdirs,network_fn,
                                   embed_fn = embed_fn,
                                   embeddirs_fn = embeddirs_fn,
                       # number of pts sent through network in parallel
                                   netchunk = 1024*64)

    optimizer = torch.optim.Adam(params=grad_vars,
                                 lr = 5e-4,
                                 betas=(0.9,0.999))

    start = 0

    render_kwargs_train = {
        'network_query_fn' : network_query_fn,
        'perturb' : 1.,
        'N_importance' : 64,
        'network_fine' : model_fine,
        'N_samples' : 64,
        'network_fn' : model,
        'use_viewdirs' : True,
        # set to render synthetic data on a white bkgd (always use for dvoxels)
        'white_bkgd' : False,
        # std dev of noise added to regularize sigma_a output, 1e0 recommended
        'raw_noise_std': 1e0,
    }

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw, dists, act_fn = F.relu : 1.-torch.exp(-act_fn(raw) * dists)

    # distances between each sampled neighbor point
    dists = z_vals[...,1:] - z_vals[...,:-1]
    # add distance between last point and infinity
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape)], -1)

    # torch.Size([N_rays, N_samples]) * torch.Size([N_rays, 1])
    # former dists comes from z_vals = time, means time interval
    # when time = 1, ray hit the n=1 in NDC
    # after the execution, dists becomes the real distance
    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    # rgb = [N_rays, N_samples, 3]
    # fall between [0, 1]
    rgb = torch.sigmoid(raw[...,:3])
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        if pytest:
            np.random.seed(0)
            noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
            noise = torch.Tensor(noise)

    # alpha = [N_rays, N_samples]
    # feed volume density
    alpha = raw2alpha(raw[...,3] + noise, dists)

    # cumulative product
    # alpha * T_i for each ray
    # weights: [N_rays, N_samples]
    weights = alpha * torch.cumprod( \
        torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), \
        -1)[:, :-1]

    # C = \sum{T_i * alpha * c}
    # rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
    rgb_map = torch.sum(weights[...,None] * rgb, -2)

    depth_map = torch.sum(weights * z_vals, -1)
    # z_vals * (single weights / accumulative weights of each ray)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    # accumulative weights along each rays
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        # the ray with low accumulative weights means that
        # there is no high opacity.
        # so rgb value of that ray will be bigger than or equal to 1, means white
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return rgb_map, disp_map, acc_map, weights, depth_map

def render_rays(ray_batch,
                network_fn,
                network_query_fn,
                N_samples,
                retraw=False,
                lindisp=False,
                perturb=0.,
                N_importance=0,
                network_fine=None,
                white_bkgd=False,
                raw_noise_std=0.,
                verbose=False,
                pytest=False):
    """Volumetric rendering.
    Args:
      ray_batch: array of shape [batch_size, ...]. All information necessary
        for sampling along a ray, including: ray origin, ray direction, min
        dist, max dist, and unit-magnitude viewing direction.
      network_fn: function. Model for predicting RGB and density at each point
        in space.
      network_query_fn: function used for passing queries to network_fn.
      N_samples: int. Number of different times to sample along each ray.
      retraw: bool. If True, include model's raw, unprocessed predictions.
      lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
      perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
        random points in time.
      N_importance: int. Number of additional times to sample along each ray.
        These samples are only passed to network_fine.
      network_fine: "fine" network with same spec as network_fn.
      white_bkgd: bool. If True, assume a white background.
      raw_noise_std: ...
      verbose: bool. If True, print more debugging info.
    Returns:
      rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
      disp_map: [num_rays]. Disparity map. 1 / depth.
      acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
      raw: [num_rays, num_samples, 4]. Raw predictions from model.
      rgb0: See rgb_map. Output for coarse model.
      disp0: See disp_map. Output for coarse model.
      acc0: See acc_map. Output for coarse model.
      z_std: [num_rays]. Standard deviation of distances along ray for each
        sample.
    """
    N_rays = ray_batch.shape[0]
    # [N_rays, 3] each
    rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6]
    viewdirs = ray_batch[:,-3:] if ray_batch.shape[-1] > 8 else None
    # bounds [N_rays, 2]
    bounds = torch.reshape(ray_batch[...,6:8],[-1,1,2])
    near, far = bounds[...,0], bounds[...,1]

    t_vals = torch.linspace(0., 1., steps = N_samples)
    if not lindisp:
        z_vals = near * (1.-t_vals) + far * t_vals
    else:
        z_vals = 1./(1./near * (1-t_vals) + 1./far * (t_vals))

    # sample N_samples number each ray
    z_vals = z_vals.expand([N_rays, N_samples])

    # perturbation
    if perturb > 0:
        # get intervals between samples
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)

        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape)

        # Pytest, overwrite u with numpy's fixed random numbers
        if pytest:
            np.random.seed(0)
            t_rand = np.random.rand(*list(z_vals.shape))
            t_rand = torch.Tensor(t_rand)

        z_vals = lower + (upper - lower) * t_rand

    # rays_o = [N_rays, 3]
    # z_vals = [N_rays, N_sample]
    # z_vals can be seen as time
    # pts = [N_rays, N_sample, 3]
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

    raw = network_query_fn(pts, viewdirs, network_fn)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map

        z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        # z_samples [N_rays, N_important]
        # the sample points satisfy the cdf of the weights predicted by the coarse network,
        # more valid points will be sampled according to the coarse network prediction.
        z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], N_importance, det=(perturb==0.), pytest=pytest)
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)

        # pts [N_rays, N_samples + N_importance, 3]
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]

        run_fn = network_fn if network_fine is None else network_fine

        raw = network_query_fn(pts, viewdirs, run_fn)

        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d,
                                                                     raw_noise_std, white_bkgd,
                                                                     pytest=pytest)
    ret = {'rgb_map' : rgb_map, 'disp_map' : disp_map, 'acc_map' : acc_map}

    if retraw:
        ret['raw'] = raw
    if N_importance > 0:
        ret['rgb0'] = rgb_map_0
        ret['disp0'] = disp_map_0
        ret['acc0'] = acc_map_0
        ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)

    for k in ret:
        if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
            print(f"! [Numerical Error] {k} contains nan or inf.")
    return ret

def render(H, W, K, chunk=batch_chunk, rays=None, c2w=None, ndc=True,
           near=0., far=1.,
           use_viewdirs=False, c2w_staticcam=None,
           **kwargs):
    """Render rays
    Args:
      H: int. Height of image in pixels.
      W: int. Width of image in pixels.
      focal: float. Focal length of pinhole camera.
      chunk: int. Maximum number of rays to process simultaneously. Used to
        control maximum memory usage. Does not affect final results.
      rays: array of shape [2, batch_size, 3]. Ray origin and direction for
        each example in batch.
      c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
      ndc: bool. If True, represent ray origin, direction in NDC coordinates.
      near: float or array of shape [batch_size]. Nearest distance for a ray.
      far: float or array of shape [batch_size]. Farthest distance for a ray.
      use_viewdirs: bool. If True, use viewing direction of a point in space in model.
      c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
       camera while using other c2w argument for viewing directions.
    Returns:
      rgb_map: [batch_size, 3]. Predicted RGB values for rays.
      disp_map: [batch_size]. Disparity map. Inverse of depth.
      acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
      extras: dict with everything returned by render_rays().
    """
    if c2w is not None:
        # special case to render full image
        rays_o, rays_d = get_rays(H, W, K, c2w)
    else:
        # use provided ray batch
        rays_o, rays_d = rays

    if use_viewdirs:
        # provide ray directions as input
        viewdirs = rays_d
        if c2w_staticcam is not None:
            # special case to visualize effect of viewdirs
            rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
        # viewdirs = [batch, 3]
        # normalize along the 3 within one ray's direction
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1,3]).float()

    # shape = [batch, 3]
    sh = rays_d.shape

    if ndc:
        # for forward facing scenes
        rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

    # Create ray batch
    rays_o = torch.reshape(rays_o, [-1,3]).float()
    rays_d = torch.reshape(rays_d, [-1,3]).float()

    near, far = near * torch.ones_like(rays_d[...,:1]), far * torch.ones_like(rays_d[...,:1])
    # [batch, 8]
    rays = torch.cat([rays_o, rays_d, near, far], -1)
    if use_viewdirs:
        rays = torch.cat([rays, viewdirs], -1)

    all_ret = batchify_rays(rays, chunk, **kwargs)
    for k in all_ret:
        k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
        all_ret[k] = torch.reshape(all_ret[k], k_sh)

    k_extract = ['rgb_map', 'disp_map', 'acc_map']
    ret_list = [all_ret[k] for k in k_extract]
    ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
    return ret_list + [ret_dict]

def render_path(render_poses, hwf, K, chunk, render_kwargs,
                gt_imgs=None, savedir=None, render_factor=0):
    H, W, focal = hwf

    if render_factor != 0:
        # Render downsampled for speed
        H = H // render_factor
        W = W // render_factor
        focal = focal / render_factor

    rgbs = []
    disps = []

    t = time.time()
    for i, c2w in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3,:4], **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

        if savedir is not None:
            rgb8 = to8b(rgbs[-1])
            filename = os.path.join(savedir, '{:03d}.png'.format(i))
            imageio.imwrite(filename, rgb8)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps

def train():
    images, poses, bds, render_poses, i_test = load_llff_data("./data/nerf_llff_data/fern", 8,
                                                              recenter=True, bd_factor=.75,
                                                              spherify=False)
    hwf = poses[0,:3,-1]
    # poses = [ right up back translate] = c2w
    poses = poses[:,:3,:4]  #[20,3,4]
    print('Loaded llff', images.shape, render_poses.shape)
    if not isinstance(i_test, list):
        i_test = [i_test]

    llffhold = 8
    print("Auto LLFF holdout 8 images")
    i_test = np.arange(images.shape[0])[::8]

    i_val = i_test
    i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])

    print("Defining Bounds")

    near = 0.
    far = 1.
    print("Near Far",near,far)

    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [ H, W, focal]

    K = None
    if K is None:
        K = np.array([
            [focal,      0, 0.5*W],
            [    0,  focal, 0.5*H],
            [    0,      0,     1]
        ])

    # basedir = "./"
    # expname = "config"
    # os.makedirs(os.path.join(basedir,expname),exist_ok=True)

    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf()
    global_step = start

    bds_dict = {
        'near' : near,
        'far' : far,
    }

    render_kwargs_test.update(bds_dict)
    render_kwargs_train.update(bds_dict)

    render_poses = torch.Tensor(render_poses)

    # batch size (number of random rays per gradient step)
    N_rand = 1024
    use_batching = True
    if use_batching:
        print("get rays")
        # poses : [20,3,4]
        # for takes [3,4]
        # rays : [N, ro+rd, H, W, 3]
        #  = (20, 2, 378, 504, 3)
        rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:,:3,:4]], 0)
        print('done, concats')

        # images            (N, H, W, 3)
        # images[:,None]    (N, rgb, H, W, 3)
        rays_rgb = np.concatenate([rays, images[:,None]], 1)

        # [N, H, W, ro+rd+rgb, 3]
        rays_rgb = np.transpose(rays_rgb, [0,2,3,1,4])

        # train images only
        rays_rgb = np.stack([rays_rgb[i] for i in i_train],0)
        # [N*H*W, ro+rd+rgb, 3]
        rays_rgb = np.reshape(rays_rgb, [-1,3,3])

        rays_rgb = rays_rgb.astype(np.float32)
        print('shuffle rays')
        np.random.shuffle(rays_rgb)

        print('done')
        i_batch = 0

    if use_batching:
        images = torch.Tensor(images)
    poses = torch.Tensor(poses)
    if use_batching:
        rays_rgb = torch.Tensor(rays_rgb)
    # N_iters = 200000 + 1
    N_iters = 1000 + 1
    print("Begin")
    print("Train views are", i_train)
    print("Test views are", i_test)
    print("Val views are", i_val)

    start = start + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        # Sample random ray batch
        if use_batching:
            # Random over all images
            # [ray batch, ray_o+ray_d+rgb, 3]
            batch = rays_rgb[i_batch:i_batch+N_rand]
            # [ray_o+ray_d+rgb, ray batch, 3]
            batch = torch.transpose(batch, 0 ,1)
            # [rays_o, rays_d], real rgb
            batch_rays, target_s = batch[:2], batch[2]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                i_batch = 0

        # chunk:
        # number of rays processed in parallel
        rgb, disp, acc, extras = render(H, W, K, chunk=batch_chunk, rays=batch_rays,
                                 verbose = i < 10, retraw = True,
                                 **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr(img_loss)

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward();
        optimizer.step()

        decay_rate = 0.1
        decay_steps = lrate_decay * 1000
        new_lrate = lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        dt = time.time() - time0
        if i % i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            torch.save({
                'global_step' : global_step,
                'network_fn_state_dict' : render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict' : render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
            }, path)
            print("Saved checkpoints at ",path)

        if i % i_video == 0 and i > 0:
            # Turn on testing mode
            with torch.no_grad():
                rgbs, disps = render_path(render_poses, hwf, K, batch_chunk, render_kwargs_test)
            print('Done, saving', rgbs.shape, disps.shape)
            moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname,i))
            imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)
            imageio.mimwrite(moviebase + 'disp.mp4', to8b(disps / np.max(disps)), fps=30, quality=8)

        if i % i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06}'.format(i))
            os.makedirs(testsavedir, exist_ok = True)
            print('test poses shape', poses[i_test].shape)
            with torch.no_grad():
                render_path(torch.Tensor(poses[i_test]), hwf, K, batch_chunk, render_kwargs_test, gt_imgs=images[i_test], savedir=testsavedir)
            print('Saved test set')

        if i % i_print == 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item()} PSNR: {psnr.item()}")

        global_step += 1

if __name__ =='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print(device)
    train()