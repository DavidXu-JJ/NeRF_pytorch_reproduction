

import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Mean squared error
img2mse = lambda x, y : torch.mean((x - y) ** 2)

# Peak Signal-to-Noise Ratio
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(device))

# change float [0,1] to RGB [0,255]
to8b= lambda x : (255 * np.clip(x,0,1)).astype(np.uint8)


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            # 2^{ [0,L-1] }
            # log sampling
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            # linear sampling
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)


        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self,inputs):
        # -1 means concat on the last dim
        return torch.cat( [fn(inputs) for fn in self.embed_fns ], -1)

def get_embedder(multires, i=0):
    if i == -1:
        return nn.Identity(), 3
    embed_kwargs = {
        'include_input' :   True,
        'input_dims'    :   3,
        'max_freq_log2' :   multires-1,
        'num_freqs'     :   multires,
        'log_sampling'  :   True,
        'periodic_fns'  :   [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x , eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

class NeRF(nn.Module):
    def __init__(self,
                 D              =   8,
                 W              =   256,
                 # 3D location
                 input_ch       =   3,
                 # 3D Cartesian unit vector
                 input_ch_views =   3,
                 output_ch      =   4,
                 skips          =   [4],
                 use_viewdirs   =   False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch,W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)]
        )

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self,x):
        # 3D positions, 3D views
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], -1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                # concat the x to channel dim
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs

# Ray helpers
def get_rays(H, W, K, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def get_rays_np(H, W, K, c2w):
    # the proportion of H,W with focal, define the maximum angle of rays' direction
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    # [H, W, 3]
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    # [H, W, 1, 3] * [3 * 3] = [H, W, 3, 3]
    # [3 * 3] broadcast multiplied by [1 * 3], multiply on the column
    # equals to c2w @ dirs
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1],np.shape(rays_d))
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # Shift ray origins to near plane
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    rays_o = torch.stack([o0,o1,o2], -1)
    rays_d = torch.stack([d0,d1,d2], -1)

    return rays_o, rays_d

# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    # weights in volume rendering
    weights = weights + 1e-5
    # probability density function, normalize to 1
    # [N_rays, N_important-1]
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    # cumulative distribution function
    cdf = torch.cumsum(pdf, -1)
    # [N_rays, N_important-1]
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Pytest, overwrite u with numpy's fixed random numbers
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = torch.Tensor(u).to(device)

    u = u.contiguous()
    indices = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(indices-1), indices-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(indices), indices)
    # cdf[below] < u < cdf[above]
    # if u < cdf[0], below = 0
    # if u > cdf[size - 1], above = size - 1
    # [N_rays, N_important, 2]
    indices_g = torch.stack([below, above], -1)

    # [N_rays, N_important, N_important-1]
    matched_shape = [indices_g.shape[0], indices_g.shape[1], cdf.shape[-1]]

    # inputs: [N_rays, 1, N_important-1] -> [N_rays, N_important, N_important-1]
    # indices: [N_rays, N_important, 2]
    # outputs: [N_rays, N_important, 2]
    # for each random sampled pts[N_important],
    # take below and above value from cdf[N_important-1]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, indices_g)

    # outputs: [N_rays, N_important, 2]
    # for point with cdf = u, its z_val lies between [0] and [1]
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, indices_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    # if denom == 0, means u is smaller or bigger than any element in each ray's cdf
    # when denom == 0, bins_g[...,1] = bins_g[...,0], t can takes any number
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    # if denom != 0, t = (u-cdf_g[...,0])/(cdf_g[...,1]-cdf_g[...,0])
    # samples takes an interpolation
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples