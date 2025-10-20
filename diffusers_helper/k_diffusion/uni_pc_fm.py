# Better Flow Matching UniPC by Lvmin Zhang
# (c) 2025
# CC BY-SA 4.0
# Attribution-ShareAlike 4.0 International Licence


import os

import torch

from tqdm.auto import trange


def expand_dims(v, dims):
    return v[(...,) + (None,) * (dims - 1)]


class FlowMatchUniPC:
    def __init__(self, model, extra_args, variant='bh1'):
        self.model = model
        self.variant = variant
        self.extra_args = extra_args

    def model_fn(self, x, t):
        return self.model(x, t, **self.extra_args)

    def update_fn(self, x, model_prev_list, t_prev_list, t, order):
        assert order <= len(model_prev_list)
        dims = x.dim()

        t_prev_0 = t_prev_list[-1]
        lambda_prev_0 = - torch.log(t_prev_0)
        lambda_t = - torch.log(t)
        model_prev_0 = model_prev_list[-1]

        h = lambda_t - lambda_prev_0

        rks = []
        D1s = []
        for i in range(1, order):
            t_prev_i = t_prev_list[-(i + 1)]
            model_prev_i = model_prev_list[-(i + 1)]
            lambda_prev_i = - torch.log(t_prev_i)
            rk = ((lambda_prev_i - lambda_prev_0) / h)[0]
            rks.append(rk)
            D1s.append((model_prev_i - model_prev_0) / rk)

        rks.append(1.)
        rks = torch.tensor(rks, device=x.device)

        R = []
        b = []

        hh = -h[0]
        h_phi_1 = torch.expm1(hh)
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.variant == 'bh1':
            B_h = hh
        elif self.variant == 'bh2':
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError('Bad variant!')

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= (i + 1)
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R)
        b = torch.tensor(b, device=x.device)

        use_predictor = len(D1s) > 0

        if use_predictor:
            D1s = torch.stack(D1s, dim=1)
            if order == 2:
                rhos_p = torch.tensor([0.5], device=b.device)
            else:
                rhos_p = torch.linalg.solve(R[:-1, :-1], b[:-1])
        else:
            D1s = None
            rhos_p = None

        if order == 1:
            rhos_c = torch.tensor([0.5], device=b.device)
        else:
            rhos_c = torch.linalg.solve(R, b)

        x_t_ = expand_dims(t / t_prev_0, dims) * x - expand_dims(h_phi_1, dims) * model_prev_0

        if use_predictor:
            pred_res = torch.tensordot(D1s, rhos_p, dims=([1], [0]))
        else:
            pred_res = 0

        x_t = x_t_ - expand_dims(B_h, dims) * pred_res
        model_t = self.model_fn(x_t, t)

        if D1s is not None:
            corr_res = torch.tensordot(D1s, rhos_c[:-1], dims=([1], [0]))
        else:
            corr_res = 0

        D1_t = (model_t - model_prev_0)
        x_t = x_t_ - expand_dims(B_h, dims) * (corr_res + rhos_c[-1] * D1_t)

        return x_t, model_t

    def sample(self, x, sigmas, callback=None, disable_pbar=False):
        order = min(3, len(sigmas) - 2)
        model_prev_list, t_prev_list = [], []
        # Get rank ID from environment variable
        rank = int(os.environ.get('RANK', 0))
        
        for i in trange(len(sigmas) - 1, disable=disable_pbar, desc=f"[Rank {rank}] Sampling"):
            vec_t = sigmas[i].expand(x.shape[0])

            if i == 0:
                model_prev_list = [self.model_fn(x, vec_t)]
                t_prev_list = [vec_t]
            elif i < order:
                init_order = i
                x, model_x = self.update_fn(x, model_prev_list, t_prev_list, vec_t, init_order)
                model_prev_list.append(model_x)
                t_prev_list.append(vec_t)
            else:
                x, model_x = self.update_fn(x, model_prev_list, t_prev_list, vec_t, order)
                model_prev_list.append(model_x)
                t_prev_list.append(vec_t)

            model_prev_list = model_prev_list[-order:]
            t_prev_list = t_prev_list[-order:]

            if callback is not None:
                callback({'x': x, 'i': i, 'denoised': model_prev_list[-1]})

        return model_prev_list[-1]


def sample_unipc(model, noise, sigmas, extra_args=None, callback=None, disable=False, variant='bh1'):
    assert variant in ['bh1', 'bh2']
    return FlowMatchUniPC(model, extra_args=extra_args, variant=variant).sample(noise, sigmas=sigmas, callback=callback, disable_pbar=disable)
