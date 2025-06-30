import torch

# adds enough dimensions so x has target_dims number of dimensions
def append_dims(x, target_dims):
    return x[(...,) + (None,) * (target_dims - x.ndim)]

# prevents cfg from blowing up or vanishing noise scale
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=1.0):
    if guidance_rescale == 0:
        return noise_cfg

    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1.0 - guidance_rescale) * noise_cfg
    return noise_cfg


def fm_wrapper(transformer, t_scale=1000.0):
    def k_model(x, sigma, **extra_args): # x = noised image
        dtype = extra_args['dtype']
        cfg_scale = extra_args['cfg_scale']
        cfg_rescale = extra_args['cfg_rescale']
        concat_latent = extra_args['concat_latent']

        original_dtype = x.dtype
        sigma = sigma.float()

        x = x.to(dtype)
        timestep = (sigma * t_scale).to(dtype)

        if concat_latent is None:
            hidden_states = x
        else:
            hidden_states = torch.cat([x, concat_latent.to(x)], dim=1)
        # positive = w/ prompt condition, negative = w/o 
        pred_positive = transformer(hidden_states=hidden_states, timestep=timestep, return_dict=False, **extra_args['positive'])[0].float()

        if cfg_scale == 1.0:
            pred_negative = torch.zeros_like(pred_positive)
        else:
            pred_negative = transformer(hidden_states=hidden_states, timestep=timestep, return_dict=False, **extra_args['negative'])[0].float()

        pred_cfg = pred_negative + cfg_scale * (pred_positive - pred_negative)
        pred = rescale_noise_cfg(pred_cfg, pred_positive, guidance_rescale=cfg_rescale) 

        x0 = x.float() - pred.float() * append_dims(sigma, x.ndim)

        return x0.to(dtype=original_dtype)

    return k_model
