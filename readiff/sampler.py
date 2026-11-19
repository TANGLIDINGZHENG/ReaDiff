"""UniPC data prediction with adaptive projected guidance."""
import torch

from .model import valid_mask


def projected_guidance(positive, negative, scale, mask, parallel_weight=1.0,
                       momentum=0.0, norm_threshold=0.0, previous=None):
    delta = (positive - negative).masked_fill(~mask[..., None], 0)
    running = delta if previous is None else delta + momentum * previous
    update = running
    if norm_threshold:
        norm = torch.linalg.vector_norm(update, dim=(1, 2), keepdim=True)
        update = update * (norm_threshold / norm.clamp_min(1e-12)).clamp(max=1)
    positive = positive.masked_fill(~mask[..., None], 0)
    if parallel_weight != 1:
        # Accumulate dot products in double precision before casting back.
        direction = positive.double()
        coefficient = ((update.double() * direction).sum((1, 2), keepdim=True)
                       / direction.square().sum((1, 2), keepdim=True).clamp_min(1e-12))
        parallel = (coefficient * direction).to(update.dtype)
        update = update - (1 - parallel_weight) * parallel
    return positive + (scale - 1) * update, running


def unipc_update(x, predictions, alphas, target_alpha, order, predict, correct=True, variant="bh2"):
    """Upstream UniPC's data-prediction B(h) update, generalized to motion tensors."""
    a0 = alphas[-1].sqrt()
    s0 = (1 - alphas[-1]).sqrt()
    at = target_alpha.sqrt()
    st = (1 - target_alpha).sqrt()
    l0 = a0.log() - s0.log()
    h = at.log() - st.log() - l0
    hh = -h
    phi1 = torch.expm1(hh)
    bh = hh if variant == "bh1" else phi1
    rks, differences = [], []
    for i in range(1, order):
        ai = alphas[-i - 1]
        rk = (0.5 * (ai.log() - torch.log1p(-ai)) - l0) / h
        rks.append(rk)
        differences.append((predictions[-i - 1] - predictions[-1]) / rk)
    rks = torch.stack(rks + [torch.ones_like(h)])
    phi_k, factorial, bs = phi1 / hh - 1, 1, []
    for i in range(1, order + 1):
        bs.append(phi_k * factorial / bh)
        factorial *= i + 1
        phi_k = phi_k / hh - 1 / factorial
    matrix = torch.stack([rks ** i for i in range(order)])
    rhs = torch.stack(bs)
    base = st / s0 * x - at * phi1 * predictions[-1]
    if differences:
        wp = (rhs.new_tensor([.5]) if order == 2
              else torch.linalg.solve(matrix[:-1, :-1], rhs[:-1]))
        predicted = base - at * bh * sum(w * d for w, d in zip(wp, differences))
    else:
        predicted = base
    if not correct:
        return predicted, None
    value = predict(predicted)
    wc = rhs.new_tensor([.5]) if order == 1 else torch.linalg.solve(matrix, rhs)
    correction = sum((w * d for w, d in zip(wc[:-1], differences)), 0)
    corrected = base - at * bh * (correction + wc[-1] * (value - predictions[-1]))
    return corrected, value


@torch.no_grad()
def sample(model, features, lengths, config, generator):
    global_video, local_video = features
    condition = model.denoiser.prepare_condition(local_video)
    options = config["sampling"]
    grid = options["timestep_indices"]
    solver_order = options["solver_order"]
    size = int(lengths.max())
    mask = valid_mask(lengths, size)
    momentum = None

    def predict(x, index):
        nonlocal momentum
        times = torch.full_like(lengths, index)
        x = x.masked_fill(~mask[..., None], 0)
        positive = model.denoiser(x, times, global_video, condition, lengths)
        negative = model.denoiser(x, times, global_video, condition, lengths, force_uncond=True)
        guided, momentum = projected_guidance(
            positive, negative, config["scale"], mask, options["apg_parallel"],
            options["apg_momentum"], options["apg_norm"], momentum
        )
        return guided

    x = torch.randn((len(lengths), size, model.latent_mean.numel()),
                    device=lengths.device, generator=generator)
    x.masked_fill_(~mask[..., None], 0)
    predictions = [predict(x, grid[0])]
    alphas = [model.alpha_bar[grid[0]]]
    for step, index in enumerate(grid[1:], start=1):
        target = model.alpha_bar[index]
        order = min(solver_order, len(predictions), len(grid) - step)
        last = step == len(grid) - 1
        x, value = unipc_update(x, predictions, alphas, target, order,
                                lambda y: predict(y, index), correct=not last)
        x.masked_fill_(~mask[..., None], 0)
        if not last:
            predictions = (predictions + [value])[-solver_order:]
            alphas = (alphas + [target])[-solver_order:]
    clean = predict(x, grid[-1])
    return (clean * model.latent_std + model.latent_mean).masked_fill(~mask[..., None], 0)
