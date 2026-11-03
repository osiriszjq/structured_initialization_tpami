"""Structured initialization and position-reference entropy calibration.

Copyright Jianqiao Zheng. Retains the original impulse/inverse/SVD construction.
Final FE: fixed18 * trainable PE base, one-shot position entropy .90.
Native timm attention forward is unchanged. See docs/CHANGES.md.
"""
from __future__ import annotations
import copy
import hashlib
import math
import random
from numbers import Real
from dataclasses import dataclass
from typing import Iterator, List, Optional
from contextlib import contextmanager
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import parametrize
from timm.layers import Attention, trunc_normal_
from timm.models.vision_transformer import VisionTransformer
METHOD_VERSION = "structured_impulse_v1"
SOURCE_SHA256 = "846dcf8a54fa48d7a0b7b6eabfe3241eae85eb8ec95025921b2572647011dd48"
_APPLIED_ATTRIBUTE = "_structured_initialization"
_OFFSET_ORDER = (4, 1, 5, 3, 7, 0, 2, 6, 8)
VERSION = "position_entropy_calibration_v1"
BASE_NAME = 'parametrizations.pos_embed.original'
ALPHA_NAME = 'parametrizations.pos_embed.0.alpha'
REPRESENTATIONS = ('S','F','L')


@dataclass(frozen=True)
class AttentionTarget:
    path: str
    module: Attention
    query: slice
    key: slice
    value: slice

def iter_native_attention_targets(model: nn.Module) -> Iterator[AttentionTarget]:
    """Yield native timm Attention modules and their fused-QKV row slices."""
    for path, module in model.named_modules():
        if not isinstance(module, Attention):
            continue
        dim = module.qkv.in_features
        if module.qkv.out_features != 3 * dim:
            raise ValueError(f"{path}.qkv does not use the expected fused [Q;K;V] layout")
        if dim % module.num_heads:
            raise ValueError(f"{path} has an embedding dimension not divisible by its head count")
        yield AttentionTarget(
            path=path,
            module=module,
            query=slice(0, dim),
            key=slice(dim, 2 * dim),
            value=slice(2 * dim, 3 * dim),
        )

def validate_native_vit_contract(model: nn.Module, expected_depth: Optional[int] = None) -> List[AttentionTarget]:
    """Return validated targets or raise before a later initializer mutates weights."""
    targets = list(iter_native_attention_targets(model))
    if not targets:
        raise ValueError("model contains no native timm Attention modules")
    if expected_depth is not None and len(targets) != expected_depth:
        raise ValueError(f"expected {expected_depth} attention blocks, found {len(targets)}")
    return targets

def _positive_scale(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be an explicit positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result

def _finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values; no alternative solver is applied")

def _validate_model(model: nn.Module):
    if type(model) is not VisionTransformer:
        raise ValueError("release-style initialization requires a native timm VisionTransformer")
    if getattr(model, _APPLIED_ATTRIBUTE, None) is not None:
        raise ValueError("release-style initialization has already been applied to this model")
    if getattr(model, "dynamic_img_size", False):
        raise ValueError("dynamic image grids are outside this reference adapter")
    if getattr(model, "num_reg_tokens", 0):
        raise ValueError("register tokens are outside this reference adapter")
    prefix_count = int(model.num_prefix_tokens)
    if prefix_count not in (0, 1) or prefix_count != int(model.cls_token is not None):
        raise ValueError("only zero prefix tokens or one class token are supported")
    if not isinstance(model.norm_pre, nn.Identity):
        raise ValueError("pre-normalized ViT is outside this reference adapter")

    for name, parameter in model.named_parameters():
        if parameter.device.type != "cpu" or parameter.dtype != torch.float32:
            raise ValueError(f"{name}: initialize on CPU in float32 before device/precision conversion")
        _finite(name, parameter)

    positions = model.pos_embed
    if not isinstance(positions, nn.Parameter) or positions.ndim != 3 or positions.shape[0] != 1:
        raise ValueError("a learned positional parameter of shape [1, tokens, dim] is required")
    if not positions.requires_grad:
        raise ValueError("the positional parameter must remain trainable")
    grid = tuple(int(size) for size in model.patch_embed.grid_size)
    if len(grid) != 2 or min(grid) <= 0:
        raise ValueError("a fixed two-dimensional patch grid is required")
    patch_count = grid[0] * grid[1]
    positional_prefix = 0 if model.no_embed_class else prefix_count
    if positions.shape[1:] != (patch_count + positional_prefix, model.embed_dim):
        raise ValueError("positional embedding length does not match the patch grid/prefix contract")

    targets = validate_native_vit_contract(model, expected_depth=len(model.blocks))
    if [target.path for target in targets] != [f"blocks.{index}.attn" for index in range(len(model.blocks))]:
        raise ValueError("only the native transformer blocks may contain attention targets")
    for index, target in enumerate(targets):
        attention = target.module
        if any(not isinstance(norm, nn.Identity) for norm in (attention.q_norm, attention.k_norm, attention.norm)):
            raise ValueError("Q/K or attention-output normalization is outside this reference adapter")
        if not isinstance(model.blocks[index].norm1, nn.LayerNorm):
            raise ValueError("the reference expects ordinary pre-attention LayerNorm")
        if attention.qkv.in_features != model.embed_dim or attention.proj.weight.shape != (model.embed_dim, model.embed_dim):
            raise ValueError("all attention blocks must use the model embedding dimension")
        if not math.isclose(attention.scale, attention.head_dim ** -0.5, rel_tol=1e-12):
            raise ValueError("the reference expects the standard inverse-square-root attention scale")
        if not attention.qkv.weight.requires_grad or not attention.proj.weight.requires_grad:
            raise ValueError("QKV and output projections must remain trainable")
    return targets, grid, positional_prefix

def _mimetic_factors(dim: int, alpha: float, beta: float, sign: int):
    # Preserve NumPy's float64 draw/SVD and the donor's global RNG consumption.
    matrix = alpha * np.random.uniform(
        size=(dim, dim), low=-3 ** 0.5 / dim ** 0.5, high=3 ** 0.5 / dim ** 0.5
    ) + sign * beta * np.eye(dim)
    left, singular_values, right_t = np.linalg.svd(matrix)
    root = np.diag(np.sqrt(singular_values))
    return left @ root, root @ right_t

def _impulse_factors(positions: torch.Tensor, grid: tuple[int, int], heads: int, head_dim: int):
    height, width = grid
    target = torch.zeros((heads, height * width, height * width), dtype=torch.float32)
    for head in range(heads):
        flat_offset = _OFFSET_ORDER[head % len(_OFFSET_ORDER)]
        row_offset, column_offset = flat_offset // 3 - 1, flat_offset % 3 - 1
        row_block = torch.zeros((width, width), dtype=torch.float32)
        for column in range(0 - min(0, column_offset), width - max(0, column_offset)):
            row_block[column, column + column_offset] = 1
        for row in range(0 - min(0, row_offset), height - max(0, row_offset)):
            target[head, row * width:(row + 1) * width,
                   (row + row_offset) * width:(row + row_offset + 1) * width] += row_block

    # The initial uniform draw is deliberately retained even though overwritten:
    # removing it changes the noise and every later draw for a fixed seed.
    background = torch.rand_like(target)
    trunc_normal_(background, std=0.6)
    score_target = 1000.0 * target + 3.0 * (background - 0.0) * (target == 0)
    pseudo = F.layer_norm(positions, [positions.shape[-1]])  # donor eps = 1e-5
    gram = torch.matmul(pseudo.transpose(-1, -2), pseudo)
    left, singular_values, right = torch.svd(gram)
    _finite("pseudo-input Gram spectrum", singular_values)
    # A guard, not a regularizer: reject numerically unidentified inverse
    # directions rather than silently substituting a ridge or another rank.
    floor = torch.finfo(gram.dtype).eps * max(gram.shape) * singular_values[0]
    if head_dim > singular_values.numel() or singular_values[head_dim - 1] <= floor:
        raise ValueError("pseudo-input Gram has insufficient reliable rank for head_dim")
    smallest_retained_gram_value = float(singular_values[head_dim - 1])
    inverse_values = torch.zeros_like(singular_values)
    inverse_values[:head_dim] = 1 / singular_values[:head_dim]
    gram_inverse = torch.matmul(torch.matmul(right, torch.diag_embed(inverse_values)), left.transpose(-1, -2))
    pseudo_inverse = torch.matmul(gram_inverse, pseudo.transpose(-1, -2))
    product = torch.matmul(torch.matmul(pseudo_inverse, score_target), pseudo_inverse.transpose(-1, -2))
    _finite("lifted impulse product", product)
    left, singular_values, right = torch.svd(product)
    root = torch.diag_embed(singular_values[:, :head_dim].sqrt())
    query = left[:, :, :head_dim] @ root
    key_t = (right[:, :, :head_dim] @ root).transpose(-1, -2)
    query_norm = query.norm("fro", dim=(-1, -2), keepdim=True)
    key_norm = key_t.norm("fro", dim=(-1, -2), keepdim=True)
    if bool((query_norm <= 0).any()) or bool((key_norm <= 0).any()):
        raise ValueError("impulse factor has zero norm")
    query = query / query_norm
    key_t = key_t / key_norm
    _finite("impulse query", query)
    _finite("impulse key", key_t)
    return query.numpy(), key_t.numpy(), {
        "retained_gram_rank": head_dim,
        "smallest_retained_gram_value": smallest_retained_gram_value,
        "gram_reliability_floor": float(floor),
    }

@torch.no_grad()
def apply_release_style_init(
    model: nn.Module,
    *,
    method: str,
    pe_scale: float,
    qk_scale: float,
    vo_scale: float,
    construction_weight_init: str,
    allow_native_base: bool = False,
) -> dict:
    """Apply one explicitly configured reference recipe to a fresh CPU ViT.

    ``construction_weight_init`` is a caller assertion checked against 'skip';
    the adapter cannot reconstruct the constructor's history from tensors.
    Global Torch/NumPy RNGs are consumed, not reseeded or reset. Set the run
    seed before model construction. Updates are staged before copying so an
    unsupported model or failed solve does not leave partially changed weights.
    All returned metadata describe this new run, never a historical experiment.
    """
    if method not in ("default", "mimetic", "impulse"):
        raise ValueError("method must be default, mimetic, or impulse")
    if construction_weight_init != "skip" and not (allow_native_base and construction_weight_init == ""):
        raise ValueError("create the model with weight_init='skip' before applying this reference; native base requires explicit opt-in")
    scales = {name: _positive_scale(name, value) for name, value in (
        ("pe_scale", pe_scale), ("qk_scale", qk_scale), ("vo_scale", vo_scale)
    )}
    pe_scale, qk_scale, vo_scale = (scales[name] for name in ("pe_scale", "qk_scale", "vo_scale"))
    targets, grid, positional_prefix = _validate_model(model)
    scaled_positions = model.pos_embed.detach() * pe_scale
    _finite("scaled positions", scaled_positions)
    patch_positions = scaled_positions[0, positional_prefix:]
    dim = model.embed_dim
    staged, layer_metadata = [], []
    for target in targets:
        attention = target.module
        heads, head_dim = attention.num_heads, attention.head_dim
        qkv = attention.qkv.weight.detach().clone()
        projection = attention.proj.weight.detach().clone()
        diagnostics = {}
        if method == "default":
            qkv[:2 * dim] = qk_scale * qkv[:2 * dim]
            qkv[2 * dim:] = vo_scale * qkv[2 * dim:]
            projection = vo_scale * projection
        else:
            if method == "mimetic":
                for head in range(heads):
                    query, key_t = _mimetic_factors(dim, 0.7, 0.7, 1)
                    query, key = query[:, :head_dim], key_t.T[:, :head_dim]
                    qkv[head * head_dim:(head + 1) * head_dim] = qk_scale * torch.tensor(query.T).float()
                    qkv[dim + head * head_dim:dim + (head + 1) * head_dim] = qk_scale * torch.tensor(key.T).float()
            else:
                query, key_t, diagnostics = _impulse_factors(patch_positions, grid, heads, head_dim)
                for head in range(heads):
                    qkv[head * head_dim:(head + 1) * head_dim] = qk_scale * torch.tensor(query[head].T).float()
                    qkv[dim + head * head_dim:dim + (head + 1) * head_dim] = qk_scale * torch.tensor(key_t[head]).float()
            value, output = _mimetic_factors(dim, 0.4, 0.4, -1)
            # Preserve the donor's distinct V/O orientation in these branches.
            if method == "impulse":
                value, output = value.T, output.T
            qkv[2 * dim:] = vo_scale * torch.tensor(value).float()
            projection = vo_scale * torch.tensor(output).float()
        _finite(f"{target.path}.qkv", qkv)
        _finite(f"{target.path}.proj", projection)
        staged.append((attention, qkv, projection))
        layer_metadata.append({
            "path": target.path, "num_heads": heads, "head_dim": head_dim,
            "attention_scale": float(attention.scale), **diagnostics,
        })

    metadata = {
        "method_version": METHOD_VERSION,
        "method": method,
        **scales,
        "construction_weight_init": construction_weight_init,
        "native_base_comparison_opt_in": allow_native_base,
        "construction_check": "caller declaration; trainer must validate model_kwargs before construction",
        "source": {
            "kind": "post-publication local reference",
            "file": "pytorch-image-models_v1.0.22/timm/models/vision_transformer.py",
            "file_last_change_commit": "e620752",
            "file_last_change_date": "2025-12-19",
            "file_sha256": SOURCE_SHA256,
            "license": "Apache-2.0 (inspected source tree)",
        },
        "evidence_scope": "new validation run; historical NeurIPS run equivalence is unverified",
        "solver_device": "cpu", "solver_dtype": "torch.float32",
        "mimetic_svd_dtype": "numpy.float64",
        "rng_policy": "consume existing global Torch and NumPy streams in reference order",
        "grid_size": list(grid), "num_prefix_tokens": int(model.num_prefix_tokens),
        "positional_prefix_tokens": positional_prefix,
        "class_token_adaptation": "exclude actual positional prefix count; no patch removed when CLS is absent",
        "bias_policy": "unchanged from model construction",
        "vo_orientation": {"default": "scale existing", "mimetic": "V, Proj", "impulse": "V.T, Proj.T"}[method],
        "layers": layer_metadata,
    }
    if method == "impulse":
        metadata["impulse"] = {
            "kernel_size": 3, "flat_offset_order": list(_OFFSET_ORDER),
            "boundary": "zero outside grid in score target; no self fallback or wrap",
            "target_scale": 1000.0, "background_scale": 3.0,
            "background_distribution": "timm trunc_normal_(std=0.6, a=-2, b=2); masked to H==0",
            "fit_layer_norm_eps": 1e-5,
            "inverse": "top-head_dim Gram SVD; no ridge or solver substitution",
            "rank_guard": "retained Gram value must exceed eps(float32)*D*largest Gram value",
        }
    model.pos_embed.copy_(scaled_positions)
    for attention, qkv, projection in staged:
        attention.qkv.weight.copy_(qkv)
        attention.proj.weight.copy_(projection)
    setattr(model, _APPLIED_ATTRIBUTE, metadata)
    return metadata

def tensor_digest(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(name.encode())
        result.update(str(value.dtype).encode())
        result.update(str(tuple(value.shape)).encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()

@torch.no_grad()
def mimetic_paper(model):
    """Paper equations, not B1's uniform-noise/native-PE Mimetic variant.

    Native adaptation explicitly uses a trainable 1D sinusoidal absolute PE
    over CLS+row-major patches. The paper requires sinusoidal PE but does not
    establish this adapter's CLS/index/trainability choices as author defaults.
    Q/K use per-head rank-d factors of .7*Z+.7*I; V/O factor .4*Z-.4*I,
    Z_ij~N(0,1/D). Stored Linear weights transpose row-vector factors.
    """
    dim = model.embed_dim
    tokens = model.pos_embed.shape[1]
    position = torch.arange(tokens).unsqueeze(1).double()
    frequency = torch.exp(torch.arange(0, dim, 2).double() * (-np.log(10000.) / dim))
    pe = torch.empty(tokens, dim, dtype=torch.float64)
    pe[:, 0::2] = (position * frequency).sin()
    pe[:, 1::2] = (position * frequency).cos()
    model.pos_embed.copy_(pe.float().unsqueeze(0))
    checks = []
    def factors(alpha, beta, sign):
        matrix = alpha * torch.randn(dim, dim, dtype=torch.float64) / dim ** 0.5 + sign * beta * torch.eye(dim, dtype=torch.float64)
        u, values, vh = torch.linalg.svd(matrix)
        return u * values.sqrt().unsqueeze(0), values.sqrt().unsqueeze(1) * vh, matrix
    for block in model.blocks:
        attention = block.attn
        width = attention.head_dim
        for head in range(attention.num_heads):
            q, kt, _ = factors(.7, .7, 1)
            attention.qkv.weight[head * width:(head + 1) * width].copy_(q[:, :width].T.float())
            attention.qkv.weight[dim + head * width:dim + (head + 1) * width].copy_(kt[:width].float())
        v, o, target = factors(.4, .4, -1)
        attention.qkv.weight[2 * dim:].copy_(v.T.float())
        attention.proj.weight.copy_(o.T.float())
        effective = attention.qkv.weight[2 * dim:].T.double() @ attention.proj.weight.T.double()
        error = float((effective - target).abs().max())
        if error > 1e-5:
            raise ValueError("Mimetic V/O orientation does not realize the intended product")
        checks.append(error)
    return {"paper": "https://proceedings.mlr.press/v202/trockman23a/trockman23a.pdf",
            "scope": "independent paper-based native adaptation; not author-run reproduction",
            "pe": "trainable 1D sin/cos, temperature10000, CLS index0, patch indices1..196",
            "qk_target": ".7*N(0,1/D)+.7*I, rank-head_dim factors independently per head",
            "vo_target": ".4*N(0,1/D)-.4*I, full balanced factors, stored transposes",
            "vo_reconstruction_max_abs_errors": checks,
            "author_code_verified": False}

def attention_logits(attention, x):
    b, n, _ = x.shape
    qkv = attention.qkv(x).reshape(b, n, 3, attention.num_heads,
                                  attention.head_dim).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)
    return ((q * attention.scale) @ k.transpose(-1, -2))

def normalized_entropy(scores):
    log_p = scores.log_softmax(-1)
    return -(log_p.exp() * log_p).sum(-1) / math.log(scores.shape[-1])

def entropy_scale(scores, target, tolerance=1e-5, iterations=60):
    """Solve s with H(softmax(s^2 * scores))/log(N) = target.

    Mean across all heads and patch queries, including the CLS key. The bounded
    positive search preserves each row's argmax mathematically. Unreachable
    targets fail explicitly; no clipping, seed search or solver substitution.
    """
    if not 0 < target < 1 or scores.shape[-1] < 2:
        raise ValueError("invalid entropy target or key count")
    if not torch.isfinite(scores).all():
        raise ValueError("nonfinite calibration logits")
    low, high = 1e-3, 1e3
    def value(scale):
        return float(normalized_entropy(scores * scale**2).mean())
    maximum, minimum = value(low), value(high)
    if not minimum <= target <= maximum:
        raise ValueError(f"unreachable target {target}; bounded entropy range [{minimum}, {maximum}]")
    for _ in range(iterations):
        scale = math.sqrt(low * high)
        achieved = value(scale)
        if abs(achieved - target) <= tolerance:
            return scale, achieved
        if achieved > target:
            low = scale
        else:
            high = scale
    raise ValueError(f"entropy solve failed: achieved {achieved}, target {target}")

@torch.no_grad()
def calibrate(model, target):
    staged = []
    for layer, block in enumerate(model.blocks):
        a = block.attn
        if not all(isinstance(v, nn.Identity) for v in (a.q_norm, a.k_norm, a.norm)):
            raise ValueError("Q/K and output normalization unsupported")
        scores = attention_logits(a, block.norm1(model.pos_embed))
        scale, _ = entropy_scale(scores[..., 1:, :], target)
        staged.append((layer, a, scale, scores))
    report = []
    for layer, a, scale, before in staged:
        dim = model.embed_dim
        a.qkv.weight[:2 * dim].mul_(scale)
        if a.qkv.bias is not None:
            a.qkv.bias[:2 * dim].mul_(scale)
        after = attention_logits(a, model.blocks[layer].norm1(model.pos_embed))
        torch.testing.assert_close(after, before * scale**2, atol=2e-4, rtol=2e-5)
        achieved = float(normalized_entropy(after[..., 1:, :]).mean())
        if abs(achieved - target) > 3e-5:
            raise ValueError("stored Q/K failed target check")
        report.append({"layer": layer, "qk_multiplier_from_norm7_reference": scale,
                       "target": target, "achieved": achieved,
                       "before_entropy_by_head": normalized_entropy(before[..., 1:, :]).mean((0, 2)).tolist(),
                       "after_entropy_by_head": normalized_entropy(after[..., 1:, :]).mean((0, 2)).tolist(),
                       "argmax_changed_fraction": float((before.argmax(-1) != after.argmax(-1)).float().mean())})
    return {"version": VERSION, "layers": report,
            "input": "actual block.norm1(P), positional CLS key included; patch queries only",
            "scope": "counterfactual positional input at every layer, not propagated real features",
            "calibration_data": "none; no images or labels",
            "intervention": "one positive Q/K multiplier per layer; Q/K biases scaled consistently"}

def check_pair(reference, candidate):
    """Full state equality outside Q/K rows, including V/O, PE, norms, FFN, bias."""
    dim = reference.embed_dim
    ref, new = reference.state_dict(), candidate.state_dict()
    for name in ref:
        if name.endswith("attn.qkv.weight") or name.endswith("attn.qkv.bias"):
            if not torch.equal(ref[name][2 * dim:], new[name][2 * dim:]):
                raise ValueError(f"value rows changed: {name}")
        elif not torch.equal(ref[name], new[name]):
            raise ValueError(f"undeclared change: {name}")

class ScalarPE(nn.Module):
    def __init__(self, alpha, learnable):
        super().__init__()
        value = torch.tensor(float(alpha))
        if learnable:
            self.alpha = nn.Parameter(value)
        else:
            self.register_buffer('alpha', value)

    def forward(self, base):
        return self.alpha * base

def install(model, representation, p0, alpha):
    if representation not in REPRESENTATIONS:
        raise ValueError('unknown PE representation')
    if parametrize.is_parametrized(model, 'pos_embed'):
        raise ValueError('PE already parametrized')
    effective = model.pos_embed.detach().clone()
    if not torch.equal(effective, p0 * alpha):
        raise ValueError('native initialized effective PE differs from alpha*P0')
    if representation != 'S':
        with torch.no_grad():
            model.pos_embed.copy_(p0)
        parametrize.register_parametrization(model, 'pos_embed', ScalarPE(alpha, representation == 'L'))
    if not torch.equal(model.pos_embed, effective):
        raise ValueError('product installation changed effective PE')
    return model

def components(model):
    if parametrize.is_parametrized(model, 'pos_embed'):
        return model.parametrizations.pos_embed.original, model.parametrizations.pos_embed[0].alpha
    return model.pos_embed, model.pos_embed.new_tensor(1.)

def folded(model):
    # remove_parametrizations on a deepcopy can mutate the shared generated class
    # and break the live training model. Build an independent ordinary ViT.
    from timm.models.vision_transformer import VisionTransformer
    with torch.random.fork_rng(devices=[]):
        result = VisionTransformer(img_size=model.patch_embed.img_size,
            patch_size=model.patch_embed.patch_size, in_chans=model.patch_embed.proj.in_channels,
            num_classes=model.num_classes, embed_dim=model.embed_dim, depth=len(model.blocks),
            num_heads=model.blocks[0].attn.num_heads, qkv_bias=model.blocks[0].attn.qkv.bias is not None,
            global_pool='token', drop_path_rate=.1, weight_init='')
    state = {k:v.detach().clone() for k,v in model.state_dict().items() if not k.startswith('parametrizations.pos_embed.')}
    state['pos_embed'] = model.pos_embed.detach().clone()
    result.load_state_dict(state, strict=True)
    exemplar=next(model.parameters())
    return result.to(device=exemplar.device,dtype=exemplar.dtype).train(model.training)

def optimizer_groups(model, args):
    """Preserve timm layer groups; explicitly exempt PE and scalar from WD."""
    from timm.optim._param_groups import param_groups_layer_decay
    # Same layer_decay=1.0 policy as the preserved resolved recipe.
    groups = param_groups_layer_decay(model, weight_decay=args.weight_decay,
                                      no_weight_decay_list=model.no_weight_decay(), layer_decay=args.layer_decay)
    named = dict(model.named_parameters())
    special = {'pos_embed', BASE_NAME, ALPHA_NAME} & set(named)
    ids = {id(named[n]) for n in special}
    for group in groups:
        group['params'] = [p for p in group['params'] if id(p) not in ids]
    groups = [g for g in groups if g['params']]
    groups.append(dict(params=[named[n] for n in sorted(special)], weight_decay=0., lr_scale=1.))
    assigned = [id(p) for g in groups for p in g['params']]
    if len(assigned) != len(set(assigned)) or set(assigned) != {id(p) for p in named.values()}:
        raise ValueError('optimizer membership mismatch')
    return groups

@contextmanager
def isolated_cpu_rng(seed):
    py,np_state=random.getstate(),np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            random.seed(seed);np.random.seed(seed);torch.random.default_generator.manual_seed(seed)
            yield
    finally:random.setstate(py);np.random.set_state(np_state)

@torch.no_grad()
def position_summary(model,layers=None):
    """Effective positional pseudo-input, all keys/heads, never real images."""
    layers=range(len(model.blocks)) if layers is None else layers;rows=[]
    for i in layers:
        b=model.blocks[i];a=b.attn;scores=attention_logits(a,b.norm1(model.pos_embed))
        if not torch.isfinite(scores).all():raise ValueError('nonfinite positional diagnostic logits')
        entropy=normalized_entropy(scores[:,:,1:,:]).mean((0,2))
        cp=scores[:,:,1:,1:].softmax(-1);n=cp.shape[-1];grid=model.patch_embed.grid_size[0]
        xy=torch.stack(torch.meshgrid(torch.arange(grid),torch.arange(grid),indexing='ij'),-1).reshape(-1,2).to(scores.device)
        local=(xy[:,None]-xy[None,:]).abs().amax(-1)<=1;local.fill_diagonal_(False)
        qkv=a.qkv.weight.reshape(3,a.num_heads,a.head_dim,model.embed_dim)
        rows.append(dict(layer=i,normalized_full_entropy_mean=float(entropy.mean()),
            normalized_full_entropy_by_head=entropy.tolist(),
            conditional_patch_entropy_by_head=(-(cp*cp.clamp_min(1e-30).log()).sum(-1)/math.log(n)).mean((0,2)).tolist(),
            local_nonself_conditional_mass_by_head=(cp*local).sum(-1).mean((0,2)).tolist(),
            cls_key_mass_by_head=scores.softmax(-1)[:,:,1:,0].mean((0,2)).tolist(),
            q_fro_by_head=qkv[0].norm(dim=(-1,-2)).tolist(),k_fro_by_head=qkv[1].norm(dim=(-1,-2)).tolist()))
    return rows

METHODS = ('default', 'mimetic', 'structured', 'F', 'L', 'FE', 'LE')


def restore_topology(model, method):
    """Create PE storage only; no initialization or calibration on resume."""
    if method not in METHODS or parametrize.is_parametrized(model, 'pos_embed'):
        raise ValueError('Fresh model and supported method required')
    if method in ('F','L','FE','LE'):
        parametrize.register_parametrization(model, 'pos_embed', ScalarPE(18., method in ('L','LE')))
    model._structured_ready = True


@torch.no_grad()
def initialize(model, method, seed):
    if method not in METHODS or getattr(model, '_structured_ready', False):
        raise ValueError('Unknown method or initialization reentry')
    if parametrize.is_parametrized(model,'pos_embed'):
        raise ValueError('Fresh unparametrized model required')
    _validate_model(model)
    p0=model.pos_embed.detach().clone()
    report=dict(method=method,seed=seed,base_sha256=tensor_digest(model.state_dict()),calibration=None)
    with isolated_cpu_rng(seed+10000):
        if method=='mimetic':
            report['mimetic']=mimetic_paper(model)
        elif method!='default':
            report['structured']=apply_release_style_init(model,method='impulse',pe_scale=18.,
                qk_scale=7.,vo_scale=1.,construction_weight_init='',allow_native_base=True)
            if method in ('FE','LE'):
                ref=copy.deepcopy(model)
                report['calibration']=calibrate(model,.90)
                check_pair(ref,model)
            if method in ('F','L','FE','LE'):
                install(model,'L' if method in ('L','LE') else 'F',p0,18.)
    model._structured_ready=True
    report['state_sha256']=tensor_digest(model.state_dict())
    return report
