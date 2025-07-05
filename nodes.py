import torch
import kornia
import math
from kornia.geometry.transform import build_laplacian_pyramid

def project(v0: torch.Tensor, v1: torch.Tensor):
    """Projects tensor v0 onto v1 and returns parallel and orthogonal components."""
    dtype = v0.dtype
    v0, v1 = v0.double(), v1.double()
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3])
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3], keepdim=True) * v1
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)

def build_image_from_pyramid(pyramid):
    """Reconstructs image from laplacian pyramid."""
    img = pyramid[-1]
    for i in range(len(pyramid) - 2, -1, -1):
        img = kornia.geometry.pyrup(img) + pyramid[i]
    return img

def laplacian_guidance(
    pred_cond: torch.Tensor,
    pred_uncond: torch.Tensor,
    guidance_scale=[1.0, 1.0],
    parallel_weights=None,
):
    """Applies laplacian guidance using laplacian pyramids."""
    levels = len(guidance_scale)
    if parallel_weights is None:
        parallel_weights = [1.0] * levels
    original_size = pred_cond.shape[-2:]
    pred_cond_pyramid = build_laplacian_pyramid(pred_cond, levels)
    pred_uncond_pyramid = build_laplacian_pyramid(pred_uncond, levels)
    pred_guided_pyramid = []
    
    parameters = zip(
        pred_cond_pyramid,
        pred_uncond_pyramid,
        guidance_scale,
        parallel_weights
    )
    

    for idx, (p_cond, p_uncond, scale, par_weight) in enumerate(parameters):
        """Crop the padding area added by build_laplacian_pyramid"""
        level_size = (original_size[0] // (2 ** idx), original_size[1] // (2 ** idx))
        p_cond = p_cond[..., :level_size[0], :level_size[1]]
        p_uncond = p_uncond[..., :level_size[0], :level_size[1]]
        diff = p_cond - p_uncond
        diff_parallel, diff_orthogonal = project(diff, p_cond)
        diff = par_weight * diff_parallel + diff_orthogonal
        p_guided = p_cond + (scale - 1) * diff
        pred_guided_pyramid.append(p_guided)
    pred_guided = build_image_from_pyramid(pred_guided_pyramid)
    
    return pred_guided.to(pred_cond.dtype)


def create_guidance_scales(high_scale, low_scale, levels, interpolation_type="linear"):
    """Creates interpolated guidance scale array."""
    if levels == 1:
        return [high_scale]

    scales = torch.zeros(levels)
    if interpolation_type == "linear":
        scales = torch.linspace(high_scale, low_scale, levels)
    elif interpolation_type == "cosine":
        # Cosine interpolation from high_scale down to low_scale
        for i in range(levels):
            t = i / (levels - 1) if levels > 1 else 0
            cosine_t = (1 - math.cos(t * math.pi)) / 2  # ranges 0 to 1
            scales[i] = high_scale * (1 - cosine_t) + low_scale * cosine_t
    elif interpolation_type == "quadratic_ease_in":
        # Eases in (starts slow, ends fast)
        for i in range(levels):
            t = i / (levels - 1) if levels > 1 else 0
            scales[i] = high_scale + (low_scale - high_scale) * (t ** 2)
    elif interpolation_type == "quadratic_ease_out":
        # Eases out (starts fast, ends slow)
        for i in range(levels):
            t = i / (levels - 1) if levels > 1 else 0
            scales[i] = high_scale + (low_scale - high_scale) * (1 - (1 - t) ** 2)
    elif interpolation_type == "step":
        mid_point = levels // 2
        scales[:mid_point] = high_scale
        scales[mid_point:] = low_scale
        if levels == 1: # Ensure single level gets high_scale
             scales[0] = high_scale
    else: # Default to linear
        scales = torch.linspace(high_scale, low_scale, levels)

    return scales.tolist()

def create_parallel_weights(high_weight, low_weight, levels, interpolation_type="linear"):
    """Creates interpolated parallel weights array."""
    # Using the same logic as create_guidance_scales for now
    # Potentially could have different/simpler interpolation for weights
    return create_guidance_scales(high_weight, low_weight, levels, interpolation_type)


class FDGNode:
    INTERPOLATION_TYPES = ["linear", "cosine", "quadratic_ease_in", "quadratic_ease_out", "step"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "guidance_scale_high": ("FLOAT", {"default": 7.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "guidance_scale_low": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "levels": ("INT", {"default": 3, "min": 1, "max": 8, "step": 1}), # Min levels 1 now
                "scale_interpolation": (s.INTERPOLATION_TYPES, {"default": "linear"}),
                "parallel_weight_high": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "parallel_weight_low": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "weight_interpolation": (s.INTERPOLATION_TYPES, {"default": "linear"}),
                "fdg_steps": ("INT", {"default": 2, "min": 0, "max": 1000, "step": 1}) # fdg_steps 0 means FDG off
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/model"
    
    def patch(self, model, guidance_scale_high, guidance_scale_low, levels, scale_interpolation,
              parallel_weight_high, parallel_weight_low, weight_interpolation, fdg_steps):
        
        if fdg_steps == 0: # Allow disabling FDG
            m = model.clone()
            # To fully disable, we might need to ensure no cfg function is set or a default one.
            # For now, returning the original model if no steps are to be FDG'd.
            # Or, ensure fdg_function respects this by only applying CFG.
            # The current fdg_function already handles CFG if sigma is outside fdg_steps range,
            # so fdg_steps = 0 effectively means it always uses else branch (standard CFG).
            # However, the set_model_sampler_cfg_function is still called.
            # A cleaner way might be to not patch if fdg_steps is 0.
            # For now, the existing logic within fdg_function should suffice.
            pass


        guidance_scale = create_guidance_scales(guidance_scale_high, guidance_scale_low, levels, scale_interpolation)
        parallel_weights = create_parallel_weights(parallel_weight_high, parallel_weight_low, levels, weight_interpolation)

        if not guidance_scale: # Should not happen if levels >= 1
            guidance_scale = [guidance_scale_high]
        if not parallel_weights:
            parallel_weights = [parallel_weight_high]

        def fdg_function(args):
            cond = args["cond"]
            uncond = args["uncond"]
            cond_scale = guidance_scale_high if math.isclose(args["cond_scale"], 1.0) else args["cond_scale"]
            sample_sigmas = args["model_options"]["transformer_options"]["sample_sigmas"]
            sigma = args["sigma"]
            step_limits = fdg_steps
            """Use CFG after limited FDG application for early steps."""
            if uncond is not None:
                if step_limits >= (len(sample_sigmas) - 1):
                    step_limits = len(sample_sigmas) - 1
                if sigma.item() > sample_sigmas[step_limits].item():
                    return laplacian_guidance(
                        cond,
                        uncond,
                        guidance_scale,
                        parallel_weights
                    )
                else: 
                    cond = uncond + (cond - uncond) * cond_scale
                    return cond
            else:
                return cond
        
        m = model.clone()
        m.set_model_sampler_cfg_function(fdg_function, disable_cfg1_optimization=True)
        return (m,)

