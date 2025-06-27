## ComfyUI-FDG
Implementation of [Guidance in the Frequency Domain Enables High-Fidelity Sampling at Low CFG Scales](https://arxiv.org/abs/2506.19713) for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

Thanks to the FDG researchers for making their code public.

FDG improves image quality at low guidance scales and avoids the drawbacks of high CFG scales by design.

Paper: https://arxiv.org/abs/2506.19713




## Usage
The node is in advanced/model.

To use FDG, just place the `FDGNode` (located in the advanced/model category) in front of the `KSampler` node in your workflow.

Basically disables cfg values ​​within KSampler.

Testing on SDXL only.

## Inputs

- `guidance_scale_high`: Guidance scale for high-frequency details.
- `guidance_scale_low`: Guidance scale for low-frequency structures.
- `levels`: Number of pyramid levels for frequency decomposition. For levels higher than 2, it operates using linear interpolation for each frequency.
- `fdg_steps`: Number of initial steps to apply FDG before switching to CFG. After the limit, the cfg value of the KSampler node is used. If the cfg is set 1, the value of guidance_scale_high is applied.