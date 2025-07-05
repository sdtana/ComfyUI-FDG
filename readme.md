## ComfyUI-FDG
Implementation of [Guidance in the Frequency Domain Enables High-Fidelity Sampling at Low CFG Scales](https://arxiv.org/abs/2506.19713) for [ComfyUI](https://github.com/comfyanonymous/ComfyUI).

Thanks to the FDG researchers for making their code public.

FDG improves image quality at low guidance scales and avoids the drawbacks of high CFG scales by design. This implementation expands upon the basic concept by offering flexible interpolation methods for guidance strengths and parallel component weights across frequency levels.

Paper: https://arxiv.org/abs/2506.19713

## Usage

To use FDG, place the `FDGNode` (located in the `advanced/model` category) in front of the `KSampler` node in your workflow. The FDG node patches the model to apply frequency-decoupled guidance. Note that the CFG value set in the KSampler will be used by FDG if `fdg_steps` is configured to switch to standard CFG, or if `fdg_steps` is 0.

Testing primarily on SDXL.

## Inputs

The `FDGNode` provides the following input parameters:

-   **`model`**: The model to patch with FDG.
-   **`guidance_scale_high`**: (Float, Default: 7.5)
    The guidance strength applied to the highest frequency components (finest details).
-   **`guidance_scale_low`**: (Float, Default: 1.0)
    The guidance strength applied to the lowest frequency components (coarsest structures).
-   **`levels`**: (Int, Default: 3, Min: 1, Max: 8)
    Number of pyramid levels for frequency decomposition. More levels mean finer frequency bands.
    - If `levels` is 1, `guidance_scale_high` and `parallel_weight_high` are used for all frequencies.
-   **`scale_interpolation`**: (Enum, Default: "linear")
    The interpolation method used to transition from `guidance_scale_high` to `guidance_scale_low` across the frequency `levels`.
    Options:
    -   `linear`: Smooth linear transition.
    -   `cosine`: Cosine curve for smoother transition.
    -   `quadratic_ease_in`: Starts slow, accelerates towards `guidance_scale_low`. High frequencies get stronger guidance for more levels.
    -   `quadratic_ease_out`: Starts fast, decelerates towards `guidance_scale_low`. High frequencies quickly transition away from `guidance_scale_high`.
    -   `step`: Applies `guidance_scale_high` to the first half of levels and `guidance_scale_low` to the second half. A more direct split.
-   **`parallel_weight_high`**: (Float, Default: 1.0, Min: 0.0, Max: 2.0)
    The weight for the component of guidance that is parallel to the conditional prediction, applied at the highest frequency.
    - A weight of 1.0 uses the parallel component unmodified.
    - A weight > 1.0 amplifies the parallel component.
    - A weight < 1.0 dampens the parallel component, relatively increasing the influence of the orthogonal component.
    - A weight of 0.0 uses only the orthogonal component for guidance at this frequency extreme.
-   **`parallel_weight_low`**: (Float, Default: 1.0, Min: 0.0, Max: 2.0)
    The weight for the parallel component of guidance, applied at the lowest frequency.
-   **`weight_interpolation`**: (Enum, Default: "linear")
    The interpolation method used to transition from `parallel_weight_high` to `parallel_weight_low` across frequency `levels`. Uses the same options as `scale_interpolation`.
-   **`fdg_steps`**: (Int, Default: 2, Min: 0, Max: 1000)
    Number of initial sampling steps where FDG is applied.
    - Beyond this step count (approximated by sigma values), standard Classifier-Free Guidance (CFG) using the KSampler's CFG value is applied.
    - If `fdg_steps` is 0, FDG is effectively disabled, and standard CFG is used for all steps (though the model is still patched).
    - If `fdg_steps` is very high (e.g., exceeds total sampling steps), FDG is applied to all steps.
    - When FDG switches to standard CFG, or if `fdg_steps` is 0, and the KSampler's CFG value is 1.0, the `guidance_scale_high` value from this node is used as the CFG scale by the internal FDG logic. This specific behavior for CFG=1 is a nuance of the underlying `fdg_function`.

## Citation

If you use this implementation in your research, please cite the original paper:
```
@misc{sadat2025guidance,
    title={Guidance in the Frequency Domain Enables High-Fidelity Sampling at Low CFG Scales},
    author={Seyedmorteza Sadat and Tobias Vontobel and Farnood Salehi and Romann M. Weber},
    year={2025},
    eprint={2506.19713},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
