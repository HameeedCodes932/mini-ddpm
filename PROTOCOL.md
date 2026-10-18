# mini-ddpm experimental plan

Author: Fazal Hameed. Written before training on 18 October 2026.

- Train an unconditional, time-conditioned small U-Net from random initialization.
  Use 28x28 MNIST, input scaling to [-1,1], a 1,000-step linear beta schedule
  from 0.0001 to 0.02, and uniformly sampled timestep noise-prediction MSE.
- A fixed seed-31415 split uses 55,000 of the 60,000 training images for fitting
  and 5,000 for validation. Do not fit the diffusion model on the official test set.
- Use seed 42, base width 32, batch 256, AdamW learning rate 0.0002, weight
  decay 0.0001, gradient clipping 1.0, and EMA decay 0.995. Train for 12 epochs.
  Use the final EMA checkpoint, without selecting epochs using sample quality.
- Run on an available free Colab T4 for speed. The program defaults to CPU and
  uses FP32 on CPU. Record actual hardware, precision and measured runtime;
  GPU timing does not establish laptop CPU runtime.
- Compare DDPM ancestral sampling with 1,000, 200 and 50 model evaluations using
  the same weights and the entire original noise range. For shorter chains,
  preserve original alpha-bar marginals and use the exact respaced Gaussian
  posterior; never just stop a 1,000-step chain early or reuse adjacent-step
  coefficients when skipping timesteps.
- Bonus: implement deterministic DDIM with 50 evaluations using the same weights.
- Sample 512 images for each setting with seed 2026. Show the first 64 as an
  unfiltered 8x8 grid. Save the complete 8-bit batches for metric reproduction.
- Measure batch-64 sampling in three repeats, rotating execution order. Warm up
  kernels first, synchronize CUDA around timing, and exclude plotting/file I/O.
- Train a separate small digit classifier from scratch on all 60,000 original
  training labels for three epochs (seed 2718). Report its accuracy on all
  10,000 test images. Use its 64-dimensional features for an explicitly
  nonstandard MNIST feature-distance diagnostic, with a fixed 2,048-image test
  reference (seed 1618). Include a separate real-512 vs reference calibration.
  Classifier confidence and predicted digit coverage are diagnostics, not proof
  of sample correctness or substitutes for visual inspection. Do not call this
  metric standard Inception FID or claim MNIST classifier confidence is accuracy
  on unlabeled generated digits.
- Retain full loss history, configuration, source and data checksums, weights,
  sample batches, timing repetitions and evaluator weights. Report actual
  problems and limitations, including one training seed and finite sample size.
- References: DDPM https://arxiv.org/abs/2006.11239 ;
  DDIM https://arxiv.org/abs/2010.02502 ;
  respaced DDPM reference https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/respace.py ;
  MNIST mirror/checksums https://github.com/pytorch/vision/blob/main/torchvision/datasets/mnist.py .
