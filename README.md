# Neural Style Transfer Project Guide

**Video** · [Watch the demo on YouTube](https://youtu.be/PzY8KnlngYU)

[![Neural Style Transfer demo thumbnail](https://img.youtube.com/vi/PzY8KnlngYU/hqdefault.jpg)](https://youtu.be/PzY8KnlngYU)



**Image Highlights**

![Content image](Source%20code/images/real/content.jpg)
![Style image](Source%20code/images/real/style.jpg)
![VGG19 sample output](Source%20code/vgg19.jpg)
![ViT sample output](Source%20code/vit.jpg)

## Table of Contents
- [Project Overview](#project-overview)
- [Documentation and Media](#documentation-and-media)
- [Repository Layout](#repository-layout)
- [Environment and Dependencies](#environment-and-dependencies)
- [Environment Setup Checklist](#environment-setup-checklist)
- [Data and Asset Preparation](#data-and-asset-preparation)
- [Quick Start Workflow](#quick-start-workflow)
- [Evaluation Snapshot](#evaluation-snapshot)
- [Future Extensions](#future-extensions)

## Project Overview
- The project implements neural style transfer using two architectures: VGG19 and a hybrid approach that couples VGG19 with Vision Transformers (ViT).
- Experiments compare model variants on content fidelity, style adaptation, image quality, runtime, and GPU memory usage.
- Source assets include training scripts, evaluation utilities, and curated datasets for reproducible benchmarks.

[Back to Table of Contents](#table-of-contents)

## Documentation and Media
- [g10.pdf](g10.pdf) · Research paper covering background, methodology, and experimental analysis.
- [G10_Technical_Report.pdf](G10_Technical_Report.pdf) · Environment setup and execution guidelines.
- [NST-Group10-vedio recording.mp4](NST-Group10-vedio%20recording.mp4) · Local copy of the walkthrough video.
- [YouTube demo](https://youtu.be/PzY8KnlngYU) · Streaming version of the project demonstration.

[Back to Table of Contents](#table-of-contents)

## Repository Layout
- `Source code/` · Core Python scripts, notebooks, and generated artifacts.
  - `style_transfer_VGG19.py`, `hybrid_vgg19+VIT.py` · Primary inference pipelines.
  - `VIT_pretrained.py`, `VIT_training.py`, `Vgg19_training.py` · Model preparation and fine-tuning utilities.
  - `evaluation.py` · Computes SSIM, PSNR, FID, runtime, and GPU memory.
  - `images/` · Reference assets.
    - `real/` · Source content and style images.
    - `fake_vgg19/`, `fake_vit/` · Representative outputs for each model family.
  - `Hybrid_outputimage/`, `VGG19output/`, `VIToutput/` · Intermediate and final renders.
  - `save model/`, `VGG19_output/` · Stored weights and additional outputs.
- `First and second ppt/` · Presentation decks.
- `Neural_Style_Transfer_Report_Final_With_Literary_Review.docx` · Full written report.

[Back to Table of Contents](#table-of-contents)

## Environment and Dependencies
- Python 3.x (recommended with Anaconda or Miniconda).
- VS Code or Jupyter Notebook for interactive exploration.
- PyTorch 2.3.0 with CUDA 12.1 when using GPU acceleration.
- Supporting packages: `torchvision`, `Pillow`, `matplotlib`, `vit-pytorch`, `scikit-image`, `pytorch-fid`.
- Suggested hardware: Intel i5 (or similar), 8 GB RAM, 256 GB SSD or better.

[Back to Table of Contents](#table-of-contents)

## Environment Setup Checklist
1. Install Anaconda or Miniconda, then create and activate a dedicated environment.
2. Install VS Code or run `pip install notebook` for Jupyter Notebook support.
3. Install PyTorch via the official selector, matching the local CUDA toolkit.
4. Run `pip install vit-pytorch scikit-image pytorch-fid matplotlib pillow`.

[Back to Table of Contents](#table-of-contents)

## Data and Asset Preparation
- Download the CIFAR-100 dataset from `https://www.cs.toronto.edu/~kriz/cifar.html` for training or fine-tuning experiments.
- Place the working content and style images in `Source code/images/real/` using the expected filenames `content.jpg` and `style.jpg`.
- Ensure generated outputs are organized under `Source code/images/fake_vgg19/` and `Source code/images/fake_vit/` for evaluation.

[Back to Table of Contents](#table-of-contents)

## Quick Start Workflow
1. **Initialize checkpoints** · Run `VIT_pretrained.py` to prepare ViT resources.
2. **Train models**
   - Execute `VIT_training.py` to train the ViT-based style transfer.
   - Optionally fine-tune VGG19 with `Vgg19_training.py`.
3. **Generate stylized images**
   - Launch `hybrid_vgg19+VIT.py` or `style_transfer_VGG19.py` for scripted runs.
   - Open `style_transfer_VGG19.ipynb` for notebook-based experimentation.
4. **Evaluate outputs** · Run `evaluation.py` to compute SSIM, PSNR, FID, runtime, and GPU memory metrics (requires `pytorch-fid` and `scikit-image`).

> All scripts assume execution from within `Source code/` so that relative paths resolve correctly.

[Back to Table of Contents](#table-of-contents)

## Evaluation Snapshot
| Model | SSIM | PSNR | FID | Time (s) | GPU Memory (MB) |
| --- | --- | --- | --- | --- | --- |
| VGG19 | 0.85 | 26.5 | 18.2 | 12.8 | 512.3 |
| ViT | 0.82 | 25.8 | 22.5 | 15.4 | 840.7 |

- VGG19 delivers stronger structural preservation and style fidelity while using less GPU memory and finishing faster.
- ViT produces smoother outputs but with higher computational cost and slightly reduced content retention.

[Back to Table of Contents](#table-of-contents)

## Future Extensions
- Explore lighter-weight transformer variants to reduce runtime and memory requirements.
- Combine quantitative metrics with human evaluation for a more holistic assessment.
- Investigate text-guided controls (for example, CLIP guidance) to expand creative flexibility.

[Back to Table of Contents](#table-of-contents)
