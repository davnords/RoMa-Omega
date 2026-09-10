<div align="center">
<h1>RoMa-Ω: What Feed-Forward 3D Models Know About Image Matching</h1>


<a href="https://arxiv.org/abs/2609.09507"><img src="https://img.shields.io/badge/arXiv-2609.09507-b31b1b" alt="arXiv"></a>

[David Nordström<sup>1</sup>](https://scholar.google.com/citations?user=-vJPE04AAAAJ),
[Xinyue Zhang<sup>2</sup>](https://scholar.google.fi/citations?user=WvixLxcAAAAJ),
[Thibaut Loiseau<sup>3</sup>](https://scholar.google.com/citations?user=qDSlhTUAAAAJ),
[Vincent Lepetit<sup>3</sup>](https://scholar.google.com/citations?user=h0a5q3QAAAAJ),
[Fredrik Kahl<sup>1</sup>](https://scholar.google.com/citations?user=P_w6UgMAAAAJ)

<sup>1</sup> **Chalmers University of Technology**<br>
<sup>2</sup> **Mobile Perception Lab, ShanghaiTech University**<br>
<sup>3</sup> **LIGM, École des Ponts, Univ. Gustave Eiffel, CNRS**
</div>

<p align="center">
    <img src="assets/teaser.png" alt="example" width=95%>
    <br>
    <em>We replace the DINOv3 backbone of RoMa v2 with features from the VGGT-Ω architecture. This small change leads to our model, RoMa-Ω, which achieves state-of-the-art performance on a wide range of benchmarks, surpassing the current best matchers RoMa and RoMa v2. In particular, \ours~surpasses RoMa on the difficult matching benchmarks WxBS and HardMatch, which RoMa v2 did not.</em>
</p>

## Overview
RoMa-Ω is a powerful but slow dense matcher that builds on [RoMa v2](https://github.com/Parskatt/RoMaV2). We change the DINOv3 backbone in RoMa v2 for VGGT-Ω and retrain. The resulting model is slightly stronger than RoMa v2 on most benchmarks and quite a bit stronger on difficult matching (like HardMatch and WxBS). As part of our paper, we also explore the zero-shot matching abilities of VGGT-Ω, which we release code for in this repo.

## Updates
- [August 6, 2026] Initial public release of the code.

## Setup/Install

Clone with submodules (this repo depends on [VGGT-Ω](https://github.com/facebookresearch/vggt-omega) via `third_party/vggtomega`):
```bash
git clone --recurse-submodules <this-repo-url>
# or, if already cloned:
git submodule update --init --recursive
```

Download the VGGT-Ω checkpoint from [this](https://huggingface.co/facebook/VGGT-Omega/blob/main/vggt_omega_1b_512.pt) link and place it at `vggtomega.pt` in the repo root (or pass `--checkpoint` explicitly).

In your python environment (tested on Linux python 3.12), run:
```bash
uv sync
```

## Evaluation

We supply a script to evaluate on [WxBS]() and [HardMatch](https://github.com/davnords/HardMatch) that automatically downloads the datasets. [RUBIK](https://github.com/thibautloiseau/RUBIK) is also supported but requires downloading the data in accordance with the repo.
```bash
python experiments/eval_sparse.py --name roma_omega --benchmark wxbs # mAA_10px: 0.7198
python experiments/eval_sparse.py --name roma_omega --benchmark hardmatch # mAA_10px: 0.5110
```
These results closely replicate those reported in the paper. You can also test the matching abilities of VGGT-Ω by using `--name vggt_omega`.

Dense PCK/EPE benchmarks:
```bash
python experiments/eval_dense.py
```
Note, the dense benchmarks require you to download the data yourself. For MegaDepth, follow the instructions in [DKM](https://github.com/parskatt/dkm).

## Training

Unfortunately, for now you will have to download the datasets on your own. You can follow [DKM](https://github.com/parskatt/dkm) for MegaDepth.

Both stages are run with `torchrun` on 8 GPUs (validated on A100s) and take roughly 3 days each on that setup:
```bash
torchrun --nproc_per_node=8 experiments/train_matcher.py --name my-run
torchrun --nproc_per_node=8 experiments/train_refiners.py --name my-run --matcher-run-path experiments/vggtroma/runs/my-run/<timestamp>
```

## License
All our code is MIT license.

## Acknowledgement
Code is based on [RoMaV2](https://github.com/Parskatt/RoMaV2) by [Parskatt](https://github.com/Parskatt).

## BibTeX
If you find our models useful, please consider citing our papers!
```bibtex
@inproceedings{nordstrom2026romaomega,
      title={RoMa-$\Omega$: What Feed-Forward 3D Models Know About Image Matching}, 
      author={David Nordström and Xinyue Zhang and Thibaut Loiseau and Vincent Lepetit and Fredrik Kahl},
      booktitle={Proceedings of the European Conference on Computer Vision (ECCV) Workshops},
      year={2026}
}
```
