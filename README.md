# NSC-Net

Pre-release PyTorch code for:

> **Neural-Inspired Modeling of Auditory Selection and Compensation for Audio-Visual Speech Separation**

Accepted by **ICML 2026**.

NSC-Net is an audio-visual speech separation model built around explicit auditory selection and cross-modal compensation. The auditory selection mechanism suppresses interference under visual guidance, while the cross-modal compensation mechanism uses aligned visual cues to recover weak or missing auditory information.

The main model is implemented in:

```text
nsc_net/models/NSCNet.py
```

## Repository Overview

```text
configs/          Example configuration files
data_preprocess/  Dataset preprocessing utilities
nsc_net/          Model, data, loss, metric, and training modules
train_nsc.py      Training entrypoint
evaluate_nsc.py   Evaluation entrypoint
infer_nsc.py      Demo inference script
```

Large datasets, checkpoints, generated audio/video files, and experiment outputs are not included.

## Basic Setup

```bash
pip install -r requirements.txt
```

Please adjust dataset paths, visual frontend paths, checkpoint paths, and experiment settings in `configs/` before running experiments.

## Basic Running

Training:

```bash
python train_nsc.py --conf_dir configs/lrs2_nsc.yaml
```

Evaluation:

```bash
python evaluate_nsc.py --conf_dir configs/lrs2_nsc.yaml
```

Inference:

```bash
python infer_nsc.py --help
```

For other datasets, replace the configuration file accordingly.

## Model Usage

```python
from nsc_net.models import NSCNet

model = NSCNet()
```

The forward interface is:

```python
denoised_mag, denoised_phase, est_spec, estimated_wav = model(input_wav, mouth_emb)
```

## Citation

If this repository is useful for your research, please cite:

```bibtex
@inproceedings{xu2026nscnet,
  title     = {Neural-Inspired Modeling of Auditory Selection and Compensation for Audio-Visual Speech Separation},
  author    = {Xu, Xinmeng and Xie, Haoran and Tao, Xiaohui and Li, Lin and Qin, S. Joe},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```

## Note

This repository is provided as an early code release. Documentation, checkpoints, and exact reproduction instructions may be updated later.
