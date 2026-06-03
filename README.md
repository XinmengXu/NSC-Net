# NSC-Net

Pre-release PyTorch code for:

> **Neural-Inspired Modeling of Auditory Selection and Compensation for Audio-Visual Speech Separation**

NSC-Net is an audio-visual speech separation model built around explicit auditory selection and cross-modal compensation. 
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

Please adjust dataset paths, visual frontend paths, and experiment settings in `configs/` before running experiments.

## Model Usage

```python
from nsc_net.models import NSCNet

model = NSCNet()
```

The forward interface is:

```python
denoised_mag, denoised_phase, est_spec, estimated_wav = model(input_wav, mouth_emb)
```

## Note

This repository is provided as an early code release. Documentation, checkpoints, and exact reproduction instructions may be updated later.
