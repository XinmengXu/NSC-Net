import argparse
import os
import warnings

import torch
import yaml
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn, TransferSpeedColumn

import nsc_net.datas
import nsc_net.models
import nsc_net.videomodels
from nsc_net.metrics import MetricsTracker
from nsc_net.utils import (
    BatchesProcessedColumn,
    MyMetricsTextColumn,
    RichProgressBarTheme,
    tensors_to_device,
)

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf_dir",
    default="configs/lrs2_nsc.yaml",
    help="Path to the experiment configuration.",
)


def main(config):
    train_conf = config["train_conf"]
    train_conf.setdefault("main_args", {})
    train_conf["main_args"]["exp_dir"] = os.path.join(
        os.getcwd(), "Experiments", "checkpoint", train_conf["exp"]["exp_name"]
    )
    model_path = os.path.join(train_conf["main_args"]["exp_dir"], "best_model.pth")

    model = getattr(nsc_net.models, train_conf["audionet"]["audionet_name"]).from_pretrain(
        model_path,
        sample_rate=train_conf["datamodule"]["data_config"]["sample_rate"],
        **train_conf["audionet"]["audionet_config"],
    )
    video_model = getattr(nsc_net.videomodels, train_conf["videonet"]["videonet_name"])(
        **train_conf["videonet"]["videonet_config"],
    )

    device = "cuda" if train_conf["training"].get("gpus") and torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    video_model.to(device).eval()
    model_device = next(model.parameters()).device

    datamodule = getattr(nsc_net.datas, train_conf["datamodule"]["data_name"])(
        **train_conf["datamodule"]["data_config"]
    )
    datamodule.setup()
    _, _, test_set = datamodule.make_sets

    ex_save_dir = os.path.join(train_conf["main_args"]["exp_dir"], "results")
    os.makedirs(ex_save_dir, exist_ok=True)
    metrics = MetricsTracker(save_file=os.path.join(ex_save_dir, "metrics.csv"))

    metricscolumn = MyMetricsTextColumn(style=RichProgressBarTheme.metrics)
    progress = Progress(
        TextColumn("[bold blue]Testing", justify="right"),
        BarColumn(bar_width=None),
        "|",
        BatchesProcessedColumn(style=RichProgressBarTheme.batch_progress),
        "|",
        TransferSpeedColumn(),
        "|",
        TimeRemainingColumn(),
        "|",
        metricscolumn,
    )

    with torch.no_grad(), progress:
        for idx in progress.track(range(len(test_set))):
            mix, sources, mouth, key = tensors_to_device(test_set[idx], device=model_device)
            mouth_tensor = torch.from_numpy(mouth[None, None]).float().to(model_device)
            mouth_emb = video_model(mouth_tensor)
            est_outputs = model(mix[None], mouth_emb)
            est_sources = est_outputs[-1] if isinstance(est_outputs, (tuple, list)) else est_outputs

            metrics(
                mix=mix,
                clean=sources.unsqueeze(0),
                estimate=est_sources.squeeze(0),
                key=key,
            )
            if idx % 50 == 0:
                metricscolumn.update(metrics.update())
    metrics.final()


if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.conf_dir, "rb") as f:
        train_conf = yaml.safe_load(f)
    main({"train_conf": train_conf})
