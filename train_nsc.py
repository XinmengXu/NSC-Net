import argparse
import json
import os
import warnings

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies.ddp import DDPStrategy

import nsc_net.datas
import nsc_net.losses
import nsc_net.models
import nsc_net.system
import nsc_net.videomodels
from nsc_net.system import make_optimizer
from nsc_net.utils import MyRichProgressBar, RichProgressBarTheme, print_only

warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser()
parser.add_argument(
    "--conf_dir",
    default="configs/lrs2_nsc.yaml",
    help="Path to a YAML experiment configuration.",
)


def _resolve_trainer_runtime(training_config):
    requested_gpus = training_config.get("gpus")
    use_gpu = torch.cuda.is_available() and bool(requested_gpus)
    devices = requested_gpus if use_gpu else 1
    accelerator = "gpu" if use_gpu else "cpu"
    strategy = (
        DDPStrategy(find_unused_parameters=True)
        if use_gpu and isinstance(requested_gpus, list) and len(requested_gpus) > 1
        else "auto"
    )
    return accelerator, devices, strategy, use_gpu


def main(config):
    pl.seed_everything(config["training"].get("seed", 2026), workers=True)

    print_only(f"Instantiating datamodule <{config['datamodule']['data_name']}>")
    datamodule = getattr(nsc_net.datas, config["datamodule"]["data_name"])(
        **config["datamodule"]["data_config"]
    )
    datamodule.setup()
    train_loader, val_loader, test_loader = datamodule.make_loader

    print_only(f"Instantiating AudioNet <{config['audionet']['audionet_name']}>")
    model = getattr(nsc_net.models, config["audionet"]["audionet_name"])(
        sample_rate=config["datamodule"]["data_config"]["sample_rate"],
        **config["audionet"]["audionet_config"],
    )
    video_model = getattr(nsc_net.videomodels, config["videonet"]["videonet_name"])(
        **config["videonet"]["videonet_config"],
    )

    print_only(f"Instantiating optimizer <{config['optimizer']['optim_name']}>")
    optimizer = make_optimizer(model.parameters(), **config["optimizer"])

    scheduler = None
    if config["scheduler"].get("sche_name"):
        print_only(f"Instantiating scheduler <{config['scheduler']['sche_name']}>")
        scheduler = getattr(torch.optim.lr_scheduler, config["scheduler"]["sche_name"])(
            optimizer=optimizer,
            **config["scheduler"]["sche_config"],
        )

    config.setdefault("main_args", {})
    config["main_args"]["exp_dir"] = os.path.join(
        os.getcwd(), "Experiments", "checkpoint", config["exp"]["exp_name"]
    )
    exp_dir = config["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "conf.yml"), "w", encoding="utf-8") as outfile:
        yaml.safe_dump(config, outfile)

    print_only(
        "Instantiating loss, train <{}>, val <{}>".format(
            config["loss"]["train"]["sdr_type"], config["loss"]["val"]["sdr_type"]
        )
    )
    loss_func = {
        "train": getattr(nsc_net.losses, config["loss"]["train"]["loss_func"])(
            getattr(nsc_net.losses, config["loss"]["train"]["sdr_type"]),
            **config["loss"]["train"]["config"],
        ),
        "val": getattr(nsc_net.losses, config["loss"]["val"]["loss_func"])(
            getattr(nsc_net.losses, config["loss"]["val"]["sdr_type"]),
            **config["loss"]["val"]["config"],
        ),
    }

    system = getattr(nsc_net.system, config["training"]["system"])(
        audio_model=model,
        video_model=video_model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        config=config,
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=exp_dir,
            filename="{epoch}",
            monitor="val_loss/dataloader_idx_0",
            mode="min",
            save_top_k=5,
            verbose=True,
            save_last=True,
        ),
        MyRichProgressBar(theme=RichProgressBarTheme()),
    ]
    if config["training"].get("early_stop"):
        callbacks.append(EarlyStopping(**config["training"]["early_stop"]))

    accelerator, devices, strategy, use_gpu = _resolve_trainer_runtime(config["training"])
    logger_dir = os.path.join(os.getcwd(), "Experiments", "tensorboard_logs")
    logger = TensorBoardLogger(logger_dir, name=config["exp"]["exp_name"])

    trainer = pl.Trainer(
        max_epochs=config["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        devices=devices,
        accelerator=accelerator,
        strategy=strategy,
        gradient_clip_val=config["training"].get("gradient_clip_val", 5.0),
        gradient_clip_algorithm=config["training"].get("gradient_clip_algorithm", "norm"),
        logger=logger,
        sync_batchnorm=use_gpu,
        num_sanity_val_steps=0,
        precision=config["training"].get("precision", "32-true"),
        deterministic=config["training"].get("deterministic", False),
    )
    trainer.fit(system)
    print_only("Finished training")

    checkpoint = callbacks[0]
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w", encoding="utf-8") as f:
        json.dump(best_k, f, indent=2)

    state_dict = torch.load(checkpoint.best_model_path, map_location="cpu")
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()
    torch.save(system.audio_model.serialize(), os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    from nsc_net.utils.parser_utils import prepare_parser_from_dict, parse_args_as_dict

    args = parser.parse_args()
    with open(args.conf_dir, encoding="utf-8") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    arg_dic, _ = parse_args_as_dict(parser, return_plain_args=True)
    main(arg_dic)
