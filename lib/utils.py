from __future__ import annotations
import glob
import logging
import os
import random
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torchinfo
from omegaconf import DictConfig, OmegaConf

from lib.models import MODELS
from lib.models.weight_init import weight_init

from diffusers.optimization import get_cosine_schedule_with_warmup


def create_output_directory(config: DictConfig) -> str:
    """
    Create the output directory.

    Args:
        config: Configuration dictionary.

    Returns:
        output_directory: Path to the output directory.
    """

    if 'output' in config and 'output_directory' in config.output and isinstance(config.output.output_directory, str):
        os.makedirs(config.output.output_directory, exist_ok=True)

        if 'suffix' in config.output and isinstance(config.output.suffix, str):
            # Output dir name: current datetime + suffix defined in config
            name = datetime.now().strftime('%Y-%m-%d_%H-%M') + '_' + config.output.suffix
        else:
            # Output dir name: current datetime
            name = datetime.now().strftime('%Y-%m-%d_%H-%M')

        output_directory = os.path.join(config.output.output_directory, config.method.model_type, name)
        os.makedirs(output_directory, exist_ok=True)
    else:
        output_directory = None

    return output_directory


def count_model_parameters(model) -> int:
    """
    Count the number of trainable parameters in the model.

    Args:
        model: torch model.

    Returns:
        int: Number of trainable parameters.
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_default_model_settings(model, args_model: DictConfig) -> None:
    """
    Fill args_model with the model's default parameter settings if not already specified.

    Args:
        model: The model to train.
        args_model: Dictionary storing model architecture parameters.
    """

    default_parms = []

    if isinstance(model, MODELS['utilise']):
        # Default parameter list for utilise model
        default_parms = ['encoder_widths', 'decoder_widths', 'str_conv_k', 'str_conv_s', 'str_conv_p', 'agg_mode',
                         'upconv_type', 'encoder_norm', 'decoder_norm', 'skip_norm', 'activation', 'n_head', 'd_k',
                         'bias_qk', 'attn_dropout', 'dropout', 'return_maps', 'padding_mode', 'skip_attention',
                         'output_activation', 'n_groups', 'dim_per_group', 'group_norm_eps', 'ltae_norm',
                         'str_conv_k_up', 'str_conv_p_up', 'norm_first']

    for param in default_parms:
        if param not in args_model:
            val = getattr(model, param)
            args_model[param] = val.value if isinstance(val, Enum) else val


def get_model(config: DictConfig, input_dim: int, logger: Optional[logging.Logger] = None):
    """
    Return a model instance and its parameter settings.

    Args:
        config: Configuration dictionary.
        input_dim: Number of input channels.
        logger: Logger instance.

    Returns:
        model: The model to train.
        args_model: Dictionary storing model architecture parameters.
    """

    model_type = config.method.model_type

    if model_type not in MODELS or model_type not in config:
        if logger is not None:
            logger.error(f"{model_type} model is not implemented.\n")
        else:
            raise NotImplementedError(f"ERROR: {model_type} model is not implemented.\n")

    args_model = deepcopy(config[model_type]) if model_type in config else OmegaConf.create()

    model = MODELS[model_type](**args_model)

    # get_default_model_settings(model, args_model)

    return model, args_model


def get_optimizer(config: DictConfig, model, logger: Optional[logging.Logger] = None):
    """
    Return an optimizer instance.

    Args:
        config: Configuration dictionary.
        model: The model to train.
        logger: Logger instance.

    Returns:
        optimizer: The optimizer for training.
    """

    if config.optimizer.name == 'Adam':
        betas = config.optimizer.get('betas', (0.9, 0.999))
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
            betas=betas
        )
    elif config.optimizer.name == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        )
    elif config.optimizer.name == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
            momentum=config.optimizer.momentum
        )
    else:
        if logger is not None:
            logger.error(f"{config.optimizer.name} optimizer is not implemented.\n")
            sys.exit(1)
        else:
            raise NotImplementedError(f"ERROR: {config.optimizer.name} optimizer is not implemented.\n")

    return optimizer


def get_scheduler(config: DictConfig, optimizer, len_trainset, logger: Optional[logging.Logger] = None):
    """
    Return a learning rate scheduler instance.

    Args:
        config: Configuration dictionary.
        optimizer: Optimizer instance.
        len_trainset: Length of the training set.
        logger: Logger instance.

    Returns:
        scheduler: Learning rate scheduler instance (None if disabled).
    """

    if config.scheduler.enabled:
        name = config.scheduler.name
        settings = without_keys(config.scheduler, ['name', 'enabled'])

        if name == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True, **settings)
        elif name == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, verbose=False, **settings)
        elif name == 'MultiStepLR':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **settings)
        elif name == 'ExponentialLR':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, verbose=False, **settings)
        elif name == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, verbose=False, T_max=config.training_settings.num_epochs
            )
        elif name == 'CosineWithWarmup':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=config.optimizer.lr_warmup_steps,
                num_training_steps=(len_trainset * config.training_settings.num_epochs),
            )
        else:
            if logger:
                logger.error(f"{name} learning rate scheduler is not implemented.\n")
                sys.exit(1)
            else:
                raise NotImplementedError(f"ERROR: {name} learning rate scheduler is not implemented.\n")
    else:
        scheduler = None

    return scheduler


def get_trainer(
        config: DictConfig,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        model,
        noise_scheduler,
        optimizer,
        scheduler
) -> Trainer:
    """
    Return a Trainer instance.

    Args:
        config: Configuration dictionary.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        model: The model to train.
        noise_scheduler: Noise scheduler (e.g. DDIMScheduler).
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.

    Returns:
        Trainer instance.
    """

    args = without_keys(config, ['scheduler', 'training_settings', 'misc', 'output'])
    if not isinstance(args, DictConfig):
        args = OmegaConf.create(args)

    if not config.scheduler.enabled:
        args.scheduler = OmegaConf.create()
        args.scheduler.name = config.scheduler.name
        args.scheduler.enabled = config.scheduler.enabled
    else:
        args.scheduler = deepcopy(getattr(config, 'scheduler'))

    for key in config.training_settings.keys():
        args[key] = getattr(config.training_settings, key)

    for key in config.misc.keys():
        args[key] = getattr(config.misc, key)

    args.save_dir = config.output.experiment_folder
    args.checkpoint_dir = config.output.checkpoint_dir

    if 'wandb' in args:
        args.wandb.dir = config.output.experiment_folder

    if args.get('resume', False) and args.get('pretrained_path', None) is not None:
        experiment_directory = Path(args.pretrained_path).parent.parent

        if 'wandb' in args:
            log_file = experiment_directory / 'training.log'
            if os.path.exists(log_file):
                shutil.copy(log_file, Path(args.save_dir) / 'training.log')

            path_model = Path(args.pretrained_path).parents[0] / 'Model_best.pth'
            if os.path.exists(path_model):
                shutil.copy(path_model, Path(args.checkpoint_dir) / 'Model_best.pth')
        else:
            experiment_tboard_log_dir = experiment_directory.parent / 'logs' / experiment_directory.name

            if os.path.isdir(experiment_tboard_log_dir):
                tb_files = glob.glob(os.path.join(experiment_tboard_log_dir, 'events.*'))
                for tb_file in tb_files:
                    shutil.copy(tb_file, Path(args.checkpoint_dir) / Path(tb_file).name)
    else:
        args.resume = False
        args.pretrained_path = None

    args.max_seq_length = args.data.max_seq_length

    from train.trainer import Trainer  # lazy import to avoid circular dependency
    return Trainer(args, train_loader, val_loader, model, noise_scheduler, optimizer, scheduler)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed(seed)


def write_model_structure_to_file(
        filepath: str,
        model,
        batch_size: int,
        seq_length: int,
        in_channels: int,
        image_size: Tuple[int, int]
) -> None:
    """
    Write the model architecture to a text file.

    Args:
        filepath: Output text file path.
        model: The model to train.
        batch_size: Batch size.
        seq_length: Sequence length.
        in_channels: Number of input channels.
        image_size: Image size (width, height).
    """

    original = sys.stdout
    sys.stdout = open(filepath, "w", encoding="utf-8")

    if isinstance(model, MODELS['SDT']):
        torchinfo.summary(model.cuda(), input_size=[
            (batch_size, seq_length, in_channels, *image_size),  # Input (image time series)
            (batch_size,),  # Timestep (noise step)
            (batch_size, seq_length),  # Batch positions (observation date sequence)
            (batch_size, seq_length, 3, *image_size)  # SAR condition input
        ], device='cuda', depth=5)
    else:
        torchinfo.summary(model.cuda(), input_size=(batch_size, seq_length, in_channels, *image_size), device='cuda')

    torch.cuda.empty_cache()
    print('\n\n')
    print(model)

    sys.stdout = original


def without_keys(d, ignore_keys):
    """
    Return a copy of the dictionary without the specified keys.

    Args:
        d: Original dictionary.
        ignore_keys: List of keys to ignore.

    Returns:
        Trimmed dictionary.
    """

    d_trim = {k: v for k, v in d.items() if k not in ignore_keys}

    if isinstance(d, DictConfig):
        return OmegaConf.create(d_trim)
    return d_trim
