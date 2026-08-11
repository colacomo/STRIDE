import argparse
import logging
import logging.config
import os
# python run.py
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
from argparse import ArgumentParser

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from lib import config_utils, data_utils
from train import utils
from lib.formatter import RawFormatter
from lib.logger import prepare_logger
from diffusers.schedulers import DDIMScheduler

from train.eval_tools import Imputation
from lib.metrics import EvalMetrics
from lib.logger import AverageMeter
from prodict import Prodict
from tqdm import tqdm
from torchvision import transforms
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os

#trans_scale = transforms.Normalize([0.5], [0.5])
generator = torch.manual_seed(19)



def print_stats(stats, evaluator, print_only_masked=False):
    prefix = evaluator.compute_metrics.prefix

    if print_only_masked is False:
        print('Metrics computed over all pixels:')
        for k, v in stats.items():
            if 'occluded_input_pixels' in k or 'observed_input_pixels' in k:
                pass
            else:
                metric = k.replace(prefix, '')
                print(f'{metric.upper()}: {v}')

    if evaluator.compute_metrics.eval_occluded_observed:
        print('\nMetrics computed over all masked input pixels:')
        for k, v in stats.items():
            if 'occluded_input_pixels' in k:
                metric = k.replace(prefix, '').replace('_occluded_input_pixels', '').replace('_images', '')
                print(f'{metric.upper()}: {v}')

        if print_only_masked is False:
            print('\nMetrics computed over all observed input pixels:')
            for k, v in stats.items():
                if 'observed_input_pixels' in k:
                    metric = k.replace(prefix, '').replace('_observed_input_pixels', '').replace('_images', '')
                    print(f'{metric.upper()}: {v}')

def save_stats_to_file(stats, evaluator, file_path, print_only_masked=False):
    prefix = evaluator.compute_metrics.prefix

    with open(file_path, 'w') as f:
        if print_only_masked is False:
            f.write('Metrics computed over all pixels:\n')
            for k, v in stats.items():
                if 'occluded_input_pixels' in k or 'observed_input_pixels' in k:
                    pass
                else:
                    metric = k.replace(prefix, '')
                    f.write(f'{metric.upper()}: {v}\n')

        if evaluator.compute_metrics.eval_occluded_observed:
            f.write('\nMetrics computed over all masked input pixels:\n')
            for k, v in stats.items():
                if 'occluded_input_pixels' in k:
                    metric = k.replace(prefix, '').replace('_occluded_input_pixels', '').replace('_images', '')
                    f.write(f'{metric.upper()}: {v}\n')

            if print_only_masked is False:
                f.write('\nMetrics computed over all observed input pixels:\n')
                for k, v in stats.items():
                    if 'observed_input_pixels' in k:
                        metric = k.replace(prefix, '').replace('_observed_input_pixels', '').replace('_images', '')
                        f.write(f'{metric.upper()}: {v}\n')

class Evaluator:
    def __init__(self, args: argparse.Namespace, args_test_data: DictConfig):
        self.args = args

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.args_metrics = {
            'masked_metrics': True,  # True： use real cloud mask to maskout the cloud pixels in targets
            'sam_units': 'deg',
            'eval_occluded_observed': True,
            'mae': True, 'rmse': True, 'mse': False, 'ssim': True, 'psnr': True, 'sam': True
        }

        self.compute_metrics = EvalMetrics(self.args_metrics)
        _ = torch.set_grad_enabled(False)

        if not os.path.isfile(args.config_file):
            raise FileNotFoundError(f'Cannot find the configuration file used during training: {args.config_file}\n')

        # Read config file used during training
        self.config = config_utils.read_config(args.config_file)

        # Merge generic data settings (used during training) with test-specific data settings
        self.config.data.update(args_test_data)
        self.config.data.preprocessed = True

        # Evaluate the entire image sequence
        self.config.data.max_seq_length = 35

        # Get the data loader
        dset = data_utils.get_dataset(self.config, phase='test')
        self.dataloader = torch.utils.data.DataLoader(
            dataset=dset, batch_size=1, shuffle=False, num_workers=self.config.misc.num_workers, drop_last=False
        )

        # Get the imputation model
        self.imputation = Imputation(
            config_file_train=self.args.config_file,
            method=self.args.method,
            mode=args.mode,
            checkpoint=self.args.checkpoint,
            multigpus=True,
            num_inference_steps=args.inference_steps,
            ifDate=True,    # False or True
            ifCond=True,
            generator=generator,
            visualize=True,  # Enable visualization
            vis_dir='./visualization_results',  # Visualization output directory
            vis_freq=1  # Visualize every reconstruction
        )

    def evaluate(self):
        self._initialize_stats()

        for i, batch in enumerate(tqdm(self.dataloader, leave=False)):
            if torch.any(batch['masks']):

                _, y_pred = self.imputation.impute_sample(batch, vis_prefix="test_sample")

                # Evaluation
                metrics = self.compute_metrics(batch, y_pred)
                self.imputation.create_summary_report(batch, y_pred, "final_report")

                for key, value in metrics.items():
                    self.stats[key].update(value)

        # Average metrics over all samples
        for metric in self.stats.keys():
            self.stats[metric] = self.stats[metric].avg

        return self.stats

    def _initialize_stats(self):
        stats = Prodict()
        eval_occluded_observed = self.args_metrics.get('eval_occluded_observed', True)

        for metric, val in self.args_metrics.items():
            if metric in ['masked_metrics', 'sam_units', 'eval_occluded_observed']:
                pass
            elif val:
                metric_name = f'masked_{metric}' if (
                        self.args_metrics['masked_metrics'] and 'ssim' not in metric
                ) else metric
                stats[metric_name] = AverageMeter()

                if eval_occluded_observed and 'ssim' not in metric:
                    stats[f'{metric_name}_occluded_input_pixels'] = AverageMeter()
                    stats[f'{metric_name}_observed_input_pixels'] = AverageMeter()

                if eval_occluded_observed and 'ssim' in metric:
                    stats[f'{metric_name}_images_occluded_input_pixels'] = AverageMeter()
                    stats[f'{metric_name}_images_observed_input_pixels'] = AverageMeter()

        self.stats = stats

# Create command line argument parser
parser = ArgumentParser(
    description='STRIDE: Reconstructing cloud-free Sentinel-2 time series under complex degradations with state-driven spatio-temporal modeling',
    formatter_class=RawFormatter
)
# Add command line arguments
parser.add_argument(
    '--config_file', type=str, default='./train/config_train.yaml', help='YAML config file to augment/override settings in configs/default.yaml'
)
parser.add_argument(
    '--save_dir', type=str, default='./results/', help='Directory path for saving models and logs'
)
parser.add_argument(
    '--resume_from', type=str, default=None,
    help='Resume training from the specified checkpoint'
)
parser.add_argument('--wandb', action='store_true', default=False,
                    help='Use Weights & Biases instead of TensorBoard')
parser.add_argument('--wandb_project', type=str, default='STRIDE',
                    help='Wandb project name')

# Evaluation-related arguments (from run_eval)
parser.add_argument('--config_file_eval', metavar='config-file', type=str, default='/media/amax/disk4/STRIDE-main/train/default_test.yaml', required=False, help='Configuration file generated by run_train.py (runtime config during training)')
parser.add_argument(
    '--method', type=str, required=False, default='STRIDE')
parser.add_argument(
    '--checkpoint', type=str, default=None, help='Model checkpoint'
)
parser.add_argument(
    '--mode', type=str, required='trivial' in sys.argv, help='Mode for non-learned baselines',
    choices=['last', 'next', 'closest', 'linear_interpolation']
)

# Optional arguments to overwrite the data settings in the config file
parser.add_argument('--test_config', type=str, default='/media/amax/disk4/STRIDE-main/train/config_test.yaml', required=False, help='YAML configuration file, test-specific settings')
parser.add_argument('--test-data.data-dir', type=str, required=False, help='Root directory of the dataset')
parser.add_argument(
    '--test-data.hdf5-file', type=str, required=False, help='HDF5 test dataset, path relative to <data-dir>')
parser.add_argument('--test-data.split', type=str, required=False, help='Data split')
parser.add_argument('--test-data.mode', type=str, required=False, help='Data mode (for EarthNet2021 only)')

parser.add_argument('--test-data.rescale', type=bool, default=False, help='rescale to [-1,1], for diffusion models only')
parser.add_argument('--inference_steps', type=int, default=1, help='(nums of inference steps, for diffusion models only')


def main(args: argparse.Namespace) -> None:
    """Main training function."""

    # Print program title
    prog_name = 's12topksar (Training with Integrated Evaluation)'
    print('\n{}\n{}\n'.format(prog_name, '=' * len(prog_name)))

    if not os.path.exists(args.config_file):
        raise FileNotFoundError(f'ERROR: Cannot find the yaml configuration file: {args.config_file}')

    cfg_custom = config_utils.read_config(args.config_file)

    if not cfg_custom:
        sys.exit(1)

    cfg_default = config_utils.read_config('train/default_train.yaml')
    config = OmegaConf.merge(cfg_default, cfg_custom)

    if args.resume_from:
        config.training_settings.resume = True
        config.training_settings.pretrained_path = args.resume_from
        print(f"Resuming training from: {args.resume_from}")
    else:
        config.training_settings.resume = False
        config.training_settings.pretrained_path = None

    config.output.output_directory = args.save_dir

    if args.wandb:
        config.wandb = OmegaConf.create()
        config.wandb.project = args.wandb_project

    config.output.experiment_folder = utils.create_output_directory(config)

    log_file = os.path.join(config.output.experiment_folder, 'run.log') if config.output.experiment_folder else None
    logger = prepare_logger('root_logger', level=logging.INFO, log_to_console=True, log_file=log_file)

    logger.info('Configuration file: %s', args.config_file)
    logger.info('\nSettings\n--------\n')
    config_utils.print_config(config, logger=logger)

    if config.misc.random_seed is not None:
        utils.set_seed(config.misc.random_seed)

    # --------------------------------------------------- Data loaders --------------------------------------------------- #
    logger.info('\nInitialize data loader (training set)...')
    train_loader = data_utils.get_dataloader_simple_sampling(
        config, phase='train', pin_memory=config.misc.pin_memory,
        drop_last=True, logger=logger, sample_ratio=0.5
    )
    logger.info('Initialize data loader (validation set)...\n')
    val_loader = None

    logger.info('Number of training samples: %d', train_loader.dataset.__len__())

    # ------------------------------------------- Prepare output directories ------------------------------------------- #
    logger.info('\nPrepare output folders and files\n--------------------------------\n')

    config.output.checkpoint_dir = os.path.join(config.output.experiment_folder, 'checkpoints')
    os.makedirs(config.output.checkpoint_dir, exist_ok=True)
    logger.info('Model weights will be stored in: %s\n', config.output.checkpoint_dir)

    config_file = os.path.join(config.output.experiment_folder, 'config.yaml')
    config_utils.write_config(config, config_file)

    # -------------------------------------------------- Define model -------------------------------------------------- #
    logger.info('\nModel Architecture\n------------------\n')
    logger.info('Architecture: %s', config.method.model_type)

    # input_dim = train_loader.dataset.num_channels
    input_dim = 10  # 4 for RGB_NIR

    model, args_model = utils.get_model(config, input_dim, logger)

    logger.info('Number of trainable parameters: %d\n', utils.count_model_parameters(model))

    config_file = os.path.join(config.output.experiment_folder, 'model_config.yaml')
    config_utils.write_config(OmegaConf.create({config.method.model_type: args_model}), config_file)

    if config.output.plot_model_txt:
        file = os.path.join(config.output.experiment_folder, 'model_parameters.txt')
        logger.info('Writing model architecture to file: %s\n', file)
        """utils.write_model_structure_to_file(
            file, model, config.training_settings.batch_size,
            train_loader.dataset.max_seq_length, input_dim,
            train_loader.dataset.image_size
        )"""

    model = nn.DataParallel(model)

    # ------------------------------------------------- Training preparation ------------------------------------------- #
    logger.info('\nPrepare training\n----------------\n')
    logger.info('Python version: %s', sys.version)
    logger.info('Torch version: %s', torch.__version__)
    logger.info('CUDA version: %s\n', torch.version.cuda)

    optimizer = utils.get_optimizer(config, model, logger)
    scheduler = utils.get_scheduler(config, optimizer, train_loader.dataset.__len__(), logger)

    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)

    if config.misc.random_seed is not None:
        utils.set_seed(config.misc.random_seed)

    trainer = utils.get_trainer(
        config, train_loader, val_loader, model, noise_scheduler,
        optimizer, scheduler
    )
    #print("inargs", args.config_file_eval)
    trainer.train(args)

if __name__ == '__main__':
    main(parser.parse_args())

