import copy
import logging
import random
from dataclasses import dataclass
import time
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from typing import Any, Dict, Optional, Tuple
from torch import Tensor
from PIL import Image
import numpy as np
import prodict
from prodict import Prodict
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from lib import logger, visutils, config_utils
from lib.logger import AverageMeter
from lib.loss import TrainLoss
from lib.data_utils import to_device
import wandb  # experiment tracking
from .run_eval import Evaluator, print_stats, save_stats_to_file
from tqdm.auto import tqdm  # progress bar
from pathlib import Path
import os
import matplotlib.pyplot as plt
import seaborn as sns  # optional, nicer heatmaps




def seconds_to_dd_hh_mm_ss(seconds_elapsed: int) -> Tuple[int, int, int, int]:
    days = seconds_elapsed // (24 * 3600)
    seconds_remainder = seconds_elapsed % (24 * 3600)
    hours = seconds_remainder // 3600
    seconds_remainder %= 3600
    minutes = seconds_remainder // 60
    seconds_remainder %= 60
    seconds = seconds_remainder

    return days, hours, minutes, seconds


def make_grid(images, rows, cols):
    """Arrange images into a grid."""
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(output_dir, epoch, pipeline, val_batch):
    """Evaluation function, generates sample images."""
    # Sample images from random noise (reverse diffusion process): 1. Unconditional sampling (random noise -> image)
    images = pipeline(
        batch_size=1,
        generator=torch.manual_seed(0),
    ).images
    # 2. Conditional sampling (given cloudy image + mask -> reconstruction)
    output = pipeline(val_batch['y'], val_batch['masks'], generator=torch.manual_seed(0)).images
    # 3. Assemble grid and save
    # Arrange images into a grid
    image_grid = make_grid(images, rows=4, cols=4)
    # Save image
    test_dir = os.path.join(output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


class Trainer:
    """Diffusion model trainer for satellite image time series reconstruction."""

    def __init__(
            self,
            args: DictConfig,
            train_loader: torch.utils.data.dataloader.DataLoader,
            val_loader: torch.utils.data.dataloader.DataLoader,
            model,
            noise_scheduler,  # Diffusion model noise scheduler
            optimizer,
            scheduler
    ):
        self.args = args
        self.use_wandb = bool('wandb' in args)  # Whether to use Weights & Biases
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.attn_cache = {}
        self.CTHW = self.args.data.ifCTHW  # Whether data format is CTHW (Channel-Time-Height-Width)

        # Data loaders
        self.dataloader = {'train': train_loader, 'val': val_loader}

        # Model and optimizer
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model.to(self.device)

        # Gradient accumulation
        self.args.accum_iter = self.args.get('accum_iter', 1)  # Accumulate gradients over accum_iter iterations

        latent_cfg = self.args.get('latent_mask', {})
        self.latent_loss_weight = float(latent_cfg.get('loss_weight', 0.01))
        self.latent_positive_weight = float(latent_cfg.get('positive_weight', 9.0))
        self.latent_alpha_max = float(latent_cfg.get('alpha_max', 1.0))
        self.latent_alpha_ramp_epochs = int(latent_cfg.get('alpha_ramp_epochs', 100))
        if self.latent_loss_weight < 0:
            raise ValueError('latent_mask.loss_weight must be non-negative')
        if self.latent_positive_weight <= 0:
            raise ValueError('latent_mask.positive_weight must be positive')
        if not 0.0 <= self.latent_alpha_max <= 1.0:
            raise ValueError('latent_mask.alpha_max must be in [0, 1]')
        if self.latent_alpha_ramp_epochs < 0:
            raise ValueError('latent_mask.alpha_ramp_epochs must be non-negative')
        self.latent_ramp_origin_epoch = 0

        # Loss functions
        self.compute_losses = TrainLoss(self.args.loss)  # Multi-task loss
        self.loss_fn = F.mse_loss  # MSE loss for diffusion main loss (noise prediction)
        # self.compute_metrics = EvalMetrics(self.args.metrics)

        # Initialize statistics
        self.train_stats = self._stats_meter(stats_type='loss')  # Training loss average meter
        self.val_stats = self._stats_meter(stats_type='loss')

        # Initialize metrics
        self.train_metrics = self._stats_meter(stats_type='metrics')
        self.val_metrics = self._stats_meter(stats_type='metrics')

        # Best loss record
        self.best_loss = np.inf
        self.epoch_best_loss = np.nan

        # Create directories
        os.makedirs(self.args.save_dir, exist_ok=True)
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        self.args.path_model_best = os.path.join(self.args.checkpoint_dir, 'Model_best.pth')
        self.args.path_model_last = os.path.join(self.args.checkpoint_dir, 'Model_last.pth')

        # Setup logging
        self.logger = logger.prepare_logger('train_logger', level=logging.INFO, log_to_console=True,
                                            log_file=os.path.join(args.save_dir, 'training.log'))

        # Noise scheduler
        self.noise_scheduler = noise_scheduler

        # Setup wandb
        if self.use_wandb:
            os.makedirs(self.args.wandb.dir, exist_ok=True)
            wandb.init(**self.args.wandb, settings=wandb.Settings(start_method="fork"))
            wandb.config.update(OmegaConf.to_container(self.args))
            self.writer = None

            # Define wandb summary metrics
            # for key, value in self.args.metrics.items():
            #     if key == 'masked_metrics':
            #         pass
            #     elif value:
            #         wandb.define_metric(f"train_metrics/{key}", summary=OBJECTIVE[key])
            #         wandb.define_metric(f"val_metrics/{key}", summary=OBJECTIVE[key])

            # wandb.define_metric('train/total_loss', summary=OBJECTIVE['total_loss'])
            # wandb.define_metric('val/total_loss', summary=OBJECTIVE['total_loss'])
        else:
            # Use TensorBoard
            os.makedirs(os.path.join(self.args.save_dir, 'tb'), exist_ok=True)
            self.writer = SummaryWriter(log_dir=os.path.join(self.args.save_dir, 'tb'))

        # Resume training
        if self.args.resume and self.args.pretrained_path:
            self._resume(path=self.args.pretrained_path)
        else:
            self.logger.info('\nTraining from scratch.\n')
            self.epoch = 0
            self.iter = 0

    def _stats_meter(self, stats_type: str) -> prodict.Prodict:
        """Create statistics meter."""
        meters = Prodict()
        stats = self._stats_dict(stats_type)
        for key, _ in stats.items():
            meters[key] = AverageMeter()  # Running average meter

        return meters

    def _get_lr(self, group: int = 0) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[group]['lr']

    def _resume(self, path: str) -> None:
        """
        Resume training.

        Args:
            path: Path to pretrained model weights.
        """

        if not os.path.isfile(path):
            raise FileNotFoundError(f'No checkpoint found at {path}\n')

        self.logger.info(f'\nLoading checkpoint from: {path}')
        checkpoint = torch.load(path, map_location=self.device)
        latent_head = checkpoint.get('latent_mask_head')
        if latent_head is None or latent_head.get('type') != 'date_avgmax_binary_v1':
            raise RuntimeError(
                'The checkpoint uses an incompatible latent-mask head. '
                'Start a new experiment for the date-level avg/max pooling head.'
            )

        # Load model state
        if 'model_state_dict' in checkpoint:
            # Handle DataParallel wrapped model
            state_dict = checkpoint['model_state_dict']
            # Remove 'module.' prefix (if model is DataParallel wrapped)
            if list(state_dict.keys())[0].startswith('module.'):
                # If current model is not DataParallel wrapped, but checkpoint is
                if not isinstance(self.model, nn.DataParallel):
                    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    self.model.load_state_dict(new_state_dict, strict=False)
                else:
                    self.model.load_state_dict(state_dict, strict=False)
            else:
                # If current model is DataParallel wrapped, but checkpoint is not
                if isinstance(self.model, nn.DataParallel):
                    new_state_dict = {'module.' + k: v for k, v in state_dict.items()}
                    self.model.load_state_dict(new_state_dict, strict=False)
                else:
                    self.model.load_state_dict(state_dict, strict=False)
        else:
            # Compatible with legacy checkpoints
            self.model.load_state_dict(checkpoint, strict=False)

        # Load optimizer state
        if 'optimizer_state_dict' in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info('Optimizer state loaded.')

        # Load scheduler state
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            if self.args.get('load_scheduler_state_dict', True):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.logger.info('Scheduler state loaded.')

        # Load training state
        self.epoch = checkpoint.get('epoch', 0)
        self.start_epoch = self.epoch  # Record start epoch
        self.iter = checkpoint.get('iter', 0)
        self.best_loss = checkpoint.get('best_loss', np.inf)
        self.epoch_best_loss = checkpoint.get('best_epoch', 0)

        latent_schedule = checkpoint.get('latent_mask_schedule')
        if latent_schedule is None:
            # Legacy checkpoint: mask head was not 0/1 calibrated, restart gradual ramp from resume point.
            self.latent_ramp_origin_epoch = self.epoch
            self.logger.info(
                'Legacy checkpoint: restarting latent-mask attention ramp at epoch %d.', self.epoch
            )
        else:
            self.latent_ramp_origin_epoch = int(latent_schedule.get('ramp_origin_epoch', 0))
            self.logger.info(
                'Restored latent-mask attention ramp origin: epoch %d.',
                self.latent_ramp_origin_epoch
            )
        
        # Adjust total training epochs
        self.logger.info(f'Resuming training from epoch {self.epoch}, iteration {self.iter}')
        self.logger.info(f'Best loss so far: {self.best_loss:.4f} at epoch {self.epoch_best_loss}')
        self.logger.info(f'Total training epochs: {self.args.num_epochs}')

    def _log_iter_epoch(self) -> None:
        """Log iteration and epoch info."""
        if self.use_wandb:
            wandb.log({'epoch': self.epoch}, step=self.iter)
        else:
            self.writer.add_scalar('epoch', self.epoch, self.iter)

    def _log_learning_rate(self) -> None:
        """Log learning rate."""
        if self.use_wandb:
            wandb.log({'log_lr': np.log10(self._get_lr()), 'epoch': self.epoch}, step=self.iter)
        else:
            self.writer.add_scalar('log_lr', np.log10(self._get_lr()), self.epoch)
            
    def _get_latent_mask_alpha(self) -> float:
        if self.latent_alpha_ramp_epochs == 0:
            return self.latent_alpha_max
        progress = (self.epoch - self.latent_ramp_origin_epoch) / self.latent_alpha_ramp_epochs
        progress = min(max(progress, 0.0), 1.0)
        return self.latent_alpha_max * progress

    def compute_latent_mask_loss(self, date_logits, md_rec):
        """Date-level supervision: degraded dates are 1, clean dates are 0."""
        expected_shape = (*md_rec.shape, 1, 1, 1)
        if md_rec.ndim != 2 or tuple(date_logits.shape) != expected_shape:
            raise ValueError(
                f'date logits shape {tuple(date_logits.shape)} must be {expected_shape} '
                f'for md_rec shape {tuple(md_rec.shape)}'
            )

        frame_target = (md_rec > 0).to(device=date_logits.device, dtype=torch.float32)
        date_target = frame_target[:, :, None, None, None]
        raw_bce = F.binary_cross_entropy_with_logits(
            date_logits.float(), date_target, reduction='none'
        )
        positive_weight = torch.as_tensor(
            self.latent_positive_weight, device=raw_bce.device, dtype=raw_bce.dtype
        )
        date_weight = torch.where(date_target > 0.5, positive_weight, torch.ones_like(raw_bce))
        loss_latent = (raw_bce * date_weight).sum() / date_weight.sum().clamp_min(1.0)

        with torch.no_grad():
            probabilities = torch.sigmoid(date_logits.float())
            positive = date_target > 0.5
            negative = ~positive
            prob_degraded = probabilities[positive].mean() if positive.any() else probabilities.new_zeros(())
            prob_clean = probabilities[negative].mean() if negative.any() else probabilities.new_zeros(())

        stats = {
            'latent_bce': loss_latent.detach(),
            'latent_prob_degraded': prob_degraded,
            'latent_prob_clean': prob_clean,
            'latent_positive_rate': frame_target.mean(),
        }
        return loss_latent, stats

    def _stats_dict(self, stats_type: str) -> prodict.Prodict:
        """Create statistics dictionary."""
        stats = Prodict()

        if stats_type == 'metrics':
            # Handle metrics statistics
            masked_metrics = self.args.metrics.masked_metrics
            for key, value in self.args.metrics.items():
                if key == 'masked_metrics':
                    pass
                elif value:
                    if masked_metrics and key != 'ssim':
                        stats[f'masked_{key}'] = np.inf
                    else:
                        stats[key] = np.inf

        elif stats_type == 'loss':
            # Handle loss statistics
            for key, value in self.args.loss.items():
                # Exclude weight keys
                if value and isinstance(value, bool):
                    stats[key] = np.inf
            stats.total_loss = np.inf
            stats.reconstruction_loss = np.inf
            stats.latent_bce = np.inf
            stats.latent_prob_degraded = np.inf
            stats.latent_prob_clean = np.inf
            stats.latent_positive_rate = np.inf
            stats.latent_alpha = np.inf

        return stats

    def _save_checkpoint(self, filepath: str) -> None:
        """Save checkpoint."""
        state = {
            'epoch': self.epoch,
            'iter': self.iter,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'best_epoch': self.epoch_best_loss,
            'latent_mask_head': {
                'version': 2,
                'type': 'date_avgmax_binary_v1',
            },
            'latent_mask_schedule': {
                'version': 1,
                'ramp_origin_epoch': self.latent_ramp_origin_epoch,
                'alpha_max': self.latent_alpha_max,
                'alpha_ramp_epochs': self.latent_alpha_ramp_epochs,
            }
        }

        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(state, filepath)

    def _log_stats_meter(self, phase: str) -> None:
        """Log statistics meter."""
        if self.use_wandb:
            if phase == 'train':
                wandb.log({
                    'train_losses/' + k: v.avg for k, v in self.train_stats.items()
                }, step=self.iter)
            else:
                stats = {'val_losses/' + k: v.avg for k, v in self.val_stats.items()}
                stats['epoch'] = self.epoch
                wandb.log(stats, step=self.iter)

        else:
            if phase == 'train':
                for k, v in self.train_stats.items():
                    self.writer.add_scalar('train_losses/' + k, v.avg, self.iter)
            else:
                for k, v in self.val_stats.items():
                    self.writer.add_scalar('val_losses/' + k, v.avg, self.iter)

    def train(self, cli_args) -> None:
        """Main training loop."""
        # Prepare evaluation args (extracted from cli_args, similar to run_eval)
        if cli_args.test_config is not None:
            if not os.path.isfile(cli_args.test_config):
                raise FileNotFoundError(f'Cannot find the test configuration file: {cli_args.test_config}\n')
            args_test_data = config_utils.read_config(cli_args.test_config).data
        else:
            args_test_data = OmegaConf.create()
            # ... other parameter handling code unchanged ...

        if self.use_wandb and self.args.get('log_gradients', False):
            wandb.watch(self.model, log='all')

        self.logger.info('\nStart training...\n')
        start_time = time.time()

        # Calculate remaining epochs
        remaining_epochs = self.args.num_epochs - self.epoch
        self.logger.info(f'Remaining epochs to train: {remaining_epochs}')

        # Use progress bar to show remaining training epochs
        with tqdm(range(self.epoch, self.args.num_epochs), leave=True) as tnr:
            tnr.set_description("Epoch")
            tnr.set_postfix(epoch=self.epoch, training_loss=np.nan)

            for current_epoch in tnr:
                # Update current epoch (use loop variable)
                self.epoch = current_epoch

                if self.scheduler is not None:
                    self._log_learning_rate()

                # -------------------------------- Training phase -------------------------------- #
                self.train_epoch(tnr)  # Run one epoch

                # Save latest weights for this round to ensure periodic eval doesn't read previous round's Model_last.
                self._save_checkpoint(self.args.path_model_last)

                if (self.epoch + 1) % 100 == 0:
                    completed_epoch = self.epoch + 1
                    self.logger.info(f'\nStarting evaluation at epoch {completed_epoch}...')
                    python_rng_state = random.getstate()
                    numpy_rng_state = np.random.get_state()
                    torch_rng_state = torch.get_rng_state()
                    cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                    evaluator = None
                    stats = None

                    try:
                        random.seed(19)
                        np.random.seed(19)
                        torch.manual_seed(19)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(19)

                        eval_args = copy.copy(cli_args)
                        runtime_config = os.path.join(self.args.save_dir, 'config.yaml')
                        eval_args.checkpoint = self.args.path_model_last
                        eval_args.config_file = runtime_config
                        eval_args.config_file_eval = runtime_config
                        torch.cuda.empty_cache()

                        with torch.no_grad():
                            evaluator = Evaluator(eval_args, args_test_data)
                            eval_since = time.time()
                            stats = evaluator.evaluate()
                            diagnostics_dir = os.path.join(
                                self.args.save_dir, 'eval', f'epoch_{completed_epoch:04d}', 'latent_mask'
                            )
                            evaluator.collect_mask_diagnostics(
                                diagnostics_dir,
                                completed_epoch,
                                latent_mask_alpha=self._get_latent_mask_alpha(),
                            )
                            eval_time_elapsed = time.time() - eval_since

                        self.logger.info(
                            'Evaluation completed in {:.0f}m {:.0f}s'.format(
                                eval_time_elapsed // 60, eval_time_elapsed % 60
                            )
                        )
                        #self.logger.info('Mask diagnostics saved to: %s', diagnostics_dir)
                        self.logger.info('Statistics:\n===========')
                        print_stats(stats, evaluator)
                        eval_file_path = os.path.join(self.args.save_dir, f'eval_epoch_{completed_epoch}.txt')
                        save_stats_to_file(stats, evaluator, eval_file_path)

                        if self.use_wandb:
                            wandb.log({'eval_stats': stats}, step=self.iter)
                        else:
                            for key, value in stats.items():
                                self.writer.add_scalar(f'eval/{key}', value, self.epoch)
                    finally:
                        if evaluator is not None:
                            del evaluator
                        torch.cuda.empty_cache()
                        random.setstate(python_rng_state)
                        np.random.set_state(numpy_rng_state)
                        torch.set_rng_state(torch_rng_state)
                        if cuda_rng_states is not None:
                            torch.cuda.set_rng_state_all(cuda_rng_states)
                        torch.set_grad_enabled(True)
                        self.model.train()
                        self.optimizer.zero_grad()

                # Update learning rate scheduler after epoch ends
                if self.scheduler is not None:
                    self._log_learning_rate()

                    if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                        self.scheduler.step(self.val_stats.total_loss.avg)
                    else:
                        self.scheduler.step()

                # Save model at specified interval
                if (self.epoch + 1) % self.args.checkpoint_every_n_epochs == 0:
                    name = 'Model_after_' + str(self.epoch + 1) + '_epochs.pth'
                    self._save_checkpoint(os.path.join(self.args.checkpoint_dir, name))

        # Training completed
        time_elapsed = int(time.time() - start_time)
        self.logger.info(
            '\n\nTraining finished!\nTraining time: %dd %dh %dm %ds' % seconds_to_dd_hh_mm_ss(time_elapsed))
        self.logger.info('\nBest model at epoch: %d', self.epoch_best_loss)
        self.logger.info(f'Validation loss of the best model: {self.best_loss:.4f}')

        # Save final model
        self._save_checkpoint(self.args.path_model_last)

        if self.use_wandb:
            wandb.finish()

    def train_epoch(self, tnr=None) -> None:
        """Single training epoch."""
        # Initialize statistics meters
        self.train_stats = self._stats_meter(stats_type='loss')
        self.model.train()

        # Clear gradients
        self.optimizer.zero_grad()

        # Training loop
        with tqdm(self.dataloader['train'], leave=False) as tnr_train:
            tnr_train.set_description(f"Training (Epoch {self.epoch})")
            tnr_train.set_postfix(epoch=self.epoch, training_loss=np.nan, best_loss=self.best_loss)

            for i, batch in enumerate(tnr_train):
                self._log_iter_epoch()  # Log global iter
                loss = self.inference_one_batch(batch)  # Forward pass and loss computation
                if np.isnan(loss.detach().item()):
                   del loss
                   torch.cuda.empty_cache()
                   self.optimizer.zero_grad()
                   continue
                loss_value = loss.detach().item()
                loss_dict = Prodict()
                # Keep original fields for compatibility with existing logs and best-loss logic.
                loss_dict.l1_loss_occluded_input_pixels = loss_value
                loss_dict.total_loss = loss_value
                for key, value in self.batch_loss_stats.items():
                    loss_dict[key] = value

                    # Update statistics meters
                for key, value in loss_dict.items():
                    self.train_stats[key].update(value)

                    # Gradient accumulation
                loss = loss / self.args.accum_iter
                loss.backward()

                    # Gradient accumulation: update params when accumulation count is reached
                if ((i + 1) % self.args.accum_iter == 0) or (i + 1 == len(self.dataloader['train'])):
                # Gradient clipping
                    if getattr(self.args, 'gradient_clip_norm', False) and self.args.gradient_clip_norm > 0.:
                       torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clip_norm)
                    elif getattr(self.args, 'gradient_clip_value', False) and self.args.gradient_clip_value > 0.:
                       torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.gradient_clip_value)

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                self.iter += 1
                tnr_train.set_postfix(epoch=self.epoch,
                    training_loss=self.train_stats.l1_loss_occluded_input_pixels.avg,
                    best_loss=self.best_loss)
                

            # Update main progress bar display
            if tnr is not None:
                tnr.set_postfix(
                    epoch=self.epoch + 1,  # Show next epoch, since current epoch is completed
                    training_loss=self.train_stats.l1_loss_occluded_input_pixels.avg,
                    best_loss=self.best_loss
                )

            # Log training stats
            self.logger.info((f'Train:\tEpoch: {self.epoch}\t' +
                              f'learning rate: {self._get_lr():.8f}\t' +
                              ''.join([f'{k}: {v.avg:.6f}\t' for k, v in self.train_stats.items()])))
            self._log_stats_meter('train')

            # Update best model
            current_loss = self.train_stats.l1_loss_occluded_input_pixels.avg
            print("cur_loss",current_loss)
            if self.best_loss > current_loss:
                self.best_loss = current_loss
                self.epoch_best_loss = self.epoch
                self._save_checkpoint(self.args.path_model_best)
                self.logger.info(f'New best model saved with loss: {self.best_loss:.6f}')

            # Reset statistics
            for key in self.train_stats:
                self.train_stats[key].reset()

    def inference_one_batch(
            self, batch: Dict[str, Any]
    ) -> Tensor:
        """Forward pass and loss computation for a single batch."""

        # Move data to device
        batch = to_device(batch, self.device)
        x_0 = batch['x']  # Multi-degraded + cloudy image: (B, T, C, H, W)
        y_0 = batch['y']  # Clean image: (B, T, C, H, W)
        cond = batch['cond']  # Condition (SAR image): (B, T, 3, H, W)
        mask = batch['masks']  # Mask: (B, T, 1, H, W)
        date = batch['position_days']  # Date: (B, T)
        md_rec = batch['md_rec']
        #print(torch.max(y_0), torch.min(y_0))

        # Adjust data format
        if self.CTHW:
            y_0 = y_0.permute(0, 2, 1, 3, 4)  # Convert to: (B, C, T, H, W)
            mask = mask.permute(0, 2, 1, 3, 4)  # Convert to: (B, 1, T, H, W)

        # Mixed precision training (commented out)
        # with torch.cuda.amp.autocast(enabled=self.args.use_amp):
        x_0 = torch.clamp(x_0, 0, 1)

        # Model prediction: conditional input on masked regions. Masked region = noise, visible region = original. model_input = noisy_images * mask + (1. - mask) * y_0
        _, mt, _, _, _ = mask.shape
        _, yt, _, _, _ = y_0.shape
        #print(date.shape, cond.shape)
        _, dt = date.shape
        _, ct, _, _, _ = cond.shape


        latent_alpha = self._get_latent_mask_alpha()
        pred_hat, latent_mask, mask_total, date_logits = self.model(
            x_0,
            date=date,
            cond=cond,
            cloud_mask=mask,
            return_masks=True,
            return_mask_logits=True,
            latent_mask_alpha=latent_alpha,
        )

        loss_rec = self.loss_fn(y_0, pred_hat)
        loss_latent, latent_stats = self.compute_latent_mask_loss(date_logits, md_rec)
        loss = loss_rec + self.latent_loss_weight * loss_latent

        self.batch_loss_stats = {
            'reconstruction_loss': loss_rec.detach().item(),
            'latent_bce': latent_stats['latent_bce'].item(),
            'latent_prob_degraded': latent_stats['latent_prob_degraded'].item(),
            'latent_prob_clean': latent_stats['latent_prob_clean'].item(),
            'latent_positive_rate': latent_stats['latent_positive_rate'].item(),
            'latent_alpha': latent_alpha,
        }
        return loss

    @staticmethod
    def linear_2pct_stretch(arr, nodata=None):
        """
        Apply 2%-98% linear stretch to input array and output 8-bit result.
        Edge case: if all valid values are 1, return all 255 directly.
        """
        valid = arr if nodata is None else arr[arr != nodata]
        if valid.size == 0:  # All values are nodata
            return np.full(arr.shape, 255, dtype=np.uint8)
        if np.all(valid == 1):  # Edge case: all values are 1
            return np.full(arr.shape, 255, dtype=np.uint8)

        p2, p98 = np.percentile(valid, (2, 98))
        out = np.clip(arr, p2, p98)
        out = (out - p2) / (p98 - p2 + 1e-8) * 255
        return out.astype(np.uint8)


                            

