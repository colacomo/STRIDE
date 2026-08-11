import argparse
import csv
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
if not hasattr(torch, 'compiler'):
    torch.compiler = type(sys)('compiler')
    torch.compiler.disable = lambda fn: fn
if not hasattr(torch, 'float8_e4m3fn'):
    torch.float8_e4m3fn = None
if not hasattr(torch, 'float8_e5m2'):
    torch.float8_e5m2 = None
from prodict import Prodict
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from lib import config_utils
from lib.arguments import eval_parser
from lib.data_utils import get_dataset
# from lib.eval_tools import Imputation
from train.eval_tools import Imputation
from lib.logger import AverageMeter
from lib.metrics import EvalMetrics
from torchvision import transforms

trans_scale = transforms.Normalize([0.5], [0.5])
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
        print("config_file",args.config_file,args.config_file_eval)
        self.config = config_utils.read_config(args.config_file_eval)
        print("bu", self.config)

        # Merge generic data settings (used during training) with test-specific data settings
        self.config.data.update(args_test_data)
        print(args_test_data)
        print("au", self.config)
        self.config.data.preprocessed = True

        # Evaluate the entire image sequence
        #self.config.data.max_seq_length = None

        # Get the data loader
        dset = get_dataset(self.config, phase='test')
        self.dataloader = torch.utils.data.DataLoader(
            dataset=dset, batch_size=1, shuffle=False, num_workers=self.config.misc.num_workers, drop_last=False
        )

        # Get the imputation model
        self.imputation = Imputation(
            config_file_train=self.args.config_file_eval,
            method=self.args.method,
            mode=args.mode,
            checkpoint=self.args.checkpoint,
            multigpus=True,
            num_inference_steps=args.inference_steps,
            ifDate=True,  # False or True
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
                #print("main", torch.max(batch['x']), torch.max(y_pred), torch.mean(batch['x'] - y_pred),torch.mean(batch['y'] - y_pred), y_pred.shape)

                # Evaluation
                metrics = self.compute_metrics(batch, y_pred)
                self.imputation.create_summary_report(batch, y_pred, "final_report")

                for key, value in metrics.items():
                    self.stats[key].update(value)

        # Average metrics over all samples
        for metric in self.stats.keys():
            self.stats[metric] = self.stats[metric].avg

        return self.stats

    def collect_mask_diagnostics(self, output_dir, epoch, latent_mask_alpha=1.0, max_sequences=256):
        """Collect date-level degradation probabilities and show distribution differences across Clean/Degraded/Cloud."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        model = self.imputation.model
        model.eval()
        records = []

        with torch.no_grad():
            for sequence_index, batch in enumerate(tqdm(self.dataloader, desc='Mask diagnostics', leave=False)):
                if sequence_index >= max_sequences:
                    break

                x = batch['x'].to(self.device)
                date = batch['position_days'].to(self.device)
                cond = batch['cond'].to(self.device) if batch.get('cond') is not None else None
                input_mask = batch['masks'].to(self.device)
                real_cloud = batch['cloud_mask'].to(self.device)
                md_rec = batch['md_rec'].to(self.device)

                original_length = x.shape[1]
                window = model.num_frames
                valid_length = min(original_length, window)
                if original_length > window:
                    x, date, input_mask, real_cloud, md_rec = (
                        tensor[:, :window] for tensor in (x, date, input_mask, real_cloud, md_rec)
                    )
                    if cond is not None:
                        cond = cond[:, :window]
                elif original_length < window:
                    pad_length = window - original_length
                    x = torch.cat([x, x.new_zeros(x.shape[0], pad_length, *x.shape[2:])], dim=1)
                    date = torch.cat([date, date.new_zeros(date.shape[0], pad_length)], dim=1)
                    input_mask = torch.cat([
                        input_mask,
                        input_mask.new_zeros(input_mask.shape[0], pad_length, *input_mask.shape[2:])
                    ], dim=1)
                    real_cloud = torch.cat([
                        real_cloud,
                        real_cloud.new_zeros(real_cloud.shape[0], pad_length, *real_cloud.shape[2:])
                    ], dim=1)
                    md_rec = torch.cat([md_rec, md_rec.new_zeros(md_rec.shape[0], pad_length)], dim=1)
                    if cond is not None:
                        cond = torch.cat([
                            cond, cond.new_zeros(cond.shape[0], pad_length, *cond.shape[2:])
                        ], dim=1)

                _, latent_mask, mask_total = model(
                    x,
                    date=date,
                    cond=cond,
                    cloud_mask=input_mask,
                    return_masks=True,
                    latent_mask_alpha=latent_mask_alpha,
                )
                latent_probability = latent_mask[:, :, 0, 0, 0]
                total_mean = mask_total.mean(dim=(2, 3, 4))
                cloud_coverage = (real_cloud > 0).float().mean(dim=(2, 3, 4))
                input_coverage = (input_mask > 0).float().mean(dim=(2, 3, 4))

                sample_indices = batch.get('sample_index', [str(sequence_index)])
                for b in range(x.shape[0]):
                    sample_id = sample_indices[b] if isinstance(sample_indices, (list, tuple)) else sample_indices
                    for t in range(valid_length):
                        degradation = int(md_rec[b, t].item())
                        cloud = float(cloud_coverage[b, t].item()) > 0
                        degraded = degradation > 0
                        class_name = 'Cloud' if cloud else ('Degraded' if degraded else 'Clean')
                        records.append({
                            'epoch': int(epoch),
                            'sequence_index': sequence_index,
                            'sample_id': str(sample_id),
                            'time_index': t,
                            'position_day': float(date[b, t].item()),
                            'md_rec': degradation,
                            'class_name': class_name,
                            'overlap': bool(cloud and degraded),
                            'latent_probability': float(latent_probability[b, t].item()),
                            'cloud_coverage': float(cloud_coverage[b, t].item()),
                            'input_mask_coverage': float(input_coverage[b, t].item()),
                            'mask_total_mean': float(total_mean[b, t].item()),
                            'latent_alpha': float(latent_mask_alpha),
                        })

        if not records:
            raise RuntimeError('No valid dates were available for mask diagnostics.')

        self._save_mask_diagnostics(records, output_dir, epoch, latent_mask_alpha)
        return output_dir

    @staticmethod
    def _save_mask_diagnostics(records, output_dir, epoch, latent_mask_alpha):
        fieldnames = list(records[0].keys())
        with open(output_dir / 'frame_metrics.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(records)

        classes = ('Clean', 'Degraded', 'Cloud')
        colors = {'Clean': '#1baf7a', 'Degraded': '#eb6834', 'Cloud': '#2a78d6'}
        markers = {'Clean': 'o', 'Degraded': '^', 'Cloud': 's'}
        arrays = {key: np.asarray([record[key] for record in records]) for key in fieldnames}
        np.savez_compressed(
            output_dir / 'mask_diagnostics.npz',
            **arrays,
            class_order=np.asarray(classes),
            class_colors=np.asarray([colors[name] for name in classes]),
            histogram_bins=np.linspace(0, 1, 21),
        )

        summaries = []
        for class_name in classes:
            values = np.asarray([
                record['latent_probability'] for record in records if record['class_name'] == class_name
            ], dtype=np.float32)
            total_values = np.asarray([
                record['mask_total_mean'] for record in records if record['class_name'] == class_name
            ], dtype=np.float32)
            row = {'class_name': class_name, 'count': len(values)}
            for prefix, data in (('latent', values), ('mask_total', total_values)):
                if len(data):
                    quantiles = np.quantile(data, [0.05, 0.25, 0.5, 0.75, 0.95])
                    row.update({
                        f'{prefix}_mean': float(data.mean()), f'{prefix}_std': float(data.std()),
                        f'{prefix}_q05': float(quantiles[0]), f'{prefix}_q25': float(quantiles[1]),
                        f'{prefix}_median': float(quantiles[2]), f'{prefix}_q75': float(quantiles[3]),
                        f'{prefix}_q95': float(quantiles[4]),
                    })
                else:
                    row.update({f'{prefix}_{name}': np.nan for name in (
                        'mean', 'std', 'q05', 'q25', 'median', 'q75', 'q95'
                    )})
            summaries.append(row)

        with open(output_dir / 'class_summary.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(summaries[0].keys()))
            writer.writeheader()
            writer.writerows(summaries)

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        bins = np.linspace(0, 1, 21)
        for class_name in classes:
            subset = [record for record in records if record['class_name'] == class_name]
            probabilities = np.asarray([record['latent_probability'] for record in subset])
            if len(probabilities):
                axes[0, 0].hist(
                    probabilities, bins=bins, weights=np.ones(len(probabilities)) / len(probabilities),
                    histtype='step', linewidth=2, color=colors[class_name],
                    label=f'{class_name} (n={len(probabilities)})'
                )
                axes[0, 1].scatter(
                    [record['latent_probability'] for record in subset],
                    [record['cloud_coverage'] for record in subset],
                    s=24, alpha=0.4, marker=markers[class_name], color=colors[class_name],
                    label=class_name,
                )
                axes[1, 0].scatter(
                    [record['latent_probability'] for record in subset],
                    [record['mask_total_mean'] for record in subset],
                    s=24, alpha=0.4, marker=markers[class_name], color=colors[class_name],
                    label=class_name,
                )

        overlap = [record for record in records if record['overlap']]
        if overlap:
            axes[0, 1].scatter(
                [record['latent_probability'] for record in overlap],
                [record['cloud_coverage'] for record in overlap],
                s=34, marker='x', linewidths=1.4, color='#0b0b0b', label='Cloud+Degraded'
            )
            axes[1, 0].scatter(
                [record['latent_probability'] for record in overlap],
                [record['mask_total_mean'] for record in overlap],
                s=34, marker='x', linewidths=1.4, color='#0b0b0b', label='Cloud+Degraded'
            )

        axes[0, 0].set(title='Latent probability by class', xlabel='Latent probability', ylabel='Within-class fraction')
        axes[0, 1].set(title='Latent probability vs real cloud coverage', xlabel='Latent probability', ylabel='Cloud coverage')
        axes[1, 0].set(title='Latent probability vs mask_total mean', xlabel='Latent probability', ylabel='mask_total mean')
        for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
            axis.grid(True, color='#d8d8d5', linewidth=0.6, alpha=0.6)
            axis.legend()

        axes[1, 1].axis('off')
        lines = [f'Epoch {epoch}   alpha={latent_mask_alpha:.3f}', f'Dates: {len(records)}']
        for row in summaries:
            lines.append(
                f"{row['class_name']}: n={row['count']}, "
                f"latent median={row['latent_median']:.3f}, "
                f"IQR=[{row['latent_q25']:.3f}, {row['latent_q75']:.3f}]"
            )
        lines.append(f"Cloud+Degraded overlap: {len(overlap)}")
        axes[1, 1].text(0.02, 0.98, '\n'.join(lines), va='top', fontsize=11, family='monospace')
        axes[1, 1].set_title('Summary')

        fig.suptitle(f'Mask diagnostics after epoch {epoch}', fontsize=15)
        fig.tight_layout()
        fig.savefig(output_dir / 'mask_dashboard.png', dpi=180, bbox_inches='tight')
        plt.close(fig)

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


if __name__ == '__main__':

    #if len(sys.argv) < 2:
    #    eval_parser.print_help()
    #    sys.exit(1)

    args = eval_parser.parse_args()

    # Extract settings w.r.t. test data
    if args.test_data.test_config is not None:
        if not os.path.isfile(args.test_data.test_config):
            raise FileNotFoundError(f'Cannot find the test configuration file: {args.test_data.test_config}\n')
        args_test_data = config_utils.read_config(args.test_data.test_config).data
    else:
        args_test_data = OmegaConf.create()

        if args.test_data.data_dir is not None:
            if not os.path.exists(args.test_data.data_dir):
                raise ValueError(f'Cannot find the data directory: {args.test_data.data_dir}\n')
            args_test_data.root = args.test_data.data_dir
        if args.test_data.hdf5_file is not None:
            if not os.path.isfile(os.path.join(args_test_data.root, args.test_data.hdf5_file)):
                raise FileNotFoundError(
                    f'Cannot find the data file: {os.path.join(args_test_data.root, args.test_data.hdf5_file)}\n')
            args_test_data.hdf5_file = args.test_data.hdf5_file
        if args.test_data.split is not None:
            args_test_data.split = args.test_data.split
        if args.test_data.mode is not None:
            args_test_data.mode = args.test_data.mode

        args_test_data.rescale = args.test_data.rescale

    evaluator = Evaluator(args, args_test_data)

    since = time.time()
    stats = evaluator.evaluate()
    time_elapsed = time.time() - since

    print('Evaluation completed in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))

    print('Statistics:\n===========')
    # for k, v in stats.items():
    #     print(f'{k}: {v}')
    print_stats(stats, evaluator)
