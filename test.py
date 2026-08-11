import argparse
import os
import sys
import time
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nestargs import NestedArgumentParser
from lib.formatter import RawFormatter

from lib import config_utils
from lib.data_utils import get_dataset
from train.test_tools import Imputation
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
        self.config = config_utils.read_config(args.config_file)

        # Merge generic data settings (used during training) with test-specific data settings
        self.config.data.update(args_test_data)
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
            config_file_train=self.args.config_file,
            method=self.args.method,
            mode=args.mode,
            checkpoint=self.args.checkpoint,
            multigpus=True,
            num_inference_steps=args.inference_steps,
            ifDate=True,  # False or True
            ifCond=True,
            generator=generator,
            visualize=True,  # Enable visualization
            vis_dir='/media/amax/disk4/STRIDE-main/test_visualization_results',  # Visualization output directory
            vis_freq=1  # Visualize every reconstruction
        )

    def evaluate(self):
        self._initialize_stats()

        for i, batch in enumerate(tqdm(self.dataloader, leave=False)):
            if torch.any(batch['masks']):

                _, y_pred = self.imputation.impute_sample(batch, vis_prefix="test_sample")

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


if __name__ == '__main__':

    # Evaluation arguments
    eval_parser = NestedArgumentParser(
        description='STRIDE (Evaluation)',
        formatter_class=RawFormatter
    )

    eval_parser.add_argument('--config_file', metavar='config-file', type=str, default='/media/amax/disk4/STRIDE-main/train/default_test.yaml', help='Configuration file generated by run_train.py (runtime config during training)')
    eval_parser.add_argument('--method', type=str, default='STRIDE', choices=['trivial', 'STRIDE'])

    eval_parser.add_argument('--checkpoint', type=str, default='/media/amax/disk4/STRIDE-main/results/STRIDE/2026-08-10_23-34/checkpoints/Model_best.pth', help='Model checkpoint')
    eval_parser.add_argument('--mode', type=str, required='trivial' in sys.argv, help='Mode for non-learned baselines', choices=['last', 'next', 'closest', 'linear_interpolation'])
    # Optional arguments to overwrite the data settings in the config file
    eval_parser.add_argument('--test-data.test-config', type=str, default='/media/amax/disk4/STRIDE-main/train/config_test.yaml', required=False, help='YAML configuration file, test-specific settings')
    eval_parser.add_argument('--test-data.data-dir', type=str, required=False, help='Root directory of the dataset')
    eval_parser.add_argument('--test-data.hdf5-file', type=str, required=False, help='HDF5 test dataset, path relative to <data-dir>')
    eval_parser.add_argument('--test-data.split', type=str, required=False, help='Data split')
    eval_parser.add_argument('--test-data.mode', type=str, required=False, help='Data mode')

    eval_parser.add_argument('--test-data.rescale', type=bool, default=False, help='rescale to [-1,1], for diffusion models only')
    eval_parser.add_argument('--inference_steps', type=int, default=1, help='(nums of inference steps, for diffusion models only')

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
    print_stats(stats, evaluator)
