import math
import matplotlib
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from enum import Enum
from matplotlib import pyplot as plt
from torch import Tensor, nn
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from lib import config_utils, data_utils, utils, visutils
from lib.models import MODELS
from lib.visutils import COLORMAPS
from diffusers.schedulers import DDIMScheduler, DPMSolverMultistepScheduler





class Method(Enum):
    STRIDE = 'STRIDE'


class Mode(Enum):
    LAST = 'last'
    NEXT = 'next'
    CLOSEST = 'closest'
    LINEAR_INTERPOLATION = 'linear_interpolation'
    NONE = None


class Imputation:
    def __init__(
            self,
            config_file_train,  #: str | None,
            method: Literal['trivial', 'STRIDE'] = 'STRIDE',
            mode: Literal['last', 'next', 'closest', 'linear_interpolation'] = None,
            # Literal['last', 'next', 'closest', 'linear_interpolation'] | None = None,
            checkpoint: str = None,  # str | None = None,
            multigpus: True | False = False,
            num_inference_steps: int = 1,
            ifDate: bool = False,
            ifCond: bool = False,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            visualize: bool = False,  # Whether to visualize
            vis_dir: str = None,  # Visualization output directory
            vis_freq: int = 1,  # Visualization frequency
    ):

        self.method = Method(method)
        self.method_name = method
        self.mode = Mode(mode)
        self.checkpoint = checkpoint
        self.config_file_train = config_file_train
        self.multigpus = multigpus
        self.num_inference_steps = num_inference_steps
        self.ifDate = ifDate
        self.ifCond = ifCond
        self.visualize = visualize  # Visualization switch
        self.vis_dir = vis_dir  # Visualization output directory
        self.vis_freq = vis_freq  # Visualization frequency
        self.vis_count = 0  # Visualization counter

        # Create visualization directory
        if self.visualize and self.vis_dir:
            os.makedirs(self.vis_dir, exist_ok=True)
            self.vis_count = 0  # Visualization counter

        if self.method == Method.STRIDE:
            if self.checkpoint is None:
                raise ValueError('No checkpoint specified.\n')

            if self.config_file_train is None:
                raise ValueError('No training configuration file specified.\n')

            if not os.path.isfile(self.config_file_train):
                raise FileNotFoundError(
                    f'Cannot find the configuration file used during training: {self.config_file_train}\n')

            if not os.path.isfile(self.checkpoint):
                raise FileNotFoundError(f'Cannot find the model weights: {self.checkpoint}\n')

            # Read the configuration file used during training
            self.config = config_utils.read_config(self.config_file_train)

            # Extract the temporal window size and the number of channels used during training
            self.temporal_window = self.config.data.max_seq_length
            self.num_channels = data_utils.get_dataset(self.config, phase=self.config.misc.run_mode).num_channels

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        _ = torch.set_grad_enabled(False)

        # Get the model
        if self.method == Method.STRIDE:
            self.model, _ = utils.get_model(self.config, self.num_channels)
            self._resume()
            self.model.to(self.device).eval()

    def impute_sample(
            self,
            batch: Dict[str, Any],
            t_start: Optional[int] = None,
            t_end: Optional[int] = None,
            return_att: Optional[bool] = False,
            vis_prefix: str = "sample"  # Visualization prefix
    ):  # -> Tuple[Dict[str, Any], Tensor, Tensor] | Tuple[Dict[str, Any], Tensor]:
        if t_start is not None and t_end is not None:
            # Choose a subsequence
            batch['x'] = batch['x'][:, t_start:t_end, ...]

            for key in ['y', 'masks', 'cloud_mask', 'masks_valid_obs']:
                if key in batch:
                    batch[key] = batch[key][:, t_start:t_end, ...]

            for key in ['days', 'position_days']:
                if key in batch:
                    batch[key] = batch[key][:, t_start:t_end]

        # Impute the given satellite image time series
        # implement impute_sequence for STRIDE model
        batch = data_utils.to_device(batch, self.device)

        y_pred = impute_sequence_STRIDE(self.model, batch, self.temporal_window, self.num_inference_steps, ifDate=self.ifDate, ifCond=self.ifCond)
        batch = data_utils.to_device(batch, 'cpu')
        y_pred = y_pred.cpu()

        # Visualize output results

        if self.visualize and self.vis_count % self.vis_freq == 0:
            self._visualize_outputs(batch, y_pred, vis_prefix)

        self.vis_count += 1  # Increment counter

        return batch, y_pred

    def _resume(self) -> None:
        checkpoint = torch.load(self.checkpoint)
        if self.multigpus:
            self.model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f'Checkpoint \'{self.checkpoint}\' loaded.')
        print(f"Chosen epoch: {checkpoint['epoch']}\n")
        del checkpoint

    @staticmethod
    def linear_2pct_stretch(arr, nodata=None):
        """
        Perform a 2%-98% linear stretch on the input array and output an 8-bit result.
        Special case: if all valid values are 1, return all 255 directly.
        """
        valid = arr if nodata is None else arr[arr != nodata]
        if valid.size == 0:  # All nodata
            return np.full(arr.shape, 255, dtype=np.uint8)
        if np.all(valid == 1):  # All values are 1
            return np.full(arr.shape, 255, dtype=np.uint8)

        p2, p98 = np.percentile(valid, (2, 98))
        out = np.clip(arr, p2, p98)
        out = (out - p2) / (p98 - p2 + 1e-8) * 255
        return out.astype(np.uint8)

    def _visualize_outputs(self, batch: Dict[str, Any], y_pred: Tensor, prefix: str):
        """Visualize output results."""
        if not self.visualize or not self.vis_dir:
            return

        # Create subdirectory
        output_dir = Path(self.vis_dir) / f"{self.method_name}" / f"{prefix}_{self.vis_count}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_year_dir = Path(self.vis_dir) / f"{self.method_name}"

        # Extract data
        x = batch['x'].cpu().numpy()  # Input image
        mask = batch['masks'].cpu().numpy()  # Mask
        y_pred_np = y_pred.cpu().numpy()  # Prediction result
        sar = batch['cond'].cpu().numpy()

        # Target data (if available)
        y = None
        if 'y' in batch and batch['y'] is not None:
            y = batch['y'].cpu().numpy()

        B, T = x.shape[:2]
        rgb_idx = [0, 1, 2]  # RGB channel indices

        # Save evaluation info
        with open(output_dir / "evaluation_info.txt", "w") as f:
            f.write(f"Prediction shape: {y_pred_np.shape}\n")
            f.write(f"Prediction range: [{y_pred_np.min():.4f}, {y_pred_np.max():.4f}]\n")
            if y is not None:
                mse = np.mean((y_pred_np - y) ** 2)
                f.write(f"MSE: {mse:.6f}\n")

        # Visualize each sample and timestep
        for b in range(min(B, 2)):  # Only visualize first 2 samples
            for t in range(T):  # Visualize timestep by timestep
                # Visualize prediction results
                if y_pred_np.shape[2] >= 3:
                    pred_rgb = y_pred_np[b, t, rgb_idx].copy()
                    for c in range(3):
                        pred_rgb[c] = self.linear_2pct_stretch(pred_rgb[c])
                    pred_rgb = pred_rgb.transpose(1, 2, 0)
                    pred_rgb = np.clip(pred_rgb, 0, 255).astype(np.uint8)
                    Image.fromarray(pred_rgb).save(output_dir / f'pred_b{b}_t{t}.png')

                    # Create comparison: Original vs Input vs Prediction
                    if x.shape[2] >= 3 and y_pred_np.shape[2] >= 3:
                        images_to_concat = []

                        # 1. Original image (if available, typically a clean frame with ground truth reference)
                        if 'ori' in batch and batch['ori'] is not None:
                            ori_rgb = batch['ori'].cpu().numpy()[b, t, rgb_idx].copy()
                            for c in range(3):
                                ori_rgb[c] = self.linear_2pct_stretch(ori_rgb[c])
                            ori_rgb = ori_rgb.transpose(1, 2, 0)
                            ori_rgb_save = np.clip(ori_rgb, 0, 255).astype(np.uint8)
                            Image.fromarray(ori_rgb_save).save(output_dir / f'ori_t{t}.png')
                            images_to_concat.append(np.clip(ori_rgb, 0, 255).astype(np.uint8))

                        # 2. Input image (input frame after masking)
                        x_rgb = x[b, t, rgb_idx].copy()
                        for c in range(3):
                            #print(x_rgb[c].shape)
                            x_rgb[c] = self.linear_2pct_stretch(x_rgb[c])
                        x_rgb = x_rgb.transpose(1, 2, 0)
                        x_rgb_save = np.clip(x_rgb, 0, 255).astype(np.uint8)
                        Image.fromarray(x_rgb_save).save(output_dir / f'input_t{t}.png')
                        images_to_concat.append(np.clip(x_rgb, 0, 255).astype(np.uint8))
                        sar_rgb = sar[b, t, rgb_idx].copy()
                        for c in range(3):
                            #print(sar_rgb[c].shape)
                            sar_rgb[c] = self.linear_2pct_stretch(sar_rgb[c])
                        sar_rgb = sar_rgb.transpose(1, 2, 0)
                        sar_rgb_save = np.clip(sar_rgb, 0, 255).astype(np.uint8)
                        Image.fromarray(sar_rgb_save).save(output_dir / f'sar_t{t}.png')

                        # 3. Predicted image (model reconstructed frame)
                        pred_rgb = y_pred_np[b, t, rgb_idx].copy()
                        for c in range(3):
                            pred_rgb[c] = self.linear_2pct_stretch(pred_rgb[c])
                        pred_rgb = pred_rgb.transpose(1, 2, 0)
                        images_to_concat.append(np.clip(pred_rgb, 0, 255).astype(np.uint8))

                        # Horizontal concatenation
                        comparison = np.concatenate(images_to_concat, axis=1)

                        # Add labels
                        labels_list = ['Original', 'Masked Input', 'Prediction'] if 'ori' in batch else ['Masked Input',
                                                                                                         'Prediction']
                        comparison_with_labels = self._add_labels_to_comparison(comparison, labels_list)
                        Image.fromarray(comparison_with_labels).save(output_dir / f'comparison_b{b}_t{t}.png')

                # Save timestep evaluation info
                with open(output_dir / f"timestep_eval_b{b}_t{t}.txt", "w") as f:
                    f.write(f"Sample {b}, Time step {t}\n")
                    if y is not None:
                        mse_t = np.mean((y_pred_np[b, t] - y[b, t]) ** 2)
                        f.write(f"MSE: {mse_t:.6f}\n")
                    f.write(f"Prediction range: [{y_pred_np[b, t].min():.4f}, {y_pred_np[b, t].max():.4f}]\n")

        # Save original images and prediction results as npy files
        npy_dir = output_dir / "npy"
        npy_dir.mkdir(parents=True, exist_ok=True)

        np.save(npy_dir / "ori.npy", x)#.cpu().numpy())
        np.save(npy_dir / "pred.npy", y_pred_np)
        np.save(npy_dir / "sar.npy", sar)

        print(f"Visualization output saved to: {output_dir}")

    def _add_labels_to_comparison(self, image: np.ndarray, labels: List[str]) -> np.ndarray:
        """Add labels to comparison image."""
        h, w = image.shape[:2]
        label_h = 30  # Label height

        # Create labeled image
        labeled_image = np.ones((h + label_h, w, 3), dtype=np.uint8) * 255
        labeled_image[:h, :, :] = image

        # Calculate width of each section
        part_w = w // len(labels)

        # Add labels
        for i, label in enumerate(labels):
            from PIL import Image, ImageDraw, ImageFont
            pil_img = Image.fromarray(labeled_image)
            draw = ImageDraw.Draw(pil_img)

            # Use default font
            try:
                font = ImageFont.truetype("arial.ttf", 20)
            except:
                font = ImageFont.load_default()

            # Calculate text position
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            x = i * part_w + (part_w - text_w) // 2
            y = h + (label_h - text_h) // 2

            # Draw text
            draw.text((x, y), label, fill=(0, 0, 0), font=font)
            labeled_image = np.array(pil_img)

        return labeled_image

    def create_summary_report(self, batch: Dict[str, Any], y_pred: Tensor, prefix: str = "summary"):
        """Create reconstruction result summary report."""
        if not self.visualize or not self.vis_dir:
            return

        summary_dir = Path(self.vis_dir) / f"{prefix}_report"
        summary_dir.mkdir(parents=True, exist_ok=True)

        # Extract data
        x = batch['x'].cpu().numpy()
        mask = batch['masks'].cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        y = batch['y'].cpu().numpy() if 'y' in batch and batch['y'] is not None else None

        B, T = x.shape[:2]
        rgb_idx = [0, 1, 2]

        # Create HTML report
        html_content = self._generate_html_report(x, y, y_pred_np, mask, B, T, prefix)

        with open(summary_dir / "report.html", "w") as f:
            f.write(html_content)

        print(f"Summary report saved to: {summary_dir / 'report.html'}")

    def _generate_html_report(self, x, y, y_pred, mask, B, T, prefix):
        """Generate HTML format summary report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reconstruction Results Summary - {prefix}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .stats {{ margin: 20px 0; }}
                .comparison {{ margin: 10px 0; }}
                img {{ max-width: 300px; margin: 5px; border: 1px solid #ddd; }}
                .row {{ display: flex; flex-wrap: wrap; }}
                .column {{ flex: 1; min-width: 300px; margin: 10px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Reconstruction Results Summary - {prefix}</h1>
                <p>Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="stats">
                <h2>Statistics</h2>
                <p>Batch size: {B}</p>
                <p>Sequence length: {T}</p>
                <p>Spatial dimensions: {x.shape[3]} x {x.shape[4]}</p>
                <p>Number of channels: {x.shape[2]}</p>
        """

        if y is not None:
            mse = np.mean((y_pred - y) ** 2)
            html += f"<p>Overall MSE: {mse:.6f}</p>"
            html += f"<p>Average mask coverage: {mask.mean():.4f}</p>"

        html += """
            </div>

            <div class="comparison">
                <h2>Sample Comparison</h2>
        """

        # Add comparison for each sample
        for b in range(min(B, 3)):
            html += f"<h3>Sample {b}</h3>"
            html += '<div class="row">'

            for t in range(min(T, 5)):
                comparison_path = f"../{prefix}_{self.vis_count}/outputs/comparison_b{b}_t{t}.png"
                if os.path.exists(
                        Path(self.vis_dir) / f"{prefix}_{self.vis_count}" / "outputs" / f"comparison_b{b}_t{t}.png"):
                    html += f'<div class="column">'
                    html += f'<h4>Timestep {t}</h4>'
                    html += f'<img src="{comparison_path}" alt="Comparison b{b} t{t}">'

                    # Add evaluation info
                    eval_file = Path(
                        self.vis_dir) / f"{prefix}_{self.vis_count}" / "outputs" / f"timestep_eval_b{b}_t{t}.txt"
                    if eval_file.exists():
                        with open(eval_file, 'r') as f:
                            eval_info = f.read()
                        html += f'<pre>{eval_info}</pre>'

                    html += '</div>'

            html += '</div>'

        html += """
            </div>
        </body>
        </html>
        """

        return html
        
def evaluate_FM(model, x_obs, mask, date, cond):
    """
    Plug-and-Play Conditional OT Flow Matching
    Image-based inference using the toy_eval logic.

    Pipeline:
    1) Data consistency gradient step
    2) OT interpolation noise step
    3) Neural flow denoising step
    """

    model.eval()
    device = x_obs.device
    bs = x_obs.shape[0]

    # =====================
    # Initialization
    # =====================
    x_0 = x_obs                     # Observation / GT (non-cloud areas are reliable)
    #print("input",x_0.shape,date.shape,cond.shape,mask.shape)
    pred_hat = model(x_0, date=date, cond=cond, cloud_mask=mask)

    # =====================
    # Output
    # =====================
    #final_image = x * mask + x1 * (1 - mask)
    return torch.clamp(pred_hat, 0, 1)


def impute_sequence_STRIDE(
        model, batch: Dict[str, Any], temporal_window: int, num_inference_steps, ifDate: bool = False,
        ifCond: bool = False,
):  # -> Tensor | Tuple[Tensor, Tensor]:
    
    #Sliding-window imputation of satellite image time series using STRIDE.

    #Assumption: `batch` consists of a single sample.
    
    generator = torch.manual_seed(0)
    # for PASTIS dataset
    if ifCond:
        x = batch['x']
        cond = batch['cond']
    else:
        x = batch['x']
        cond = None

    if ifDate:
        date = batch['position_days']
    else:
        date = None

    mask = batch['masks']
    seq_length = x.shape[1]
    y_pred: Tensor
    att: Tensor

    # Pad the sequence with zeros
    if seq_length < temporal_window:
        pad = torch.zeros(
            (x.shape[0], temporal_window - x.shape[1], x.shape[2], x.shape[3], x.shape[4]), device=x.device
        )
        x = torch.cat([x, pad], dim=1)  # x and mask need to do the padding
        mask = torch.cat([mask, pad[:, :, :1, :, :]], dim=1)
        if date is not None:
            pad = torch.zeros((date.shape[0], temporal_window - date.shape[1]), device=date.device)
            date = torch.cat([date, pad], dim=1)
        if cond is not None:
            pad = torch.zeros(
                (cond.shape[0], temporal_window - cond.shape[1], cond.shape[2], cond.shape[3], cond.shape[4]),
                device=cond.device
            )
            cond = torch.cat([cond, pad], dim=1)
        y_pred = evaluate_FM(model, x, mask, date, cond)
        y_pred = y_pred[:, :seq_length]

    elif seq_length == temporal_window:
        # Process the entire sequence in one go
        # y_pred = model(x, batch_positions=positions)
        #print("=pipeline")
        y_pred = evaluate_FM(model, x, mask, date, cond)
        #print("y_pred", y_pred.shape)

    else:
        t_start = 0
        t_end = temporal_window
        t_max = x.shape[1]
        cloud_coverage = torch.mean(batch['masks'], dim=(0, 2, 3, 4))
        reached_end = False

        while not reached_end:
            # y_pred_chunk = model(x[:, t_start:t_end], batch_positions=positions[:, t_start:t_end])
            if date is not None:
                y_pred_chunk = evaluate_FM(model, x[:, t_start:t_end], mask[:, t_start:t_end], date[:, t_start:t_end],
                                        cond[:, t_start:t_end])
            else:
                y_pred_chunk = evaluate_FM(model, x[:, t_start:t_end], mask[:, t_start:t_end], date=None,
                                        cond=cond[:, t_start:t_end])

            if t_start == 0:
                # Initialize the full-length output sequence
                B, T, _, H, W = x.shape
                C = y_pred_chunk.shape[2]
                y_pred = torch.zeros((B, T, C, H, W), device=x.device)

                y_pred[:, t_start:t_end] = y_pred_chunk

                # Move the temporal window
                t_start_old = t_start
                t_end_old = t_end
                t_start, t_end = move_temporal_window_next(t_start, t_max, temporal_window, cloud_coverage)
            else:
                # Find the indices of those frames that have been processed by both the previous and the current
                # temporal window
                t_candidates = torch.Tensor(
                    list(set(torch.arange(t_start_old, t_end_old).tolist()) & set(
                        torch.arange(t_start, t_end).tolist()))
                ).long().to(x.device)

                # Find the frame for which the difference between the previous and the current prediction is
                # the lowest:
                # use this frame to switch from the previous imputation results to the current imputation results
                error = torch.mean(
                    torch.abs(y_pred[:, t_candidates] - y_pred_chunk[:, t_candidates - t_start]),
                    dim=(0, 2, 3, 4)
                )
                t_switch = error.argmin().item() + t_start
                y_pred[:, t_switch:t_end] = y_pred_chunk[:, (t_switch - t_start)::]

                if t_end == t_max:
                    reached_end = True
                else:
                    # Move the temporal window
                    t_start_old = t_start
                    t_end_old = t_end
                    t_start, t_end = move_temporal_window_next(
                        t_start_old, t_max, temporal_window, cloud_coverage
                    )

    return y_pred


def impute_sequence_per_frame(
        model, batch: Dict[str, Any], temporal_window: int,
        num_neighbor: int = 1, ifDate: bool = False, ifCond: bool = False
) -> torch.Tensor:
    """
    Build subsequences frame-by-frame and predict:
    For each frame in the sequence, extract itself, temporally adjacent `num_neighbor` frames,
    and select the clearest non-bad_idx frames from the remaining sequence to form a subsequence
    of length temporal_window. Finally extract only the target frame's prediction and combine for output.
    """
    x = batch['x']
    mask = batch['masks']
    date = batch['position_days'] if ifDate else None
    cond = batch['cond'] if ifCond else None

    B, T, C, H, W = x.shape
    y_pred_full = torch.zeros_like(x)

    # Calculate cloud mask ratio (less cloud = smaller value), used for selecting distant clear frames beyond neighbors
    # mask shape: [B, T, 1, H, W]
    cloud_ratio = mask.mean(dim=(0, 2, 3, 4))

    # If bad_idx exists, force its cloud mask ratio to infinity to ensure it is never selected as context
    bad_idx = batch.get('bad_idx', [])
    if isinstance(bad_idx, list) and len(bad_idx) > 0:
        for b_idx in bad_idx:
            if b_idx < T:
                cloud_ratio[b_idx] = float('inf')

    for t in range(T):
        # 1. Select neighboring temporal frames of the target frame (num_neighbor frames before and after, handling boundaries)
        neighbors = []
        for step in range(1, num_neighbor + 1):
            if t - step >= 0: neighbors.append(t - step)
            if t + step < T: neighbors.append(t + step)

        # 2. Select other relatively clear frames to fill the sequence up to temporal_window length
        existing_indices = set([t] + neighbors)
        remaining_indices = [i for i in range(T) if i not in existing_indices]

        # Sort by cloud coverage (clearer frames first)
        remaining_indices.sort(key=lambda i: cloud_ratio[i].item())

        needed = temporal_window - len(existing_indices)
        if needed > 0:
            selected_others = remaining_indices[:needed]
        else:
            selected_others = []

        # 3. Combine final indices and ensure strict temporal order
        sub_indices = [t] + neighbors[:temporal_window-1] + selected_others
        sub_indices = sorted(list(set(sub_indices)))

        # Record the relative index of the target frame in the subsequence
        t_local = sub_indices.index(t)

        # 4. Extract subsequence data
        idx_tensor = torch.tensor(sub_indices, device=x.device)
        x_sub = x[:, idx_tensor]
        mask_sub = mask[:, idx_tensor]
        date_sub = date[:, idx_tensor] if date is not None else None
        cond_sub = cond[:, idx_tensor] if cond is not None else None

        # 5. Padding (for cases where the entire sequence length is less than temporal_window)
        seq_length = len(sub_indices)
        if seq_length < temporal_window:
            pad_len = temporal_window - seq_length
            pad_x = torch.zeros((B, pad_len, C, H, W), device=x.device)
            x_sub = torch.cat([x_sub, pad_x], dim=1)
            mask_sub = torch.cat([mask_sub, torch.zeros((B, pad_len, 1, H, W), device=mask.device)], dim=1)
            if date_sub is not None:
                date_sub = torch.cat([date_sub, torch.zeros((B, pad_len), device=date.device)], dim=1)
            if cond_sub is not None:
                pad_cond = torch.zeros((B, pad_len, cond_sub.shape[2], H, W), device=cond_sub.device)
                cond_sub = torch.cat([cond_sub, pad_cond], dim=1)

        # 6. Feed into model for evaluation
        y_pred_sub = evaluate_FM(model, x_sub, mask_sub, date_sub, cond_sub)

        # 7. Extract only the current target frame's prediction and assemble into the global container
        y_pred_full[:, t] = y_pred_sub[:, t_local]

    return y_pred_full

def move_temporal_window_end(t_max: int, temporal_window: int) -> Tuple[int, int]:
    """
    Moves the temporal window for evaluation such that the last frame of the temporal window coincides with the
    last frame of the image sequence.

    Args:
        t_max:              int, sequence length of the image sequence
        temporal_window:    int, length of the subsequence passed to U-TILISE for processing

    Returns:
        t_start:            int, frame index, start of the subsequence
        t_end:              int, frame index, end of the subsequence
    """

    t_start = t_max - temporal_window
    t_end = t_max

    return t_start, t_end


def move_temporal_window_next(
        t_start: int, t_max: int, temporal_window: int, cloud_coverage: Tensor
) -> Tuple[int, int]:
    """
    Moves the temporal window for evaluation by half of the temporal window size (= stride).
    If the first frame within the STRIDE temporal window is cloudy (cloud coverage above 10%), the temporal window is
    shifted by at most half the stride (backward or forward) such that the first frame is as least cloudy as
    possible.

    Args:
        t_start:            int, frame index, start of the subsequence for processing
        t_max:              int, frame index, t_max - 1 is the last frame of the subsequence for processing
        temporal_window:    int, length of the subsequence passed to U-TILISE for processing
        cloud_coverage:     torch.Tensor, (T,), cloud coverage [-] per frame

    Returns:
        t_start:            int, frame index, start of the subsequence
        t_end:              int, frame index, end of the subsequence
    """

    stride = temporal_window // 2
    t_start += stride

    if t_start + temporal_window > t_max:
        # Reduce the stride such that the end of the temporal window coincides with the end of the entire sequence
        t_start, t_end = move_temporal_window_end(t_max, temporal_window)
    else:
        # Check if the start of the next temporal window is mostly cloud-free
        if cloud_coverage[t_start] <= 0.1:
            # Keep the default stride and ensure that the temporal window does not exceed the sequence length
            t_end = t_start + temporal_window
            if t_end > t_max:
                t_start, t_end = move_temporal_window_end(t_max, temporal_window)
        else:
            # Find the least cloudy frame within [t_start + stride - dt, t_start + stride + dt]
            dt = math.ceil(stride / 2)
            left = max(0, t_start - dt)
            right = min(t_start + dt + 1, t_max)

            # Frame(s) with the lowest cloud coverage within [t_start + stride - dt, t_start + stride + dt]
            t_candidates = (cloud_coverage[left:right] == cloud_coverage[left:right].min()).nonzero(as_tuple=True)[
                               0] + left

            # Take the frame closest to the standard stride
            t_start = t_candidates[torch.abs(t_candidates - t_start).argmin()].item()

            # Ensure that the temporal window does not exceed the sequence length
            t_end = t_start + temporal_window
            if t_end > t_max:
                t_start, t_end = move_temporal_window_end(t_max, temporal_window)

    return t_start, t_end


# Add missing import
import time
