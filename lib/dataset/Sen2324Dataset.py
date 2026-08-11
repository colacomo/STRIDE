"""
Author: Vivien Sainte Fare Garnot (github.com/VSainteuf)
Refactored for Performance & Stability
"""

import json
import os
import warnings
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata
from torchvision import transforms
from typing import Any, Dict, List, Optional, Tuple
from lib.datasets.Degra_utils import Degradation
import ast

# Suppress annoying TypedStorage warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.storage")

class Sen2324Dataset(tdata.Dataset):
    def __init__(
        self,
        root: str,
        split: str = 'train',
        channels: str = 'bgr-nir',
        filter_settings = None,
        crop_settings = None,
        pe_strategy: str = 'day-of-year',
        mask_kwargs = None,
        augment: bool = False,
        max_seq_length: Optional[int] = None,
        rescale: bool = True,
        ifTestClip: bool = False,
        ifCTHW: bool = False,
        norm=True,
        target="semantic",
        cache=False,
        mem16=False,
        reference_date="2018-09-01",
        class_mapping=None,
        mono_date=None,
        sats=["S2"],
        date_rescale=False,
    ):
        super(Sen2324Dataset, self).__init__()
        if filter_settings is None:
            filter_settings = {'type': 'cloud-free', 'min_length': 10, 'return_valid_obs_only': True, 'max_t_sampling': None}
        self.filter_settings = filter_settings
        self.max_seq_length = max_seq_length
        self.rescale = rescale
        self.channels = channels
        self.root = root
        self.split = split
        self.mask_path = os.path.join(root, "REAL_MASKS")
        self.norm = norm
        self.reference_date = datetime(*map(int, reference_date.split("-")))
        self.cache = cache
        self.mem16 = mem16
        self.mono_date = None
        
        if mono_date is not None:
             self.mono_date = (
                datetime(*map(int, mono_date.split("-")))
                if "-" in mono_date
                else int(mono_date)
            )
            
        self.memory = {}
        self.memory_dates = {}
        self.class_mapping = (
            np.vectorize(lambda x: class_mapping[x])
            if class_mapping is not None
            else class_mapping
        )
        self.target = target
        self.sats = sats
        self.date_rescale = date_rescale

        # Image size
        self.crop_settings = crop_settings
        # Fix: add robustness check for crop_settings type
        if hasattr(crop_settings, 'enabled') and crop_settings.enabled:
             self.image_size = crop_settings.shape 
        else:
             self.image_size = (128, 128)
             
        self.crop_size = 128

        self.variable_seq_length = False
        if filter_settings and filter_settings.get('type', None) is not None:
            self.variable_seq_length = filter_settings.get('return_valid_obs_only', False)

        print("Reading patch metadata . . .")
        self.meta_patch = gpd.read_file(os.path.join(root, "metadata.geojson"))
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(str)
        self.meta_patch.sort_index(inplace=True)
        
        self.date_tables = {s: None for s in sats}
        self.date_range = self._compute_date_range()

        # Optimize date table construction
        for s in sats:
            dates = self.meta_patch["dates-{}".format(s)]
            date_table = pd.DataFrame(
                index=self.meta_patch.index, columns=self.date_range, dtype=int
            )
            for pid, date_seq in dates.items():
                if isinstance(date_seq, str):
                    try:
                        date_seq = json.loads(date_seq)
                    except json.JSONDecodeError:
                        date_seq = ast.literal_eval(date_seq)
                
                # Fast date conversion
                d_list = []
                for x in date_seq.values():
                    str_x = str(x)
                    dt = datetime(int(str_x[:4]), int(str_x[4:6]), int(str_x[6:]))
                    d_list.append((dt - self.reference_date).days)
                
                if d_list:
                    date_table.loc[pid, d_list] = 1
            
            date_table = date_table.fillna(0)
            self.date_tables[s] = {
                index: np.array(list(d.values()))
                for index, d in date_table.to_dict(orient="index").items()
            }

        print("Done.")
        if split == "train":
            folds = [1, 2, 3, 4]
        elif split == "test":
            folds = [5]
        else:
            folds = [1, 2, 3, 4, 5] # default

        self.meta_patch = pd.concat(
            [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
        )

        if norm:
            self.norms = {}
            for s in self.sats:
                if s != "S2":
                    norm_file = os.path.join(root, "NORM_{}_patch.json".format(s))
                    if os.path.exists(norm_file):
                        with open(norm_file, "r") as file:
                            normvals = json.loads(file.read())
                        selected_folds = folds
                        means = [normvals["Fold_{}".format(f)]["mean"] for f in selected_folds]
                        stds = [normvals["Fold_{}".format(f)]["std"] for f in selected_folds]
                        
                        m = np.stack(means).mean(axis=0)
                        std = np.stack(stds).mean(axis=0)
                        
                        # Important fix: prevent NaN from zero std
                        std[std == 0] = 1e-6 
                        
                        self.norms[s] = (
                            torch.from_numpy(m).float(),
                            torch.from_numpy(std).float(),
                        )
                    else:
                        self.norms[s] = (torch.zeros(3), torch.ones(3))
        else:
            self.norms = None

        # Bad frames
        bad_frames_path = os.path.join(root, "bad_frames.json")
        if os.path.exists(bad_frames_path):
            with open(bad_frames_path, "r") as file:
                bad_frames = json.load(file)
            self.bad_frames = {k: v for k, v in bad_frames.items()}
        else:
            self.bad_frames = {}

        self.samples = []  # Store (pid, segment_idx)
        min_len = self.filter_settings.get('min_length', 10)
        self.min_length = min_len
        initial_ids = self.meta_patch.index

        print(f"Filtering dataset with min_length={min_len} and expanding multiples...")
        for pid in initial_ids:
            s2_dates = self.get_dates(pid, 'S2')
            bad_idx = self.bad_frames.get(pid, [])

            actual_len = len(s2_dates) - len(bad_idx)

            if actual_len >= min_len:
                if self.split == 'train':
                    num_segments = actual_len // min_len
                else:
                    num_segments = 1

                for i in range(num_segments):
                    self.samples.append({'id': pid, 'segment_idx': i})

        print(f"Original patches: {len(initial_ids)}, Expanded samples: {len(self.samples)}")
        self.len = len(self.samples)

        # Channels setup
        if 'bgr' == self.channels[:3]:
            self.num_channels = 3
            self.c_index_rgb = torch.Tensor([2, 1, 0]).long()
            self.s2_channels = [0, 1, 2]
        else:
            self.num_channels = 10
            self.c_index_rgb = torch.Tensor([2, 1, 0]).long()
            self.s2_channels = list(np.arange(10))

        if '-nir' in self.channels:
            self.num_channels += 1
            self.c_index_nir = torch.Tensor([3]).long()
            self.s2_channels += [6]
        elif 'all' in self.channels:
            self.c_index_nir = torch.Tensor([6]).long()
        else:
            self.c_index_nir = torch.from_numpy(np.array(np.nan))

        print("Dataset ready.")
        
    def _compute_date_range(self):
        return np.array(range(-200, 5000))  # Expand range slightly just in case

    def __len__(self):
        return self.len

    def get_dates(self, id_patch, sat):
        return self.date_range[np.where(self.date_tables[sat][id_patch] == 1)[0]]

    def _random_crop(self, data_dict, cloud_mask):
        """Randomly crop data to specified size."""
        T, C, H, W = data_dict["S2"].shape

        # Compute random crop start position
        if H > self.crop_size:
            h_start = torch.randint(0, H - self.crop_size, (1,)).item()
        else:
            h_start = 0

        if W > self.crop_size:
            w_start = torch.randint(0, W - self.crop_size, (1,)).item()
        else:
            w_start = 0

        # Crop all data
        for key in data_dict:
            if data_dict[key] is not None:
                data_dict[key] = data_dict[key][:, :, h_start:h_start + self.crop_size,
                                 w_start:w_start + self.crop_size]

        # Crop cloud mask
        cloud_mask = cloud_mask[:, :, h_start:h_start + self.crop_size, w_start:w_start + self.crop_size]

        return data_dict, cloud_mask

    def _center_crop(self, data_dict, cloud_mask):
        """Center crop data to specified size."""
        T, C, H, W = data_dict["S2"].shape

        # Compute center crop start position
        h_start = (H - self.crop_size) // 2
        w_start = (W - self.crop_size) // 2

        # Crop all data
        for key in data_dict:
            if data_dict[key] is not None:
                data_dict[key] = data_dict[key][:, :, h_start:h_start + self.crop_size,
                                 w_start:w_start + self.crop_size]

        # Crop cloud mask
        cloud_mask = cloud_mask[:, :, h_start:h_start + self.crop_size, w_start:w_start + self.crop_size]

        return data_dict, cloud_mask

    def build_s11(self, datas1):
        p2, p98 = np.percentile(datas1, (2, 98))
        out = np.clip(datas1, p2, p98)
        # Fix: prevent division by zero
        denominator = p98 - p2
        if denominator == 0: denominator = 1e-8
        out = (out - p2) / denominator
        return out

    def __getitem__(self, item):
        sample_info = self.samples[item]
        id_patch = sample_info['id']
        segment_idx = sample_info['segment_idx']

        # Load satellite data
        if not self.cache or item not in self.memory.keys():
            data = {}
            for satellite in self.sats:
                data_file = os.path.join(
                    self.root,
                    "DATA_{}".format(satellite),
                    "{}_{}.npy".format(satellite, id_patch),
                )
                if os.path.exists(data_file):
                    data[satellite] = np.load(data_file).astype(np.float32)
                    if satellite == 'S1A':
                        data[satellite] = self.build_s11(data[satellite])
                else:
                    data[satellite] = None

            data = {s: torch.from_numpy(a) if a is not None else None for s, a in data.items()}

            #print('data', data["S2"].shape, data["S1A"].shape)

            # Data normalization
            if self.norm:
                for s, d in data.items():
                    if d is not None:
                        if s == "S2":
                            # Sentinel-2 data normalization
                            #print("data", torch.max(d), torch.min(d))
                            data[s] = torch.clamp(d, 0, 8000) / 8000
                            #print("data", torch.max(data[s]), torch.min(data[s]))
                        else:
                            # SAR data standardization
                            #mid = (d - self.norms[s][0][None, :, None, None]) / self.norms[s][1][None, :, None, None]
                            data[s] = d#torch.clamp(mid, -2, 2) / 2
                            #print(data[s].shape, torch.max(data[s]), torch.min(data[s]))
                            if self.rescale:
                               mean = self.norms[s][0].view(1, -1, 1, 1)
                               std = self.norms[s][1].view(1, -1, 1, 1)
                               mid = (d - mean) / (std + 1e-8)
                               data[s] = torch.clamp(mid, -2, 2) / 2
                            else:
                               data[s] = self.build_s11(data[s])

            # Load cloud mask
            cloud_mask_file = os.path.join(self.mask_path, "{}.npy".format(id_patch))
            if os.path.exists(cloud_mask_file):
                cloud_mask = np.load(cloud_mask_file)
                cloud_mask = torch.from_numpy(cloud_mask).float()
                #print("cloud_mask",torch.max(cloud_mask))
            else:
                # If no cloud mask, create all-clear mask
                T, C, H, W = data["S2"].shape
                cloud_mask = torch.zeros((T, 1, H, W))

            # Cache data
            if self.cache:
                if self.mem16:
                    self.memory[item] = [{k: v.half() if v is not None else None for k, v in data.items()},
                                         cloud_mask.half()]
                else:
                    self.memory[item] = [data, cloud_mask]

        else:
            # Load from cache
            data, cloud_mask = self.memory[item]
            if self.mem16:
                data = {k: v.float() if v is not None else None for k, v in data.items()}
                cloud_mask = cloud_mask.float()

        # Get date sequences
        if not self.cache or id_patch not in self.memory_dates.keys():
            dates = {}
            for s in self.sats:
                if data[s] is not None:
                    dates[s] = torch.from_numpy(self.get_dates(id_patch, s))
                else:
                    dates[s] = None
            if self.cache:
                self.memory_dates[id_patch] = dates
        else:
            dates = self.memory_dates[id_patch]

        #print("dates", dates["S2"], dates["S1A"])

        # Crop data to specified size
        original_shape = data["S2"].shape
        if original_shape[2] > self.crop_size or original_shape[3] > self.crop_size:
            if self.split=="train":
                data, cloud_mask = self._random_crop(data, cloud_mask)
            else:
                data, cloud_mask = self._center_crop(data, cloud_mask)

        # ==========================================================
        # Stage 3: Time subsampling & dates
        # ==========================================================
        # Get dates
        if not self.cache or id_patch not in self.memory_dates.keys():
            dates = {s: torch.from_numpy(self.get_dates(id_patch, s)) for s in self.sats}
            if self.cache: self.memory_dates[id_patch] = dates
        else:
            dates = self.memory_dates[id_patch]

        bad_idx = self.bad_frames.get(id_patch, [])

        # Pass segment_idx for splitting
        #t_sampled = self._subsample_sequence(data["S2"], bad_idx, segment_idx)
        t_sampled = self._subsample_sequence(data["S2"], bad_idx, segment_idx)
        
        # Guard against empty sequences
        if len(t_sampled) == 0:
            t_sampled = torch.arange(min(len(data["S2"]), self.max_seq_length or 10))

        if 'S2' in self.sats:
            data["S2"] = data["S2"][t_sampled] # [T_sub, C, H, W]
            data["S2"] = data["S2"][:, self.s2_channels]
            dates["S2"] = dates["S2"][t_sampled]
            
            if self.rescale:
                trans_scale = transforms.Normalize([0.5], [0.5])
                data['S2'] = trans_scale(data['S2'])

        # Match SAR
        if 'S1D' in self.sats:
            dates["S1D"], t_sampled_SAR_D = self.match_sequences_vectorized(dates['S2'], dates['S1D'])
            data["S1D"] = data["S1D"][t_sampled_SAR_D]
        else:
            dates["S1D"], data["S1D"] = None, None

        if 'S1A' in self.sats:
            dates["S1A"], t_sampled_SAR_A = self.match_sequences_vectorized(dates['S2'], dates['S1A'])
            data["S1A"] = data["S1A"][t_sampled_SAR_A]
        else:
            dates["S1A"], data["S1A"] = None, None

        cond = data["S1D"] if data["S1D"] is not None else data["S1A"]
        position_days_cond = dates["S1D"] if dates["S1D"] is not None else dates["S1A"]

        # Randomly select mask frames (Vectorized)
        if self.split == 'train':
            masks = self.select_random_frames_vectorized(cloud_mask, len(t_sampled))
        else:
            masks = self.select_fixed_frames(cloud_mask, len(t_sampled))

        masks = masks * 0.5
        masks = torch.where(masks > 0.5, torch.tensor(1.0), masks)
        masks = torch.where(masks <= 0.5, torch.tensor(0.0), masks)

        cloud_mask_sup = cloud_mask[t_sampled].clone()
        cloud_mask_ins = torch.where(cloud_mask_sup > 0.5, torch.tensor(1.0), cloud_mask_sup)
        cloud_mask_sup = torch.where(cloud_mask_sup > 0, torch.tensor(1.0), cloud_mask_sup)
        
        masks = cloud_mask_ins + masks
        masks = torch.where(masks > 1, torch.tensor(1.0), masks)
        # ==========================================================
        # Stage 5: Degradation & Output
        # ==========================================================
        frames_input = data["S2"].clone()
        T_curr = frames_input.shape[0]
        
        degrader = Degradation()
        if self.split == 'test':
            import hashlib
            # Generate a unique hash value as random seed based on patch_id and segment_idx
            seed_str = f"{id_patch}_{segment_idx}"
            seed = int(hashlib.md5(seed_str.encode('utf-8')).hexdigest()[:8], 16)
            frames_input, md_record = degrader.add_random_degradation_to_sequence(frames_input, T_curr, seed=seed)
        else:
            frames_input, md_record = degrader.add_random_degradation_to_sequence(frames_input, T_curr)
        
        flag = (cloud_mask_ins == 1).expand_as(frames_input)
        frames_input[flag] = 1
        flag = (masks == 1).expand_as(frames_input)
        frames_input[flag] = 1 

        days = dates['S2'] - dates['S2'][0]
        if self.date_rescale:
            dates['S2'] = ((dates['S2'] / 10).round() * 10).int()
        #print("in",frames_input.shape,torch.max(frames_input),torch.max(data["S2"]),torch.max(cond),torch.min(cond))
        return {
            'x': frames_input,
            'cond': cond,
            'y': data["S2"],
            'masks': masks,
            'position_days': dates['S2'],
            'position_days_cond': position_days_cond,
            'days': days,
            'sample_index': id_patch,
            'c_index_rgb': self.c_index_rgb,
            'c_index_nir': self.c_index_nir,
            'cloud_mask': cloud_mask_sup,
            'md_rec': md_record,
        }

    def _subsample_sequence(self, sample: torch.Tensor, bad_index: list, segment_idx: int = 0) -> torch.Tensor:
        """
        Subsample the sequence. Supports deterministic segmentation based on min_length.
        """
        seq_length = sample.shape[0]

        # 1. Find all valid frame indices
        if self.filter_settings.get('type') == 'cloud-free':
            masks_valid_obs = torch.ones(seq_length, dtype=torch.bool)
            if bad_index:
                masks_valid_obs[bad_index] = False
        else:
            masks_valid_obs = torch.ones(seq_length, dtype=torch.bool)

        if self.filter_settings.get('return_valid_obs_only', True):
            all_valid_indices = torch.nonzero(masks_valid_obs).squeeze()
            if all_valid_indices.ndim == 0 and all_valid_indices.numel() == 1:
                all_valid_indices = all_valid_indices.unsqueeze(0)
        else:
            all_valid_indices = torch.arange(seq_length)

        if all_valid_indices.numel() == 0:
            return torch.tensor([], dtype=torch.long)

        # ==========================================================
        # Slice by segment_idx
        # ==========================================================
        # Compute segment start/end: step size is min_length
        if self.split == 'train':
            step = self.min_length
            start_idx = segment_idx * step
            start_idx = max(0, len(all_valid_indices) - step)
            end_idx = start_idx + self.max_seq_length

            t_sampled = all_valid_indices[start_idx:end_idx]
            #print(len(t_sampled), len(all_valid_indices), self.max_seq_length)

        else:
            t_sampled = all_valid_indices
            #print(len(t_sampled), len(all_valid_indices),self.max_seq_length)

        # If sliced length exceeds max_seq_length, do a random crop
        if self.max_seq_length is not None and len(t_sampled) > self.max_seq_length:
            # --- Fix: disable np.random in test mode, use deterministic offset ---
            if self.split == 'train':
                t_start_offset = np.random.randint(0, len(t_sampled) - self.max_seq_length + 1)
            else:
                t_start_offset = 0  # Test mode: always start at offset 0 for determinism
            # --- End fix ---

            t_sampled = t_sampled[t_start_offset: t_start_offset + self.max_seq_length]

        return t_sampled

    def select_random_frames_vectorized(self, mask, num_frames):
        T = mask.shape[0]
        if T == 0: return mask # Edge case

        # mask shape: [T, 1, H, W]
        # Compute cloud-free ratio per frame (Vectorized)
        # 0 is clear, >0 is cloud
        # sum over (C, H, W) -> dim (1, 2, 3)
        cloud_pixels = (mask > 0).sum(dim=(1, 2, 3))
        total_pixels = mask.shape[2] * mask.shape[3]
        cloud_free_ratios = 1.0 - (cloud_pixels.float() / total_pixels)

        # Get indices
        clear_indices = torch.nonzero(cloud_free_ratios >= 0.95).squeeze()
        # Handle scalar output from nonzero if only 1 match
        if clear_indices.ndim == 0 and clear_indices.numel() == 1:
            clear_indices = clear_indices.unsqueeze(0)

        if clear_indices.numel() == 0:
            # No clear frames, select the best ones
            _, sorted_idx = torch.sort(cloud_free_ratios, descending=True)
            clear_indices = sorted_idx

        clear_indices = clear_indices.tolist()

        # Require at least 60% clear frames
        min_clear = max(6, int(0.6 * num_frames))

        selected_indices = []

        # 1. Select clear frames
        if len(clear_indices) >= min_clear:
            selected_indices.extend(np.random.choice(clear_indices, min_clear, replace=False))
        else:
            selected_indices.extend(clear_indices)  # Take all available
            # Fill the remaining clear quota (no more clear frames available)
            pass

        # 2. Fill remaining frames
        remaining_needed = num_frames - len(selected_indices)
        if remaining_needed > 0:
            all_indices = set(range(T))
            used_indices = set(selected_indices)
            avail_indices = list(all_indices - used_indices)

            if len(avail_indices) > 0:
                if len(avail_indices) >= remaining_needed:
                    selected_indices.extend(np.random.choice(avail_indices, remaining_needed, replace=False))
                else:
                    # Not enough, allow repeated sampling
                    selected_indices.extend(np.random.choice(avail_indices, remaining_needed, replace=True))
            else:
                # Very rare: total frames < num_frames, must resample from already selected
                selected_indices.extend(np.random.choice(list(used_indices), remaining_needed, replace=True))

        # Truncate and shuffle
        selected_indices = np.array(selected_indices[:num_frames])
        np.random.shuffle(selected_indices)

        return mask[selected_indices]

    def select_fixed_frames(self, mask, num_frames):
        """
        In test/val mode, ensure at least min_clear clear frames are selected.
        Args:
            mask: [T, 1, H, W] tensor (already subsampled)
            num_frames: number of frames to return (usually equals len(mask))
        """
        T = mask.shape[0]
        if T == 0: return mask

        # 1. Compute min_clear threshold
        min_clear = max(6, int(0.6 * num_frames))

        # 2. Compute per-frame clarity
        cloud_pixels = (mask > 0).sum(dim=(1, 2, 3))
        total_pixels = mask.shape[2] * mask.shape[3]
        cloud_free_ratios = 1.0 - (cloud_pixels.float() / total_pixels)

        # 3. Find clear frames (threshold >= 0.95)
        is_clear = cloud_free_ratios >= 0.95
        clear_indices = torch.nonzero(is_clear).squeeze()

        # Handle scalar/empty results
        if clear_indices.ndim == 0 and clear_indices.numel() == 1:
            clear_indices = clear_indices.unsqueeze(0)

        if clear_indices.numel() == 0:
            # If no frame meets 0.95 threshold, take the clearest ones
            _, sorted_idx = torch.sort(cloud_free_ratios, descending=True)
            # Even if quality is low, treat the best as "clear" candidates
            clear_indices = sorted_idx

        clear_indices = clear_indices.tolist()

        selected_indices = []

        # 4. Fill clear frame quota
        if len(clear_indices) >= min_clear:
            # Enough unique clear frames: take the first min_clear for determinism
            selected_indices.extend(clear_indices[:min_clear])
        else:
            # Not enough clear frames, must repeat them
            # e.g. need 4, only have 2 [A, B] -> [A, B, A, B]
            while len(selected_indices) < min_clear:
                for idx in clear_indices:
                    if len(selected_indices) < min_clear:
                        selected_indices.append(idx)
                    else:
                        break

        # 5. Fill remaining frames (cloudy frames allowed)
        remaining = num_frames - len(selected_indices)

        if remaining > 0:
            all_indices = list(range(T))
            # Prefer frames not yet selected
            used_set = set(selected_indices)
            candidates = [i for i in all_indices if i not in used_set]

            if len(candidates) >= remaining:
                selected_indices.extend(candidates[:remaining])
            else:
                # Not enough candidates (num_frames > T), add all candidates first
                selected_indices.extend(candidates)
                # Fill the rest by cycling from the full set
                still_needed = num_frames - len(selected_indices)
                for i in range(still_needed):
                    selected_indices.append(all_indices[i % T])

        # 6. Sort indices to maintain temporal order
        selected_indices = sorted(selected_indices)

        return mask[torch.tensor(selected_indices)]
    """def select_fixed_frames(self, mask, num_frames):
        # Simple truncation or padding
        if mask.shape[0] >= num_frames:
            return mask[:num_frames]
        else:
            # Pad by repeating
            indices = torch.cat([torch.arange(mask.shape[0]), torch.randint(0, mask.shape[0], (num_frames - mask.shape[0],))])
            return mask[indices]"""

    def match_sequences_vectorized(self, A, B):
        """
        Fully vectorized sequence matching, replacing match_sequences_tensor.
        Find the closest value in B for every element in A.
        A: [M] (S2 dates)
        B: [N] (SAR dates)
        Returns:
            C: [M] (Matched values from B)
            indices: [M] (Indices in B)
        """
        if B is None or len(B) == 0:
            return None, None

        # Broadcast to compute absolute difference matrix [M, N]
        # diff[i, j] = |A[i] - B[j]|
        diff = torch.abs(A.unsqueeze(1) - B.unsqueeze(0))

        # Find index of minimum per row [M]
        min_indices = torch.argmin(diff, dim=1)

        # Gather matched values
        C = B[min_indices]

        return C, min_indices
