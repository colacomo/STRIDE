"""
Author: Vivien Sainte Fare Garnot (github.com/VSainteuf)
Refactored for Performance, Stability & Reproducibility (Hybrid Mutually Exclusive Split)
"""

import json
import os
import warnings
import random
import hashlib
import contextlib
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata
from torchvision import transforms
from typing import Any, Dict, List, Optional, Tuple
from lib.datasets.Degra_utils import Degradation

# Suppress annoying TypedStorage warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.storage")


@contextlib.contextmanager
def deterministic_rng(seed):
    """
    Context manager: fixes the seed on entry, restores global random state on exit.
    Ensures reproducibility for specific operations without polluting the external random stream.
    """
    st_np = np.random.get_state()
    st_py = random.getstate()
    st_tr = torch.get_rng_state()

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    try:
        yield
    finally:
        np.random.set_state(st_np)
        random.setstate(st_py)
        torch.set_rng_state(st_tr)


class Sen2324Datasetdj(tdata.Dataset):
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
        reference_date="2023-01-01",
        class_mapping=None,
        mono_date=None,
        sats=["S2"],
        date_rescale=False,
    ):
        super(Sen2324Datasetdj, self).__init__()
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

        # =======================================================
        # Hybrid site split logic (test_n: no mutual exclusion, test_s: mutual exclusion)
        # =======================================================
        def get_site_id(row):
            region = row.get("region", "Unknown")
            orig_loc = row.get("original_loc_id", "")
            loc_id = orig_loc.split("_")[0] if "_" in orig_loc else orig_loc
            return f"{region}_{loc_id}"

        self.meta_patch["site_id"] = self.meta_patch.apply(get_site_id, axis=1)
        unique_sites = sorted(self.meta_patch["site_id"].unique().tolist())

        # Ensure consistent random seed for each split
        rng = np.random.RandomState(42)
        rng.shuffle(unique_sites)

        # Assign 1/6 of sites as test_s (Fold 6), strictly mutually exclusive
        test_s_sites = set(unique_sites[::6])

        def assign_fold(row):
            if row["site_id"] in test_s_sites:
                return 6  # test_s: site-level mutual exclusion
            else:
                # Remaining sites for Fold 1-5, split by Patch (no site-level mutual exclusion)
                # Prefer existing Patch Fold from original dataset for reproducibility
                if "Fold" in row and pd.notnull(row["Fold"]):
                    return int(row["Fold"])
                else:
                    patch_hash = int(hashlib.md5(str(row.name).encode('utf-8')).hexdigest(), 16)
                    return (patch_hash % 5) + 1

        self.meta_patch["Assigned_Fold"] = self.meta_patch.apply(assign_fold, axis=1)

        # Assign fold based on split parameter
        if split == "train":
            folds = [1, 2, 3, 4]
        elif split == "test_n":
            folds = [5]
        elif split == "test_s":
            folds = [6]
        else:
            folds = [1, 2, 3, 4, 5, 6]  # default

        self.meta_patch = self.meta_patch[self.meta_patch["Assigned_Fold"].isin(folds)]
        print(f"[{split.upper()}] Split complete: selected {len(self.meta_patch)} sample patches from {len(set(self.meta_patch['site_id']))} independent sites.")
        # =======================================================

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
                        import ast
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

        # Load Normalization
        if norm:
            self.norms = {}
            for s in self.sats:
                if s != "S2":
                    norm_file = os.path.join(root, "NORM_{}_patch.json".format(s))
                    if os.path.exists(norm_file):
                        with open(norm_file, "r") as file:
                            normvals = json.loads(file.read())

                        # Use train set (Fold 1-4) statistics to avoid KeyError from test_s (Fold 6)
                        train_folds = [1, 2, 3, 4]
                        means = [normvals["Fold_{}".format(f)]["mean"] for f in train_folds if "Fold_{}".format(f) in normvals]
                        stds = [normvals["Fold_{}".format(f)]["std"] for f in train_folds if "Fold_{}".format(f) in normvals]

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

        # Filter out sequences that are too short
        self.samples = []
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
        return np.array(range(-200, 5000))

    def __len__(self):
        return self.len

    def get_dates(self, id_patch, sat):
        return self.date_range[np.where(self.date_tables[sat][id_patch] == 1)[0]]

    def _random_crop(self, data_dict, cloud_mask):
        T, C, H, W = data_dict["S2"].shape

        if H > self.crop_size:
            h_start = torch.randint(0, H - self.crop_size, (1,)).item()
        else:
            h_start = 0

        if W > self.crop_size:
            w_start = torch.randint(0, W - self.crop_size, (1,)).item()
        else:
            w_start = 0

        for key in data_dict:
            if data_dict[key] is not None:
                data_dict[key] = data_dict[key][:, :, h_start:h_start + self.crop_size,
                                 w_start:w_start + self.crop_size]

        cloud_mask = cloud_mask[:, :, h_start:h_start + self.crop_size, w_start:w_start + self.crop_size]
        return data_dict, cloud_mask

    def _center_crop(self, data_dict, cloud_mask):
        T, C, H, W = data_dict["S2"].shape

        h_start = (H - self.crop_size) // 2
        w_start = (W - self.crop_size) // 2

        for key in data_dict:
            if data_dict[key] is not None:
                data_dict[key] = data_dict[key][:, :, h_start:h_start + self.crop_size,
                                 w_start:w_start + self.crop_size]

        cloud_mask = cloud_mask[:, :, h_start:h_start + self.crop_size, w_start:w_start + self.crop_size]
        return data_dict, cloud_mask

    def build_s11(self, datas1):
        p2, p98 = np.percentile(datas1, (2, 98))
        out = np.clip(datas1, p2, p98)
        denominator = p98 - p2
        if denominator == 0: denominator = 1e-8
        out = (out - p2) / denominator
        return out

    def __getitem__(self, item):
        sample_info = self.samples[item]
        id_patch = sample_info['id']
        segment_idx = sample_info['segment_idx']

        # Unique traceable hash seed (ensures reproducible random ops on the same patch)
        seed_str = f"repro_{id_patch}_{segment_idx}"
        deterministic_seed = int(hashlib.md5(seed_str.encode('utf-8')).hexdigest(), 16) % (2**32)

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

            # Data normalization
            if self.norm:
                for s, d in data.items():
                    if d is not None:
                        if s == "S2":
                            data[s] = torch.clamp(d, 0, 8000) / 8000
                        else:
                            data[s] = d
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
            else:
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
            data, cloud_mask = self.memory[item]
            if self.mem16:
                data = {k: v.float() if v is not None else None for k, v in data.items()}
                cloud_mask = cloud_mask.float()

        # Get date sequence
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

        # Crop operation
        original_shape = data["S2"].shape
        if original_shape[2] > self.crop_size or original_shape[3] > self.crop_size:
            if self.split=="train":
                data, cloud_mask = self._random_crop(data, cloud_mask)
            else:
                data, cloud_mask = self._center_crop(data, cloud_mask)

        bad_idx = self.bad_frames.get(id_patch, [])

        with deterministic_rng(deterministic_seed):
            t_sampled = self._subsample_sequence(data["S2"], bad_idx, segment_idx)

            if len(t_sampled) == 0:
                t_sampled = torch.arange(min(len(data["S2"]), self.max_seq_length or 10))

        if 'S2' in self.sats:
            data["S2"] = data["S2"][t_sampled]
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

        with deterministic_rng(deterministic_seed):
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

        frames_input = data["S2"].clone()
        T_curr = frames_input.shape[0]

        with deterministic_rng(deterministic_seed):
            degrader = Degradation()
            frames_input, md_record = degrader.add_random_degradation_to_sequence(frames_input, T_curr)

        flag = (cloud_mask_ins == 1).expand_as(frames_input)
        frames_input[flag] = 1
        flag = (masks == 1).expand_as(frames_input)
        frames_input[flag] = 1

        days = dates['S2'] - dates['S2'][0]
        if self.date_rescale:
            dates['S2'] = ((dates['S2'] / 10).round() * 10).int()

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
        seq_length = sample.shape[0]

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

        if self.split == 'train':
            step = self.min_length
            start_idx = segment_idx * step
            start_idx = min(start_idx, max(0, len(all_valid_indices) - step))
            end_idx = start_idx + (self.max_seq_length if self.max_seq_length else step)
            t_sampled = all_valid_indices[start_idx:end_idx]
        else:
            t_sampled = all_valid_indices

        if self.max_seq_length is not None and len(t_sampled) > self.max_seq_length:
            t_start_offset = np.random.randint(0, len(t_sampled) - self.max_seq_length + 1)
            t_sampled = t_sampled[t_start_offset: t_start_offset + self.max_seq_length]

        return t_sampled

    def select_random_frames_vectorized(self, mask, num_frames):
        T = mask.shape[0]
        if T == 0: return mask

        cloud_pixels = (mask > 0).sum(dim=(1, 2, 3))
        total_pixels = mask.shape[2] * mask.shape[3]
        cloud_free_ratios = 1.0 - (cloud_pixels.float() / total_pixels)

        clear_indices = torch.nonzero(cloud_free_ratios >= 0.95).squeeze()
        if clear_indices.ndim == 0 and clear_indices.numel() == 1:
            clear_indices = clear_indices.unsqueeze(0)

        if clear_indices.numel() == 0:
            _, sorted_idx = torch.sort(cloud_free_ratios, descending=True)
            clear_indices = sorted_idx

        clear_indices = clear_indices.tolist()
        min_clear = max(6, int(0.6 * num_frames))
        selected_indices = []

        if len(clear_indices) >= min_clear:
            selected_indices.extend(np.random.choice(clear_indices, min_clear, replace=False))
        else:
            selected_indices.extend(clear_indices)
            pass

        remaining_needed = num_frames - len(selected_indices)
        if remaining_needed > 0:
            all_indices = set(range(T))
            used_indices = set(selected_indices)
            avail_indices = list(all_indices - used_indices)

            if len(avail_indices) > 0:
                if len(avail_indices) >= remaining_needed:
                    selected_indices.extend(np.random.choice(avail_indices, remaining_needed, replace=False))
                else:
                    selected_indices.extend(np.random.choice(avail_indices, remaining_needed, replace=True))
            else:
                selected_indices.extend(np.random.choice(list(used_indices), remaining_needed, replace=True))

        selected_indices = np.array(selected_indices[:num_frames])
        np.random.shuffle(selected_indices)

        return mask[selected_indices]

    def select_fixed_frames(self, mask, num_frames):
        T = mask.shape[0]
        if T == 0: return mask

        min_clear = max(6, int(0.6 * num_frames))
        cloud_pixels = (mask > 0).sum(dim=(1, 2, 3))
        total_pixels = mask.shape[2] * mask.shape[3]
        cloud_free_ratios = 1.0 - (cloud_pixels.float() / total_pixels)

        is_clear = cloud_free_ratios >= 0.95
        clear_indices = torch.nonzero(is_clear).squeeze()

        if clear_indices.ndim == 0 and clear_indices.numel() == 1:
            clear_indices = clear_indices.unsqueeze(0)

        if clear_indices.numel() == 0:
            _, sorted_idx = torch.sort(cloud_free_ratios, descending=True)
            clear_indices = sorted_idx

        clear_indices = clear_indices.tolist()
        selected_indices = []

        if len(clear_indices) >= min_clear:
            selected_indices.extend(clear_indices[:min_clear])
        else:
            while len(selected_indices) < min_clear:
                for idx in clear_indices:
                    if len(selected_indices) < min_clear:
                        selected_indices.append(idx)
                    else:
                        break

        remaining = num_frames - len(selected_indices)
        if remaining > 0:
            all_indices = list(range(T))
            used_set = set(selected_indices)
            candidates = [i for i in all_indices if i not in used_set]

            if len(candidates) >= remaining:
                selected_indices.extend(candidates[:remaining])
            else:
                selected_indices.extend(candidates)
                still_needed = num_frames - len(selected_indices)
                for i in range(still_needed):
                    selected_indices.append(all_indices[i % T])

        selected_indices = sorted(selected_indices)
        return mask[torch.tensor(selected_indices)]

    def match_sequences_vectorized(self, A, B):
        if B is None or len(B) == 0:
            return None, None

        diff = torch.abs(A.unsqueeze(1) - B.unsqueeze(0))
        min_indices = torch.argmin(diff, dim=1)
        C = B[min_indices]

        return C, min_indices
