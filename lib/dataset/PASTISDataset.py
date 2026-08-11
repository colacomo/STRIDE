"""
Author: Vivien Sainte Fare Garnot (github.com/VSainteuf)
License MIT
"""

import json
import os
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata
from torchvision import transforms
from typing import Any, Dict, List, Optional, Tuple
from omegaconf import DictConfig, ListConfig, OmegaConf
from lib.datasets.Degra_utils import Degradation


class PASTISDataset(tdata.Dataset):
    def __init__(
        self,
        root: str,
        split: str = 'train',
        channels: str = 'bgr-nir',
        filter_settings = None,#: Optional[Dict | DictConfig] = None,
        crop_settings = None,#: Optional[Dict | DictConfig] = None,
        pe_strategy: str = 'day-of-year',
        mask_kwargs = None,#: Optional[Dict | DictConfig] = None,
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
        """
        Pytorch Dataset class to load samples from the PASTIS dataset, for semantic and
        panoptic segmentation.

        The Dataset yields ((data, dates), target) tuples, where:
            - data contains the image time series
            - dates contains the date sequence of the observations expressed in number
              of days since a reference date
            - target is the semantic or instance target

        Args:
            root (str): Path to the dataset
            norm (bool): If true, images are standardised using pre-computed
                channel-wise means and standard deviations.
            reference_date (str, Format : 'YYYY-MM-DD'): Defines the reference date
                based on which all observation dates are expressed. Along with the image
                time series and the target tensor, this dataloader yields the sequence
                of observation dates (in terms of number of days since the reference
                date). This sequence of dates is used for instance for the positional
                encoding in attention based approaches.
            target (str): 'semantic' or 'instance'. Defines which type of target is
                returned by the dataloader.
                * If 'semantic' the target tensor is a tensor containing the class of
                  each pixel.
                * If 'instance' the target tensor is the concatenation of several
                  signals, necessary to train the Parcel-as-Points module:
                    - the centerness heatmap,
                    - the instance ids,
                    - the voronoi partitioning of the patch with regards to the parcels'
                      centers,
                    - the (height, width) size of each parcel
                    - the semantic label of each parcel
                    - the semantic label of each pixel
            cache (bool): If True, the loaded samples stay in RAM, default False.
            mem16 (bool): Additional argument for cache. If True, the image time
                series tensors are stored in half precision in RAM for efficiency.
                They are cast back to float32 when returned by __getitem__.
            folds (list, optional): List of ints specifying which of the 5 official
                folds to load. By default (when None is specified) all folds are loaded.
            class_mapping (dict, optional): Dictionary to define a mapping between the
                default 18 class nomenclature and another class grouping, optional.
            mono_date (int or str, optional): If provided only one date of the
                available time series is loaded. If argument is an int it defines the
                position of the date that is loaded. If it is a string, it should be
                in format 'YYYY-MM-DD' and the closest available date will be selected.
            sats (list): defines the satellites to use. If you are using PASTIS-R, you have access to
                Sentinel-2 imagery and Sentinel-1 observations in Ascending and Descending orbits,
                respectively S2, S1A, and S1D.
                For example use sats=['S2', 'S1A'] for Sentinel-2 + Sentinel-1 ascending time series,
                or sats=['S2', 'S1A','S1D'] to retrieve all time series.
                If you are using PASTIS, only  S2 observations are available.
        """
        super(PASTISDataset, self).__init__()
        if filter_settings is None:
            filter_settings = {'type': 'cloud-free', 'min_length': 10, 'return_valid_obs_only': True, 'max_t_sampling': None}
        self.filter_settings = filter_settings
        self.max_seq_length = max_seq_length
        self.rescale = rescale
        #print("rescale",self.rescale)
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
        # Handle single-date parameter, supports date string or index
            self.mono_date = (
                datetime(*map(int, mono_date.split("-")))
                if "-" in mono_date
                else int(mono_date)
            )
        self.memory = {}
        self.memory_dates = {}
        # Class mapping function
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
        self.image_size = crop_settings.shape if self.crop_settings.enabled else (128, 128)

        # Fixed sequence length? If yes, the `collate_fn` function of the data loader pads samples to the same temporal
        # length before collating them to a batch
        # Whether sequence length is variable
        if filter_settings and filter_settings.get('type', None) is not None:
            self.variable_seq_length = filter_settings.return_valid_obs_only
        else:
            self.variable_seq_length = False

        # Get metadata
        print("Reading patch metadata . . .")
        self.meta_patch = gpd.read_file(os.path.join(root, "metadata.geojson"))
        self.meta_patch.index = self.meta_patch["ID_PATCH"].astype(int)
        self.meta_patch.sort_index(inplace=True)
        self.date_tables = {s: None for s in sats}
        self.date_range = np.array(range(-200, 600))

        for s in sats:
            dates = self.meta_patch["dates-{}".format(s)]
            date_table = pd.DataFrame(
                index=self.meta_patch.index, columns=self.date_range, dtype=int
            )
            for pid, date_seq in dates.items():
                #print(date_seq)
                if isinstance(date_seq, str):
                    try:
                        import json
                        date_seq = json.loads(date_seq)
                    except json.JSONDecodeError:
                        import ast
                        date_seq = ast.literal_eval(date_seq)
                #print(date_seq)
                #date_seq = {k: int(v) for k,v in date_seq.items()}
                d = pd.DataFrame().from_dict(date_seq, orient="index")
                #d = pd.Series(date_seq,name=0).to_frame()
                #print(d)
                d = d[0].apply(
                    lambda x: (
                        datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                        - self.reference_date
                    ).days
                )
                date_table.loc[pid, d.values] = 1
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

        # Select samples for the corresponding folds
        self.meta_patch = pd.concat(
            [self.meta_patch[self.meta_patch["Fold"] == f] for f in folds]
        )

        #self.len = self.meta_patch.shape[0]
        #self.id_patches = self.meta_patch.index
        
        # Get normalisation values
        if norm:
            self.norms = {}
            for s in self.sats:
                if s != "S2":
                    with open(
                        os.path.join(root, "NORM_{}_patch.json".format(s)), "r"
                    ) as file:
                        normvals = json.loads(file.read())
                    selected_folds = folds if folds is not None else range(1, 6)
                    means = [normvals["Fold_{}".format(f)]["mean"] for f in selected_folds]
                    stds = [normvals["Fold_{}".format(f)]["std"] for f in selected_folds]
                    self.norms[s] = np.stack(means).mean(axis=0), np.stack(stds).mean(axis=0)
                    self.norms[s] = (
                        torch.from_numpy(self.norms[s][0]).float(),
                        torch.from_numpy(self.norms[s][1]).float(),
                    )
        else:
            self.norms = None

        # Get bad frames
        with open(os.path.join(root, "bad_frames.json"), "r") as file:
            bad_frames = json.load(file)
        self.bad_frames = {int(k): v for k, v in bad_frames.items()}
        
        # --- Core modification: remove sequences shorter than min_length ---
        valid_ids = []
        min_len = self.filter_settings.get('min_length', 20)
        self.min_length=min_len
        initial_ids = self.meta_patch.index
        
        for pid in initial_ids:
            # Get the number of raw S2 observation dates
            s2_dates = self.get_dates(pid, 'S2')
            bad_idx = self.bad_frames.get(int(pid), [])
            # Compute actual usable length after filtering bad frames
            actual_len = len(s2_dates) - len(bad_idx)
            if actual_len >= min_len:
                valid_ids.append(pid)
            else:
                print("filter!!!!")

        self.id_patches = pd.Index(valid_ids)
        self.len = len(self.id_patches)

        # Save the number of channels, the indices of the RGB channels, and the index of the NIR channel
        if 'bgr' == self.channels[:3]:
            # self.channels in ['bgr', 'bgr-nir', 'bgr-mask', 'bgr-nir-mask']
            self.num_channels = 3
            self.c_index_rgb = torch.Tensor([2, 1, 0]).long()
            self.s2_channels = [0, 1, 2]                         # B2, B3, B4
        else:
            # self.channels in ['all', 'all-mask']
            self.num_channels = 10
            self.c_index_rgb = torch.Tensor([2, 1, 0]).long()
            self.s2_channels = list(np.arange(10))               # all 10 bands

        if '-nir' in self.channels:
            # self.channels in ['bgr-nir', 'bgr-nir-mask']
            self.num_channels += 1
            self.c_index_nir = torch.Tensor([3]).long()
            self.s2_channels += [6]                              # B8
        elif 'all' in self.channels:
            self.c_index_nir = torch.Tensor([6]).long()
        else:
            self.c_index_nir = torch.from_numpy(np.array(np.nan))

        print("Dataset ready.")

    def __len__(self):
        return self.len

    def get_dates(self, id_patch, sat):
        """Get the date sequence for a given patch and satellite."""
        return self.date_range[np.where(self.date_tables[sat][id_patch] == 1)[0]]

    def build_s11(self, datas1):
        #print("ds1",datas1.shape)
        p2, p98 = np.percentile(datas1, (2, 98))
        out = np.clip(datas1, p2, p98)
        out = (out - p2) / (p98 - p2 + 1e-8)
        return out

    def __getitem__(self, item):
        id_patch = self.id_patches[item]

        # Retrieve and prepare satellite data
        if not self.cache or item not in self.memory.keys():
            data = {
                satellite: np.load(
                    os.path.join(
                        self.root,
                        "DATA_{}".format(satellite),
                        "{}_{}.npy".format(satellite, id_patch),
                    )
                ).astype(np.float32)
                for satellite in self.sats
            }  # T x C x H x W arrays

            data = {s: torch.from_numpy(a) for s, a in data.items()}

            # Data normalization
            if self.norm:
                for s, d in data.items():
                    if s == "S2":
                        # Sentinel-2 data: normalize to [0,1]
                        data[s] = torch.clamp(d, 0, 8000) / 8000
                    else:
                        if self.rescale:
                            # SAR data: standardize and clip to [-1,1]
                            mid = (d - self.norms[s][0][None, :, None, None]) / self.norms[s][1][None, :, None, None]
                            data[s] = torch.clamp(mid, -2, 2) / 2  # SAR data is rescale to [-1, 1]
                        else:
                            data[s] = self.build_s11(data[s])


            # Load target data
            if self.target == "semantic":
                # Semantic segmentation target
                target = np.load(
                    os.path.join(
                        self.root, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch)
                    )
                )
                target = torch.from_numpy(target[0].astype(int))

                if self.class_mapping is not None:
                    target = self.class_mapping(target)

            elif self.target == "instance":
                # Instance segmentation target - contains multiple components
                heatmap = np.load(  # Centerness heatmap
                    os.path.join(
                        self.root,
                        "INSTANCE_ANNOTATIONS",
                        "HEATMAP_{}.npy".format(id_patch),
                    )
                )

                instance_ids = np.load(
                    os.path.join(
                        self.root,
                        "INSTANCE_ANNOTATIONS",
                        "INSTANCES_{}.npy".format(id_patch),
                    )
                )
                pixel_to_object_mapping = np.load(
                    os.path.join(
                        self.root,
                        "INSTANCE_ANNOTATIONS",
                        "ZONES_{}.npy".format(id_patch),
                    )
                )

                pixel_semantic_annotation = np.load(
                    os.path.join(
                        self.root, "ANNOTATIONS", "TARGET_{}.npy".format(id_patch)
                    )
                )

                if self.class_mapping is not None:
                    pixel_semantic_annotation = self.class_mapping(
                        pixel_semantic_annotation[0]
                    )
                else:
                    pixel_semantic_annotation = pixel_semantic_annotation[0]

                size = np.zeros((*instance_ids.shape, 2))
                object_semantic_annotation = np.zeros(instance_ids.shape)
                for instance_id in np.unique(instance_ids):
                    if instance_id != 0:
                        h = (instance_ids == instance_id).any(axis=-1).sum()
                        w = (instance_ids == instance_id).any(axis=-2).sum()
                        size[pixel_to_object_mapping == instance_id] = (h, w)
                        object_semantic_annotation[
                            pixel_to_object_mapping == instance_id
                        ] = pixel_semantic_annotation[instance_ids == instance_id][0]

                target = torch.from_numpy(
                    np.concatenate(
                        [
                            heatmap[:, :, None],  # 0
                            instance_ids[:, :, None],  # 1
                            pixel_to_object_mapping[:, :, None],  # 2
                            size,  # 3-4
                            object_semantic_annotation[:, :, None],  # 5
                            pixel_semantic_annotation[:, :, None],  # 6
                        ],
                        axis=-1,
                    )
                ).float()
            # Cache data
            if self.cache:
                if self.mem16:
                    self.memory[item] = [{k: v.half() for k, v in data.items()}, target]
                else:
                    self.memory[item] = [data, target]

        else:
            # Load data from cache
            data, target = self.memory[item]
            if self.mem16:
                data = {k: v.float() for k, v in data.items()}

        # Retrieve date sequences
        # Retrieve date sequences
        if not self.cache or id_patch not in self.memory_dates.keys():
            dates = {
                s: torch.from_numpy(self.get_dates(id_patch, s)) for s in self.sats
            }
            if self.cache:
                self.memory_dates[id_patch] = dates
        else:
            dates = self.memory_dates[id_patch]
        # Single-date mode handling
        if self.mono_date is not None:
            if isinstance(self.mono_date, int):
                # Select single date by index
                data = {s: data[s][self.mono_date].unsqueeze(0) for s in self.sats}
                dates = {s: dates[s][self.mono_date] for s in self.sats}
            else:
                # Select closest observation by date
                mono_delta = (self.mono_date - self.reference_date).days
                mono_date = {
                    s: int((dates[s] - mono_delta).abs().argmin()) for s in self.sats
                }
                data = {s: data[s][mono_date[s]].unsqueeze(0) for s in self.sats}
                dates = {s: dates[s][mono_date[s]] for s in self.sats}
        # Sentinel-1 data sampling (experimental)
        sample_from_S1 = True
        if sample_from_S1:
            Num = 35
            t_sampled_ablation = torch.linspace(0, len(data['S1A']) - 1, steps=Num, dtype=torch.long)

            data['S1A'] = data['S1A'][t_sampled_ablation, :, :, :]
            dates['S1A'] = dates['S1A'][t_sampled_ablation]

        # Temporally subsample/trim the sequence
        # Temporal subsample/trim sequence
        #print("bsub",data["S2"].shape)
        t_sampled = self._subsample_sequence(data["S2"], self.bad_frames[id_patch])
        #print("asub", data["S2"].shape,t_sampled.shape)
        # Extract subsampled time steps
        if 'S2' in self.sats:
            data["S2"] = data["S2"][t_sampled, :, :, :]
            data["S2"] = data["S2"][:, self.s2_channels, :, :]
            dates["S2"] = dates["S2"][t_sampled]

            if self.rescale:
                trans_scale = transforms.Normalize([0.5], [0.5])
                data['S2'] = trans_scale(data['S2']) # [-1, 1]
                #print("s2",torch.max(data['S2']),torch.min(data['S2']))
        #print("asub1", data["S2"].shape)
        # Extract matched SAR frames
        if 'S1D' in self.sats:
            dates["S1D"], t_sampled_SAR = match_sequences_tensor(dates['S2'], dates['S1D'])
            data["S1D"] = data["S1D"][t_sampled_SAR, :, :, :]
        else:
            dates["S1D"] = None
            data["S1D"] = None

        if 'S1A' in self.sats:
            dates["S1A"], t_sampled_SAR = match_sequences_tensor(dates['S2'], dates['S1A'])
            data["S1A"] = data["S1A"][t_sampled_SAR, :, :, :]
        else:
            dates["S1A"] = None
            data["S1A"] = None

        cond = data["S1D"] if data["S1D"] is not None else data["S1A"]
        position_days_cond = dates["S1D"] if dates["S1D"] is not None else dates["S1A"]


        if self.mem16:
            data = {k: v.float() for k, v in data.items()}

        # masks for training
        # Load and process cloud masks
        masks_pool = np.load(self.mask_path + "/{}.npy".format(id_patch)) # [T, 1, H, W]
        masks_pool = torch.from_numpy(masks_pool).float()
        # randomly select "len(t_sampled)" frames from masks (T frames)
        # Random or fixed frame selection
        if self.split == 'train':
            masks = self.select_random_frames(masks_pool, len(t_sampled)) # [len(t_sampled), 1, H, W]
        else:
            masks = self.select_fixed_frames(masks_pool, len(t_sampled))
        # masks: (0: clear; 1: Thick cloud; 2: Thin cloud; 3: Cloud shadow)
        # Binarize mask (0: clear; 1: cloud)
        #masks = torch.where(masks > 0, 1.0, masks)
        masks = torch.where(masks > 0, torch.tensor(1.0, dtype=masks.dtype), masks)

        # Real cloud_mask for evaluation (used to maskout the cloud pixels in the target)
        # Real cloud mask for evaluation
        cloud_mask = masks_pool[t_sampled].clone()
        #cloud_mask = torch.where(cloud_mask > 0, 1.0, cloud_mask)
        cloud_mask = torch.where(cloud_mask > 0, torch.tensor(1.0, dtype=cloud_mask.dtype), cloud_mask)

        # frames_input
        # Input frames (fill masked pixels with 1)
        frames_input = data["S2"].clone()
        T,_,_,_ = frames_input.shape

        # Multi-degradation (record degraded sequence and frame positions)
        degrader = Degradation()
        if self.split == 'test':
            import hashlib
            # Generate a unique hash from patch_id as random seed to ensure fully deterministic degradation in the test set
            seed_str = f"{id_patch}"
            seed = int(hashlib.md5(seed_str.encode('utf-8')).hexdigest()[:8], 16)
            frames_input, md_record = degrader.add_random_degradation_to_sequence(frames_input, T,
                                                                                  seed=seed)
        else:
            frames_input, md_record = degrader.add_random_degradation_to_sequence(frames_input, T)
        #print(md_record)
        flag = (masks == 1).expand_as(frames_input)
        frames_input[flag] = 1    # fill masked pixels with 1

        # days since the first observation in the sequence
        # Days since the first observation in the sequence
        days = dates['S2'] - dates['S2'][0]

        # if date_rescale: rescale each date to the closest multiple of 10
        if self.date_rescale:
            dates['S2'] = ((dates['S2'] / 10).round() * 10).int()
            # print(dates['S2'])

        # Assemble output #print("in",frames_input.shape,torch.max(frames_input),torch.min(frames_input),torch.max(data["S2"]),torch.min(data["S2"]),torch.max(cond),torch.min(cond))
        """print('x', frames_input.shape,
            'cond', cond.shape,
            'y', data["S2"].shape,
            'masks', masks.shape,      # (T x 1 x H x W)
            'position_days', dates['S2'],
            'position_days_cond', position_days_cond,
            'days', days,  # number of days since the first observation in the sequence, (T, )
            'sample_index', id_patch,
            'label', target.shape,
            'c_index_rgb', self.c_index_rgb,
            'c_index_nir', self.c_index_nir,
            'cloud_mask', cloud_mask.shape,)
        """
        out = {
            'x': frames_input,
            'cond': cond,
            'y': data["S2"],
            'masks': masks,      # (T x 1 x H x W)
            'position_days': dates['S2'],
            'position_days_cond': position_days_cond,
            'days': days,  # number of days since the first observation in the sequence, (T, )
            'sample_index': id_patch,
            'label': target,
            'c_index_rgb': self.c_index_rgb,
            'c_index_nir': self.c_index_nir,
            'cloud_mask': cloud_mask,
            'md_rec':md_record,
        }

        return out


    def _subsample_sequence(self, sample: torch.Tensor, bad_index: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Filters/Subsamples the image time series stored in `sample` as follows (cf. `self.filter_settings` and
        `self.max_seq_length`):
        1) Extracts cloud-free images or extracts the longest consecutive cloud-free subsequence,
        2) selects a subsequence of cloud-free images such that the temporal difference between consecutive cloud-free
           images is at most `self.filter_settings.max_t_sampling` days,
        3) trims the sequence to a maximum temporal length.

        Args:
            sample:           torch.Tensor, (T, C, H, W), image time series.
            bad_index:        list, indices of the invalid frames.

        Returns:
            t_sampled:        torch.Tensor, length T.
        """

        # Generate a mask to exclude invalid frames:
        # a value of 1 indicates a valid frame, whereas a value of 0 marks an invalid frame
        seq_length = sample.shape[0]

        if self.filter_settings.type == 'cloud-free':
            # Indices of available and cloud-free images
            masks_valid_obs = torch.ones(seq_length, )
            masks_valid_obs[bad_index] = 0
            # masks_valid_obs = torch.from_numpy(sample['valid_obs'][:])

        # elif self.filter_settings.type == 'cloud-free_consecutive':
        #     subseq = self._longest_consecutive_seq(sample)
        #     masks_valid_obs = torch.from_numpy(sample['valid_obs'][:])
        #     masks_valid_obs[:subseq['start']] = 0
        #     masks_valid_obs[subseq['end'] + 1:] = 0
        else:
            masks_valid_obs = torch.ones(seq_length, )

        if self.filter_settings.get('return_valid_obs_only', True):
            t_sampled = masks_valid_obs.nonzero().view(-1)
        else:
            t_sampled = torch.arange(0, len(masks_valid_obs))

        if self.max_seq_length is not None and len(t_sampled) > self.max_seq_length:
            # Randomly select `self.max_seq_length` consecutive frames
            t_start = np.random.choice(np.arange(0, len(t_sampled) - self.max_seq_length + 1))
            t_end = t_start + self.max_seq_length
            t_sampled = t_sampled[t_start:t_end]

        return t_sampled

    """def select_random_frames(self, mask, num_frames):
        T = mask.shape[0]
        indices = torch.randperm(T)[:num_frames]
        selected_frames = torch.index_select(mask, 0, indices)
        return selected_frames"""

    def select_random_frames(self, mask, num_frames):
        """
        Randomly select frames, ensuring at least 20% of frames have cloud-free ratio >= 95%.

        Args:
            mask: torch.Tensor, shape (T, 1, H, W), cloud mask (0: clear, 1: cloud)
            num_frames: int, number of frames to select

        Returns:
            selected_frames: torch.Tensor, shape (num_frames, 1, H, W)
        """
        T = mask.shape[0]

        # Compute cloud-free pixel ratio for each frame
        cloud_free_ratios = []
        for t in range(T):
            frame_mask = mask[t]
            total_pixels = frame_mask.numel()
            cloud_free_pixels = (frame_mask == 0).sum().item()
            cloud_free_ratio = cloud_free_pixels / total_pixels
            cloud_free_ratios.append(cloud_free_ratio)

        # Split frames into two categories: clear frames (cloud-free ratio >= 95%) and other frames
        clear_frames = []  # clear frame indices
        other_frames = []  # other frame indices

        for t in range(T):
            if cloud_free_ratios[t] >= 0.95:  # cloud-free ratio >= 95%
                clear_frames.append(t)
            else:
                other_frames.append(t)

        # Compute required number of clear frames (at least 20%)
        min_clear_frames = max(2, int(0.3 * num_frames))  # at least 1 frame

        # If insufficient clear frames, log a warning
        if len(clear_frames) < min_clear_frames:
            print(f"Warning: Only {len(clear_frames)} frame(s) with cloud-free ratio >= 95%, but at least {min_clear_frames} are required")
            # If no clear frames at all, use fallback strategy
            if len(clear_frames) == 0:
                print("Warning: No frames found with cloud-free ratio >= 95%, will select frames with highest cloud-free ratio")
                # Select frames with highest cloud-free ratio
                sorted_indices = np.argsort(cloud_free_ratios)[::-1]  # descending order
                selected_clear_frames = sorted_indices[:min_clear_frames].tolist()
            else:
                # Use all available clear frames
                selected_clear_frames = clear_frames.copy()
                # Need to supplement with some relatively clear frames
                remaining_needed = min_clear_frames - len(clear_frames)
                # Select frames with highest cloud-free ratio from others
                other_frame_ratios = [(idx, cloud_free_ratios[idx]) for idx in other_frames]
                other_frame_ratios.sort(key=lambda x: x[1], reverse=True)
                selected_clear_frames.extend([idx for idx, _ in other_frame_ratios[:remaining_needed]])
        else:
            # Randomly select enough clear frames
            selected_clear_frames = np.random.choice(clear_frames, size=min_clear_frames, replace=False).tolist()

        # Need to select remaining frames
        remaining_frames_needed = num_frames - len(selected_clear_frames)

        # Randomly select from remaining frames (excluding already selected clear frames)
        available_frames = list(set(range(T)) - set(selected_clear_frames))

        if remaining_frames_needed > 0:
            if remaining_frames_needed > len(available_frames):
                # If needed frames exceed available frames, allow repeated selection
                additional_frames = np.random.choice(available_frames, size=remaining_frames_needed,
                                                     replace=True).tolist()
            else:
                # Randomly select without replacement
                additional_frames = np.random.choice(available_frames, size=remaining_frames_needed,
                                                     replace=False).tolist()
        else:
            additional_frames = []

        # Merge all selected frame indices
        selected_indices = selected_clear_frames + additional_frames

        # If more frames were selected than needed (may happen in edge cases), truncate to required number
        if len(selected_indices) > num_frames:
            selected_indices = selected_indices[:num_frames]

        # Shuffle order (optional, depends on whether you want clear frames concentrated at the front)
        np.random.shuffle(selected_indices)

        # Select frames by index
        selected_frames = mask[selected_indices]

        # Validate selection results
        clear_count = 0
        for idx in selected_indices:
            if cloud_free_ratios[idx] >= 0.95:
                clear_count += 1

        if clear_count < min_clear_frames:
            print(f"Warning: Only {clear_count} clear frame(s) were selected, falling short of the target of {min_clear_frames}")

        return selected_frames
    def select_fixed_frames(self, mask, num_frames):
        indices = torch.arange(0, num_frames)
        selected_frames = torch.index_select(mask, 0, indices)
        return selected_frames

def prepare_dates(date_dict, reference_date):
    """Date formatting."""
    d = pd.DataFrame().from_dict(date_dict, orient="index")
    d = d[0].apply(
        lambda x: (
            datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
            - reference_date
        ).days
    )
    return d.values


def match_sequences_tensor(A, B):
    M = A.size(0)
    N = B.size(0)

    # Initialize C with a large negative integer as a placeholder
    placeholder = -10 ** 6
    C = torch.full_like(A, placeholder)
    indices = torch.full_like(A, -1, dtype=torch.long)  # Indices of elements in B used for C

    # Step 1: Fix positions for exact matches
    for i in range(M):
        a = A[i].item()
        if a in B:
            pos_in_b = (B == a).nonzero(as_tuple=True)[0].item()
            C[i] = a
            indices[i] = pos_in_b

    # Step 2: Fill the gaps for non-exact matches
    for i in range(M):
        if C[i].item() == placeholder:
            a = A[i].item()
            # Find the best match for a in B
            pos = torch.searchsorted(B, torch.tensor([a]), right=False).item()

            # Check to the left and right to find the closest element in B
            left_pos = pos - 1
            right_pos = pos if pos < N else N - 1

            left_diff = float('inf') if left_pos < 0 else abs(B[left_pos].item() - a)
            right_diff = float('inf') if right_pos >= N else abs(B[right_pos].item() - a)

            # Choose the position with the smallest difference
            if left_diff <= right_diff and left_pos >= 0:
                best_pos = left_pos
            else:
                best_pos = right_pos

            # Assign the best match to C
            C[i] = B[best_pos]
            indices[i] = best_pos

    return C, indices
