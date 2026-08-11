import torch
import torch.nn.functional as F
import random
import numpy as np
import cv2
import os
import scipy.io as sio


class Degradation(object):
    def __init__(self, args=None):
        super(Degradation, self).__init__()
        self.args = args
        self.downsample_factor = None
        self.intensity = None

    # ---------------------- Basic degradation methods ----------------------
    def _add_gaussian_noise(self, clean_patch, min_sigma, max_sigma):
        sigma = np.random.uniform(min_sigma, max_sigma) / 255
        noise = np.random.randn(*clean_patch.shape) * sigma
        noisy_patch = clean_patch + noise
        return noisy_patch.astype(np.float32)

    def _add_impulse_noise(self, clean_patch, amount, salt_vs_pepper=0.5):
        B, H, W = clean_patch.shape
        num_bands_fraction = 1 / 3
        num_bands = int(np.floor(num_bands_fraction * B))
        bands = np.random.permutation(B)[:num_bands]
        for b in bands:
            p = amount
            q = salt_vs_pepper
            flipped = np.random.choice([True, False], size=(H, W), p=[p, 1 - p])
            salted = np.random.choice([True, False], size=(H, W), p=[q, 1 - q])
            peppered = ~salted
            clean_patch[b, flipped & salted] = 1
            clean_patch[b, flipped & peppered] = 0
        return clean_patch.astype(np.float32)

    def _apply_gaussian_blur(self, clean_patch, kernel_size):
        B, H, W = clean_patch.shape
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        x = torch.arange(kernel_size, dtype=torch.float32)
        mean = (kernel_size - 1) / 2
        kernel_1d = torch.exp(-((x - mean) ** 2) / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.unsqueeze(0)

        input_tensor = torch.from_numpy(clean_patch).float().unsqueeze(0)
        kernel_2d = kernel_2d.repeat(input_tensor.shape[1], 1, 1, 1)
        blurred_image = F.conv2d(input_tensor, kernel_2d,
                                 padding=kernel_size // 2, stride=1,
                                 groups=input_tensor.shape[1])
        return blurred_image.squeeze(0).detach().numpy().astype(np.float32)

    def _apply_motion_blur(self, clean_patch, kernel_size, angle):
        motion_kernel = np.zeros((kernel_size, kernel_size))
        motion_kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        motion_kernel /= kernel_size

        rotation_matrix = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
        kernel = cv2.warpAffine(motion_kernel, rotation_matrix, (kernel_size, kernel_size))

        kernel_2d = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0).float()
        input_tensor = torch.from_numpy(clean_patch).float().unsqueeze(0)
        kernel_2d = kernel_2d.repeat(input_tensor.shape[1], 1, 1, 1)

        blurred_image = F.conv2d(input_tensor, kernel_2d,
                                 padding=kernel_size // 2, groups=input_tensor.shape[1])
        return blurred_image.squeeze(0).detach().numpy().astype(np.float32)

    def _simulate_haze(self, hsi, omega=0.2, gamma=1.0, top_percent=0.01):
        """Simulate haze using a randomly selected atmospheric scattering layer."""
        cirrus_band_file = os.path.join(r'/media/amax/disk4/landsat8trans1',
                                        str(random.randint(1001, 1340)) + ".png")
        cirrus_band = cv2.imread(cirrus_band_file, cv2.IMREAD_GRAYSCALE)
        #print("cirrus", cirrus_band_file, cirrus_band, np.max(cirrus_band))
        C, H, W = hsi.shape
        cirrus_band = cv2.resize(cirrus_band, (W, H), interpolation=cv2.INTER_LINEAR)/255
        #print("cirrus",cirrus_band.shape,np.max(cirrus_band))

        Bandwavelength = [497, 560, 665, 704, 740, 783, 835, 865, 1614, 2202]
        num_pixels = H * W
        top_k = max(int(num_pixels * top_percent / 100), 1)
        atmospheric_light = np.zeros(C)
        for i in range(C):
            band = hsi[i, :, :].flatten()
            top_pixels = np.partition(band, -top_k)[-top_k:]
            atmospheric_light[i] = np.mean(top_pixels)
            #print(i, atmospheric_light[i],np.min(band))

        t1 = 1 - omega * cirrus_band
        t1 = np.where(t1 <= 0, 1e-10, t1)
        hazy_hsi = np.zeros_like(hsi)
        #print("t1",t1.shape,np.max(t1),np.min(t1))

        for band in range(C):
            lambda_ratio = Bandwavelength[2] / Bandwavelength[band]
            transmission = np.exp((lambda_ratio ** gamma) * np.log(t1))
            #print(band, lambda_ratio, np.max(transmission), np.max(atmospheric_light[band]))
            hazy_hsi[band] = hsi[band] * transmission + atmospheric_light[band] * (1 - transmission)

        return hazy_hsi.astype(np.float32)

    # ---------------------- High-level interface ----------------------
    def _degrade_by_type(self, clean_patch, degrade_type, degrade_range):
        if degrade_type == 'gaussianN':
            degraded_patch = self._add_gaussian_noise(clean_patch, 5, 25)
        elif degrade_type == 'impulse':
            degraded_patch = self._add_impulse_noise(clean_patch, amount=random.uniform(*degrade_range[0]))
        elif degrade_type == 'blur':
            kernel_size = random.choice(degrade_range[0])
            degraded_patch = self._apply_gaussian_blur(clean_patch, kernel_size)
        elif degrade_type == 'motion_blur':
            kernel_size, angle = random.choice(degrade_range[0])
            degraded_patch = self._apply_motion_blur(clean_patch, kernel_size, angle)
        elif degrade_type == 'haze':
            omega = random.choice(degrade_range[0])
            degraded_patch = self._simulate_haze(hsi=clean_patch, omega=omega)
        elif degrade_type == 'dark':
            factor = random.uniform(0.3, 0.6)
            degraded_patch = clean_patch * factor
        else:
            raise ValueError(f"Invalid degradation type: {degrade_type}")
        return degraded_patch, clean_patch

    def single_degrade(self, clean_patch, degrade_type=None, degrade_range=None, name=None):
        """Apply a single degradation type to a single-frame image."""
        if degrade_type == 'complexN':
            degrad_patch_1, _ = self._degrade_by_type(clean_patch, degrade_type, degrade_range)
        else:
            degrad_patch_1, _ = self._degrade_by_type(clean_patch, degrade_type, degrade_range)
        return degrad_patch_1, self.intensity

        # ---------------------- Random degradation for full sequence ----------------------

    def add_random_degradation_to_sequence(self, sequence, max_seq_length, bad_index=None, seed=None):
        """
        Randomly apply degradation to 10% of clear frames in the input tensor sequence
        (avoiding bad frames). Input and output are both Tensors.
        """
        if seed is not None:
            py_state = random.getstate()
            np_state = np.random.get_state()
            random.seed(seed)
            np.random.seed(seed)

        try:
            if not isinstance(sequence, torch.Tensor):
                raise TypeError("Expected sequence as torch.Tensor, got {}".format(type(sequence)))

            device = sequence.device
            num_frames = sequence.shape[0]
            degraded_sequence = sequence.clone()

            degrade_record = torch.zeros(num_frames, dtype=torch.int64, device=device)

            if bad_index is None:
                bad_index = []

            good_indices = [i for i in range(num_frames) if i not in bad_index]
            if len(good_indices) == 0:
                return degraded_sequence, degrade_record

            num_to_degrade = max(1, int(len(good_indices) * 0.1))
            selected_indices = random.sample(good_indices, num_to_degrade)

            degrade_types = {
                'gaussianN': 1,
                'impulse': 2,
                'blur': 3,
                'motion_blur': 4,
                #'haze': 5,
            }

            degrade_type_list = list(degrade_types.keys())

            for idx in selected_indices:
                if idx < max_seq_length:
                    degrade_type = random.choice(degrade_type_list)
                    frame_np = sequence[idx].detach().cpu().numpy()

                    if degrade_type == 'gaussianN':
                        degraded_frame, _ = self.single_degrade(frame_np, 'gaussianN', [(5, 25)])
                    elif degrade_type == 'impulse':
                        degraded_frame, _ = self.single_degrade(frame_np, 'impulse', [[0.01, 0.05]])
                    elif degrade_type == 'blur':
                        degraded_frame, _ = self.single_degrade(frame_np, 'blur', [[3, 5, 7]])
                    elif degrade_type == 'motion_blur':
                        degraded_frame, _ = self.single_degrade(frame_np, 'motion_blur',
                                                                [[(5, 0), (7, 45), (9, 90)]])
                    #elif degrade_type == 'haze':
                    #    degraded_frame, _ = self.single_degrade(frame_np, 'haze', [[1.2, 2.4, 3.6]])
                    else:
                        continue

                    degraded_tensor = torch.from_numpy(np.clip(degraded_frame, 0, 1)).to(device).float()
                    degraded_sequence[idx] = degraded_tensor

                    degrade_record[idx] = degrade_types[degrade_type]

            return degraded_sequence, degrade_record

        finally:
            if seed is not None:
                random.setstate(py_state)
                np.random.set_state(np_state)
