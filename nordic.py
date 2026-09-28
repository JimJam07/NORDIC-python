from base_classes import base_nordic
import numpy as np
import nibabel as nib
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.signal.windows import tukey
from g_factor import g_factor_processing
from utils import *
from LLR import sub_LLR_Processing
import time
import os
import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d as jax_conv2d
from scipy.io import savemat

class Nordic(base_nordic):
    def __init__(self, IN_MAG, IN_PHASE,*, Args={}):
        super().__init__()
        self.Args.update(Args)

        t0 = time.perf_counter()
        self.data_load(IN_MAG, IN_PHASE)
        print(f"data_load took {time.perf_counter() - t0:.3f} seconds")

        if self.II.shape[3] < self.Args['min_vols']:
            raise ValueError("Too few volumes")

        t1 = time.perf_counter()
        self.phase_preparation()
        print(f"phase_preparation took {time.perf_counter() - t1:.3f} seconds")

        t2 = time.perf_counter()
        self.gfactor = g_factor_processing(self.II, self.Args)
        print(f"g_factor_processing took {time.perf_counter() - t2:.3f} seconds")

        t3 = time.perf_counter()
        self.NORDIC_processing()
        print(f"NORDIC_processing took {time.perf_counter() - t3:.3f} seconds")

        t4 = time.perf_counter()
        self.data_post_processing()
        print(f"data_post_processing took {time.perf_counter() - t4:.3f} seconds")

        t5 = time.perf_counter()
        self.output_generation(Args['FN_OUT'])
        print(f"output_generation took {time.perf_counter() - t5:.3f} seconds")

    def data_load(self, fn_magn_in, fn_phase_in):
        """
        Loads magnitude and phase NIfTI files, normalizes phase,
        combines them into a complex matrix, and applies absolute scaling.

        self.Args: A dictionary containing parameters (equivalent to MATLAB struct).
             Requires at least the key 'magnitude_only'.
        """
        # 1. Load magnitude image and safely cast directly to float32 to halve memory usage
        info = nib.load(fn_magn_in)
        I_M = info.get_fdata(dtype=np.float32)
        np.abs(I_M, out=I_M)  # In-place absolute value to save memory

        info_phase = None

        # Check if we need complex data (magnitude_only != 1)
        if not self.Args.get('magnitude_only', 0) == 1:
            info_phase = nib.load(fn_phase_in)
            I_P = info_phase.get_fdata(dtype=np.float64)

            p_min = I_P.min()
            p_max = I_P.max()
            p_range = p_max - p_min

            print('Phase should be -pi to pi...')

            # 2. In-place mathematical scaling maps phase to [-pi, pi] without intermediate arrays
            if p_range != 0:
                I_P -= p_min
                I_P *= (2.0 * np.pi / p_range)
                I_P -= np.pi
            else:
                I_P.fill(0)

            print(f"Phase data range is {I_P.min():.2f} to {I_P.max():.2f}")

            # 3. Memory-efficient complex array construction
            II = np.empty(I_M.shape, dtype=np.complex128)

            # .real and .imag are writeable views in numpy. We calculate and multiply directly.
            np.cos(I_P, out=II.real)
            II.real *= I_M

            np.sin(I_P, out=II.imag)
            II.imag *= I_M

            del I_P

        else:
            # Magnitude only
            info_phase = None
            II = I_M

        # 4. Handle 3D vs 4D data slicing
        if II.ndim >= 4:
            tempvol = np.abs(II[..., 0])
        else:
            tempvol = np.abs(II)

        # 5. Calculate absolute scale
        nonzero_mask = tempvol != 0
        if np.any(nonzero_mask):
            self.Args['ABSOLUTE_SCALE'] = float(tempvol[nonzero_mask].min())
        else:
            self.Args['ABSOLUTE_SCALE'] = 1.0  # Fallback to prevent division by zero

        # Clean up scaling intermediates
        del tempvol, nonzero_mask

        # 6. Apply global scale in-place
        self.II = II/self.Args['ABSOLUTE_SCALE']
        self.info_phase = info_phase
        self.info = info

    def phase_preparation(self):
        """
        Phase preparation pipeline utilizing JAX for accelerated batched convolution.
        II: 4D complex numpy array (X, Y, Slices, Time)
        self.Args: dictionary containing configuration parameters
        """
        KSP2 = self.II.copy()
        nx, ny, nslices, nframes = KSP2.shape

        print('estimating slice-dependent phases ...')
        noise_vol = self.Args.get('noise_volume_last', 0)

        if noise_vol > 0:
            meanphase = np.mean(KSP2[..., :-noise_vol], axis=3)
        else:
            meanphase = np.mean(KSP2, axis=3)

        meanphase *= self.Args.get('phase_slice_average_for_kspace_centering', 1.0)
        phases = {'meanphase': meanphase}

        # Remove mean phase
        KSP2 *= np.exp(-1j * np.angle(meanphase))[..., np.newaxis]

        DD_phase = np.zeros_like(KSP2)
        temporal_phase = self.Args.get('temporal_phase', 0)

        if temporal_phase > 0:
            filter_name = self.Args.get('phase_filter_name', '')

            if filter_name == 'tukey':
                alpha = self.Args.get('phase_filter_width', 0.5)
                wy = tukey(ny, alpha=alpha).reshape(1, ny, 1, 1)
                wx = tukey(nx, alpha=alpha).reshape(nx, 1, 1, 1)
                window2d = wx * wy

                tmp = ifftshift(KSP2, axes=(0, 1))
                tmp = ifft2(tmp, axes=(0, 1))
                tmp = ifftshift(tmp, axes=(0, 1))

                tmp *= window2d

                tmp = fftshift(tmp, axes=(0, 1))
                tmp = fft2(tmp, axes=(0, 1))
                DD_phase = fftshift(tmp, axes=(0, 1))

            elif filter_name == 'DK':
                # Reshape to (Batch, X, Y) where Batch = slices * frames
                flat_ksp2 = KSP2.transpose(2, 3, 0, 1).reshape(nslices * nframes, nx, ny)
                j_ksp2 = jnp.array(flat_ksp2)
                j_kernel = jnp.array(self.Args['decorr_kernel'])

                # Same conv
                batched_conv = jax.vmap(lambda img, krn: jax_conv2d(img, krn, mode='same'), in_axes=(0, None))
                flat_dd_phase = batched_conv(j_ksp2, j_kernel)

                # Retrieve from JAX and reshape back to (X, Y, Slices, Time)
                DD_phase = np.array(flat_dd_phase).reshape(nslices, nframes, nx, ny).transpose(2, 3, 0, 1)

        if temporal_phase == 2:
            phase_diff = np.angle(KSP2 / DD_phase)
            mask = np.abs(phase_diff) > 1
            DD_phase[mask] = KSP2[mask]

        phases['DD_phase'] = DD_phase

        if temporal_phase == 3:
            phase_diff = np.angle(KSP2 / DD_phase)
            mask = np.abs(phase_diff) > 1

            DD_phase3 = KSP2.copy()

            shifted_sum = (
                    np.roll(KSP2, 1, axis=0) + np.roll(KSP2, -1, axis=0) +
                    np.roll(KSP2, 1, axis=1) + np.roll(KSP2, -1, axis=1)
            )

            DD_phase3[mask] = shifted_sum[mask]

            KSP2 = np.abs(KSP2) * np.exp(1j * np.angle(DD_phase3))
            II = np.abs(self.II) * np.exp(1j * np.angle(DD_phase3)) * np.exp(1j * meanphase)[..., np.newaxis]

            phases['DD_phase3'] = DD_phase3
            print('NOT USED !!!')

        # Remove temporal phase
        self.KSP2 = KSP2*np.exp(-1j * np.angle(DD_phase))
        self.phases = phases
    # NOTE: Ensure `calculate_NORDIC_threshold` and `updated_sub_LLR_Processing_v2`
    # are defined/imported in your module.

    def NORDIC_processing(self):
        """
        Main NORDIC processing orchestrator. Applies g-factor, handles empty voxels,
        calculates thresholds, and runs the parallel LLR reconstruction.
        """

        self.Args['matdim'] = self.II.shape
        nt = self.II.shape[3]

        # 1. Determine Default Spatial Kernel Size
        default_k = int(np.round((nt * 11) ** (1 / 3)))
        self.Args['kernel_size'] = [default_k, default_k, default_k]

        # 2. Apply g-factor normalization
        # Broadcast the 3D g-factor across the 4D temporal dimension safely
        gfactor_exp = self.gfactor[..., np.newaxis]
        mask_g = gfactor_exp != 0
        np.divide(self.II, gfactor_exp, out=self.II, where=mask_g)

        # 3. Extract Noise Volume
        noise_vol_last = self.Args.get('noise_volume_last', 0)
        if noise_vol_last > 0:
            # MATLAB: end+1-ARG.noise_volume_last => Python: -noise_vol_last
            KSP2_NOISE = self.II[..., -noise_vol_last]
        else:
            KSP2_NOISE = None

        # 4. Handle Zero-Elements (100% Vectorized)
        if self.Args.get('data_has_zero_elements') == 1:
            # Find spatial voxels where the sum across time is exactly 0
            zero_mask = np.sum(np.abs(self.II), axis=3) == 0
            num_zeros = np.sum(zero_mask)

            if num_zeros > 0:
                # Vectorized generation for all time-volumes simultaneously (No loop required)
                noise_r = np.random.randn(num_zeros, nt)
                noise_i = np.random.randn(num_zeros, nt)
                self.II[zero_mask, :] = (noise_r + 1j * noise_i) / np.sqrt(2)

        # 5. Calculate Threshold
        self.Args = calculate_NORDIC_threshold(KSP2_NOISE, self.Args)

        print('starting NORDIC ...')

        # 6. Execute Parallel LLR Processing
        KSP_recon, self.Args, KSP_weight, NOISE, Component_threshold, energy_removed, SNR_weight = \
            sub_LLR_Processing(self.II, self.Args)

        # 7. Normalize accumulated patches by the patch weighting
        mask_w = KSP_weight > 0
        weight_exp = KSP_weight[..., np.newaxis]

        # Safely reconstruct image
        np.divide(KSP_recon, weight_exp, out=KSP_recon, where=(mask_w[..., np.newaxis]))

        # Normalize and package aux metrics
        self.Args['NOISE'] = np.zeros_like(NOISE)
        np.divide(NOISE, KSP_weight, out=self.Args['NOISE'], where=mask_w)
        np.sqrt(self.Args['NOISE'], out=self.Args['NOISE'])

        self.Args['Component_threshold'] = np.zeros_like(Component_threshold)
        np.divide(Component_threshold, KSP_weight, out=self.Args['Component_threshold'], where=mask_w)

        self.Args['energy_removed'] = np.zeros_like(energy_removed)
        np.divide(energy_removed, KSP_weight, out=self.Args['energy_removed'], where=mask_w)

        self.Args['SNR_weight'] = np.zeros_like(SNR_weight)
        np.divide(SNR_weight, KSP_weight, out=self.Args['SNR_weight'], where=mask_w)

        self.IMG2 = KSP_recon

        print('completing NORDIC ...')

        # 8. Calculate Residuals (Removed dead 'if 0' code)
        self.Args['Residual'] = KSP_recon - self.II


    def data_post_processing(self):
        """
        Post-processing to re-apply phase components, g-factor, and absolute scale.
        Note: 'II' is passed to maintain API compatibility but is no longer used
        in this updated algorithm version.
        """

        self.Residual = self.Args.get('Residual', np.zeros_like(self.IMG2)).copy()

        abs_scale = self.Args.get('ABSOLUTE_SCALE', 1.0)

        static_scale_3d = self.gfactor * abs_scale

        # ---------------------------------------------------------
        #  process Residuals
        # ---------------------------------------------------------
        self.Residual *= static_scale_3d[..., np.newaxis]

        # ---------------------------------------------------------
        # Process IMG2
        # ---------------------------------------------------------
        meanphase = self.phases.get('meanphase')
        DD_phase = self.phases.get('DD_phase')

        static_multiplier_3d = static_scale_3d * np.exp(1j * np.angle(meanphase))

        self.IMG2 *= static_multiplier_3d[..., np.newaxis]

        if DD_phase is not None:
            self.IMG2 *= np.exp(1j * np.angle(DD_phase))

        np.nan_to_num(self.IMG2, copy=False, nan=0.0)

    def output_generation(self, fn_out):
        """
        Generates all requested NIfTI and MATLAB outputs based on ARG configuration.
        Features in-place memory optimization, dynamic range scaling, and native nibabel header management.
        """

        # ---------------------------------------------------------
        # Helper 2: NIfTI File Writer
        # ---------------------------------------------------------
        def save_nifti(data_array, prefix, suffix, source_info):
            """Wraps array in nibabel NIfTI object preserving original headers."""
            if data_array is None or source_info is None:
                return

            # nibabel naturally preserves qform/sform from the source_info
            out_nii = nib.Nifti1Image(data_array, source_info.affine, source_info.header)
            out_nii.header.set_data_dtype(data_array.dtype)

            out_path = os.path.join(dir_out, f"{prefix}{base_name}{suffix}{ext}")
            nib.save(out_nii, out_path)
            print(f"Saved: {out_path}")

        dir_out = self.Args.get('DIROUT', './')
        os.makedirs(dir_out, exist_ok=True)

        write_gz = self.Args.get('write_gzipped_niftis', 0)
        ext = '.nii.gz' if write_gz else '.nii'
        base_name = fn_out.replace('.nii.gz', '').replace('.nii', '')

        # =========================================================
        # 1. Pre-NORDIC Maps (g-factor)
        # =========================================================
        if self.Args.get('save_gfactor_map') == 1:
            g_img = np.abs(self.gfactor).astype(np.float32)
            np.nan_to_num(g_img, copy=False, nan=0.0)
            save_nifti(g_img, "gfactor_", "", self.info)

        # =========================================================
        # 2. Main Image Outputs (Magnitude & Phase)
        # =========================================================
        if self.Args.get('make_complex_nii') == 1:
            # Save Magnitude
            magn_scaled = apply_gain_scaling(self.IMG2, self.Args)
            save_nifti(magn_scaled, "", "_magn", self.info)

            # Save Phase (Requires re-mapping to original phase scale)
            if hasattr(self, 'info_phase') and self.info_phase is not None:
                phase_array = np.angle(self.IMG2)

                # Revert phase to original scaling range if captured during load
                range_norm = self.Args.get('phase_range_norm', 2 * np.pi)
                range_center = self.Args.get('phase_range_center', 0.5)

                phase_scaled = (phase_array / (2 * np.pi) + range_center) * range_norm
                phase_scaled = phase_scaled.astype(np.float32)
                save_nifti(phase_scaled, "", "_phase", self.info_phase)

        else:
            # Standard Magnitude Only
            img_scaled = apply_gain_scaling(self.IMG2, self.Args)
            save_nifti(img_scaled, "", "", self.info)

        # =========================================================
        # 3. Residual Outputs
        # =========================================================
        if self.Args.get('save_residual_matlab') == 1:
            mat_path = os.path.join(dir_out, f"RESIDUAL_{base_name}.mat")
            savemat(mat_path, {'Residual': self.Residual})
            print(f"Saved: {mat_path}")

        if self.Args.get('save_residual_NIFTI') == 1:
            res_scaled = apply_gain_scaling(self.Residual, self.Args)
            save_nifti(res_scaled, "RESIDUAL_", "", self.info)

        # =========================================================
        # 4. Auxiliary Metrics & Logging
        # =========================================================
        if self.Args.get('save_add_info') == 1:
            mat_path = os.path.join(dir_out, f"{base_name}_info.mat")
            # Sanitize Args dict to remove large arrays before saving metadata
            safe_args = {k: v for k, v in self.Args.items() if not isinstance(v, np.ndarray)}
            savemat(mat_path, {'ARG': safe_args})

        # Optional NORDIC Auxiliary Maps
        aux_maps = [
            ('save_add_info_NOISE', 'NOISE', 'gfactor_post_normalization_'),
            ('save_add_info_Component_threshold', 'Component_threshold', 'Component_threshold_'),
            ('save_add_info_energy_removed', 'energy_removed', 'energy_removed_'),
            ('save_add_info_SNR_weight', 'SNR_weight', 'SNR_weight_')
        ]

        for toggle, arg_key, prefix in aux_maps:
            if self.Args.get(toggle) == 1 and arg_key in self.Args:
                aux_data = np.abs(self.Args[arg_key]).astype(np.float32)
                np.nan_to_num(aux_data, copy=False, nan=0.0)
                save_nifti(aux_data, prefix, "", self.info)





