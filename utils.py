import scipy.linalg
import numpy as np


def get_numpy_dtype(nib_obj):
    """Safely extracts the original NIfTI data type."""
    if nib_obj is None:
        return np.float32
    return nib_obj.get_data_dtype()

def apply_gain_scaling(data_array, Args):
    """Replicates MATLAB's 99th percentile 32k gain scaling."""
    data_abs = np.abs(data_array)
    np.nan_to_num(data_abs, copy=False, nan=0.0)

    gain_level = 0.0
    if Args.get('full_dynamic_range', 0) != 0:
        sn_scale = 2.0 * np.percentile(data_abs, 99)
        if sn_scale > 0:
            gain_level = np.floor(np.log2(32000.0 / sn_scale))

    # Apply scaling and strictly cast to float32 (single)
    scaled_data = (data_abs * (2 ** gain_level)).astype(np.float32)
    return scaled_data





def calculate_NORDIC_threshold(KSP2_NOISE, arg):
    """
    Calculates the baseline noise standard deviation and simulates the
    expected Marchenko-Pastur threshold using Monte Carlo SVD.
    """

    ### 1. Base Noise Estimation ###
    if arg.get('noise_volume_last', 0) > 0 and KSP2_NOISE is not None:
        # Create a boolean mask of valid data without mutating the array
        valid_mask = np.isfinite(KSP2_NOISE) & (KSP2_NOISE != 0)

        if np.any(valid_mask):
            # ddof=1 matches MATLAB's default behavior for std()
            arg['measured_noise'] = np.std(KSP2_NOISE[valid_mask], ddof=1)
        else:
            arg['measured_noise'] = 1.0
    else:
        arg['measured_noise'] = 1.0

    # Magnitude correction for complex data
    use_magn = arg.get('use_magn_for_gfactor', 1)
    mag_only = arg.get('magnitude_only', 0)

    if use_magn == 0 and not (mag_only == 1):
        arg['measured_noise'] /= np.sqrt(2)

    ### 2. Kernel Size Adjustments ###
    ks_pca = arg.get('kernel_size_PCA', None)
    if ks_pca is not None and len(ks_pca) > 0:
        arg['kernel_size'] = list(ks_pca)

    matdim = arg['matdim']  # (x, y, z, t, ...)
    nt = matdim[3]
    nz = matdim[2]

    # Adjust if the number of slices is less than or equal to cubic kernel depth
    if nz <= arg['kernel_size'][2]:
        default_k2 = int(np.round((nt * 11 / nz) ** 0.5))
        arg['kernel_size'] = [default_k2, default_k2, nz]

    ### 3. Setup Processing Parameters ###
    arg['patch_average'] = 0
    arg['patch_average_sub'] = arg.get('NORDIC_patch_overlap', 1)
    arg['LLR_scale'] = 1
    arg['soft_thrs'] = None

    ### 4. Monte Carlo SVD Simulation ###
    prod_k = int(np.prod(arg['kernel_size']))
    nvr_thresh_sum = 0.0

    # scipy.linalg.svdvals is drastically faster than full SVD
    for _ in range(10):
        rand_mat = np.random.randn(prod_k, nt)
        s_vals = scipy.linalg.svdvals(rand_mat)
        nvr_thresh_sum += s_vals[0]

    ### 5. Final Threshold Scaling ###
    factor_error = arg.get('factor_error', 1.0)

    if mag_only != 1:
        # sqrt(2) due to complex, factor_error for g-factor underestimate
        arg['NVR_threshold'] = (nvr_thresh_sum / 10.0) * np.sqrt(2) * arg['measured_noise'] * factor_error
    else:
        arg['NVR_threshold'] = (nvr_thresh_sum / 10.0) * arg['measured_noise'] * factor_error

    ### 6. Threshold Overrides ###
    if arg.get('MP', 0) > 0:
        arg['soft_thrs'] = 10

    soft_thrs_in = arg.get('soft_thrs_in', None)
    if soft_thrs_in is not None:
        arg['soft_thrs'] = soft_thrs_in

    return arg