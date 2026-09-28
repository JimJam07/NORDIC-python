import numpy as np
from LLR import sub_LLR_Processing


def g_factor_processing(KSP2, arg):
    """
    Prepares the data and calculates the g-factor using the v2 LLR processing.

    Parameters:
        KSP2: 4D or 5D numpy array of complex image data
        arg: dictionary containing configuration parameters
    Returns:
        gfactor: processed noise/g-factor map
    """

    ks_g = arg.get('kernel_size_gfactor', [])


    if len(ks_g) < 4:
        t_limit = 90
    else:
        # MATLAB index 4 -> Python index 3
        t_limit = int(ks_g[3])

    t_max = min(t_limit, KSP2.shape[3])

    if KSP2.ndim >= 5:
        KSP2_sub = KSP2[:, :, :, :t_max, 0].copy()
    else:
        KSP2_sub = KSP2[:, :, :, :t_max].copy()

    np.nan_to_num(KSP2_sub, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    if len(ks_g) >= 3:
        arg['kernel_size'] = [ks_g[0], ks_g[1], ks_g[2]]

    arg['patch_average_sub'] = arg.get('gfactor_patch_overlap')
    arg['soft_thrs'] = 10

    print('estimating g-factor ...')

    KSP_recon, arg, KSP_weight, NOISE, Component_threshold, energy_removed, SNR_weight = \
        sub_LLR_Processing(KSP2_sub, arg)

    print('completed estimating g-factor')

    mask = KSP_weight > 0
    gfactor = np.zeros_like(NOISE)

    np.divide(NOISE, KSP_weight, out=gfactor, where=mask)
    np.sqrt(gfactor, out=gfactor)

    if np.any(gfactor == 0):
        np.nan_to_num(gfactor, copy=False, nan=0.0)

        # Calculate median of non-zero elements
        nonzero_mask = gfactor != 0
        if np.any(nonzero_mask):
            med_val = np.median(gfactor[nonzero_mask])
            # Apply median to all values < 1 (which includes the 0s)
            gfactor[gfactor < 1] = med_val

        arg['data_has_zero_elements'] = 1

    if arg.get('MP') == 2:
        gfactor = np.ones_like(gfactor)


    return gfactor