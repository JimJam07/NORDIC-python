import numpy as np
import itertools
import concurrent.futures
import scipy.linalg

# NOTE: Ensure `process_subset` is imported/defined in your module.
# from your_module import process_subset

def sub_LLR_Processing(A, arg):
    """
    Parallel Local Low-Rank (LLR) patch processing orchestrator (v2).
    """

    ### 1. REGION CONFIGURATION GENERATION ###
    w = np.asarray(arg['kernel_size'][:3], dtype=int)
    patch_avg_sub = arg.get('patch_average_sub', 1)


    steps = np.floor(w / patch_avg_sub).astype(int)
    o = [np.arange(0, w[i], steps[i]) if steps[i] > 0 else np.array([0]) for i in range(3)]

    A_shape0 = A.shape[0]
    x_configs = []

    for ostepx in o[0]:
        s1 = (A_shape0 - ostepx) // w[0]

        # 0-based nstepx indexing
        for nstepx in range(s1):
            x_configs.append([nstepx, ostepx])

        range_used_end = (s1 * w[0]) + ostepx
        if range_used_end < A_shape0:
            edgewidth = A_shape0 - range_used_end
            x_configs.append([s1, edgewidth - w[0]])

    all_combinations = itertools.product(x_configs, o[1], o[2])
    REGIONS = np.array([[x[0], x[1], y, z] for x, y, z in all_combinations])

    ### 2. ARRAY INITIALIZATION ###
    OUT = np.zeros_like(A)
    spatial_shape = A.shape[:3]

    # Initialise arrays
    AUX_OUT = {
        'KSP2_weight': np.zeros(spatial_shape, dtype=np.float32),
        'NOISE': np.zeros(spatial_shape, dtype=np.float32),
        'KSP2_tmp_update_threshold': np.zeros(spatial_shape, dtype=np.float32),
        'energy_removed': np.zeros(spatial_shape, dtype=np.float32),
        'SNR_weight': np.zeros(spatial_shape, dtype=np.float32)
    }

    arg_in = {
        'kernel_size': arg['kernel_size'],
        'Component_threshold_to_use': None,
        'soft_thrs': arg.get('soft_thrs', None),
        'NVR_threshold': arg.get('NVR_threshold', 0),
        'calculate_residual': arg.get('calculate_residual', 0)
    }

    ### 3. PARALLEL EXECUTION & ACCUMULATION ###
    def worker(region_range):
        """Worker closure to avoid passing massive A matrix repeatedly."""
        local_arg_in = arg_in.copy()
        local_arg_in['RANGE'] = region_range
        # process_subset must return the reconstructed patch (AA) and its AUX metrics
        AA, AUX, _ = process_subset(A, local_arg_in)
        return region_range, AA, AUX

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(worker, reg): reg for reg in REGIONS}

        for future in concurrent.futures.as_completed(futures):
            region_range, AA, AUX = future.result()

            nstepx = region_range[0]
            ostepx = region_range[1]
            w1 = arg_in['kernel_size'][0]

            # Reconstruct the 0-based spatial slice along the first dimension
            start_x = (nstepx * w1) + ostepx
            slc = slice(start_x, start_x + w1)

            # Accumulate results in-place
            OUT[slc, ...] += AA
            AUX_OUT['KSP2_weight'][slc, ...] += AUX['KSP2_weight']
            AUX_OUT['KSP2_tmp_update_threshold'][slc, ...] += AUX['KSP2_tmp_update_threshold']
            AUX_OUT['energy_removed'][slc, ...] += AUX['energy_removed']
            AUX_OUT['SNR_weight'][slc, ...] += AUX['SNR_weight']
            AUX_OUT['NOISE'][slc, ...] += AUX['NOISE']

    ### 4. OUTPUT ROUTING ###
    return (
        OUT,
        arg,
        AUX_OUT['KSP2_weight'],
        AUX_OUT['NOISE'],
        AUX_OUT['KSP2_tmp_update_threshold'],
        AUX_OUT['energy_removed'],
        AUX_OUT['SNR_weight']
    )


def process_subset(IN, arg):
    """
    Extracts the spatial subset block for parallel processing.
    """

    ### 1. SUBSET SLICING ###
    w1 = int(arg['kernel_size'][0])


    nstepx = int(arg['RANGE'][0])
    ostepx = int(arg['RANGE'][1])

    start_x = (nstepx * w1) + ostepx

    # Create a zero-copy memory view of the array subset
    AA = IN[start_x: start_x + w1, ...]

    ### 2. CORE CALCULATION ###
    OUT, OUT_ARRAY = calc_subset(AA, arg)

    return OUT, OUT_ARRAY, arg

def calc_subset(IN, arg):
    """
    Core subset calculator for LLR processing.
    Processes spatial patches and accumulates NORDIC MPPCA metrics.
    """

    ### 1. PARAMETER SETUP ###
    comp_thresh_full = arg.get('Component_threshold_to_use', None)
    lambda2 = arg.get('NVR_threshold', 0)
    soft_thrs = arg.get('soft_thrs', None)

    w2 = int(arg['kernel_size'][1])
    w3 = int(arg['kernel_size'][2])

    ostepy = int(arg['RANGE'][2])
    ostepz = int(arg['RANGE'][3])

    s2 = (IN.shape[1] - ostepy) // w2
    s3 = (IN.shape[2] - ostepz) // w3

    spatial_shape = IN.shape[:3]

    IN_UPDATE = np.zeros_like(IN)
    KSP2_weight = np.zeros(spatial_shape, dtype=np.float64)
    KSP2_tmp_update_threshold = np.zeros(spatial_shape, dtype=np.float64)
    energy_removed = np.zeros(spatial_shape, dtype=np.float64)
    SNR_weight = np.zeros(spatial_shape, dtype=np.float64)
    NOISE = np.zeros(spatial_shape, dtype=np.float64)

    n2_idx = list(range(ostepy, ostepy + (s2 * w2), w2))
    edge_n2 = IN.shape[1] - w2
    if edge_n2 not in n2_idx and edge_n2 >= 0:
        n2_idx.append(edge_n2)

    n3_idx = list(range(ostepz, ostepz + (s3 * w3), w3))
    edge_n3 = IN.shape[2] - w3
    if edge_n3 not in n3_idx and edge_n3 >= 0:
        n3_idx.append(edge_n3)

    ### 4. CORE PROCESSING LOOP ###
    # itertools.product flattens the 2D loop into a fast, C-optimized iterator
    for n2, n3 in itertools.product(n2_idx, n3_idx):

        slc_2 = slice(n2, n2 + w2)
        slc_3 = slice(n3, n3 + w3)

        # Handle dynamic prunning threshold
        comp_thresh_tmp = None
        if comp_thresh_full is not None and comp_thresh_full.size > 0:
            # We slice the first dimension fully, and specify Y and Z
            comp_thresh_tmp = np.min(comp_thresh_full[:, slc_2, slc_3, :])

        # Extract the local block memory view
        local_block = IN[:, slc_2, slc_3, :]

        # Execute SVD and MPPCA tracking
        update, idx_scalar, energy_scrub_scalar, SNR_weight_scalar, NOISE_scalar = \
            LLR_NORDIC_MPPCA(local_block, lambda2, soft_thrs, comp_thresh_tmp, arg)

        ### 5. IN-PLACE ACCUMULATION ###
        # This completely replaces the slow MATLAB `subfunction_update_matrix_local`.
        # NumPy cleanly broadcasts the scalar outputs across the 3D sliced volumes.
        IN_UPDATE[:, slc_2, slc_3, :] += update
        KSP2_weight[:, slc_2, slc_3] += 1.0  # Assuming standard uniform block weight
        KSP2_tmp_update_threshold[:, slc_2, slc_3] += idx_scalar
        energy_removed[:, slc_2, slc_3] += energy_scrub_scalar
        SNR_weight[:, slc_2, slc_3] += SNR_weight_scalar
        NOISE[:, slc_2, slc_3] += NOISE_scalar

    ### 6. PACKAGE OUTPUT ###
    OUT_ARRAY = {
        'KSP2_weight': KSP2_weight,
        'KSP2_tmp_update_threshold': KSP2_tmp_update_threshold,
        'energy_removed': energy_removed,
        'SNR_weight': SNR_weight,
        'NOISE': NOISE
    }

    return IN_UPDATE, OUT_ARRAY


def LLR_NORDIC_MPPCA(IN, lambda2, soft_thrs, comp_thresh_tmp, arg):
    """
    Core SVD thresholding engine. Applies Marchenko-Pastur (MPPCA) or
    Hard/Soft thresholding to singular values of the spatial patch.
    """

    ### 1. FLATTEN AND SVD ###
    tmp1 = IN.reshape(-1, IN.shape[3])

    # Vh is already V transposed/Hermitian
    U, S, Vh = scipy.linalg.svd(tmp1, full_matrices=False)

    # Keep a copy of original S for SNR calculation later
    S_orig_sum = np.sum(S)

    # Apply override threshold index if provided (adjusting for 0-based Python indexing)
    if comp_thresh_tmp is not None:
        idx_override = max(0, min(len(S) - 1, int(comp_thresh_tmp) - 1))
        lambda2 = S[idx_override]

    ### 2. THRESHOLDING LOGIC ###
    NOISE = 0.0

    if soft_thrs is None or np.size(soft_thrs) == 0:
        # Standard NORDIC Hard Thresholding
        mask = S < lambda2
        idx = np.sum(mask)

        energy_scrub = np.sqrt(np.sum(S[mask]) / S_orig_sum) if S_orig_sum > 0 else 0.0
        S[mask] = 0
        t = idx

    elif soft_thrs == 10:
        # MPPCA Thresholding
        MM, NNN = tmp1.shape
        R = min(MM, NNN)

        scaling = (max(MM, NNN) - np.arange(R)) / NNN
        vals = (S[:R] ** 2) / NNN

        # Eq 1: First estimation of Sigma^2
        csum = np.cumsum(vals[::-1])
        cmean = csum / np.arange(1, R + 1)
        sigmasq_1 = cmean[::-1] / scaling

        # Eq 2: Second estimation of Sigma^2
        gamma = (MM - np.arange(R)) / NNN
        rangeMP = 4 * np.sqrt(gamma)
        rangeData = vals[:R] - vals[R - 1]

        with np.errstate(divide='ignore', invalid='ignore'):
            sigmasq_2 = rangeData / rangeMP

        # Find where sigmasq_2 drops below sigmasq_1
        condition = sigmasq_2 < sigmasq_1
        if np.any(condition):
            t = np.argmax(condition)
        else:
            t = R - 1

        NOISE = sigmasq_2[t]
        idx = len(S) - t

        energy_scrub = np.sqrt(np.sum(S[t:]) / S_orig_sum) if S_orig_sum > 0 else 0.0
        S[t:] = 0

    else:
        # Soft Thresholding via percentage count
        idx = np.sum(S < lambda2)
        energy_scrub = 0.0

        cut_idx = max(1, len(S) - int(np.floor(idx * soft_thrs)))
        S[cut_idx:] = 0
        t = idx

    ### 3. RECONSTRUCTION & METRICS ###

    tmp1_recon = (U * S) @ Vh

    if arg.get('calculate_residual') == 2:
        tmp1_recon -= tmp1

    tmp1_recon = tmp1_recon.reshape(IN.shape)


    t_idx = max(0, int(t) - 1)
    if S[t_idx] > 0:
        SNR_weight = S[0] / S[t_idx]
    else:
        SNR_weight = 1.0

    return tmp1_recon, idx, energy_scrub, SNR_weight, NOISE

