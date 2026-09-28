import os
import numpy as np

class base_nordic:
    def __init__(self):
        self.Args = {
            "DIROUT": os.getcwd() + "/",
            "noise_volume_last": 0,
            "factor_error": 1.0,
            "full_dynamic_range": 0,
            "temporal_phase": 1,
            "data_has_zero_elements": 0,
            "write_gzipped_niftis": 0,
            "phase_filter_width": 3,
            "NORDIC_patch_overlap": 2,
            "gfactor_patch_overlap": 2,
            "kernel_size_gfactor": [14, 14, 1, 90],
            "kernel_size_PCA": [],
            "phase_slice_average_for_kspace_centering": 1,
            "magnitude_only": 0,
            "save_gfactor_map": 0,
            "use_generic_NII_read": 0,
            "NORDIC": 1,
            "MP": 0,
            "LLR_scale": 1,
            "NVR_threshold": 1,
            "patch_average": 0,
            "phase_filter_name": "DK",
            "calculate_residual": 0,
            "save_residual_matlab": 0,
            "save_residual_NIFTI": 0,
            "make_complex_nii": 0,
            "save_add_info": 0,
            "use_magn_for_gfactor": 0,
            "soft_thrs_in": [],
            "save_add_info_NOISE": 0,
            "save_add_info_Component_threshold": 0,
            "save_add_info_energy_removed": 0,
            "save_add_info_SNR_weight": 0,
            "min_vols": 6,
            "decorr_kernel": np.array([
                    [0.03, 0.04, 0.05, 0.04, 0.03],
                    [0.04, 0.08, 0.01, 0.08, 0.04],
                    [0.05, 0.01, 0.00, 0.01, 0.05],
                    [0.04, 0.08, 0.01, 0.08, 0.04],
                    [0.03, 0.04, 0.05, 0.04, 0.03]
            ], dtype=np.float32)
        }

        self.Args['decorr_kernel'] /= np.sum(self.Args['decorr_kernel'])