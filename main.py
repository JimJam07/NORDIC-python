from nordic import Nordic
import os


# file Names
mag_file = "20210812_142005ejaep2dboldpt36xpt36xpt8isos028a001.nii.gz"
phase_file = "20210812_142005ejaep2dboldpt36xpt36xpt8isos029a001.nii.gz"

Dir = "./data"


mag_in = os.path.join(Dir, mag_file)
phase_in = os.path.join(Dir, phase_file)

Args = {
    "temporal_phase": 1,
    "phase_filter_width": 10,
    "FN_OUT": "NORDIC_"+ mag_file[:-7],
    'save_gfactor_map': 1,
    "DIROUT": os.getcwd() + "/data/",
}

Nordic = Nordic(mag_in, phase_in, Args=Args)
