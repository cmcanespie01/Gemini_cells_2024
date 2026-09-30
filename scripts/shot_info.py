import time
import numpy as np
import matplotlib.pyplot as plt 
from pathlib import Path
from LAMP import Experiment

# create experiment object
ROOT_FOLDER = Path(__file__).resolve().parents[1] # absolute path to the experiment config files and other subfolders - 2 directories up in this case
ex = Experiment(ROOT_FOLDER)

# below is left over from Gemini DAQ
#shot_dict = {'date': 20260821, 'run': 'run002', 'shotnum': 1}
#shot_info = ex.DAQ.get_shot_info(shot_dict)
#print(shot_info)

# Run settings json file
# currently errors out due to unexpected second line?
# run_settings = ex.DAQ.get_settings(20260821, 'run002')
# print(run_settings)

run_scalars = ex.DAQ.get_scalars(20260821, 'run002')
print(run_scalars)