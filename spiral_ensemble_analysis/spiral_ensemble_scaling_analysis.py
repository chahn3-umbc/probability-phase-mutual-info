from ensemble_scaling_analysis import ensemble_analysis_class as eac
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

DATA_PATH = 'spiral_ensemble_analysis/spiral_ensemble_data/'
RESULTS_PATH = 'spiral_ensemble_analysis/spiral_ensemble_results/'

with os.scandir(DATA_PATH) as entries:
    for entry in entries:
        if entry.name.endswith('.csv') and entry.is_file():

            ensemble_name = entry.name.split('.')[0]+'.'+entry.name.split('.')[1]

            spiral_ens = eac.EnsembleAnalysis(ensemble_name, data_file=DATA_PATH+entry.name, results_dir=RESULTS_PATH, fit_points=(7, 8, 9))
            spiral_ens.analyze(plot_type=False)