import numpy as np
from gmi_analysis_for_ED import (
    get_projected_ensemble,
    get_p_phi,
    Ensemble,
    linear_fit,
    calc_entropy,
)
from scipy.optimize import curve_fit

def load_time_avg_ensemble(file_path):
    data = np.load(file_path, allow_pickle=True)
    q = data['q']
    p = data['p']
    phi = data['phi']
    return q, p, phi

data_dir = 'deep_thermalization_analysis/projected_ensemble_data'

for N in range(8, 16, 2):
    file_path = f'{data_dir}/projected_ensemble_L_{N}_tavg_t100-5000_site{N//2}_qpphi.npz'
    q, p, phi = load_time_avg_ensemble(file_path)
    
    ensemble = Ensemble(q, p, phi, ensemble_name=f'Projected Ensemble N={N}')
    ensemble.analyze(plot=False)
    ensemble.save_ent()