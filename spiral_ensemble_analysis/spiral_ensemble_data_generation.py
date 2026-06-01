import numpy as np
import pandas as pd

SAVE_PATH = './spiral_ensemble_analysis/spiral_ensemble_data/'


def generate_spiral_ensemble_data(delta, num_points=10**5):
    q = np.ones(num_points) / num_points
    p = np.random.rand(num_points)
    epsilon = np.random.uniform(-delta, delta, num_points)
    phi = np.mod(2 * np.pi * p + epsilon, 2 * np.pi)
    data = pd.DataFrame({'q': q, 'p': p, 'phi': phi})
    data.to_csv(f'{SAVE_PATH}spiral_ensemble_delta={delta}.csv', index=False)


deltas = [0.2, 0.4, 0.6, 0.8, 1.0]
num_points = 10**6

for delta in deltas:
    generate_spiral_ensemble_data(delta, num_points)
