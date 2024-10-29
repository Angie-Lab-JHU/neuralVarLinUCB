import pickle

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from tqdm.notebook import tqdm

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(constrained_layout=True)
ax.plot([0, 1], [0, 1], "--", color="grey", label = "Ideal calibration")

conf_level_lower_bounds = np.arange(start=0.025, stop=0.5, step=0.025)
conf_levels = 1 - 2 * conf_level_lower_bounds


def test(labels, rba_outputs, method, color):
    targets_pred = rba_outputs[:,0]
    stdevs = rba_outputs[:,1]
    sigma = stdevs**2
    sharpness = np.sqrt(np.mean(sigma))
    print(sharpness)
    
    targets_test = labels
    residuals = targets_pred - targets_test.reshape(-1)
    norm = stats.norm(loc=0, scale=1)

    # Computing calibration
    def calculate_density(percentile):
        """
        Calculate the fraction of the residuals that fall within the lower
        `percentile` of their respective Gaussian distributions, which are
        defined by their respective uncertainty estimates.
        """
        # Find the normalized bounds of this percentile
        upper_bound = norm.ppf(percentile)

        # Normalize the residuals so they all should fall on the normal bell curve
        normalized_residuals = residuals.reshape(-1) / stdevs.reshape(-1)

        # Count how many residuals fall inside here
        num_within_quantile = 0
        for resid in normalized_residuals:
            if resid <= upper_bound:
                num_within_quantile += 1

        # Return the fraction of residuals that fall within the bounds
        density = num_within_quantile / len(residuals)
        return density

    predicted_pi = np.linspace(0, 1, 20)
    observed_pi = [calculate_density(quantile) for quantile in tqdm(predicted_pi, desc="Calibration")]

    calibration_error = ((predicted_pi - observed_pi) ** 2).sum()
    print(calibration_error)
    ax.plot(predicted_pi, observed_pi, '-o', label=method, color = color)


with open("linear_UCB/labels_val_19999.pkl", "rb") as fp:
    linear_labels = np.squeeze(pickle.load(fp))

with open("linear_UCB/outputs_val_19999.pkl", "rb") as fp:
    linear_outputs = np.squeeze(pickle.load(fp))

with open("linear_neural_UCB/labels_val_19999.pkl", "rb") as fp:
    linear_neural_labels = np.squeeze(pickle.load(fp))

with open("linear_neural_UCB/outputs_val_19999.pkl", "rb") as fp:
    linear_neural_outputs = np.squeeze(pickle.load(fp))

with open("neural_UCB/labels_val_19999.pkl", "rb") as fp:
    neural_labels = np.squeeze(pickle.load(fp))

with open("neural_UCB/outputs_val_19999.pkl", "rb") as fp:
    neural_outputs = np.squeeze(pickle.load(fp))

with open("neural_MLE/labels_val_19999.pkl", "rb") as fp:
    ours_labels = np.squeeze(pickle.load(fp))

with open("neural_MLE/outputs_val_19999.pkl", "rb") as fp:
    ours_outputs = np.squeeze(pickle.load(fp))

neural_outputs[:,1,:] = neural_outputs[:,1,:]/np.sqrt(0.001)

test(linear_labels[:,0], linear_outputs[:,:,0], "LinUCB", "tab:blue")
test(neural_labels[:,0], neural_outputs[:,:,0], "NeuralUCB", "tab:orange")
test(linear_neural_labels[:,0], linear_neural_outputs[:,:,0], "Neural-LinUCB", "tab:red")
test(ours_labels[:,0], ours_outputs[:,:,0], "Neural-$\sigma^2$-LinUCB", "blue")

plt.legend(loc='upper left', fontsize=14)
ax.set_xlabel("Predicted Confidence Level")
ax.set_ylabel("Observed Confidence Level")
# plt.tight_layout()
plt.savefig("out/calib_main_6.pdf")