import os
from typing import Callable, Any
import statsmodels.api as sm
import numpy as np
import matplotlib

from Evaluation_Framework.predictions_evaluator.gt_pred_set import GtPredSet

matplotlib.use('Agg')  # Set the backend to 'Agg'
import matplotlib.pyplot as plt

import metric_definitions


def filter_for_error_outliers(gt: list, error: list, outlier_limit) -> (list, list):
    gt_filtered = []
    error_filtered = []
    for idx in range(len(gt)):
        if -outlier_limit < error[idx] < outlier_limit:
            gt_filtered.append(gt[idx])
            error_filtered.append(error[idx])
    return gt_filtered, error_filtered



def scatter_plot_gt_error_pairs(gt_pred_data: GtPredSet, name: str, error_function: Callable[[int, int], Any], out_filepath: str, classwise=False, outlier_limit=None):
    gt_sign, pred_sign = gt_pred_data.get_gt_pred_lists_for_class("traffic_sign")
    gt_vehicle, pred_vehicle = gt_pred_data.get_gt_pred_lists_for_class("vehicle")

    errors_sign = [error_function(gt_value, pred_value) for gt_value, pred_value in zip(gt_sign, pred_sign)]
    if outlier_limit:
        gt_sign, errors_sign = filter_for_error_outliers(gt_sign, errors_sign, outlier_limit)

    errors_vehicle = [error_function(gt_value, pred_value) for gt_value, pred_value in zip(gt_vehicle, pred_vehicle)]
    if outlier_limit:
        gt_vehicle, errors_vehicle = filter_for_error_outliers(gt_vehicle, errors_vehicle, outlier_limit)

    plt.scatter(gt_sign, errors_sign, s=0.5, color='red', label='Traffic Sign')
    plt.scatter(gt_vehicle, errors_vehicle, s=0.5, color='blue', label='Vehicle')
    plt.legend()
    plt.xlabel('Ground Truth Werte')
    plt.ylabel('Error')
    plt.title(name)
    plt.grid(True)
    plt.savefig(out_filepath, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()


def scatter_plot_gt_pred_pairs(gt_pred_data: GtPredSet, name: str,  out_filepath: str):
    gt_sign, pred_sign = gt_pred_data.get_gt_pred_lists_for_class("traffic_sign")
    gt_vehicle, pred_vehicle = gt_pred_data.get_gt_pred_lists_for_class("vehicle")
    plt.scatter(gt_sign, pred_sign, s=0.5, color='red', label='Traffic Sign')
    plt.scatter(gt_vehicle, pred_vehicle, s=0.5, color='blue', label='Vehicle')
    plt.legend()
    plt.xlabel('Ground Truth Werte')
    plt.ylabel('Prediction Werte')
    plt.title(name)
    plt.grid(True)
    plt.xlim(0, 45000)
    plt.ylim(0, 45000)
    plt.savefig(out_filepath, dpi=300, bbox_inches='tight',
                pad_inches=0.5)
    plt.close()


def line_plot_bins_gt_abs_error_pairs(gt_pred_data: GtPredSet, title: str, error_name: str,
                                      error_function: Callable[[int, int], Any], out_filepath: str, classwise=False,
                                      outlier_limit=None, num_bins=8, min_value=0, max_value=40000):
    # Base plot with all classes
    gt, pred = gt_pred_data.get_gt_pred_lists()
    errors = [error_function(gt_value, pred_value) for gt_value, pred_value in zip(gt, pred)]
    if outlier_limit:
        gt, errors = filter_for_error_outliers(gt, errors, outlier_limit)
    sorted_data = sorted(zip(gt, errors), key=lambda x: x[0])
    sorted_gt, sorted_errors = zip(*sorted_data)

    # Bins-based approach
    bins = np.linspace(min_value, max_value, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_means = [
        np.mean([error for gt_value, error in zip(sorted_gt, sorted_errors) if bins[i] <= gt_value < bins[i + 1]]) for i
        in range(num_bins)]

    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, bin_means, label='All classes', color='black')  # Changed legend title to "All classes"

    # If classwise flag is True, plot lines for each class
    if classwise:
        for object_class in gt_pred_data.get_class_names():
            gt, pred = gt_pred_data.get_gt_pred_lists(object_class)
            errors = [error_function(gt_value, pred_value) for gt_value, pred_value in zip(gt, pred)]
            if outlier_limit:
                gt, errors = filter_for_error_outliers(gt, errors, outlier_limit)
            sorted_data = sorted(zip(gt, errors), key=lambda x: x[0])
            sorted_gt, sorted_errors = zip(*sorted_data)

            # Bins-based approach for each class
            bin_means = [np.mean(
                [error for gt_value, error in zip(sorted_gt, sorted_errors) if bins[i] <= gt_value < bins[i + 1]]) for i
                         in range(num_bins)]

            plt.plot(bin_centers, bin_means, label=object_class)  # Use object_class as legend title for class lines

    plt.xlabel('Ground Truth Werte')
    plt.ylabel(error_name)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move the legend outside of the plot
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to make room for the legend outside the plot
    plt.savefig(out_filepath, dpi=300)  # Save the plot to the specified output filepath
    plt.close()


def line_plot_lowess_gt_abs_error_pairs(gt_pred_data: GtPredSet, title: str, error_name: str, error_function: Callable[[int, int], Any], out_filepath: str, classwise=False, outlier_limit=None):
    # Base plot with all classes
    gt, pred = gt_pred_data.get_gt_pred_lists()
    errors = [error_function(gt_value, pred_value) for gt_value, pred_value in zip(gt, pred)]
    if outlier_limit:
        gt, errors = filter_for_error_outliers(gt, errors, outlier_limit)
    sorted_data = sorted(zip(gt, errors), key=lambda x: x[0])
    sorted_gt, sorted_errors = zip(*sorted_data)
    lowess = sm.nonparametric.lowess(sorted_errors, sorted_gt, frac=0.66)
    smooth_gt, smooth_errors = zip(*lowess)

    plt.figure(figsize=(10, 6))
    plt.plot(smooth_gt, smooth_errors, label='All classes', color='black')  # Changed legend title to "All classes"

    # If classwise flag is True, plot lines for each class
    if classwise:
        for object_class in gt_pred_data.get_class_names():
            gt, pred = gt_pred_data.get_gt_pred_lists(object_class)
            errors = [error_function(gt_value, pred_value) for gt_value, pred_value in zip(gt, pred)]
            if outlier_limit:
                gt, errors = filter_for_error_outliers(gt, errors, outlier_limit)
            sorted_data = sorted(zip(gt, errors), key=lambda x: x[0])
            sorted_gt, sorted_errors = zip(*sorted_data)
            lowess = sm.nonparametric.lowess(sorted_errors, sorted_gt, frac=0.01)
            smooth_gt, smooth_errors = zip(*lowess)

            plt.plot(smooth_gt, smooth_errors, label=object_class)  # Use object_class as legend title for class lines

    plt.xlabel('Ground Truth Werte')
    plt.ylabel(error_name)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move the legend outside of the plot
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to make room for the legend outside the plot
    plt.savefig(out_filepath, dpi=300)  # Save the plot to the specified output filepath
    plt.close()




def plot_relative_error(gt_pred_data: GtPredSet, output_dir: str, show_plot_popup=False, num_bins=None):
    filepath_scatter = os.path.join(output_dir, "relative_error_scatter.png")
    scatter_plot_gt_error_pairs(gt_pred_data, "Relative Error", metric_definitions.relative_error, filepath_scatter)
    filepath_scatter = os.path.join(output_dir, "relative_error_scatter_outlier_th_2.png")
    scatter_plot_gt_error_pairs(gt_pred_data, "Relative Error", metric_definitions.relative_error, filepath_scatter, outlier_limit=2.0)

def plot_abs_relative_error_lines(gt_pred_data: GtPredSet, output_dir: str, show_plot_popup=False, num_bins=None):
    filepath_plot = os.path.join(output_dir, "abs_relative_error_lines_8_bins.png")
    line_plot_bins_gt_abs_error_pairs(gt_pred_data, "Abs. Relative Error (8 bins)", "Abs. relative error",
                                        metric_definitions.abs_relative_error, filepath_plot, classwise=True, num_bins=8)
    filepath_plot = os.path.join(output_dir, "abs_relative_error_lines_8_bins_outlier_th_2.png")
    line_plot_bins_gt_abs_error_pairs(gt_pred_data, "Abs. Relative Error (8 bins) (outlier th. 2.0)", "Abs. relative error",
                                        metric_definitions.abs_relative_error, filepath_plot, classwise=True, outlier_limit=2.0, num_bins=8)

    filepath_plot = os.path.join(output_dir, "abs_relative_error_lines_80_bins.png")
    line_plot_bins_gt_abs_error_pairs(gt_pred_data, "Abs. Relative Error (80 bins)", "Abs. relative error",
                                        metric_definitions.abs_relative_error, filepath_plot, classwise=True, num_bins=80)
    filepath_plot = os.path.join(output_dir, "abs_relative_error_lines_80_bins_outlier_th_2.png")
    line_plot_bins_gt_abs_error_pairs(gt_pred_data, "Abs. Relative Error (80 bins) (outlier th. 2.0)", "Abs. relative error",
                                        metric_definitions.abs_relative_error, filepath_plot, classwise=True, outlier_limit=2.0, num_bins=80)
    filepath_plot = os.path.join(output_dir, "abs_relative_error_lines_80_bins_start_50000.png")
    line_plot_bins_gt_abs_error_pairs(gt_pred_data, "Abs. Relative Error (80 bins) (starting at 5000)",
                                      "Abs. relative error",
                                      metric_definitions.abs_relative_error, filepath_plot, classwise=True,
                                      num_bins=80, min_value=5000)


def plot_abs_error_lines(gt_pred_data: GtPredSet, output_dir: str):
    filepath_plot = os.path.join(output_dir, "abs_error_lines_80_bins.png")
    line_plot_bins_gt_abs_error_pairs(gt_pred_data, "Abs. error (in mm) (80 bins)", "Abs. error", metric_definitions.abs_metric_error, filepath_plot, classwise=True, num_bins=80)
    filepath_plot = os.path.join(output_dir, "abs_error_lines_outlier_th_10000_80_bins.png")
    line_plot_bins_gt_abs_error_pairs(gt_pred_data, "Abs. error (in mm) (80 bins) (outlier th. 1000)", "Abs. error",
                                        metric_definitions.abs_metric_error, filepath_plot, classwise=True, outlier_limit=10000.0, num_bins=80)


def plot_gt_pred_values(gt_pred_data: GtPredSet, output_dir: str):
    filepath_plot = os.path.join(output_dir, "gt_pred_scatter.png")
    scatter_plot_gt_pred_pairs(gt_pred_data, "Groundtruth x Prediction", filepath_plot)
