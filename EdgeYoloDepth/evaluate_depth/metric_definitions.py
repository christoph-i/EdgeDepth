import numpy as np

# TODO load those centrally from eval framework or other unified source 

def relative_error(gt, pred):
    if gt == 0:
        raise ArithmeticError("Computing error failed. Invalid GT value == 0 present.")
    return (pred - gt) / gt

def abs_relative_error(gt, pred):
    if gt == 0:
        raise ArithmeticError("Computing error failed. Invalid GT value == 0 present.")
    return np.abs(gt - pred) / gt


def metric_error(gt, pred):
    if gt == 0:
        raise ArithmeticError("Computing error failed. Invalid GT value == 0 present.")
    return pred - gt

def abs_metric_error(gt, pred):
    if gt == 0:
        raise ArithmeticError("Computing error failed. Invalid GT value == 0 present.")
    return abs(pred - gt)


# cod partially from https://github.com/ialhashim/DenseDepth/blob/master/utils.py


def aggregate_mean_abs_rel(gt: list, pred: list):
    pred = np.stack(pred, axis=0)
    gt = np.stack(gt, axis=0)
    return np.mean(np.abs(gt - pred) / gt)

def aggregate_mean_abs_rel(gt: list, pred: list, stddev_var=False):
    pred = np.stack(pred, axis=0)
    gt = np.stack(gt, axis=0)
    abs_rel = np.abs(gt - pred) / gt
    metric = np.mean(abs_rel)
    stddev = np.std(abs_rel)
    var = np.var(abs_rel)
    if stddev_var:
        return metric, stddev, var
    else:
        return metric

def aggregate_mean_abs_metric(gt: list, pred: list):
    pred = np.stack(pred, axis=0)
    gt = np.stack(gt, axis=0)
    return np.mean(np.abs(pred - gt))

def aggregate_mean_abs_metric(gt: list, pred: list, stddev_var=False):
    pred = np.stack(pred, axis=0)
    gt = np.stack(gt, axis=0)
    abs_diff = np.abs(pred - gt)
    metric = np.mean(abs_diff)
    stddev = np.std(abs_diff)
    var = np.var(abs_diff)
    if stddev_var:
        return metric, stddev, var
    else:
        return metric

def aggregate_rmse(gt: list, pred: list):
    pred = np.stack(pred, axis=0)
    gt = np.stack(gt, axis=0)
    rmse = (gt - pred) ** 2
    return np.sqrt(rmse.mean())

def aggregate_rmse(gt: list, pred: list, stddev_var=False):
    pred = np.stack(pred, axis=0)
    gt = np.stack(gt, axis=0)

    # Calculate individual RMSEs for each sample
    squared_diff = (gt - pred) ** 2
    individual_rmses = np.sqrt(squared_diff)

    # Calculate the mean RMSE
    metric = np.sqrt(np.mean(squared_diff))

    # Calculate stddev and variance of the individual RMSEs
    stddev = np.std(individual_rmses)
    var = np.var(individual_rmses)
    if stddev_var:
        return metric, stddev, var
    else:
        return metric

def aggregate_a1_threshold(gt: list, pred: list):
    pred = np.stack(pred, axis=0)
    gt = np.stack(gt, axis=0)
    thresh = np.maximum((gt / pred), (pred / gt))
    return (thresh < 1.25).mean()

def aggregate_a1_threshold(gt: list, pred: list, stddev_var=False):
    pred = np.stack(pred, axis=0)
    gt = np.stack(gt, axis=0)
    thresh = np.maximum((gt / pred), (pred / gt))
    within_thresh = (thresh < 1.25)
    metric = within_thresh.mean()
    stddev = np.std(within_thresh)
    var = np.var(within_thresh)
    if stddev_var:
        return metric, stddev, var
    else:
        return metric

def aggregate_a2_threshold(gt: list, pred: list):
    pred = np.stack(pred, axis=0)
    gt = np.stack(gt, axis=0)
    thresh = np.maximum((gt / pred), (pred / gt))
    return (thresh < 1.25 ** 2).mean()

def aggregate_a3_threshold(gt: list, pred: list):
    pred = np.stack(pred, axis=0)
    gt = np.stack(gt, axis=0)
    thresh = np.maximum((gt / pred), (pred / gt))
    return (thresh < 1.25 ** 3).mean()



def aggregate_mean_errors(gt: list, pred: list):
    a1 = aggregate_a1_threshold(gt, pred)
    a2 = aggregate_a2_threshold(gt, pred)
    a3 = aggregate_a3_threshold(gt, pred)
    abs_rel = aggregate_mean_abs_rel(gt, pred)
    rmse = aggregate_rmse(gt, pred)

    pred_stacked = np.stack(pred, axis=0)
    gt_stacked = np.stack(gt, axis=0)
    log_10 = (np.abs(np.log10(gt_stacked)-np.log10(pred_stacked))).mean()

    return a1, a2, a3, abs_rel, rmse, log_10



# def rmse_error(gt, pred):
#     return (pred - gt) ** 2
#
# def log_rmse_error(gt, pred):
#     if gt > 0 and pred > 0:
#         return (np.log(pred + 1) - np.log(gt + 1)) ** 2
#     return None