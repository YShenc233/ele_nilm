import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def acc_precision_recall_f1_score(status, status_pred):
    """
    Calculate classification metrics for NILM status prediction.
    
    Args:
        status: Ground truth ON/OFF status
        status_pred: Predicted ON/OFF status
        
    Returns:
        acc: Accuracy scores
        precision: Precision scores
        recall: Recall scores
        f1: F1 scores
    """
    assert status.shape == status_pred.shape
    
    if type(status) != np.ndarray:
        status = status.detach().cpu().numpy().squeeze()   
    if type(status_pred) != np.ndarray: 
        status_pred = status_pred.detach().cpu().numpy().squeeze()
    
    status = status.reshape(status.shape[0], -1)
    status_pred = status_pred.reshape(status_pred.shape[0], -1)
    accs, precisions, recalls, f1_scores = [], [], [], []

    for i in range(status.shape[0]):
        # Make sure we have both classes for confusion matrix
        uniq_status = np.unique(status[i, :])
        uniq_pred = np.unique(status_pred[i, :])
        
        # Check if both ground truth and predictions contain both classes
        has_both_labels = (len(uniq_status) == 2 and len(uniq_pred) == 2)
        
        if has_both_labels:
            tn, fp, fn, tp = confusion_matrix(status[i, :], status_pred[i, :], labels=[0, 1]).ravel()
        else:
            # Handle cases where predictions or ground truth only have one class
            tp, fp, fn, tn = 0, 0, 0, 0
            for j in range(len(status[i, :])):
                if status[i, j] == 1 and status_pred[i, j] == 1:
                    tp += 1
                elif status[i, j] == 1 and status_pred[i, j] == 0:
                    fn += 1
                elif status[i, j] == 0 and status_pred[i, j] == 1:
                    fp += 1
                elif status[i, j] == 0 and status_pred[i, j] == 0:
                    tn += 1
            
        acc = (tn + tp) / (tn + fp + fn + tp)
        precision = tp / np.max((tp + fp, 1e-9))
        recall = tp / np.max((tp + fn, 1e-9))
        f1_score = 2 * (precision * recall) / np.max((precision + recall, 1e-9))

        accs.append(acc)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1_score)

    return np.array(accs), np.array(precisions), np.array(recalls), np.array(f1_scores)

def regression_errors(pred, label):
    """
    Calculate regression error metrics for energy prediction.
    
    Args:
        pred: Predicted energy values
        label: Ground truth energy values
        
    Returns:
        mae: Mean Absolute Error
        mre: Mean Relative Error
    """
    assert pred.shape == label.shape
    
    if type(pred) != np.ndarray:
        pred = pred.detach().cpu().numpy().squeeze()
    if type(label) != np.ndarray:
        label = label.detach().cpu().numpy().squeeze()  

    pred = pred.reshape(pred.shape[0], -1)
    label = label.reshape(label.shape[0], -1)
    epsilon = np.full(label.shape, 1e-9)
    mae_arr, mre_arr = [], []

    for i in range(label.shape[0]):
        abs_diff = np.abs(label[i, :] - pred[i, :])
        mae = np.mean(abs_diff)
        mre_num = np.nan_to_num(abs_diff)
        mre_den = np.max((label[i, :], pred[i, :], epsilon[i, :]), axis=0)
        mre = np.mean(mre_num / mre_den)
        mae_arr.append(mae)
        mre_arr.append(mre)

    return np.array(mae_arr), np.array(mre_arr)

def rmse(pred, label):
    """
    Calculate Root Mean Square Error (RMSE).
    
    Args:
        pred: Predicted values
        label: Ground truth values
        
    Returns:
        rmse_arr: RMSE scores
    """
    if type(pred) != np.ndarray:
        pred = pred.detach().cpu().numpy().squeeze()
    if type(label) != np.ndarray:
        label = label.detach().cpu().numpy().squeeze()
        
    pred = pred.reshape(pred.shape[0], -1)
    label = label.reshape(label.shape[0], -1)
    
    rmse_arr = []
    for i in range(label.shape[0]):
        rmse_arr.append(np.sqrt(np.mean(np.square(label[i, :] - pred[i, :]))))
    
    return np.array(rmse_arr)

def energy_accuracy(pred, label):
    """
    Calculate Energy Accuracy (EACC) - how close the total energy consumption is.
    
    Args:
        pred: Predicted values
        label: Ground truth values
        
    Returns:
        eacc_arr: Energy accuracy scores
    """
    if type(pred) != np.ndarray:
        pred = pred.detach().cpu().numpy().squeeze()
    if type(label) != np.ndarray:
        label = label.detach().cpu().numpy().squeeze()
        
    pred = pred.reshape(pred.shape[0], -1)
    label = label.reshape(label.shape[0], -1)
    
    eacc_arr = []
    for i in range(label.shape[0]):
        pred_sum = np.sum(pred[i, :])
        label_sum = np.sum(label[i, :])
        eacc = 1 - np.abs(pred_sum - label_sum) / np.max([pred_sum, label_sum, 1e-9])
        eacc_arr.append(eacc)
    
    return np.array(eacc_arr)

def nde(pred, label):
    """
    Calculate Normalized Disaggregation Error (NDE).
    
    Args:
        pred: Predicted values
        label: Ground truth values
        
    Returns:
        nde_arr: NDE scores
    """
    if type(pred) != np.ndarray:
        pred = pred.detach().cpu().numpy().squeeze()
    if type(label) != np.ndarray:
        label = label.detach().cpu().numpy().squeeze()
        
    pred = pred.reshape(pred.shape[0], -1)
    label = label.reshape(label.shape[0], -1)
    
    nde_arr = []
    for i in range(label.shape[0]):
        numerator = np.sum(np.square(label[i, :] - pred[i, :]))
        denominator = np.sum(np.square(label[i, :]))
        nde_val = np.sqrt(numerator / np.max([denominator, 1e-9]))
        nde_arr.append(nde_val)
    
    return np.array(nde_arr)

def event_detection_metrics(status, status_pred, threshold=3):
    """
    Event detection metrics for NILM - measure how well the model detects
    state changes (ON to OFF or OFF to ON).
    
    Args:
        status: Ground truth status
        status_pred: Predicted status
        threshold: Number of time steps to consider events as matching
        
    Returns:
        event_tp: True positive rate for event detection
        event_fp: False positive rate for event detection
        event_fn: False negative rate for event detection
    """
    if type(status) != np.ndarray:
        status = status.detach().cpu().numpy().squeeze()   
    if type(status_pred) != np.ndarray: 
        status_pred = status_pred.detach().cpu().numpy().squeeze()
    
    status = status.reshape(status.shape[0], -1)
    status_pred = status_pred.reshape(status_pred.shape[0], -1)
    
    event_tp_arr, event_fp_arr, event_fn_arr = [], [], []
    
    for i in range(status.shape[0]):
        # Find state change points (events)
        true_events = np.where(np.diff(status[i, :]) != 0)[0] + 1
        pred_events = np.where(np.diff(status_pred[i, :]) != 0)[0] + 1
        
        # Initialize counters
        tp, fp, fn = 0, 0, 0
        
        # Mark matched events
        matched_pred = np.zeros_like(pred_events, dtype=bool)
        
        # Calculate TP and FN
        for true_event in true_events:
            # Find the closest predicted event to the true event
            distances = np.abs(pred_events - true_event)
            if len(distances) > 0:
                min_idx = np.argmin(distances)
                min_distance = distances[min_idx]
                
                if min_distance <= threshold and not matched_pred[min_idx]:
                    tp += 1
                    matched_pred[min_idx] = True
                else:
                    fn += 1
            else:
                fn += 1
        
        # Calculate FP
        fp = np.sum(~matched_pred)
        
        # Calculate rates
        total_true_events = len(true_events)
        total_pred_events = len(pred_events)
        
        event_tp = tp / max(total_true_events, 1)
        event_fp = fp / max(total_pred_events, 1)
        event_fn = fn / max(total_true_events, 1)
        
        event_tp_arr.append(event_tp)
        event_fp_arr.append(event_fp)
        event_fn_arr.append(event_fn)
    
    return np.array(event_tp_arr), np.array(event_fp_arr), np.array(event_fn_arr)

def compute_status(data, threshold, min_on, min_off):
    """
    Compute ON/OFF status from power data with minimum duration constraints.
    
    Args:
        data: Power consumption data
        threshold: Power threshold to consider device ON
        min_on: Minimum ON duration
        min_off: Minimum OFF duration
        
    Returns:
        status: Computed status array
    """
    status = np.zeros(data.shape)
    if len(data.squeeze().shape) == 1:
        columns = 1
    else:
        columns = data.squeeze().shape[-1]

    threshold = [threshold]
    min_on = [min_on]
    min_off = [min_off]

    for i in range(columns):
        initial_status = data[:, i] >= threshold[i]
        status_diff = np.diff(initial_status)
        events_idx = status_diff.nonzero()

        events_idx = np.array(events_idx).squeeze()
        events_idx += 1

        if initial_status[0]:
            events_idx = np.insert(events_idx, 0, 0)

        if initial_status[-1]:
            events_idx = np.insert(
                events_idx, events_idx.size, initial_status.size)

        events_idx = events_idx.reshape((-1, 2))
        on_events = events_idx[:, 0].copy()
        off_events = events_idx[:, 1].copy()
        assert len(on_events) == len(off_events)

        if len(on_events) > 0:
            off_duration = on_events[1:] - off_events[:-1]
            off_duration = np.insert(off_duration, 0, 1000)
            on_events = on_events[off_duration > min_off[i]]
            off_events = off_events[np.roll(off_duration, -1) > min_off[i]]

            on_duration = off_events - on_events
            on_events = on_events[on_duration >= min_on[i]]
            off_events = off_events[on_duration >= min_on[i]]
            assert len(on_events) == len(off_events)

        temp_status = data[:, i].copy()
        temp_status[:] = 0
        for on, off in zip(on_events, off_events):
            temp_status[on: off] = 1
        status[:, i] = temp_status

    return status