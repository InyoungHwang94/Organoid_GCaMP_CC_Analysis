"""
===============================================================================
CalculateCC.py — Correlation Coefficient Analysis of Organoid Calcium Imaging
Created : 2026-03-24
Last Modified : 2026-07-20

Purpose
-------
Main analysis pipeline: loads Suite2p output, applies two-stage ROI quality
filtering, computes pairwise cross-correlation matrices with temporal lags
(vectorized), estimates a shuffled chance-level baseline, and detects
population-level synchrony events. Results saved to .h5 per recording, with
a running per-recording correlation summary CSV

Pipeline Position
-----------------
Runs after  : Suite2p segmentation and signal extraction
Runs before : GabazineComparison.py, MultiDrug_Comparison.py

Notes
-----
- Set folder_path to the directory containing recording subfolders.
- Each subfolder must contain suite2p/plane0/. Frame rate defaults to 15.023 Hz
  if no .xml file is found.
- Set ENABLE_FILTERING = False to skip the two-stage ROI quality filter.
- Output folders are recreated on each run (existing data will be overwritten).
- Cross-correlation is now vectorized (matrix ops across all lags at once)
  instead of looping per cell pair.
- Added calculate_shuffled_cross_correlation_baseline() for a chance-level
  control: circularly shifts each cell independently and recomputes mean max
  cross-correlation, repeated n_shuffles times.

===============================================================================
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import datetime
import shutil
import logging
import warnings
import gc
from scipy import ndimage, signal

import pandas as pd

from helper import TwoP, files, process_spike_data_gcamp6m

# Reduce verbose output
warnings.filterwarnings('ignore', category=FutureWarning)
logging.getLogger().setLevel(logging.WARNING)

# ============================================================================
# SECTION 1: FILTERING FUNCTIONS
# ============================================================================

def basic_signal_quality_filter(dff_data, spike_data, 
                               peak_percentile=5, 
                               variance_low_percentile=5, 
                               variance_high_percentile=95,
                               use_dff_for_filtering=False):
    """Basic signal quality filtering - RELAXED parameters"""
    
    n_cells, n_frames = dff_data.shape
    print(f"\n=== BASIC SIGNAL QUALITY FILTERING ===")
    print(f"Input: {n_cells} ROIs, {n_frames} frames")
    
    filter_data = dff_data if use_dff_for_filtering else spike_data
    data_type = "DFF" if use_dff_for_filtering else "Spike"
    print(f"Using {data_type} data for filtering")
    
    peak_amplitudes = np.zeros(n_cells)
    variances = np.zeros(n_cells)
    
    print("Calculating signal quality metrics...")
    for i in tqdm(range(n_cells)):
        roi_trace = filter_data[i, :]
        peak_amplitudes[i] = np.max(roi_trace)
        variances[i] = np.var(roi_trace)
    
    peak_threshold = np.percentile(peak_amplitudes, peak_percentile)
    var_low_threshold = np.percentile(variances, variance_low_percentile)
    var_high_threshold = np.percentile(variances, variance_high_percentile)
    
    print(f"\nThresholds calculated:")
    print(f"  Peak amplitude (>{peak_percentile}th percentile): {peak_threshold:.4f}")
    print(f"  Variance range: {var_low_threshold:.6f} to {var_high_threshold:.6f}")
    
    peak_pass = peak_amplitudes >= peak_threshold
    variance_pass = (variances > var_low_threshold) & (variances < var_high_threshold)
    filtering_mask = peak_pass & variance_pass
    
    n_filtered_pass = np.sum(filtering_mask)
    
    print(f"\nFiltering results:")
    print(f"  Failed peak amplitude: {np.sum(~peak_pass)}/{n_cells} ({np.sum(~peak_pass)/n_cells*100:.1f}%)")
    print(f"  Failed variance bounds: {np.sum(~variance_pass)}/{n_cells} ({np.sum(~variance_pass)/n_cells*100:.1f}%)")
    print(f"  Passed filtering: {n_filtered_pass}/{n_cells} ({n_filtered_pass/n_cells*100:.1f}%)")
    
    filtering_stats = {
        'input_rois': n_cells,
        'filtered_rois': n_filtered_pass,
        'pass_rate': n_filtered_pass/n_cells,
        'peak_threshold': peak_threshold,
        'variance_low_threshold': var_low_threshold,
        'variance_high_threshold': var_high_threshold,
        'peak_failures': np.sum(~peak_pass),
        'variance_failures': np.sum(~variance_pass),
        'peak_amplitudes': peak_amplitudes,
        'variances': variances,
        'filtering_method': 'basic_signal_quality',
        'data_type_used': data_type
    }
    
    return filtering_mask, filtering_stats

def detect_events_single_roi(roi_trace, method='adaptive_threshold', 
                           threshold_factor=2.0, min_prominence=None, 
                           min_duration=2, sampling_rate=10):
    """Detect calcium events in a single ROI trace"""
    
    if len(roi_trace) == 0 or np.all(np.isnan(roi_trace)) or np.var(roi_trace) < 1e-10:
        return np.zeros_like(roi_trace, dtype=bool), np.array([]), 0
    
    if method == 'adaptive_threshold':
        baseline_median = np.median(roi_trace)
        baseline_mad = np.median(np.abs(roi_trace - baseline_median))
        threshold = baseline_median + threshold_factor * baseline_mad

        above_threshold = roi_trace > threshold
        event_frames = np.zeros_like(above_threshold, dtype=bool)
        current_event_start = None
        event_peaks = []
        
        for i, is_active in enumerate(above_threshold):
            if is_active and current_event_start is None:
                current_event_start = i
            elif not is_active and current_event_start is not None:
                event_duration = i - current_event_start
                if event_duration >= min_duration:
                    event_frames[current_event_start:i] = True
                    event_segment = roi_trace[current_event_start:i]
                    event_peaks.append(np.max(event_segment))
                current_event_start = None
        
        if current_event_start is not None:
            event_duration = len(above_threshold) - current_event_start
            if event_duration >= min_duration:
                event_frames[current_event_start:] = True
                event_segment = roi_trace[current_event_start:]
                event_peaks.append(np.max(event_segment))
        
        event_peaks = np.array(event_peaks)
        n_events = len(event_peaks)
        
    elif method == 'peak_detection':
        if min_prominence is None:
            min_prominence = np.std(roi_trace) * 1.5
        
        peaks, properties = signal.find_peaks(roi_trace, 
                                     prominence=min_prominence,
                                     distance=min_duration)
        
        event_frames = np.zeros_like(roi_trace, dtype=bool)
        half_width = max(1, min_duration // 2)
        
        for peak in peaks:
            start = max(0, peak - half_width)
            end = min(len(roi_trace), peak + half_width + 1)
            event_frames[start:end] = True
        
        event_peaks = roi_trace[peaks] if len(peaks) > 0 else np.array([])
        n_events = len(peaks)
    
    return event_frames, event_peaks, n_events

def event_based_snr_filter(dff_data, spike_data, stage1_mask,
                          snr_threshold=1.2, min_events=1,
                          event_detection_method='adaptive_threshold',
                          threshold_factor=2.0, min_duration=2, 
                          sampling_rate=10, use_dff_for_snr=False):
    """Stage 2: Event-based SNR filtering"""
    
    n_cells, n_frames = dff_data.shape
    stage1_survivors = np.sum(stage1_mask)
    
    print(f"\n=== STAGE 2: Event-Based SNR Filtering ===")
    print(f"Input: {stage1_survivors} ROIs (survivors from Stage 1)")
    
    snr_data = dff_data if use_dff_for_snr else spike_data
    data_type = "DFF" if use_dff_for_snr else "Spike"
    print(f"Using {data_type} data for SNR calculation")
    print(f"Event detection method: {event_detection_method}")
    print(f"SNR threshold: {snr_threshold}")
    
    stage2_mask = np.zeros(n_cells, dtype=bool)
    snr_values = np.full(n_cells, np.nan)
    event_counts = np.zeros(n_cells, dtype=int)
    
    stage1_indices = np.where(stage1_mask)[0]
    
    print("Processing ROIs for event-based SNR...")
    valid_snr_rois = 0
    
    for idx in tqdm(stage1_indices):
        roi_trace = snr_data[idx, :]
        
        event_frames, event_peaks, n_events = detect_events_single_roi(
            roi_trace, 
            method=event_detection_method,
            threshold_factor=threshold_factor,
            min_duration=min_duration,
            sampling_rate=sampling_rate
        )
        
        event_counts[idx] = n_events
        
        if n_events >= min_events and len(event_peaks) > 0:
            quiet_frames = ~event_frames
            
            if np.sum(quiet_frames) > 5:
                quiet_data = roi_trace[quiet_frames]
                quiet_mean = np.mean(quiet_data)
                quiet_std = np.std(quiet_data)
                
                if quiet_std > 1e-10:
                    peak_response = np.max(event_peaks)
                    snr = (peak_response - quiet_mean) / quiet_std
                    snr_values[idx] = snr
                    
                    if snr >= snr_threshold:
                        stage2_mask[idx] = True
                        valid_snr_rois += 1
    
    n_stage2_pass = np.sum(stage2_mask)
    n_no_events = np.sum((event_counts[stage1_indices] < min_events))
    n_low_snr = np.sum((stage1_mask) & (~stage2_mask) & (event_counts >= min_events))
    
    print(f"\nStage 2 filtering results:")
    print(f"  ROIs with insufficient events (<{min_events}): {n_no_events}")
    print(f"  ROIs with low SNR (<{snr_threshold}): {n_low_snr}")
    print(f"  ROIs with valid SNR calculation: {valid_snr_rois}")
    print(f"  Passed Stage 2: {n_stage2_pass}/{stage1_survivors} ({n_stage2_pass/stage1_survivors*100:.1f}% of Stage 1)")
    print(f"  Overall pass rate: {n_stage2_pass}/{n_cells} ({n_stage2_pass/n_cells*100:.1f}% of original)")
    
    valid_snrs = snr_values[~np.isnan(snr_values)]
    if len(valid_snrs) > 0:
        print(f"  SNR statistics: mean={np.mean(valid_snrs):.2f}, "
              f"median={np.median(valid_snrs):.2f}, "
              f"min={np.min(valid_snrs):.2f}, max={np.max(valid_snrs):.2f}")
    
    filtering_stats = {
        'stage1_survivors': stage1_survivors,
        'stage2_survivors': n_stage2_pass,
        'overall_pass_rate': n_stage2_pass/n_cells,
        'stage2_pass_rate': n_stage2_pass/stage1_survivors if stage1_survivors > 0 else 0,
        'snr_threshold': snr_threshold,
        'min_events': min_events,
        'no_events_failures': n_no_events,
        'low_snr_failures': n_low_snr,
        'snr_values': snr_values,
        'event_counts': event_counts,
        'valid_snr_rois': valid_snr_rois,
        'event_detection_method': event_detection_method,
        'threshold_factor': threshold_factor,
        'filtering_method': 'two_stage_snr'
    }
    
    return stage2_mask, filtering_stats

# ============================================================================
# SECTION 2: PREPROCESSING FUNCTIONS
# ============================================================================

def gaussian_smoothing(data, sigma=0.5, sampling_rate=15):
    """Apply MINIMAL Gaussian smoothing"""
    
    n_cells, n_frames = data.shape
    smoothed_data = np.zeros_like(data)
    
    for i in tqdm(range(n_cells), desc="Smoothing cells"):
        smoothed_data[i, :] = ndimage.gaussian_filter1d(data[i, :], sigma)
    
    original_noise = np.mean([np.std(np.diff(data[i, :])) for i in range(n_cells)])
    smoothed_noise = np.mean([np.std(np.diff(smoothed_data[i, :])) for i in range(n_cells)])
    noise_reduction = (original_noise - smoothed_noise) / original_noise * 100
    
    smoothing_stats = {
        'sigma_frames': sigma,
        'sigma_ms': sigma / sampling_rate * 1000,
        'noise_reduction_percent': noise_reduction,
        'original_derivative_noise': original_noise,
        'smoothed_derivative_noise': smoothed_noise
    }
    
    print(f"  Derivative noise reduction: {noise_reduction:.1f}%")
    
    return smoothed_data, smoothing_stats

def temporal_binning(data, bin_size=2):
    """Bin temporal data"""
    
    n_cells, n_frames = data.shape
    n_bins = n_frames // bin_size
    
    print(f"Temporal binning: {n_frames} frames → {n_bins} bins (bin size: {bin_size})")
    
    binned_data = np.zeros((n_cells, n_bins))
    
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size
        binned_data[:, i] = np.mean(data[:, start_idx:end_idx], axis=1)
    
    original_noise = np.mean([np.std(data[i, :]) for i in range(n_cells)])
    binned_noise = np.mean([np.std(binned_data[i, :]) for i in range(n_cells)])
    noise_reduction = (original_noise - binned_noise) / original_noise * 100
    
    binning_stats = {
        'original_frames': n_frames,
        'binned_frames': n_bins,
        'bin_size': bin_size,
        'temporal_resolution_ms': bin_size * (1000 / 15),
        'noise_reduction_percent': noise_reduction,
        'original_noise_std': original_noise,
        'binned_noise_std': binned_noise
    }
    
    print(f"  Temporal resolution: {binning_stats['temporal_resolution_ms']:.1f} ms per bin")
    print(f"  Noise reduction: {noise_reduction:.1f}%")
    
    return binned_data, binning_stats

def preprocessing_pipeline(data, 
                                   temporal_bin_size=2,
                                   gaussian_sigma=0.5,
                                   sampling_rate=15,
                                   apply_temporal_binning=False,
                                   apply_gaussian_smoothing=True,
                                   use_full_timeseries=False):
    
    processed_data = data.copy()
    preprocessing_stats = {'original_shape': data.shape}
    
    # Step 1: Gaussian smoothing 
    if apply_gaussian_smoothing:
        processed_data, smoothing_stats = gaussian_smoothing(
            processed_data, sigma=gaussian_sigma, sampling_rate=sampling_rate
        )
        preprocessing_stats['smoothing'] = smoothing_stats

    
    # Step 2: Temporal binning
    if apply_temporal_binning:
        processed_data, binning_stats = temporal_binning(
            processed_data, bin_size=temporal_bin_size
        )
        preprocessing_stats['binning'] = binning_stats

    
    # Step 3: For cross-correlation, use ALL frames
    n_frames = processed_data.shape[1]
    active_mask = np.ones(n_frames, dtype=bool)
    
    active_stats = {
        'total_frames': n_frames,
        'active_frames': n_frames,
        'active_percentage': 100.0,
        'used_full_timeseries': True
    }
    preprocessing_stats['active_selection'] = active_stats
    
    preprocessing_stats['final_shape'] = processed_data.shape
    preprocessing_stats['methods_applied'] = {
        'gaussian_smoothing': apply_gaussian_smoothing,
        'temporal_binning': apply_temporal_binning,
        'active_period_selection': False,
        'used_full_timeseries': True
    }
    
    print(f"\nPreprocessing complete!")
    print(f"  Final data shape: {processed_data.shape}")
    print(f"  Using ALL {n_frames} frames for cross-correlation")
    
    return processed_data, active_mask, preprocessing_stats

# ============================================================================
# SECTION 3: CROSS-CORRELATION WITH TIME LAGS (VECTORIZED)
# ============================================================================

def calculate_cross_correlation_with_lags(data, max_lag=3, verbose=True):
    """
    Calculate cross-correlation with time lags for all cell pairs (vectorized
    across pairs and lags via matrix operations).

    Parameters:
        data: (n_cells, n_frames) array of preprocessed neural data
        max_lag: maximum time lag in frames (default: 3 frames = ±200ms at 15Hz)
        verbose: print progress and summary (set False for repeated/shuffle calls)

    Returns:
        max_corr_matrix: (n_cells, n_cells) matrix of maximum correlations
        best_lag_matrix: (n_cells, n_cells) matrix of optimal lags
        standard_corr_matrix: (n_cells, n_cells) standard correlation at lag=0
        correlation_stats: dictionary with statistics
    """

    n_cells, n_frames = data.shape

    if verbose:
        print(f"\n{'='*80}")
        print(f"CROSS-CORRELATION ANALYSIS")
        print(f"{'='*80}")

    # Initialize output matrices
    max_corr_matrix = np.zeros((n_cells, n_cells))
    best_lag_matrix = np.zeros((n_cells, n_cells), dtype=int)
    standard_corr_matrix = np.zeros((n_cells, n_cells))

    # Remove cells with no variance
    valid_cells = []
    for i in range(n_cells):
        if np.var(data[i, :]) > 1e-10:
            valid_cells.append(i)

    if verbose:
        print(f"Using {len(valid_cells)}/{n_cells} cells with sufficient variance")

    if len(valid_cells) < 2:
        if verbose:
            print("ERROR: Too few cells with variance")
        return max_corr_matrix, best_lag_matrix, standard_corr_matrix, {}

    # Calculate correlations for all cell pairs at once (vectorized across pairs and lags)
    if verbose:
        print("\nCalculating cross-correlations (vectorized)...")

    valid_idx = np.array(valid_cells)
    valid_data = data[valid_idx, :]
    means = valid_data.mean(axis=1, keepdims=True)
    stds = valid_data.std(axis=1, keepdims=True)
    norm_data = (valid_data - means) / (stds + 1e-10)

    lags = list(range(-max_lag, max_lag + 1))
    n_valid = len(valid_idx)
    all_corrs = np.zeros((len(lags), n_valid, n_valid))

    for li, lag in enumerate(lags):
        if lag < 0:
            A = norm_data[:, :lag]
            B = norm_data[:, -lag:]
        elif lag > 0:
            A = norm_data[:, lag:]
            B = norm_data[:, :-lag]
        else:
            A = norm_data
            B = norm_data

        if A.shape[1] <= 10:  # Need sufficient overlap
            continue

        Ac = A - A.mean(axis=1, keepdims=True)
        Bc = B - B.mean(axis=1, keepdims=True)
        norm_A = np.sqrt((Ac ** 2).sum(axis=1))
        norm_B = np.sqrt((Bc ** 2).sum(axis=1))
        denom = np.outer(norm_A, norm_B)

        with np.errstate(invalid="ignore", divide="ignore"):
            corr = (Ac @ Bc.T) / denom
        corr[denom == 0] = 0.0
        all_corrs[li] = np.nan_to_num(corr, nan=0.0)

    max_corr_vals = np.max(all_corrs, axis=0)
    best_lag_idx = np.argmax(all_corrs, axis=0)
    lags_arr = np.array(lags)
    best_lag_vals = lags_arr[best_lag_idx]
    standard_corr_vals = all_corrs[max_lag]  # lag == 0 slice (center of lag range)

    iu, ju = np.triu_indices(n_valid, k=1)
    gi, gj = valid_idx[iu], valid_idx[ju]

    max_corr_matrix[gi, gj] = max_corr_vals[iu, ju]
    max_corr_matrix[gj, gi] = max_corr_vals[iu, ju]

    best_lag_matrix[gi, gj] = best_lag_vals[iu, ju]
    best_lag_matrix[gj, gi] = -best_lag_vals[iu, ju]  # Opposite lag for reverse direction

    standard_corr_matrix[gi, gj] = standard_corr_vals[iu, ju]
    standard_corr_matrix[gj, gi] = standard_corr_vals[iu, ju]
    
    # Set diagonal to 1.0
    np.fill_diagonal(max_corr_matrix, 1.0)
    np.fill_diagonal(standard_corr_matrix, 1.0)
    np.fill_diagonal(best_lag_matrix, 0)
    
    # Calculate statistics
    upper_tri = np.triu_indices_from(max_corr_matrix, k=1)
    max_correlations = max_corr_matrix[upper_tri]
    standard_correlations = standard_corr_matrix[upper_tri]
    best_lags = best_lag_matrix[upper_tri]
    
    valid_max_corr = max_correlations[~np.isnan(max_correlations)]
    valid_std_corr = standard_correlations[~np.isnan(standard_correlations)]
    
    correlation_stats = {
        'n_cells_total': n_cells,
        'n_cells_valid': len(valid_cells),
        'n_frames': n_frames,
        'max_lag': max_lag,
        'max_lag_ms': max_lag * 66.7,
        
        # Max cross-correlation statistics
        'mean_max_correlation': np.mean(valid_max_corr) if len(valid_max_corr) > 0 else 0,
        'std_max_correlation': np.std(valid_max_corr) if len(valid_max_corr) > 0 else 0,
        'median_max_correlation': np.median(valid_max_corr) if len(valid_max_corr) > 0 else 0,
        'min_max_correlation': np.min(valid_max_corr) if len(valid_max_corr) > 0 else 0,
        'max_max_correlation': np.max(valid_max_corr) if len(valid_max_corr) > 0 else 0,
        
        # Standard correlation statistics (for comparison)
        'mean_standard_correlation': np.mean(valid_std_corr) if len(valid_std_corr) > 0 else 0,
        'std_standard_correlation': np.std(valid_std_corr) if len(valid_std_corr) > 0 else 0,
        
        # Improvement statistics
        'mean_improvement': np.mean(valid_max_corr - valid_std_corr) if len(valid_max_corr) > 0 else 0,
        'improvement_percentage': (np.mean(valid_max_corr) / np.mean(valid_std_corr) - 1) * 100 if np.mean(valid_std_corr) > 0 else 0,
        
        # Lag statistics
        'mean_abs_lag': np.mean(np.abs(best_lags)),
        'median_lag': np.median(best_lags),
        'lag_distribution': np.bincount(best_lags + max_lag, minlength=2*max_lag+1),
        
        'n_correlations': len(valid_max_corr),
        'percent_positive_corr': np.sum(valid_max_corr > 0.1) / len(valid_max_corr) * 100 if len(valid_max_corr) > 0 else 0
    }
    
    if verbose:
        print(f"\nCross-correlation results:")
        print(f"  Max correlation - Mean: {correlation_stats['mean_max_correlation']:.3f}, "
              f"Median: {correlation_stats['median_max_correlation']:.3f}")

    return max_corr_matrix, best_lag_matrix, standard_corr_matrix, correlation_stats


def calculate_shuffled_cross_correlation_baseline(data, max_lag=3, n_shuffles=100, verbose=True):
    """
    Estimate the chance-level cross-correlation by independently circularly
    shifting each cell's trace and recomputing the mean max cross-correlation,
    repeated n_shuffles times.

    Parameters:
        data: (n_cells, n_frames) array of preprocessed neural data
        max_lag: maximum time lag in frames (same value used for the real correlation)
        n_shuffles: number of circular-shift iterations
        verbose: print progress and summary

    Returns:
        shuffle_stats: dict with 'mean_random_max_correlation', 'std_random_max_correlation',
            and 'shuffle_mean_max_correlations' (the per-iteration values)
    """

    n_cells, n_frames = data.shape
    shuffle_mean_max_correlations = np.zeros(n_shuffles)

    if verbose:
        print(f"\nComputing shuffled (chance) cross-correlation baseline ({n_shuffles} iterations)...")

    for s in tqdm(range(n_shuffles), desc="Shuffle iterations", disable=not verbose):
        shuffled_data = np.zeros_like(data)
        for i in range(n_cells):
            shift = np.random.randint(0, n_frames)
            shuffled_data[i] = np.roll(data[i], shift)

        _, _, _, shuffled_corr_stats = calculate_cross_correlation_with_lags(
            shuffled_data, max_lag=max_lag, verbose=False
        )
        shuffle_mean_max_correlations[s] = shuffled_corr_stats.get('mean_max_correlation', 0)

    mean_random_max_correlation = float(np.mean(shuffle_mean_max_correlations))
    std_random_max_correlation = float(np.std(shuffle_mean_max_correlations))

    if verbose:
        print(f"  Random (chance) mean max correlation: "
              f"{mean_random_max_correlation:.3f} ± {std_random_max_correlation:.3f}")

    shuffle_stats = {
        'mean_random_max_correlation': mean_random_max_correlation,
        'std_random_max_correlation': std_random_max_correlation,
        'shuffle_mean_max_correlations': shuffle_mean_max_correlations,
        'n_shuffles': n_shuffles
    }

    return shuffle_stats

# ============================================================================
# SECTION 4: ROBUST SPIKE DETECTION
# ============================================================================

def detect_spike_peaks_robust(
    dff_data,
    sampling_rate=15,
    min_peak_distance_s=0.5,
    prominence_factor=2.0,
    adaptive_smoothing=True,
    detrend=True,
    verbose=True
):
    """
    Robust spike detection for GCaMP6m 2p imaging data.

    Parameters
    ----------
    dff_data : np.ndarray
        2D array (n_cells x n_frames) of ΔF/F traces.
    sampling_rate : float
        Frame rate in Hz (default 15 Hz).
    min_peak_distance_s : float
        Minimum distance between peaks (in seconds).
    prominence_factor : float
        Multiplier for noise-based prominence threshold.
    adaptive_smoothing : bool
        If True, adjusts Gaussian sigma per cell based on noise.
    detrend : bool
        If True, removes slow baseline drifts using Savitzky-Golay.
    verbose : bool
        If True, prints detection summary.

    Returns
    -------
    cell_spike_data : dict
        Per-cell dictionary of detected peaks and metrics.
    summary_stats : dict
        Global summary of spike detection.
    """

    n_cells, n_frames = dff_data.shape
    cell_spike_data = {}

    if verbose:
        print(f"\n=== ROBUST SPIKE DETECTION ===")
        print(f"Cells: {n_cells} | Frames: {n_frames} | Sampling: {sampling_rate} Hz")

    min_distance_frames = int(min_peak_distance_s * sampling_rate)

    # --- Loop through each cell ---
    for cell_idx in tqdm(range(n_cells), desc="Detecting spikes"):
        trace = np.copy(dff_data[cell_idx, :])

        # (1) Detrend slow baseline drifts
        if detrend:
            baseline = signal.savgol_filter(trace, window_length=min(301, n_frames - 1), polyorder=3)
            trace = trace - baseline

        # (2) Adaptive Gaussian smoothing based on noise level
        if adaptive_smoothing:
            diff_std = np.std(np.diff(trace))
            # Map noise level (diff_std) to smoothing sigma [0.3–1.5]
            sigma = np.clip(np.interp(diff_std, [0.005, 0.05], [0.3, 1.5]), 0.3, 1.5)
        else:
            sigma = 0.5
        trace_smooth = ndimage.gaussian_filter1d(trace, sigma)

        # (3) Robust threshold using median absolute deviation (MAD)
        baseline_median = np.median(trace_smooth)
        mad = np.median(np.abs(trace_smooth - baseline_median))
        threshold = baseline_median + 3 * mad * 1.4826  # conservative threshold
        
        # (4) Detect peaks using adaptive prominence and distance
        peaks, properties = signal.find_peaks(
            trace_smooth,
            height=threshold,
            distance=min_distance_frames,
            prominence=mad * prominence_factor
        )

        # (5) Store data (including trace for boundary detection)
        cell_spike_data[f'cell_{cell_idx}'] = {
            'peak_frames': peaks,
            'peak_amplitudes': trace_smooth[peaks] if len(peaks) > 0 else np.array([]),
            'n_peaks': len(peaks),
            'sigma_used': sigma,
            'mad': mad,
            'baseline_median': baseline_median,
            'trace_smooth': trace_smooth  # Store for boundary detection
        }

    # --- Summarize results ---
    total_peaks = sum([len(v['peak_frames']) for v in cell_spike_data.values()])
    mean_peaks_per_cell = np.mean([len(v['peak_frames']) for v in cell_spike_data.values()])
    avg_sigma = np.mean([v['sigma_used'] for v in cell_spike_data.values()])

    summary_stats = {
        'n_cells': n_cells,
        'n_frames': n_frames,
        'total_peaks': total_peaks,
        'mean_peaks_per_cell': mean_peaks_per_cell,
        'avg_sigma_used': avg_sigma,
        'min_peak_distance_s': min_peak_distance_s,
        'prominence_factor': prominence_factor,
        'sampling_rate': sampling_rate,
        'method': 'adaptive_MAD_threshold + smoothing + detrend'
    }

    if verbose:
        print("\nSpike detection summary:")
        print(f"  Avg. peaks per cell: {mean_peaks_per_cell:.1f}")
        print(f"  Avg. smoothing σ: {avg_sigma:.2f} frames")
        print(f"  Min distance: {min_peak_distance_s:.2f} s | Prominence ×{prominence_factor}")

    return cell_spike_data, summary_stats

# ============================================================================
# SECTION 5: POPULATION-LEVEL SYNCHRONY ANALYSIS
# ============================================================================

def find_transient_boundaries(trace, peak_frame, baseline_median, mad, threshold_factor=1.0):
    """
    Find start and end of a calcium transient around a detected peak
    
    Parameters:
            trace: detrended and smoothed calcium signal
            peak_frame: detected peak location (frame index)
            baseline_median: baseline level of the trace
            mad: median absolute deviation (noise level)
            threshold_factor: multiplier for threshold (default 1.0)
        
    Returns:
        start_frame, end_frame
    """
    
    # Threshold = baseline + threshold_factor * MAD
    threshold = baseline_median + threshold_factor * mad * 1.4826
    
    n_frames = len(trace)
    
    # Find START: go backward from peak until below threshold
    start_frame = peak_frame
    for i in range(peak_frame, -1, -1):
        if trace[i] < threshold:
            start_frame = i + 1  # Last frame above threshold
            break
        if i == 0:  # Reached beginning
            start_frame = 0
    
    # Find END: go forward from peak until below threshold
    end_frame = peak_frame
    for i in range(peak_frame, n_frames):
        if trace[i] < threshold:
            end_frame = i - 1  # Last frame above threshold
            break
        if i == n_frames - 1:  # Reached end
            end_frame = n_frames - 1
    
    return start_frame, end_frame


def create_population_transient_array(dff_data, cell_spike_data, sampling_rate=15, verbose=True):
    """
    Create boolean array marking full transient duration for each cell
    
    Parameters:
        dff_data: (n_cells, n_frames) preprocessed DFF data
        cell_spike_data: dict from detect_spike_peaks_robust containing peak info
        sampling_rate: frame rate in Hz
        verbose: print progress
    
    Returns:
        transient_active: (n_cells, n_frames) boolean array
        transient_boundaries: dict with start/end info for each transient
    """
    
    n_cells, n_frames = dff_data.shape
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"CREATING POPULATION TRANSIENT ARRAY")
        print(f"{'='*80}")
        print(f"Cells: {n_cells} | Frames: {n_frames}")
    
    # Initialize output array
    transient_active = np.zeros((n_cells, n_frames), dtype=bool)
    transient_boundaries = {}
    
    # Process each cell
    total_transients = 0
    
    for cell_idx in tqdm(range(n_cells), desc="Processing cell transients"):
        cell_key = f'cell_{cell_idx}'
        
        if cell_key not in cell_spike_data:
            continue
        
        # Get peak information
        peaks = cell_spike_data[cell_key]['peak_frames']
        baseline_median = cell_spike_data[cell_key]['baseline_median']
        mad = cell_spike_data[cell_key]['mad']
        trace = cell_spike_data[cell_key]['trace_smooth']
        
        if len(peaks) == 0:
            continue
        
        # Find boundaries for each peak
        cell_transients = []
        
        for peak in peaks:
            start, end = find_transient_boundaries(
                trace, peak, baseline_median, mad, threshold_factor=1.0
            )
            
            # Mark entire transient as active
            transient_active[cell_idx, start:end+1] = True
            
            # Store boundary info
            cell_transients.append({
                'peak': int(peak),
                'start': int(start),
                'end': int(end),
                'duration': int(end - start + 1)
            })
            
            total_transients += 1
        
        transient_boundaries[cell_key] = cell_transients
    
    # Calculate statistics
    frames_per_cell = np.sum(transient_active, axis=1)
    mean_active_frames = np.mean(frames_per_cell)
    mean_active_percent = mean_active_frames / n_frames * 100
    
    if verbose:
        print(f"\nTransient boundary detection complete:")
        print(f"  Total transients detected: {total_transients}")
        print(f"  Mean transients per cell: {total_transients/n_cells:.1f}")
        print(f"  Mean active frames per cell: {mean_active_frames:.1f} ({mean_active_percent:.1f}%)")
    
    return transient_active, transient_boundaries


def detect_population_synchrony_events(
    transient_active, 
    min_fraction_coincident=0.10,
    sampling_rate=15,
    compute_shuffle_baseline=True,
    n_shuffles=200,
    verbose=True
):
    """
    Detect population synchrony events based on overlapping transients
    
    Parameters:
        transient_active: (n_cells, n_frames) boolean array of active transients
        min_fraction_coincident: minimum fraction of cells required (0.10 = 10%)
        sampling_rate: frame rate in Hz
        compute_shuffle_baseline: whether to compute shuffle control
        n_shuffles: number of shuffles for baseline
        verbose: print progress
    
    Returns:
        population_sync_frames: boolean array of synchronous frames
        sync_stats: dictionary with statistics
    """
    
    n_cells, n_frames = transient_active.shape
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"DETECTING POPULATION SYNCHRONY EVENTS")
        print(f"{'='*80}")
        print(f"Threshold: ≥{min_fraction_coincident*100:.0f}% of {n_cells} cells")
    
    # Count cells with active transients per frame
    cells_active_per_frame = np.sum(transient_active, axis=0)
    
    # Define synchrony threshold
    min_coincident_cells = max(2, int(n_cells * min_fraction_coincident))
    
    # Identify synchronous frames
    population_sync_frames = cells_active_per_frame >= min_coincident_cells
    
    n_sync_frames = np.sum(population_sync_frames)
    sync_percentage = n_sync_frames / n_frames * 100
    
    if verbose:
        print(f"\nSynchrony detection:")
        print(f"  Min cells required: {min_coincident_cells}")
        print(f"  Synchronous frames: {n_sync_frames}/{n_frames} ({sync_percentage:.2f}%)")
    
    # Compute shuffle baseline
    shuffle_stats = {}
    if compute_shuffle_baseline and n_sync_frames > 0:
        if verbose:
            print("\nComputing shuffle baseline...")
        
        shuffle_counts = np.zeros(n_shuffles)
        for s in range(n_shuffles):
            # Circularly shift each cell's transient pattern
            shuffled = np.zeros_like(transient_active)
            for i in range(n_cells):
                shift = np.random.randint(0, n_frames)
                shuffled[i] = np.roll(transient_active[i], shift)
            
            # Count synchronous frames in shuffled data
            shuffled_active = np.sum(shuffled, axis=0)
            shuffled_sync = np.sum(shuffled_active >= min_coincident_cells)
            shuffle_counts[s] = shuffled_sync / n_frames * 100
        
        shuffle_mean = np.mean(shuffle_counts)
        shuffle_std = np.std(shuffle_counts)
        z_score = (sync_percentage - shuffle_mean) / (shuffle_std + 1e-10)
        
        shuffle_stats = {
            'shuffle_mean_percent': shuffle_mean,
            'shuffle_std_percent': shuffle_std,
            'z_score': z_score
        }
        
        if verbose:
            print(f"  Shuffle baseline: {shuffle_mean:.2f} ± {shuffle_std:.2f}%")
            print(f"  Z-score: {z_score:.2f}")
    
    # Create statistics dictionary
    sync_stats = {
        'n_cells': n_cells,
        'n_frames': n_frames,
        'min_fraction_coincident': min_fraction_coincident,
        'min_coincident_cells': min_coincident_cells,
        'synchronous_frames': n_sync_frames,
        'synchrony_percentage': sync_percentage,
        'cells_active_per_frame': cells_active_per_frame,
        'max_cells_active': int(np.max(cells_active_per_frame)),
        'mean_cells_active': float(np.mean(cells_active_per_frame)),
        'method': 'full_transient_overlap'
    }
    
    if shuffle_stats:
        sync_stats.update(shuffle_stats)
    
    return population_sync_frames, sync_stats


def group_consecutive_frames(frame_indices):
    """
    Group consecutive frame numbers into events
    
    Parameters:
        frame_indices: array of frame indices
    
    Returns:
        events: list of lists, each containing consecutive frames
    
    Example:
        Input:  [45, 46, 47, 48, 102, 103, 104, 250, 251, 252]
        Output: [[45, 46, 47, 48], [102, 103, 104], [250, 251, 252]]
    """
    
    if len(frame_indices) == 0:
        return []
    
    events = []
    current_event = [frame_indices[0]]
    
    for i in range(1, len(frame_indices)):
        if frame_indices[i] == frame_indices[i-1] + 1:
            # Consecutive frame
            current_event.append(frame_indices[i])
        else:
            # Gap detected - save current event and start new one
            events.append(current_event)
            current_event = [frame_indices[i]]
    
    # Add last event
    events.append(current_event)
    
    return events


def analyze_population_synchrony_events(
    population_sync_frames,
    cells_active_per_frame,
    sampling_rate=15,
    verbose=True
):
    """
    Analyze population synchrony events (duration, intervals, peak activity)
    
    Parameters:
        population_sync_frames: boolean array of synchronous frames
        cells_active_per_frame: array of cell counts per frame
        sampling_rate: frame rate in Hz
        verbose: print results
    
    Returns:
        event_data: list of dictionaries with event properties
        event_stats: summary statistics
    """
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"ANALYZING POPULATION SYNCHRONY EVENTS")
        print(f"{'='*80}")
    
    # Get synchronous frame indices
    sync_frame_indices = np.where(population_sync_frames)[0]
    
    if len(sync_frame_indices) == 0:
        if verbose:
            print("No synchronous events detected")
        return [], {}
    
    # Group consecutive frames into events
    events = group_consecutive_frames(sync_frame_indices)
    
    if verbose:
        print(f"Found {len(events)} synchronous events")
    
    # Analyze each event
    event_data = []
    
    for event_id, frames in enumerate(events):
        start_frame = frames[0]
        end_frame = frames[-1]
        duration_frames = len(frames)
        duration_ms = duration_frames / sampling_rate * 1000
        duration_s = duration_frames / sampling_rate
        
        # Find peak frame (frame with maximum cells active)
        cells_active_in_event = cells_active_per_frame[frames]
        peak_idx = np.argmax(cells_active_in_event)
        peak_frame = frames[peak_idx]
        max_cells_active = int(cells_active_in_event[peak_idx])
        mean_cells_active = float(np.mean(cells_active_in_event))
        
        # Calculate interval to next event (start to start)
        if event_id < len(events) - 1:
            next_start = events[event_id + 1][0]
            interval_frames = next_start - start_frame
            interval_s = interval_frames / sampling_rate
        else:
            interval_frames = None
            interval_s = None
        
        event_data.append({
            'event_id': event_id + 1,
            'start_frame': int(start_frame),
            'peak_frame': int(peak_frame),
            'end_frame': int(end_frame),
            'duration_frames': int(duration_frames),
            'duration_ms': float(duration_ms),
            'duration_s': float(duration_s),
            'start_time_s': float(start_frame / sampling_rate),
            'peak_time_s': float(peak_frame / sampling_rate),
            'end_time_s': float(end_frame / sampling_rate),
            'max_cells_active': max_cells_active,
            'mean_cells_active': float(mean_cells_active),
            'interval_to_next_frames': int(interval_frames) if interval_frames is not None else None,
            'interval_to_next_s': float(interval_s) if interval_s is not None else None
        })
    
    # Calculate summary statistics
    durations = [e['duration_frames'] for e in event_data]
    intervals = [e['interval_to_next_frames'] for e in event_data if e['interval_to_next_frames'] is not None]
    
    event_stats = {
        'n_events': len(events),
        'mean_duration_frames': float(np.mean(durations)),
        'std_duration_frames': float(np.std(durations)),
        'min_duration_frames': int(np.min(durations)),
        'max_duration_frames': int(np.max(durations)),
        'mean_duration_ms': float(np.mean(durations) / sampling_rate * 1000),
        'mean_interval_frames': float(np.mean(intervals)) if len(intervals) > 0 else None,
        'std_interval_frames': float(np.std(intervals)) if len(intervals) > 0 else None,
        'mean_interval_s': float(np.mean(intervals) / sampling_rate) if len(intervals) > 0 else None
    }
    
    if verbose:
        print(f"\nEvent statistics:")
        print(f"  Number of events: {event_stats['n_events']}")
        print(f"  Duration: {event_stats['mean_duration_frames']:.1f} ± {event_stats['std_duration_frames']:.1f} frames "
              f"({event_stats['mean_duration_ms']:.0f} ms)")
        print(f"  Range: {event_stats['min_duration_frames']}-{event_stats['max_duration_frames']} frames")
        if event_stats['mean_interval_s'] is not None:
            print(f"  Inter-event interval: {event_stats['mean_interval_s']:.2f} s")
    
    return event_data, event_stats


def save_population_synchrony_to_csv(
    event_data,
    event_stats,
    sync_stats,
    rec_name,
    output_folder,
    sampling_rate=15
):
    """
    Save population synchrony event data to CSV
    
    Parameters:
        event_data: list of event dictionaries
        event_stats: summary statistics
        sync_stats: synchrony detection statistics
        rec_name: recording name
        output_folder: output directory
        sampling_rate: frame rate in Hz
    
    Returns:
        csv_path: path to saved CSV file
    """
    
    if len(event_data) == 0:
        print("No synchronous events to save")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(event_data)
    
    # Create metadata header
    metadata = f"""# Population Synchrony Events - {rec_name}
#
# Detection Parameters:
#   Cells analyzed: {sync_stats['n_cells']}
#   Total frames: {sync_stats['n_frames']}
#   Min fraction coincident: {sync_stats['min_fraction_coincident']*100:.0f}%
#   Min cells required: {sync_stats['min_coincident_cells']}
#   Sampling rate: {sampling_rate} Hz
#
# Synchrony Summary:
#   Total synchronous frames: {sync_stats['synchronous_frames']} ({sync_stats['synchrony_percentage']:.2f}%)
#   Number of events: {event_stats['n_events']}
#   Mean event duration: {event_stats['mean_duration_ms']:.0f} ms ({event_stats['mean_duration_frames']:.1f} frames)
#   Mean inter-event interval: {event_stats.get('mean_interval_s', 'N/A')} s
# """
    
    metadata += "#\n# Column descriptions:\n"
    metadata += "#   event_id: Sequential event number\n"
    metadata += "#   start_frame: First frame of synchronous event\n"
    metadata += "#   peak_frame: Frame with maximum cell participation\n"
    metadata += "#   end_frame: Last frame of synchronous event\n"
    metadata += "#   duration_frames: Event duration in frames\n"
    metadata += "#   duration_ms: Event duration in milliseconds\n"
    metadata += "#   duration_s: Event duration in seconds\n"
    metadata += "#   start_time_s: Event start time in seconds\n"
    metadata += "#   peak_time_s: Event peak time in seconds\n"
    metadata += "#   end_time_s: Event end time in seconds\n"
    metadata += "#   max_cells_active: Maximum number of cells active during event\n"
    metadata += "#   mean_cells_active: Mean number of cells active during event\n"
    metadata += "#   interval_to_next_frames: Frames from this event start to next event start\n"
    metadata += "#   interval_to_next_s: Seconds from this event start to next event start\n"
    
    # Save CSV
    csv_filename = f"{rec_name}_population_synchrony_events.csv"
    csv_path = os.path.join(output_folder, csv_filename)
    
    # Write metadata then data
    with open(csv_path, 'w') as f:
        f.write(metadata + '\n')
    
    df.to_csv(csv_path, mode='a', index=False)
    
    print(f"\n{'='*80}")
    print(f"SAVED POPULATION SYNCHRONY EVENTS")
    print(f"{'='*80}")
    print(f"File: {csv_filename}")
    print(f"Events: {len(event_data)}")
    print(f"Time range: {df['start_time_s'].min():.2f}s to {df['end_time_s'].max():.2f}s")
    print(f"{'='*80}")
    
    return csv_path

# ============================================================================
# SECTION 6: VISUALIZATION FUNCTIONS
# ============================================================================

def plot_filtering_results(dff_data, spike_data, stage1_mask, stage2_mask, 
                           stage1_stats, stage2_stats, rec_name, save_path):
    """Create filtering visualization plots"""
    
    n_cells = len(stage1_mask)
    n_stage1 = np.sum(stage1_mask)
    n_stage2 = np.sum(stage2_mask)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Original data
    im1 = axes[0,0].imshow(dff_data, aspect='auto', cmap='hot', interpolation='nearest')
    axes[0,0].set_title(f'Original DFF Data\n{n_cells} ROIs')
    axes[0,0].set_ylabel('ROI Index')
    plt.colorbar(im1, ax=axes[0,0])
    
    # After Stage 1
    dff_stage1 = dff_data[stage1_mask, :]
    im2 = axes[0,1].imshow(dff_stage1, aspect='auto', cmap='hot', interpolation='nearest')
    axes[0,1].set_title(f'After Stage 1 - DFF\n{n_stage1} ROIs ({stage1_stats["pass_rate"]*100:.1f}%)')
    plt.colorbar(im2, ax=axes[0,1])
    
    # After Stage 2 (Final)
    dff_stage2 = dff_data[stage2_mask, :]
    im3 = axes[0,2].imshow(dff_stage2, aspect='auto', cmap='hot', interpolation='nearest')
    axes[0,2].set_title(f'After Stage 2 - DFF\n{n_stage2} ROIs ({stage2_stats["overall_pass_rate"]*100:.1f}%)')
    plt.colorbar(im3, ax=axes[0,2])
    
    # Spike data - FIXED SCALE 0 to 1
    im4 = axes[1,0].imshow(spike_data, aspect='auto', cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    axes[1,0].set_title(f'Original Spike Data\n{n_cells} ROIs')
    axes[1,0].set_ylabel('ROI Index')
    axes[1,0].set_xlabel('Frames')
    plt.colorbar(im4, ax=axes[1,0])
    
    spike_stage1 = spike_data[stage1_mask, :]
    im5 = axes[1,1].imshow(spike_stage1, aspect='auto', cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    axes[1,1].set_title(f'After Stage 1 - Spikes\n{n_stage1} ROIs')
    axes[1,1].set_xlabel('Frames')
    plt.colorbar(im5, ax=axes[1,1])
    
    spike_stage2 = spike_data[stage2_mask, :]
    im6 = axes[1,2].imshow(spike_stage2, aspect='auto', cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    axes[1,2].set_title(f'After Stage 2 - Spikes\n{n_stage2} ROIs')
    axes[1,2].set_xlabel('Frames')
    plt.colorbar(im6, ax=axes[1,2])
    
    plt.suptitle(f'Two-Stage Filtering Results - {rec_name}', fontsize=16)
    plt.tight_layout()
    
    filtering_plot_path = os.path.join(save_path, f"{rec_name}_two_stage_filtering.jpg")
    plt.savefig(filtering_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved two-stage filtering visualization to {filtering_plot_path}")
    
    return [filtering_plot_path]

def plot_raster_exclusion_analysis(dff_data, spike_data, stage1_mask, final_mask, 
                                   stage1_stats, stage2_stats, rec_name, save_path):
    """Create raster plots showing what was excluded at each stage"""
    
    n_cells = len(stage1_mask)
    stage1_excluded = ~stage1_mask
    stage2_excluded = stage1_mask & (~final_mask)
    
    n_stage1_excluded = np.sum(stage1_excluded)
    n_stage2_excluded = np.sum(stage2_excluded)
    n_final = np.sum(final_mask)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Final survivors
    if n_final > 0:
        dff_survivors = dff_data[final_mask, :]
        spike_survivors = spike_data[final_mask, :]
        
        dff_raster = dff_survivors > np.percentile(dff_survivors, 75)
        spike_raster = spike_survivors > 0.5  # Binary threshold
        
        axes[0,0].imshow(dff_raster, aspect='auto', cmap='Greens', interpolation='nearest', vmin=0, vmax=1)
        axes[0,0].set_title(f'Final Survivors - DFF Raster\n{n_final} ROIs (KEPT)')
        axes[0,0].set_ylabel('ROI Index')
        
        axes[1,0].imshow(spike_raster, aspect='auto', cmap='Greens', interpolation='nearest', vmin=0, vmax=1)
        axes[1,0].set_title(f'Final Survivors - Spike Raster\n{n_final} ROIs')
        axes[1,0].set_xlabel('Frames')
        axes[1,0].set_ylabel('ROI Index')
    
    # Stage 1 excluded
    if n_stage1_excluded > 0:
        dff_excluded_s1 = dff_data[stage1_excluded, :]
        spike_excluded_s1 = spike_data[stage1_excluded, :]
        
        dff_s1_raster = dff_excluded_s1 > np.percentile(dff_excluded_s1, 75)
        spike_s1_raster = spike_excluded_s1 > 0.5
        
        axes[0,1].imshow(dff_s1_raster, aspect='auto', cmap='Reds', interpolation='nearest', vmin=0, vmax=1)
        axes[0,1].set_title(f'Stage 1 Excluded - DFF Raster\n{n_stage1_excluded} ROIs\n(Poor peak amplitude or variance)')
        
        axes[1,1].imshow(spike_s1_raster, aspect='auto', cmap='Reds', interpolation='nearest', vmin=0, vmax=1)
        axes[1,1].set_title(f'Stage 1 Excluded - Spike Raster\n{n_stage1_excluded} ROIs')
        axes[1,1].set_xlabel('Frames')
    
    # Stage 2 excluded
    if n_stage2_excluded > 0:
        dff_excluded_s2 = dff_data[stage2_excluded, :]
        spike_excluded_s2 = spike_data[stage2_excluded, :]
        
        dff_s2_raster = dff_excluded_s2 > np.percentile(dff_excluded_s2, 75)
        spike_s2_raster = spike_excluded_s2 > 0.5
        
        axes[0,2].imshow(dff_s2_raster, aspect='auto', cmap='Oranges', interpolation='nearest', vmin=0, vmax=1)
        axes[0,2].set_title(f'Stage 2 Excluded - DFF Raster\n{n_stage2_excluded} ROIs\n(Passed Stage 1, failed SNR/events)')
        
        axes[1,2].imshow(spike_s2_raster, aspect='auto', cmap='Oranges', interpolation='nearest', vmin=0, vmax=1)
        axes[1,2].set_title(f'Stage 2 Excluded - Spike Raster\n{n_stage2_excluded} ROIs')
        axes[1,2].set_xlabel('Frames')
    
    plt.suptitle(f'Raster Pattern Analysis - {rec_name}', fontsize=16)
    plt.tight_layout()
    
    raster_path = os.path.join(save_path, f"{rec_name}_raster_exclusion_analysis.jpg")
    plt.savefig(raster_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved raster exclusion analysis to {raster_path}")
    return raster_path

def create_final_summary_with_synchrony(
    dff_max_corr, spike_max_corr,
    dff_corr_stats, spike_corr_stats,
    event_data, event_stats,
    dff_processed, spikes_processed, n_original_cells,
    final_mask, rec_name, save_path,
    dff_random_mean_corr=None, spike_random_mean_corr=None):

    """Create final summary plot with cross-correlation and synchrony results"""

    fig, axes = plt.subplots(1,3, figsize=(12, 4))

    # DFF max cross-correlations
    im1 = axes[0].imshow(dff_max_corr, aspect='auto', cmap='Reds',
                          interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(im1, ax=axes[0], label='Correlation')
    dff_random_text = f" | Random: {dff_random_mean_corr:.3f}" if dff_random_mean_corr is not None else ""
    axes[0].set_title(f'DFF Max Cross-Correlation\nMean: {dff_corr_stats["mean_max_correlation"]:.3f}{dff_random_text}')
    axes[0].set_xlabel('Cells')
    axes[0].set_ylabel('Cells')

    # Spike max cross-correlations
    im2 = axes[1].imshow(spike_max_corr, aspect='auto', cmap='Reds',
                          interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(im2, ax=axes[1], label='Correlation')
    spike_random_text = f" | Random: {spike_random_mean_corr:.3f}" if spike_random_mean_corr is not None else ""
    axes[1].set_title(f'Spike Max Cross-Correlation\nMean: {spike_corr_stats["mean_max_correlation"]:.3f}{spike_random_text}')
    axes[1].set_xlabel('Cells')
    axes[1].set_ylabel('Cells')

    # Summary statistics
    axes[2].axis('off')

    dff_random_line = f"- DFF random (chance) mean: {dff_random_mean_corr:.3f}\n" if dff_random_mean_corr is not None else ""
    spike_random_line = f"- Spike random (chance) mean: {spike_random_mean_corr:.3f}\n" if spike_random_mean_corr is not None else ""

    summary_text = f"""
Processing Summary - {rec_name}

Original cells: {n_original_cells}
Filtered cells: {dff_processed.shape[0]} ({np.sum(final_mask)/n_original_cells*100:.1f}%)

Cross-Correlation Results:
- DFF mean: {dff_corr_stats['mean_max_correlation']:.3f}
- Spike mean: {spike_corr_stats['mean_max_correlation']:.3f}
{dff_random_line}{spike_random_line}
Population Synchrony:
- Events detected: {len(event_data)}
"""
    
    if len(event_data) > 0:
        summary_text += f"- Mean duration: {event_stats['mean_duration_ms']:.0f} ms\n"
        if event_stats.get('mean_interval_s'):
            summary_text += f"- Mean interval: {event_stats['mean_interval_s']:.2f} s\n"
    
    summary_text += "\nData saved to processed_POPULATION_SYNC.h5"

    axes[2].text(0.05, 0.95, summary_text, transform=axes[2].transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle(f'Population Synchrony Analysis Results - {rec_name}', fontsize=14)
    plt.tight_layout()
    
    summary_path = os.path.join(save_path, f"{rec_name}_population_sync_summary.jpg")
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Analysis summary plot saved: {summary_path}")
    return summary_path

# ============================================================================
# SECTION 7: UTILITY FUNCTIONS
# ============================================================================

def convert_tuples_to_lists(obj):
    """Recursively convert tuples to lists and numpy types to Python types for HDF5 saving"""
    if isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, dict):
        return {k: convert_tuples_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_tuples_to_lists(item) for item in obj]
    elif isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    else:
        return obj


# ============================================================================
# MAIN PROCESSING LOOP
# ============================================================================

# Configuration parameters
folder_path = r'E:\HERE_SOOBINA\B3\B3_D150_GC'
subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
num_subfolders = len(subfolders)
summary_csv_path = os.path.join(folder_path, 'correlation_summary.csv')

ENABLE_FILTERING = True

# Stage 1: RELAXED Basic Signal Quality Parameters
stage1_params = {
    'peak_percentile': 10,
    'variance_low_percentile': 10,
    'variance_high_percentile': 95,
    'use_dff_for_filtering': False
}

# Stage 2: RELAXED Event-Based SNR Parameters
stage2_params = {
    'snr_threshold': 1.2,
    'min_events': 1,
    'event_detection_method': 'adaptive_threshold',
    'threshold_factor': 2.0,
    'min_duration': 3,
    'use_dff_for_snr': False
}

# Stage 3: Preprocessing Parameters (for cross-correlation)
neural_smoothing_params = {
    'enable_preprocessing': True,
    'apply_gaussian_smoothing': True,
    'gaussian_sigma': 1.5,
    'apply_temporal_binning': False,
    'temporal_bin_size': 2,
    'use_full_timeseries': True,
    'apply_to_dff': True,
    'apply_to_spikes': True,
}

# Cross-correlation parameters
cross_correlation_params = {
    'max_lag': 3,  # ±3 frames = ±200ms at 15 Hz
}

# Population synchrony parameters
synchrony_params = {
    'min_fraction_coincident': 0.10,  # 10% of cells
    'compute_shuffle_baseline': True,
    'n_shuffles': 100
}

print("="*80)
print("POPULATION-LEVEL SYNCHRONY ANALYSIS PIPELINE")
print("="*80)

# Loop through the subfolders
for subfolder in tqdm(subfolders):
    try:
        basepath = subfolder
        folder_name = os.path.basename(subfolder)
        rec_name = folder_name
        
        print(f"\n{'='*80}")
        print(f"STARTING PROCESSING: {folder_name}")
        print(f"{'='*80}")
        print(f"Basepath: {basepath}")
        
        # Create output folder
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        output_folder_name = f"{date_str}_{folder_name}_population_sync_data"
        output_folder = os.path.join(basepath, output_folder_name)
        
        try:
            if os.path.exists(output_folder):
                shutil.rmtree(output_folder)
            os.makedirs(output_folder, exist_ok=True)
            
            test_file = os.path.join(output_folder, "test_write.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            
            print(f"Created output folder: {output_folder_name}")
            
        except PermissionError:
            print(f"ERROR: Permission denied for folder: {folder_name}")
            continue
        except Exception as e:
            print(f"ERROR: Output folder creation failed: {e}")
            continue
        
        # Calculate dFF
        print("\nStep 1: Loading TwoP data...")
        twop_data = TwoP(basepath, rec_name)
        twop_data.find_files()
        twop_dict = twop_data.calc_dFF()
        
        DFF_final = twop_dict['norm_dFF'].copy()
        numFrames = DFF_final.shape[1] if DFF_final.ndim > 1 else 0
        n_cells = DFF_final.shape[0]
        print(f"Loaded: {n_cells} cells, {numFrames} frames")
        
        # Get frame rate
        xml_path = os.path.join(basepath, f"{basepath}.xml")
        if os.path.exists(xml_path):
            xml_dict = files.read_xml(xml_path)
            frameRate = 1/xml_dict["rel_time"][1]
        else:
            frameRate = 15.023
        
        # Calculate spikes
        print("\nStep 2: Processing spikes...")
        raw_spikes, norm_spikes = process_spike_data_gcamp6m(DFF_final, n_cells, numFrames, sampling_rate=frameRate)
        
        # ========================================
        # TWO-STAGE FILTERING
        # ========================================
        
        if ENABLE_FILTERING:
            print(f"\n{'='*80}")
            print("Step 3: Two-stage filtering...")
            print(f"{'='*80}")
            
            # Stage 1
            print("\nStep 3a: Stage 1 filtering...")
            stage1_mask, stage1_stats = basic_signal_quality_filter(
                DFF_final, norm_spikes, **stage1_params
            )
            print(f"Stage 1: {np.sum(stage1_mask)}/{n_cells} cells passed ({np.sum(stage1_mask)/n_cells*100:.1f}%)")

            # Stage 2
            print("\nStep 3b: Stage 2 filtering...")
            final_mask, stage2_stats = event_based_snr_filter(
                DFF_final, norm_spikes, stage1_mask,
                sampling_rate=frameRate, **stage2_params
            )
            print(f"Stage 2: {np.sum(final_mask)}/{n_cells} cells passed ({np.sum(final_mask)/n_cells*100:.1f}%)")
            
            print(f"\nFILTERING RESULTS:")
            print(f"  Original ROIs: {n_cells}")
            print(f"  Stage 1 survivors: {np.sum(stage1_mask)} ({stage1_stats['pass_rate']*100:.1f}%)")
            print(f"  Final survivors: {np.sum(final_mask)} ({stage2_stats['overall_pass_rate']*100:.1f}%)")
            
            # Create filtering visualization
            try:
                plot_filtering_results(DFF_final, norm_spikes, stage1_mask, final_mask,
                                     stage1_stats, stage2_stats, rec_name, output_folder)
                
                plot_raster_exclusion_analysis(DFF_final, norm_spikes, stage1_mask, final_mask,
                                             stage1_stats, stage2_stats, rec_name, output_folder)
                
            except Exception as e:
                print(f"ERROR: Filtering visualization failed: {e}")
            
            # Create filtered datasets
            DFF_filtered = DFF_final[final_mask, :]
            spikes_filtered = norm_spikes[final_mask, :]
            
            DFF_for_preprocessing = DFF_filtered
            spikes_for_preprocessing = spikes_filtered
            correlation_suffix = "_filtered"
            filtering_stats = stage2_stats
            
        else:
            print("Step 3: Skipping filtering...")
            DFF_for_preprocessing = DFF_final
            spikes_for_preprocessing = norm_spikes
            correlation_suffix = "_unfiltered"
            filtering_stats = {'overall_pass_rate': 1.0, 'stage2_survivors': n_cells}
            stage1_mask = np.ones(n_cells, dtype=bool)
            final_mask = np.ones(n_cells, dtype=bool)
            stage1_stats = {}
            stage2_stats = {}
        
        # ========================================
        # PREPROCESSING FOR CROSS-CORRELATION
        # ========================================
        
        if neural_smoothing_params['enable_preprocessing']:
            print(f"\n{'='*80}")
            print(f"Step 4: Preprocessing for cross-correlation...")
            print(f"{'='*80}")
            
            # Apply to DFF data
            print("\nStep 4a: Preprocessing DFF...")
            DFF_processed, DFF_active_mask, DFF_preprocessing_stats = preprocessing_pipeline(
                DFF_for_preprocessing,
                temporal_bin_size=neural_smoothing_params['temporal_bin_size'],
                gaussian_sigma=neural_smoothing_params['gaussian_sigma'],
                sampling_rate=frameRate,
                apply_temporal_binning=neural_smoothing_params['apply_temporal_binning'],
                apply_gaussian_smoothing=neural_smoothing_params['apply_gaussian_smoothing'],
                use_full_timeseries=neural_smoothing_params['use_full_timeseries']
            )
            
            print(f"DFF preprocessing complete: {DFF_processed.shape}")
            
            # Apply to spike data
            print("\nStep 4b: Preprocessing spikes...")
            spikes_processed, spikes_active_mask, spikes_preprocessing_stats = preprocessing_pipeline(
                spikes_for_preprocessing,
                temporal_bin_size=neural_smoothing_params['temporal_bin_size'],
                gaussian_sigma=neural_smoothing_params['gaussian_sigma'],
                sampling_rate=frameRate,
                apply_temporal_binning=neural_smoothing_params['apply_temporal_binning'],
                apply_gaussian_smoothing=neural_smoothing_params['apply_gaussian_smoothing'],
                use_full_timeseries=neural_smoothing_params['use_full_timeseries']
            )
            print(f"Spike preprocessing complete: {spikes_processed.shape}")
            
            # Use preprocessed data for correlation
            DFF_for_correlation = DFF_processed
            spikes_for_correlation = spikes_processed
            correlation_suffix += "_crosscorr"
            
        else:
            print("Step 4: Skipping preprocessing...")
            DFF_for_correlation = DFF_for_preprocessing
            spikes_for_correlation = spikes_for_preprocessing
            DFF_active_mask = np.ones(DFF_for_preprocessing.shape[1], dtype=bool)
            spikes_active_mask = np.ones(spikes_for_preprocessing.shape[1], dtype=bool)
            DFF_preprocessing_stats = {}
            spikes_preprocessing_stats = {}
        
        # ========================================
        # CROSS-CORRELATION ANALYSIS
        # ========================================
        print(f"\n{'='*80}")
        print(f"Step 5: CROSS-CORRELATION ANALYSIS WITH TIME LAGS")
        print(f"{'='*80}")
        print(f"Using {DFF_for_correlation.shape[0]} ROIs for correlation")
        print(f"DFF data: {DFF_for_correlation.shape}")
        print(f"Spike data: {spikes_for_correlation.shape}")
        
        # Calculate DFF cross-correlations
        print("\nCalculating DFF cross-correlations...")
        DFF_max_corr_matrix, DFF_best_lag_matrix, DFF_standard_corr_matrix, DFF_correlation_stats = \
            calculate_cross_correlation_with_lags(
                DFF_for_correlation, 
                max_lag=cross_correlation_params['max_lag']
            )
        
        print("\nCalculating spike cross-correlations...")
        spikes_max_corr_matrix, spikes_best_lag_matrix, spikes_standard_corr_matrix, spikes_correlation_stats = \
            calculate_cross_correlation_with_lags(
                spikes_for_correlation, 
                max_lag=cross_correlation_params['max_lag']
            )
        
        print(f"\nCross-correlations calculated:")
        print(f"  DFF max mean: {DFF_correlation_stats['mean_max_correlation']:.3f} "
              f"(standard: {DFF_correlation_stats['mean_standard_correlation']:.3f}, "
              f"+{DFF_correlation_stats['improvement_percentage']:.1f}%)")
        print(f"  Spike max mean: {spikes_correlation_stats['mean_max_correlation']:.3f} "
              f"(standard: {spikes_correlation_stats['mean_standard_correlation']:.3f}, "
              f"+{spikes_correlation_stats['improvement_percentage']:.1f}%)")

        # Shuffled (chance) cross-correlation baseline: circularly shift each
        # cell's trace by an independent random amount and recompute the mean
        # max cross-correlation, repeated 1000 times. No subtraction applied yet.
        print("\nCalculating DFF shuffled (chance) cross-correlation baseline...")
        DFF_shuffle_stats = calculate_shuffled_cross_correlation_baseline(
            DFF_for_correlation,
            max_lag=cross_correlation_params['max_lag'],
            n_shuffles=1000
        )

        print("\nCalculating spike shuffled (chance) cross-correlation baseline...")
        spikes_shuffle_stats = calculate_shuffled_cross_correlation_baseline(
            spikes_for_correlation,
            max_lag=cross_correlation_params['max_lag'],
            n_shuffles=1000
        )

        # Append per-recording correlation summary to CSV
        summary_row = pd.DataFrame([{
            'rec_name': rec_name,
            'dff_max_corr': DFF_correlation_stats['mean_max_correlation'],
            'spikes_max_corr': spikes_correlation_stats['mean_max_correlation'],
            'dff_random_mean_corr': DFF_shuffle_stats['mean_random_max_correlation'],
            'spike_random_mean_corr': spikes_shuffle_stats['mean_random_max_correlation']
        }])
        summary_row.to_csv(summary_csv_path, mode='a', header=not os.path.exists(summary_csv_path), index=False)

        # ========================================
        # POPULATION-LEVEL SYNCHRONY ANALYSIS
        # ========================================
        print(f"\n{'='*80}")
        print(f"Step 6: POPULATION-LEVEL SYNCHRONY ANALYSIS")
        print(f"{'='*80}")
        
        if spikes_for_correlation.shape[0] > 0:
            # Step 6a: Robust spike detection
            print("\nStep 6a: Robust spike detection...")
            cell_spike_data_robust, spike_summary = detect_spike_peaks_robust(
                dff_data=DFF_for_correlation,
                sampling_rate=frameRate,
                min_peak_distance_s=0.5,
                prominence_factor=2.0,
                adaptive_smoothing=True,
                detrend=True,
                verbose=True
            )
            
            # Step 6b: Create full-duration transient array
            print("\nStep 6b: Creating population transient array...")
            transient_active, transient_boundaries = create_population_transient_array(
                dff_data=DFF_for_correlation,
                cell_spike_data=cell_spike_data_robust,
                sampling_rate=frameRate,
                verbose=True
            )
            
            # Step 6c: Detect population synchrony events
            print("\nStep 6c: Detecting population synchrony...")
            population_sync_frames, sync_stats = detect_population_synchrony_events(
                transient_active=transient_active,
                min_fraction_coincident=synchrony_params['min_fraction_coincident'],
                sampling_rate=frameRate,
                compute_shuffle_baseline=synchrony_params['compute_shuffle_baseline'],
                n_shuffles=synchrony_params['n_shuffles'],
                verbose=True
            )
            
            # Step 6d: Analyze events (duration, intervals)
            print("\nStep 6d: Analyzing synchrony events...")
            event_data, event_stats = analyze_population_synchrony_events(
                population_sync_frames=population_sync_frames,
                cells_active_per_frame=sync_stats['cells_active_per_frame'],
                sampling_rate=frameRate,
                verbose=True
            )
            
            # Step 6e: Save to CSV
            try:
                csv_path = save_population_synchrony_to_csv(
                    event_data=event_data,
                    event_stats=event_stats,
                    sync_stats=sync_stats,
                    rec_name=rec_name,
                    output_folder=output_folder,
                    sampling_rate=frameRate
                )
            except Exception as e:
                print(f"Warning: Failed to save synchrony CSV: {e}")
                csv_path = None
            
            # Store results for saving to HDF5
            synchrony_results = {
                'transient_boundaries': transient_boundaries,
                'transient_active': transient_active,
                'population_sync_frames': population_sync_frames,
                'sync_stats': sync_stats,
                'event_data': event_data,
                'event_stats': event_stats,
                'csv_path': csv_path,
                'method': 'full_transient_overlap',
                'min_fraction_coincident': synchrony_params['min_fraction_coincident']
            }
            
        else:
            synchrony_results = {
                'note': 'No cells available for synchrony analysis'
            }
            event_data = []
            event_stats = {}

        # ========================================
        # SAVE RESULTS
        # ========================================
        print(f"\n{'='*80}")
        print(f"Step 7: SAVING CONSOLIDATED RESULTS")
        print(f"{'='*80}")
        
        consolidated_data = {
            'recording_info': {
                'recording_name': rec_name,
                'basepath': basepath,
                'output_folder': output_folder_name,
                'frame_rate': frameRate,
                'total_frames': numFrames,
                'original_cell_count': n_cells,
                'processing_date': str(np.datetime64('now')),
                'pipeline_version': 'population_synchrony_v1'
            },
            'filtering_results': {
                'filtering_applied': ENABLE_FILTERING,
                'relaxed_filtering': True,
                'stage1_mask': stage1_mask,
                'stage2_mask': final_mask,
                'stage1_survivors': np.sum(stage1_mask),
                'stage2_survivors': np.sum(final_mask),
                'stage1_stats': stage1_stats,
                'stage2_stats': stage2_stats,
                'stage1_params': stage1_params,
                'stage2_params': stage2_params,
                'final_cell_count': DFF_for_correlation.shape[0]
            },
            'preprocessing_results': {
                'preprocessing_applied': neural_smoothing_params['enable_preprocessing'],
                'neural_smoothing_params': neural_smoothing_params,
                'dff_preprocessing_stats': DFF_preprocessing_stats,
                'spikes_preprocessing_stats': spikes_preprocessing_stats,
                'preprocessing_method': 'full_timeseries_for_cross_correlation'
            },
            'cross_correlation_analysis': {
                'max_lag_frames': cross_correlation_params['max_lag'],
                'max_lag_ms': cross_correlation_params['max_lag'] * 66.7,
                'dff_max_correlation_matrix': DFF_max_corr_matrix,
                'dff_standard_correlation_matrix': DFF_standard_corr_matrix,
                'dff_best_lag_matrix': DFF_best_lag_matrix,
                'dff_correlation_stats': DFF_correlation_stats,
                'spikes_max_correlation_matrix': spikes_max_corr_matrix,
                'spikes_standard_correlation_matrix': spikes_standard_corr_matrix,
                'spikes_best_lag_matrix': spikes_best_lag_matrix,
                'spikes_correlation_stats': spikes_correlation_stats,
                'correlation_method': 'cross_correlation_with_time_lags',
                'cells_used_for_correlation': DFF_for_correlation.shape[0]
            },
            'shuffled_baseline_analysis': {
                'dff_shuffle_stats': DFF_shuffle_stats,
                'spikes_shuffle_stats': spikes_shuffle_stats,
                'n_shuffles': 1000
            },
            'population_synchrony_analysis': synchrony_results,
            'processed_data': {
                'dff_processed': DFF_for_correlation,
                'spikes_processed': spikes_for_correlation,
                'data_shape': list(DFF_for_correlation.shape),
                'temporal_resolution_ms': (neural_smoothing_params['temporal_bin_size'] * 1000 / frameRate) if neural_smoothing_params['enable_preprocessing'] else (1000 / frameRate)
            }
        }
        
        consolidated_data = convert_tuples_to_lists(consolidated_data)
        
        consolidated_filename = f"{folder_name}_processed_POPULATION_SYNC.h5"
        consolidated_path = os.path.join(output_folder, consolidated_filename)
        
        print(f"Saving consolidated data to: {consolidated_filename}")
        
        try:
            files.write_h5(consolidated_path, consolidated_data)
            
            if os.path.exists(consolidated_path):
                file_size = os.path.getsize(consolidated_path)
                print(f"Consolidated file saved ({file_size} bytes)")
                print(f"Contains:")
                print(f"   DFF max correlation matrix: {DFF_max_corr_matrix.shape}")
                print(f"   Spike max correlation matrix: {spikes_max_corr_matrix.shape}")
                print(f"   Transient boundaries: {len(transient_boundaries)} cells")
                print(f"   Population synchrony events: {len(event_data)}")
                print(f"   Complete population-level analysis")
            
        except Exception as e:
            print(f"ERROR: Consolidated file saving failed: {e}")
            import traceback
            traceback.print_exc()
        
        # ========================================
        # CREATE FINAL SUMMARY VISUALIZATION
        # ========================================
        
        print("\nCreating final summary visualization...")
        try:
            create_final_summary_with_synchrony(
                DFF_max_corr_matrix, spikes_max_corr_matrix,
                DFF_correlation_stats, spikes_correlation_stats,
                event_data, event_stats,
                DFF_for_correlation, spikes_for_correlation, n_cells,
                final_mask, rec_name, output_folder,
                dff_random_mean_corr=DFF_shuffle_stats['mean_random_max_correlation'],
                spike_random_mean_corr=spikes_shuffle_stats['mean_random_max_correlation']
            )
        except Exception as e:
            print(f"ERROR: Final summary plot creation failed: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n{'='*80}")
        print(f"PROCESSING COMPLETE FOR {folder_name}")
        print(f"{'='*80}")
        print(f"All results saved in folder: {output_folder_name}")
        print(f"Main consolidated file: {consolidated_filename}")
        print(f"\nKey Results:")
        print(f"  DFF correlation: {DFF_correlation_stats['mean_max_correlation']:.3f}")
        print(f"  Spike correlation: {spikes_correlation_stats['mean_max_correlation']:.3f}")
        print(f"  DFF random baseline: {DFF_shuffle_stats['mean_random_max_correlation']:.3f}")
        print(f"  Spike random baseline: {spikes_shuffle_stats['mean_random_max_correlation']:.3f}")
        if len(event_data) > 0:
            print(f"  Population synchrony events: {len(event_data)}")
            print(f"  Mean event duration: {event_stats['mean_duration_ms']:.0f} ms")
            if event_stats.get('mean_interval_s') is not None:
                print(f"  Mean inter-event interval: {event_stats['mean_interval_s']:.2f} s")
        else:
            print(f"  No population synchrony events detected")
        print(f"{'='*80}")
        
        gc.collect()
        
    except Exception as e:
        print(f"ERROR in {folder_name}: {e}")
        import traceback
        traceback.print_exc()
        print("CONTINUING TO NEXT FOLDER...")
        continue

print("\n" + "="*80)
print("POPULATION-LEVEL SYNCHRONY BATCH PROCESSING COMPLETE")
print("="*80)