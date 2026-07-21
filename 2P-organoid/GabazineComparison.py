"""
===============================================================================
GabazineComparison.py — Pre/Post Gabazine Neural Activity Comparison
Created : 2026-03-24
Last Modified : 2026-07-20

Purpose
-------
Computes and compares pairwise correlation coefficients before and after
Gabazine (GABA-A antagonist) application, supporting both single and
multiple post-treatment recording epochs. Cross-correlation is vectorized
and paired with a shuffled chance-level baseline; per-recording results
(PRE and POST) are cached so each unique recording within an organoid is
loaded, filtered, correlated, and shuffled only once, no matter how many
PRE x POST pairings reuse it

Pipeline Position
-----------------
Runs after  : CalculateCC.py
Runs before : (downstream statistical analysis / figure generation)

Notes
-----
- Set BASE_FOLDERS (list) to the base directories containing organoid
  subfolders; each is run through run_full_pipeline() in turn.
- PRE recording: folder name contains no '_GZ_' tag.
- POST recordings: folder names contain '_GZ_001', '_GZ_002', etc.
- calculate_cross_correlation_with_lags() is vectorized (matrix ops across
  all lags/cells at once) instead of the original per-pair loop.
- calculate_shuffled_cross_correlation_baseline() estimates a chance-level
  correlation via independent circular shifts (N_SHUFFLES, default 1000).
- get_pre_recording_data() / get_post_recording_data() cache filtering +
  correlation + shuffle results per recording path, shared across TYPE 1
  pairwise comparisons and the TYPE 2 mega-figure within an organoid.
- If a base_folder has no per-trial XML subfolders to deconcat from,
  split_combined_suite2p_equal_pre_post() falls back to an equal-halves
  PRE/POST split.
- append_summary_row() writes one row per unique recording (not per
  pairwise comparison) to SUMMARY_CSV_PATH.

===============================================================================
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\WSDup")

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from tqdm import tqdm
import datetime
import pandas as pd
import warnings
import re
warnings.filterwarnings('ignore')

from helper import files, TwoP, process_spike_data_gcamp6m

# Number of circular-shift iterations for the shuffled (chance) cross-correlation
# baseline. Change this single constant to adjust it everywhere it's used.
N_SHUFFLES = 1000

# Path to the running per-recording correlation summary CSV, one row per
# unique PRE/POST recording (not per pairwise comparison). Set once per
# batch run inside batch_process_organoids_enhanced().
SUMMARY_CSV_PATH = None

# ============================================================================
# DECONCAT FUNCTIONS - Run these first if needed
# ============================================================================

def Frame_Detector_Per_Lap(base_folder):
    """
    Automatically split all recordings into the number of recordings (pre and post treatment)
    
    Parameters:
        base_folder: Base folder containing recordings
    """
    
    print(f"\n{'='*70}")
    print("AUTO-SPLIT: Equal Pre/Post Durations")
    print(f"{'='*70}")
    print(f"Base folder: {base_folder}\n")
    
    all_filelist = os.listdir(base_folder)
    recording_folderlist = [x for x in all_filelist if not x.endswith('.csv') and 'suite2p' not in x]
    
    # Sort folders: PRE first, then POST in numerical order
    def sort_key(folder_name):
        folder_upper = folder_name.upper()
        has_gz = ('_GZ_' in folder_upper or '-GZ-' in folder_upper or 
                  '_GZ-' in folder_upper or '-GZ_' in folder_upper)
        
        if not has_gz:
            # PRE recording - should be first (return 0)
            return (0, 0)
        else:
            # POST recording - extract number after GZ
            # Match patterns like GZ-001, GZ_001, _GZ-001, etc.
            import re
            match = re.search(r'[_-]GZ[_-](\d+)', folder_upper)
            if match:
                gz_num = int(match.group(1))
                return (1, gz_num)  # (1, X) ensures all POST come after PRE
            else:
                return (1, 999)  # Fallback for malformed names
    
    recording_folderlist = sorted(recording_folderlist, key=sort_key)

    num_recordings = len(recording_folderlist)
    print(f"Subfolders found (sorted): {recording_folderlist}")

    if num_recordings == 0:
        print("No per-trial subfolders (with their own .xml) found -- "
              "nothing to derive PRE/POST frame counts from.")
        return None

    suite2p_folders = None
    for item in os.scandir(base_folder):
        if item.is_dir():
            plane0_path = os.path.join(item.path, 'plane0')
            if os.path.exists(plane0_path):
                suite2p_folders = os.path.join(base_folder, item.name)

    if suite2p_folders is None:
        print("No recordings found with Suite2p data")
        return None

    ops_path = os.path.join(suite2p_folders, 'plane0', 'ops.npy')
    ops = np.load(ops_path, allow_pickle=True).item()
    total_frames = ops['nframes']

    recording_frame_rate = np.zeros(num_recordings)
    num_frames = np.zeros(num_recordings)
    for subfolder in range(len(recording_folderlist)):
        folder_name = recording_folderlist[subfolder]
        xml_path = os.path.join(base_folder, folder_name, f"{folder_name}.xml")
        if os.path.exists(xml_path):
            try:
                xml_dict = files.read_xml(xml_path)
                recording_frame_rate[subfolder] = float(1/xml_dict["rel_time"][1])
                num_frames[subfolder] = xml_dict['num_frames']
            except:
                print(f"  XML found but could not parse")

    frame_info = {
        'recording_names': recording_folderlist,
        'num_recordings': num_recordings,
        'suite2p_folder': suite2p_folders,
        'total_frames': total_frames,
        'frame_rate': recording_frame_rate,
        'num_frames': num_frames
    }
    
    return frame_info

def deconcat_suite2p_output(frame_info, recording_folder):
    """
    Deconcatenate the suite2p output files for each trial within the session
    Input: session_directory - folder containing a single suite2p output folder for all trials within the session
    Output: deconcatenated F, Fneu, spks, iscell, stat, ops for each trial
    """
    
    # Find all trial directories
    if frame_info is not None:
        trial_dir = frame_info['recording_names']
        num_trials = len(trial_dir)
        total_frames = frame_info['total_frames']
        num_frames = frame_info['num_frames']

        # Load twoP data    
        F = np.load(os.path.join(recording_folder, r'suite2p/plane0/F.npy'), allow_pickle=True)
        Fneu = np.load(os.path.join(recording_folder, r'suite2p/plane0/Fneu.npy'), allow_pickle=True)
        iscell = np.load(os.path.join(recording_folder, r'suite2p/plane0/iscell.npy'), allow_pickle=True)
        stat = np.load(os.path.join(recording_folder, r'suite2p/plane0/stat.npy'), allow_pickle=True)
        ops = np.load(os.path.join(recording_folder, r'suite2p/plane0/ops.npy'), allow_pickle=True).item()
        spks = np.load(os.path.join(recording_folder, r'suite2p/plane0/spks.npy'), allow_pickle=True)

        usecells = iscell[:, 0] == 1

        twoP_data = {}
        twoP_data['F'] = F[usecells, :]
        twoP_data['Fneu'] = Fneu[usecells, :]
        twoP_data['stat'] = stat[usecells]
        twoP_data['ops'] = ops
        twoP_data['spks'] = spks[usecells, :]
        twoP_data['iscell'] = iscell[usecells, :]
        
        #  Verify frame counts BEFORE splitting
        num_frames_per_trial = total_frames / num_trials

        if num_frames_per_trial - np.mean(num_frames) == 1:
            print(f"  ⚠️ Frame counts are inconsistent: {num_frames_per_trial} vs {np.mean(num_frames)}")
        elif num_frames_per_trial - np.mean(num_frames) == 0:
            print(f"  Frame counts are consistent: {num_frames_per_trial} frames per trial")
            return

        total_frames_in_suite2p = twoP_data['F'].shape[1]
        total_frames_expected = total_frames
        
        print(f"\nVerifying frame counts:")
        print(f"  Suite2p actual: {total_frames_in_suite2p} frames")
        print(f"  XML expected: {total_frames_expected} frames")
        
        cumulative_frames = np.zeros(num_trials)
        for i in range(num_trials):
            print(f"  Trial {i+1}: {int(num_frames[i])} frames")
            if i == 0:
                cumulative_frames[i] = num_frames[i]-1
            else:
                cumulative_frames[i] = cumulative_frames[i-1]-1 + num_frames[i]

        # Split and save data for each trial
        for i in range(num_trials):
            print(f"\nProcessing trial {i+1}/{num_trials}...")
            
            # Create output directory
            output_dir = os.path.join(recording_folder, trial_dir[i], r'suite2p/plane0/')
            os.makedirs(output_dir, exist_ok=True)
            
            # Determine frame range for this trial
            start_frame = 0 if i == 0 else int(cumulative_frames[i-1])
            end_frame = int(cumulative_frames[i])-1
            print(f"  Saving frames {start_frame} to {end_frame}")
            
            # Process each data type
            for key in twoP_data.keys():
                if key in ['F', 'Fneu', 'spks']:
                    # Split temporal data by frames
                    data_split = twoP_data[key][:, start_frame:end_frame]
                    print(f"  {key} shape: {data_split.shape}")
                else:
                    # Non-temporal data (stat, ops, iscell) - same for all trials
                    data_split = twoP_data[key]
                
                # Save the split data
                np.save(os.path.join(output_dir, f'{key}.npy'), data_split)
            
            print(f"  Saved to {output_dir}")
        
        print("\nDeconcatenation complete!")
        print(f"Total frames distributed: {int(cumulative_frames[-1])}")
    else:
        print("No Suite2p folder found. Exiting deconcatenation.")
    return cumulative_frames


def split_combined_suite2p_equal_pre_post(organoid_folder, organoid_name):
    """
    Fallback deconcat for organoid folders that hold a single combined
    suite2p output (PRE + POST concatenated) with no per-trial XML
    subfolders to read frame counts from (Frame_Detector_Per_Lap
    returns None in that case).

    Splits the recording into two equal halves by frame count: first
    half = PRE, second half = POST. Any trailing _GZ on organoid_name
    is stripped first so the PRE subfolder's name doesn't itself look
    like a POST folder to the _GZ-based classifier used elsewhere.
    """

    suite2p_dir = os.path.join(organoid_folder, 'suite2p', 'plane0')

    F = np.load(os.path.join(suite2p_dir, 'F.npy'), allow_pickle=True)
    Fneu = np.load(os.path.join(suite2p_dir, 'Fneu.npy'), allow_pickle=True)
    spks = np.load(os.path.join(suite2p_dir, 'spks.npy'), allow_pickle=True)
    stat = np.load(os.path.join(suite2p_dir, 'stat.npy'), allow_pickle=True)
    ops = np.load(os.path.join(suite2p_dir, 'ops.npy'), allow_pickle=True).item()
    iscell = np.load(os.path.join(suite2p_dir, 'iscell.npy'), allow_pickle=True)

    usecells = iscell[:, 0] == 1
    twoP_data = {
        'F': F[usecells, :],
        'Fneu': Fneu[usecells, :],
        'spks': spks[usecells, :],
        'stat': stat[usecells],
        'ops': ops,
        'iscell': iscell[usecells, :],
    }

    total_frames = twoP_data['F'].shape[1]
    n_pre = total_frames // 2
    if total_frames % 2 != 0:
        print(f"  ⚠️  {total_frames} frames is odd -- splitting {n_pre}/{total_frames - n_pre}")

    base_name = re.sub(r'[_-]GZ$', '', organoid_name, flags=re.IGNORECASE)
    trial_ranges = {
        base_name: (0, n_pre),
        f'{base_name}_GZ-001': (n_pre, total_frames),
    }

    for trial_name, (start, end) in trial_ranges.items():
        output_dir = os.path.join(organoid_folder, trial_name, 'suite2p', 'plane0')
        os.makedirs(output_dir, exist_ok=True)
        for key, data in twoP_data.items():
            data_split = data[:, start:end] if key in ('F', 'Fneu', 'spks') else data
            np.save(os.path.join(output_dir, f'{key}.npy'), data_split)
        print(f"  ✓ Saved {trial_name}: frames {start}-{end} ({end - start} frames)")

    return trial_ranges

# ============================================================================
# NEW: Parse folder name to detect multiple post recordings
# ============================================================================

def parse_folder_for_multi_post(folder_name):
    """
    Parse folder name to detect if it has multiple post recordings.
    
    Pattern: ..._x000frames where x indicates total number of recordings
    
    Returns:
        (has_multi_post, expected_recordings)
    """
    pattern = r'_(\d+)000frames'
    match = re.search(pattern, folder_name)
    
    if match:
        expected_recordings = int(match.group(1))
        if expected_recordings > 2:  # More than 1 PRE + 1 POST
            return True, expected_recordings
    
    return False, None

# ============================================================================
# ORIGINAL FUNCTIONS - UNCHANGED
# ============================================================================

def create_roi_mask(stat_entry):
    """Create ROI boundary from Suite2p stat"""
    ypix = stat_entry['ypix']
    xpix = stat_entry['xpix']
    
    try:
        points = np.column_stack([xpix, ypix])
        hull = ConvexHull(points)
        hull_x = points[hull.vertices, 0]
        hull_y = points[hull.vertices, 1]
        return hull_y, hull_x
    except:
        return ypix, xpix

def plot_roi_outline(ax, stat_entry, color, linewidth=2.5, label_num=None):
    """Plot ROI outline"""
    ypix, xpix = create_roi_mask(stat_entry)
    ypix_closed = np.append(ypix, ypix[0])
    xpix_closed = np.append(xpix, xpix[0])
    
    ax.plot(xpix_closed, ypix_closed, color=color, linewidth=linewidth, alpha=0.9)
    
    if label_num is not None:
        centroid_y = stat_entry['med'][0]
        centroid_x = stat_entry['med'][1]
        ax.text(centroid_x, centroid_y, str(label_num), 
               color='white', fontsize=11, weight='bold',
               ha='center', va='center',
               bbox=dict(boxstyle='circle,pad=0.3', facecolor=color, 
                       edgecolor='white', linewidth=2, alpha=0.85))

def basic_signal_quality_filter(dff_data, spike_data, 
                               peak_percentile=10, 
                               variance_low_percentile=10, 
                               variance_high_percentile=95,
                               use_dff_for_filtering=False):
    """Stage 1: Basic signal quality filtering"""
    
    n_cells, n_frames = dff_data.shape
    print(f"\n=== STAGE 1: Basic Signal Quality Filtering ===")
    print(f"Input: {n_cells} ROIs, {n_frames} frames")
    
    filter_data = dff_data if use_dff_for_filtering else spike_data
    data_type = "DFF" if use_dff_for_filtering else "Spike"
    print(f"Using {data_type} data for filtering")
    
    peak_amplitudes = np.zeros(n_cells)
    variances = np.zeros(n_cells)
    
    print("Calculating signal quality metrics...")
    for i in tqdm(range(n_cells), desc="Analyzing cells"):
        roi_trace = filter_data[i, :]
        peak_amplitudes[i] = np.max(roi_trace)
        variances[i] = np.var(roi_trace)
    
    peak_threshold = np.percentile(peak_amplitudes, peak_percentile)
    var_low_threshold = np.percentile(variances, variance_low_percentile)
    var_high_threshold = np.percentile(variances, variance_high_percentile)
    
    peak_pass = peak_amplitudes >= peak_threshold
    variance_pass = (variances > var_low_threshold) & (variances < var_high_threshold)
    filtering_mask = peak_pass & variance_pass
    
    n_filtered_pass = np.sum(filtering_mask)
    
    print(f"\nFiltering results:")
    print(f"  Passed filtering: {n_filtered_pass}/{n_cells} ({n_filtered_pass/n_cells*100:.1f}%)")
    
    filtering_stats = {
        'input_rois': n_cells,
        'filtered_rois': n_filtered_pass,
        'pass_rate': n_filtered_pass/n_cells,
        'peak_threshold': peak_threshold,
        'variance_thresholds': (var_low_threshold, var_high_threshold),
    }
    
    return filtering_mask, filtering_stats

def event_based_snr_filter(dff_data, spike_data, stage1_mask,
                          snr_threshold=1.2, min_events=1,
                          threshold_factor=2.0, min_duration=3,
                          sampling_rate=10):
    """Stage 2: Event-based SNR filtering"""
    
    n_cells = dff_data.shape[0]
    stage1_survivors = np.sum(stage1_mask)
    
    print(f"\n=== STAGE 2: Event-Based SNR Filtering ===")
    print(f"Input: {stage1_survivors} ROIs (survivors from Stage 1)")
    
    stage2_mask = np.zeros(n_cells, dtype=bool)
    snr_values = np.full(n_cells, np.nan)
    
    stage1_indices = np.where(stage1_mask)[0]
    
    for idx in tqdm(stage1_indices, desc="Processing ROIs"):
        roi_trace = spike_data[idx, :]
        
        baseline_median = np.median(roi_trace)
        baseline_mad = np.median(np.abs(roi_trace - baseline_median))
        threshold = baseline_median + threshold_factor * baseline_mad
        
        above_threshold = roi_trace > threshold
        event_count = 0
        in_event = False
        event_lengths = []
        
        for val in above_threshold:
            if val and not in_event:
                in_event = True
                event_len = 1
            elif val and in_event:
                event_len += 1
            elif not val and in_event:
                if event_len >= min_duration:
                    event_count += 1
                    event_lengths.append(event_len)
                in_event = False
        
        if event_count >= min_events:
            event_frames = above_threshold
            quiet_frames = ~event_frames
            
            if np.sum(quiet_frames) > 5:
                quiet_std = np.std(roi_trace[quiet_frames])
                if quiet_std > 1e-10:
                    peak_response = np.max(roi_trace[event_frames])
                    quiet_mean = np.mean(roi_trace[quiet_frames])
                    snr = (peak_response - quiet_mean) / quiet_std
                    snr_values[idx] = snr
                    
                    if snr >= snr_threshold:
                        stage2_mask[idx] = True
    
    n_stage2_pass = np.sum(stage2_mask)
    
    print(f"\nStage 2 results:")
    print(f"  Passed Stage 2: {n_stage2_pass}/{stage1_survivors} ({n_stage2_pass/stage1_survivors*100:.1f}%)")
    
    filtering_stats = {
        'stage1_survivors': stage1_survivors,
        'stage2_survivors': n_stage2_pass,
        'snr_values': snr_values,
    }
    
    return stage2_mask, filtering_stats

def calculate_cross_correlation_with_lags(data, max_lag=3, verbose=True):
    """Calculate cross-correlation with time lags for all cell pairs (vectorized)"""

    n_cells, n_frames = data.shape

    if verbose:
        print(f"\nCalculating cross-correlations with ±{max_lag} frame lags (vectorized)...")

    max_corr_matrix = np.zeros((n_cells, n_cells))
    best_lag_matrix = np.zeros((n_cells, n_cells), dtype=int)
    standard_corr_matrix = np.zeros((n_cells, n_cells))

    means = data.mean(axis=1, keepdims=True)
    stds = data.std(axis=1, keepdims=True)
    norm_data = (data - means) / (stds + 1e-10)

    lags = list(range(-max_lag, max_lag + 1))
    all_corrs = np.zeros((len(lags), n_cells, n_cells))

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

    iu, ju = np.triu_indices(n_cells, k=1)

    max_corr_matrix[iu, ju] = max_corr_vals[iu, ju]
    max_corr_matrix[ju, iu] = max_corr_vals[iu, ju]

    best_lag_matrix[iu, ju] = best_lag_vals[iu, ju]
    best_lag_matrix[ju, iu] = -best_lag_vals[iu, ju]

    standard_corr_matrix[iu, ju] = standard_corr_vals[iu, ju]
    standard_corr_matrix[ju, iu] = standard_corr_vals[iu, ju]

    np.fill_diagonal(max_corr_matrix, 1.0)
    np.fill_diagonal(standard_corr_matrix, 1.0)
    np.fill_diagonal(best_lag_matrix, 0)

    upper_tri = np.triu_indices_from(max_corr_matrix, k=1)
    max_correlations = max_corr_matrix[upper_tri]
    valid_corr = max_correlations[~np.isnan(max_correlations)]

    correlation_stats = {
        'mean_max_correlation': np.mean(valid_corr) if len(valid_corr) > 0 else 0,
        'median_max_correlation': np.median(valid_corr) if len(valid_corr) > 0 else 0,
    }

    if verbose:
        print(f"  Mean correlation: {correlation_stats['mean_max_correlation']:.3f}")

    return max_corr_matrix, best_lag_matrix, standard_corr_matrix, correlation_stats


def calculate_shuffled_cross_correlation_baseline(data, max_lag=3, n_shuffles=1000, verbose=False):
    """
    Estimate the chance-level cross-correlation by independently circularly
    shifting each cell's trace and recomputing the mean max cross-correlation,
    repeated n_shuffles times.

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


def append_summary_row(organoid_name, rec_name, condition, result):
    """Append one row (per unique recording) to SUMMARY_CSV_PATH. No-op if
    SUMMARY_CSV_PATH hasn't been set (e.g. when calling analysis functions
    outside batch_process_organoids_enhanced)."""

    if SUMMARY_CSV_PATH is None:
        return

    row = {
        'organoid_name': organoid_name,
        'rec_name': rec_name,
        'condition': condition,
        'n_cells_filtered': int(result['spikes_filtered'].shape[0]),
        'n_cells_original': result['n_cells_original'],
        'spikes_max_corr': result['corr_stats']['mean_max_correlation'],
        'spikes_random_mean_corr': result['shuffle_stats']['mean_random_max_correlation'],
    }
    pd.DataFrame([row]).to_csv(
        SUMMARY_CSV_PATH, mode='a',
        header=not os.path.exists(SUMMARY_CSV_PATH), index=False
    )


def get_pre_recording_data(pre_path, cache, organoid_name=None, n_shuffles=N_SHUFFLES, max_lag=3):
    """
    Load, dFF, spike, filter (using this PRE recording's own data),
    correlate, and compute the shuffle baseline for a PRE recording.

    Cached by pre_path: calling this again for the same PRE (e.g. from a
    different PRE x POST pairing, or from the TYPE 2 mega-figure block)
    returns the cached result instantly instead of recomputing.
    """

    if pre_path in cache:
        return cache[pre_path]

    print(f"\n  Processing PRE recording: {os.path.basename(pre_path)}")

    F_pre = np.load(os.path.join(pre_path, 'suite2p', 'plane0', 'F.npy'), allow_pickle=True)
    Fneu_pre = np.load(os.path.join(pre_path, 'suite2p', 'plane0', 'Fneu.npy'), allow_pickle=True)
    stat = np.load(os.path.join(pre_path, 'suite2p', 'plane0', 'stat.npy'), allow_pickle=True)
    ops = np.load(os.path.join(pre_path, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()

    n_cells = F_pre.shape[0]

    pre_name = os.path.basename(pre_path)
    xml_path = os.path.join(pre_path, f"{pre_name}.xml")
    if os.path.exists(xml_path):
        try:
            xml_dict = files.read_xml(xml_path)
            frame_rate = 1 / xml_dict["rel_time"][1]
        except:
            frame_rate = 15.0
    else:
        frame_rate = 15.0

    F_pre = F_pre.astype(np.float64)
    Fneu_pre = Fneu_pre.astype(np.float64)
    dff_pre = (F_pre - 0.7 * Fneu_pre) / (F_pre - 0.7 * Fneu_pre).mean(axis=1, keepdims=True)
    _, spikes_pre = process_spike_data_gcamp6m(dff_pre, n_cells, F_pre.shape[1], sampling_rate=frame_rate)

    stage1_mask, stage1_stats = basic_signal_quality_filter(
        dff_pre, spikes_pre,
        peak_percentile=10,
        variance_low_percentile=10,
        variance_high_percentile=95,
        use_dff_for_filtering=False
    )
    final_mask, stage2_stats = event_based_snr_filter(
        dff_pre, spikes_pre, stage1_mask,
        snr_threshold=1.2,
        min_events=1,
        threshold_factor=2.0,
        min_duration=3,
        sampling_rate=frame_rate
    )

    dff_pre_filtered = dff_pre[final_mask, :]
    spikes_pre_filtered = spikes_pre[final_mask, :]
    stat_filtered = stat[final_mask]

    corr_max, corr_lag, corr_std, corr_stats = calculate_cross_correlation_with_lags(
        spikes_pre_filtered, max_lag=max_lag, verbose=False
    )
    shuffle_stats = calculate_shuffled_cross_correlation_baseline(
        spikes_pre_filtered, max_lag=max_lag, n_shuffles=n_shuffles, verbose=False
    )

    print(f"    Cells filtered: {np.sum(final_mask)}/{n_cells}  |  "
          f"Real: {corr_stats['mean_max_correlation']:.3f}  |  "
          f"Random: {shuffle_stats['mean_random_max_correlation']:.3f}")

    result = {
        'path': pre_path,
        'dff_filtered': dff_pre_filtered,
        'spikes_filtered': spikes_pre_filtered,
        'stat_filtered': stat_filtered,
        'ops': ops,
        'mask': final_mask,
        'frame_rate': frame_rate,
        'n_cells_original': n_cells,
        'corr_max': corr_max,
        'corr_stats': corr_stats,
        'shuffle_stats': shuffle_stats,
    }
    cache[pre_path] = result
    append_summary_row(organoid_name, pre_name, 'PRE', result)
    return result


def get_post_recording_data(post_path, pre_result, cache, organoid_name=None, n_shuffles=N_SHUFFLES, max_lag=3):
    """
    Load, dFF, spike a POST recording, apply the mask derived from its
    matched PRE recording (not its own), correlate, and compute the
    shuffle baseline.

    Cached by (post_path, id of the mask that produced it): the common
    case (one PRE, several POSTs) collapses to one cache entry per POST.
    If the same POST is ever paired with a different PRE (a different
    mask), it's correctly recomputed for that mask rather than reusing a
    stale result.
    """

    key = (post_path, id(pre_result['mask']))
    if key in cache:
        return cache[key]

    print(f"\n  Processing POST recording: {os.path.basename(post_path)}")

    F_post = np.load(os.path.join(post_path, 'suite2p', 'plane0', 'F.npy'), allow_pickle=True)
    Fneu_post = np.load(os.path.join(post_path, 'suite2p', 'plane0', 'Fneu.npy'), allow_pickle=True)

    n_cells_post = F_post.shape[0]
    mask = pre_result['mask']
    if n_cells_post != len(mask):
        raise ValueError(
            f"Cell count mismatch! PRE ({os.path.basename(pre_result['path'])}) has "
            f"{len(mask)} cells, POST ({os.path.basename(post_path)}) has {n_cells_post}"
        )

    frame_rate = pre_result['frame_rate']

    F_post = F_post.astype(np.float64)
    Fneu_post = Fneu_post.astype(np.float64)
    dff_post = (F_post - 0.7 * Fneu_post) / (F_post - 0.7 * Fneu_post).mean(axis=1, keepdims=True)
    _, spikes_post = process_spike_data_gcamp6m(dff_post, n_cells_post, F_post.shape[1], sampling_rate=frame_rate)

    dff_post_filtered = dff_post[mask, :]
    spikes_post_filtered = spikes_post[mask, :]

    corr_max, corr_lag, corr_std, corr_stats = calculate_cross_correlation_with_lags(
        spikes_post_filtered, max_lag=max_lag, verbose=False
    )
    shuffle_stats = calculate_shuffled_cross_correlation_baseline(
        spikes_post_filtered, max_lag=max_lag, n_shuffles=n_shuffles, verbose=False
    )

    print(f"    Cells: {np.sum(mask)}/{n_cells_post}  |  "
          f"Real: {corr_stats['mean_max_correlation']:.3f}  |  "
          f"Random: {shuffle_stats['mean_random_max_correlation']:.3f}")

    result = {
        'path': post_path,
        'dff_filtered': dff_post_filtered,
        'spikes_filtered': spikes_post_filtered,
        'frame_rate': frame_rate,
        'n_cells_original': n_cells_post,
        'corr_max': corr_max,
        'corr_stats': corr_stats,
        'shuffle_stats': shuffle_stats,
    }
    cache[key] = result
    append_summary_row(organoid_name, os.path.basename(post_path), 'POST', result)
    return result


def identify_top_synchronous_cells(correlation_matrix, n_cells=10):
    """Identify top N most synchronous cells based on mean correlation"""
    
    n = correlation_matrix.shape[0]
    sync_scores = np.zeros(n)
    
    for i in range(n):
        other_corrs = np.concatenate([correlation_matrix[i, :i], 
                                     correlation_matrix[i, i+1:]])
        sync_scores[i] = np.mean(other_corrs)
    
    top_indices = np.argsort(sync_scores)[-n_cells:][::-1]
    
    print(f"\nTop {n_cells} synchronous cells:")
    print(f"  Indices: {top_indices}")
    print(f"  Synchrony scores: {sync_scores[top_indices]}")
    
    return top_indices, sync_scores[top_indices]
def create_matched_visualization(
    dff_pre, dff_post, 
    spikes_pre, spikes_post,
    sync_cell_indices,
    stat, avg_projection,
    corr_pre_stats, corr_post_stats,
    frame_rate,
    output_path=None
):
    """
    Create side-by-side visualization of matched pre/post gabazine data
    NOW WITH BOTH GLOBAL AND TOP10 CORRELATIONS
    """
    
    # Set professional font properties
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 11
    
    n_cells = len(sync_cell_indices)
    colors = plt.cm.tab10(np.linspace(0, 1, n_cells))
    
    fig = plt.figure(figsize=(20, 22))
    
    panel_width = 0.35
    panel_height = 0.25
    colorbar_width = 0.015
    colorbar_gap = 0.003
    
    bottom_left_heatmap = 0.10
    bottom_right_heatmap = 0.55
    bottom_row_y = 0.10
    
    bottom_left_colorbar = bottom_left_heatmap + panel_width + colorbar_gap
    bottom_right_colorbar = bottom_right_heatmap + panel_width + colorbar_gap
    
    total_bottom_width = panel_width + colorbar_gap + colorbar_width
    
    bottom_center_left = bottom_left_heatmap + (total_bottom_width / 2)
    bottom_center_right = bottom_right_heatmap + (total_bottom_width / 2)
    
    top_left_col = bottom_center_left - (panel_width / 2)
    top_right_col = bottom_center_right - (panel_width / 2)
    
    top_row_y = 0.70
    middle_row_y = 0.40
    
    ax_img_pre = fig.add_axes([top_left_col+0.005+0.030, top_row_y, panel_width, panel_height])
    ax_img_post = fig.add_axes([top_right_col+0.005-0.030, top_row_y, panel_width, panel_height])

    ax_traces_pre = fig.add_axes([top_left_col+0.005+0.030, middle_row_y+0.015, panel_width, panel_height])
    ax_traces_post = fig.add_axes([top_right_col+0.005-0.030, middle_row_y+0.015, panel_width, panel_height])

    ax_corr_pre = fig.add_axes([bottom_left_heatmap+0.030, bottom_row_y, panel_width, panel_height])
    ax_corr_post = fig.add_axes([bottom_right_heatmap-0.030, bottom_row_y, panel_width, panel_height])

    cbar_ax_pre = fig.add_axes([bottom_left_colorbar+0.01, bottom_row_y, colorbar_width, panel_height])
    cbar_ax_post = fig.add_axes([bottom_right_colorbar-0.05, bottom_row_y, colorbar_width, panel_height])
    
    # ROW 1: Average projections
    ax_img_pre.imshow(avg_projection, cmap='gray', aspect='equal',
                      vmin=np.percentile(avg_projection, 1),
                      vmax=np.percentile(avg_projection, 99.5))
    
    for idx, (cell_idx, color) in enumerate(zip(sync_cell_indices, colors)):
        plot_roi_outline(ax_img_pre, stat[cell_idx], color=color, 
                        linewidth=2.5, label_num=idx+1)
    
    ax_img_pre.set_title('Pre-Gabazine: Top Synchronous Cells',
                         fontsize=18, weight='bold', pad=10)
    ax_img_pre.axis('off')
    
    ax_img_post.imshow(avg_projection, cmap='gray', aspect='equal',
                       vmin=np.percentile(avg_projection, 1),
                       vmax=np.percentile(avg_projection, 99.5))
    
    for idx, (cell_idx, color) in enumerate(zip(sync_cell_indices, colors)):
        plot_roi_outline(ax_img_post, stat[cell_idx], color=color,
                        linewidth=2.5, label_num=idx+1)
    
    ax_img_post.set_title('Post-Gabazine: Same Cells Tracked',
                          fontsize=18, weight='bold', pad=10)
    ax_img_post.axis('off')
    
    # ROW 2: DFF Traces
    dff_traces_pre = dff_pre[sync_cell_indices, :]
    time_vector_pre = np.arange(dff_traces_pre.shape[1]) / frame_rate
    
    offset_scale = 6.5
    
    for idx, (trace, color) in enumerate(zip(dff_traces_pre, colors)):
        trace_zscore = (trace - np.mean(trace)) / (np.std(trace) + 1e-10)
        trace_offset = trace_zscore + idx * offset_scale
        ax_traces_pre.plot(time_vector_pre, trace_offset, color=color,
                          linewidth=1.2, alpha=0.9)
    
    y_max = (n_cells - 1) * offset_scale + 5
    y_min = -3
    x_max = time_vector_pre.max()
    max_range = max(x_max, y_max - y_min)
    
    ax_traces_pre.set_xlim([0, max_range])
    ax_traces_pre.set_ylim([y_min, y_min + max_range])
    ax_traces_pre.set_xlabel('Time (s)', fontsize=14, weight='bold')
    ax_traces_pre.set_ylabel('ΔF/F', fontsize=14, weight='bold')
    ax_traces_pre.set_title('Pre-Gabazine: Calcium Traces', fontsize=18, weight='bold', pad=10)
    ax_traces_pre.spines['top'].set_visible(False)
    ax_traces_pre.spines['right'].set_visible(False)
    ax_traces_pre.spines['left'].set_linewidth(1.5)
    ax_traces_pre.spines['bottom'].set_linewidth(1.5)
    ax_traces_pre.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    ax_traces_pre.set_aspect('equal')
    
    dff_traces_post = dff_post[sync_cell_indices, :]
    time_vector_post = np.arange(dff_traces_post.shape[1]) / frame_rate
    
    for idx, (trace, color) in enumerate(zip(dff_traces_post, colors)):
        trace_zscore = (trace - np.mean(trace)) / (np.std(trace) + 1e-10)
        trace_offset = trace_zscore + idx * offset_scale
        ax_traces_post.plot(time_vector_post, trace_offset, color=color,
                           linewidth=1.2, alpha=0.9)
    
    x_max_post = time_vector_post.max()
    max_range_post = max(x_max_post, y_max - y_min)
    
    ax_traces_post.set_xlim([0, max_range_post])
    ax_traces_post.set_ylim([y_min, y_min + max_range_post])
    ax_traces_post.set_xlabel('Time (s)', fontsize=14, weight='bold')
    ax_traces_post.set_ylabel('ΔF/F', fontsize=14, weight='bold')
    ax_traces_post.set_title('Post-Gabazine: Calcium Traces', fontsize=18, weight='bold', pad=10)
    ax_traces_post.spines['top'].set_visible(False)
    ax_traces_post.spines['right'].set_visible(False)
    ax_traces_post.spines['left'].set_linewidth(1.5)
    ax_traces_post.spines['bottom'].set_linewidth(1.5)
    ax_traces_post.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
    ax_traces_post.set_aspect('equal')
    
    # ROW 3: Correlation Heatmaps
    corr_pre_subset = np.zeros((n_cells, n_cells))
    corr_matrix_pre_full = np.corrcoef(dff_pre)
    for i, idx_i in enumerate(sync_cell_indices):
        for j, idx_j in enumerate(sync_cell_indices):
            corr_pre_subset[i, j] = corr_matrix_pre_full[idx_i, idx_j]
    
    upper_tri_pre = np.triu_indices_from(corr_pre_subset, k=1)
    mean_corr_pre_top10 = np.mean(corr_pre_subset[upper_tri_pre])
    mean_corr_pre_global = corr_pre_stats['mean_max_correlation']
    
    im_pre = ax_corr_pre.imshow(corr_pre_subset, cmap='Reds', aspect='equal',
                                 vmin=0, vmax=1, interpolation='nearest')
    ax_corr_pre.set_title(
        f'Pre: Correlation Matrix\n'
        f'Global mean: r={mean_corr_pre_global:.3f}\n'
        f'Top 10 mean: r={mean_corr_pre_top10:.3f}',
        fontsize=16, weight='bold', pad=10
    )
    ax_corr_pre.set_xlabel('Cell #', fontsize=14, weight='bold')
    ax_corr_pre.set_ylabel('Cell #', fontsize=14, weight='bold')
    ax_corr_pre.set_xticks(range(n_cells))
    ax_corr_pre.set_yticks(range(n_cells))
    ax_corr_pre.set_xticklabels(range(1, n_cells + 1))
    ax_corr_pre.set_yticklabels(range(1, n_cells + 1))
    
    cbar_pre = plt.colorbar(im_pre, cax=cbar_ax_pre)
    cbar_pre.set_label('Correlation', fontsize=14, weight='bold')
    cbar_pre.ax.tick_params(labelsize=14)

    corr_post_subset = np.zeros((n_cells, n_cells))
    corr_matrix_post_full = np.corrcoef(dff_post)
    for i, idx_i in enumerate(sync_cell_indices):
        for j, idx_j in enumerate(sync_cell_indices):
            corr_post_subset[i, j] = corr_matrix_post_full[idx_i, idx_j]
    
    upper_tri_post = np.triu_indices_from(corr_post_subset, k=1)
    mean_corr_post_top10 = np.mean(corr_post_subset[upper_tri_post])
    mean_corr_post_global = corr_post_stats['mean_max_correlation']
    
    im_post = ax_corr_post.imshow(corr_post_subset, cmap='Reds', aspect='equal',
                                   vmin=0, vmax=1, interpolation='nearest')
    ax_corr_post.set_title(
        f'Post: Correlation Matrix\n'
        f'Global mean: r={mean_corr_post_global:.3f}\n'
        f'Top 10 mean: r={mean_corr_post_top10:.3f}',
        fontsize=16, weight='bold', pad=10
    )
    ax_corr_post.set_xlabel('Cell #', fontsize=14, weight='bold')
    ax_corr_post.set_ylabel('Cell #', fontsize=14, weight='bold')
    ax_corr_post.set_xticks(range(n_cells))
    ax_corr_post.set_yticks(range(n_cells))
    ax_corr_post.set_xticklabels(range(1, n_cells + 1))
    ax_corr_post.set_yticklabels(range(1, n_cells + 1))
    
    cbar_post = plt.colorbar(im_post, cax=cbar_ax_post)
    cbar_post.set_label('Correlation', fontsize=14, weight='bold')
    cbar_post.ax.tick_params(labelsize=10)
    
    # ROW 4: Legend
    ax_legend = fig.add_axes([0.1, 0.02, 0.8, 0.06])
    ax_legend.axis('off')
    
    legend_elements = [
        plt.Line2D([0], [0], color=color, linewidth=3.5,
                  label=f'Cell {idx+1} (ROI #{cell_idx})')
        for idx, (color, cell_idx) in enumerate(zip(colors, sync_cell_indices))
    ]
    
    legend = ax_legend.legend(handles=legend_elements, loc='center', ncol=5,
                             frameon=True, fontsize=14, fancybox=True, 
                             shadow=False, framealpha=0.9,
                             columnspacing=1.5, handlelength=2.5)
    legend.get_frame().set_linewidth(1.5)
    
    fig.text(0.5, 0.98, 'Matched Pre/Post Gabazine Analysis: Same Cells Tracked Across Conditions',
             ha='center', fontsize=25, weight='bold', family='Arial')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"\nSaved matched visualization to: {output_path}")
    
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    
    correlation_metrics = {
        'pre_global': mean_corr_pre_global,
        'pre_top10': mean_corr_pre_top10,
        'post_global': mean_corr_post_global,
        'post_top10': mean_corr_post_top10
    }
    
    return fig, correlation_metrics

# ============================================================================
# NEW: Type 2 Mega-Figure (All recordings in one figure)
# ============================================================================
def create_mega_figure_all_recordings(
    dff_data_list,
    spike_data_list,
    sync_cell_indices,
    stat, avg_projection,
    corr_stats_list,
    frame_rate,
    recording_labels,
    output_path=None
):
    """
    Create mega-figure showing all recordings side-by-side
    NOW WITH BOTH GLOBAL AND TOP10 CORRELATIONS
    """
    
    n_recordings = len(dff_data_list)
    n_cells = len(sync_cell_indices)
    colors = plt.cm.tab10(np.linspace(0, 1, n_cells))
    
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 9
    
    fig, axes = plt.subplots(3, n_recordings, figsize=(6*n_recordings, 18))
    
    if n_recordings == 1:
        axes = axes.reshape(-1, 1)
    
    all_correlation_metrics = []
    
    for col_idx, (dff, spike, corr_stats, label) in enumerate(
        zip(dff_data_list, spike_data_list, corr_stats_list, recording_labels)):
        
        dff_subset = dff[sync_cell_indices, :]
        spike_subset = spike[sync_cell_indices, :]
        
        # ROW 0: Field of View
        ax_img = axes[0, col_idx]
        ax_img.imshow(avg_projection, cmap='gray', aspect='equal',
                      vmin=np.percentile(avg_projection, 1),
                      vmax=np.percentile(avg_projection, 99.5))
        
        for idx, (cell_idx, color) in enumerate(zip(sync_cell_indices, colors)):
            plot_roi_outline(ax_img, stat[cell_idx], color=color, 
                            linewidth=2, label_num=idx+1)
        
        ax_img.set_title(f'{label}\nTop {n_cells} Synchronous Cells',
                        fontsize=14, weight='bold', pad=8)
        ax_img.axis('off')
        
        # ROW 1: Calcium Traces
        ax_traces = axes[1, col_idx]
        
        time_vector = np.arange(dff_subset.shape[1]) / frame_rate
        offset_scale = 6.5
        
        for idx, (trace, color) in enumerate(zip(dff_subset, colors)):
            trace_zscore = (trace - np.mean(trace)) / (np.std(trace) + 1e-10)
            trace_offset = trace_zscore + idx * offset_scale
            ax_traces.plot(time_vector, trace_offset, color=color,
                          linewidth=1.0, alpha=0.9)
        
        y_max = (n_cells - 1) * offset_scale + 5
        y_min = -3
        
        ax_traces.set_xlim([0, time_vector.max()])
        ax_traces.set_ylim([y_min, y_max])
        ax_traces.set_xlabel('Time (s)', fontsize=11, weight='bold')
        ax_traces.set_ylabel('ΔF/F', fontsize=11, weight='bold')
        ax_traces.set_title(f'{label}: Calcium Traces', 
                           fontsize=14, weight='bold', pad=8)
        ax_traces.spines['top'].set_visible(False)
        ax_traces.spines['right'].set_visible(False)
        ax_traces.grid(True, alpha=0.2, linestyle=':', linewidth=0.5)
        
        # ROW 2: Correlation Matrix
        ax_corr = axes[2, col_idx]
        
        corr_subset = np.zeros((n_cells, n_cells))
        corr_matrix_full = np.corrcoef(dff)
        for i, idx_i in enumerate(sync_cell_indices):
            for j, idx_j in enumerate(sync_cell_indices):
                corr_subset[i, j] = corr_matrix_full[idx_i, idx_j]
        
        upper_tri = np.triu_indices_from(corr_subset, k=1)
        mean_corr_top10 = np.mean(corr_subset[upper_tri])
        mean_corr_global = corr_stats['mean_max_correlation']
        
        all_correlation_metrics.append({
            'label': label,
            'global': mean_corr_global,
            'top10': mean_corr_top10
        })
        
        im = ax_corr.imshow(corr_subset, cmap='Reds', aspect='equal',
                           vmin=0, vmax=1, interpolation='nearest')
        
        ax_corr.set_title(
            f'{label}: Correlation\n'
            f'Global: r={mean_corr_global:.3f}\n'
            f'Top 10: r={mean_corr_top10:.3f}',
            fontsize=13, weight='bold', pad=8
        )
        ax_corr.set_xlabel('Cell #', fontsize=11, weight='bold')
        ax_corr.set_ylabel('Cell #', fontsize=11, weight='bold')
        ax_corr.set_xticks(range(n_cells))
        ax_corr.set_yticks(range(n_cells))
        ax_corr.set_xticklabels(range(1, n_cells + 1))
        ax_corr.set_yticklabels(range(1, n_cells + 1))
        
        if col_idx == n_recordings - 1:
            cbar = plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)
            cbar.set_label('Correlation', fontsize=11, weight='bold')
    
    # Legend at bottom
    fig.subplots_adjust(bottom=0.08)
    ax_legend = fig.add_axes([0.1, 0.02, 0.8, 0.04])
    ax_legend.axis('off')
    
    legend_elements = [
        plt.Line2D([0], [0], color=color, linewidth=3,
                  label=f'Cell {idx+1} (ROI #{cell_idx})')
        for idx, (color, cell_idx) in enumerate(zip(colors, sync_cell_indices))
    ]
    
    if len(legend_elements) > 0:
        legend = ax_legend.legend(handles=legend_elements, loc='center', 
                                 ncol=min(5, n_cells),
                                 frameon=True, fontsize=11, fancybox=True, 
                                 shadow=False, framealpha=0.9,
                                 columnspacing=1.5, handlelength=2.5)
        legend.get_frame().set_linewidth(1.5)
    
    fig.text(0.5, 0.98, 
             f'Complete Gabazine Time Series: {n_recordings} Recordings Tracked',
             ha='center', fontsize=20, weight='bold', family='Arial')
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"\n✓ Saved mega-figure to: {output_path}")
    
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    
    return fig, all_correlation_metrics

# ============================================================================
# ORIGINAL ANALYSIS FUNCTIONS - UNCHANGED
# ============================================================================

def analyze_matched_pregaba_recording(
    pre_data,
    post_data,
    rec_name,
    output_folder,
    n_sync_cells=10,
    max_lag=3
):
    """Complete analysis pipeline for matched pre/post gabazine data.

    Filtering, correlation, and shuffle baselines are computed once by
    get_pre_recording_data / get_post_recording_data (and cached there) --
    this function only consumes the already-computed results.
    """

    os.makedirs(output_folder, exist_ok=True)

    dff_pre_filtered = pre_data['dff_filtered']
    spikes_pre_filtered = pre_data['spikes_filtered']
    dff_post_filtered = post_data['dff_filtered']
    spikes_post_filtered = post_data['spikes_filtered']
    stat_filtered = pre_data['stat_filtered']
    ops = pre_data['ops']
    frame_rate = pre_data['frame_rate']
    final_mask = pre_data['mask']

    corr_pre_max, corr_pre_stats = pre_data['corr_max'], pre_data['corr_stats']
    corr_post_max, corr_post_stats = post_data['corr_max'], post_data['corr_stats']
    pre_shuffle_stats = pre_data['shuffle_stats']
    post_shuffle_stats = post_data['shuffle_stats']

    print(f"\n{'='*80}")
    print(f"MATCHED PRE/POST GABAZINE ANALYSIS: {rec_name}")
    print(f"{'='*80}")
    print(f"Pre-gabazine:  {dff_pre_filtered.shape[0]} cells (filtered), {dff_pre_filtered.shape[1]} frames")
    print(f"Post-gabazine: {dff_post_filtered.shape[0]} cells (filtered), {dff_post_filtered.shape[1]} frames")
    print(f"Frame rate: {frame_rate:.2f} Hz")
    print(f"  **SAME MASK APPLIED TO POST-GABAZINE** ({np.sum(final_mask)}/{pre_data['n_cells_original']} cells)")
    print(f"\nUsing cached filtering + correlation + shuffle baseline:")
    print(f"  PRE  correlation: real={corr_pre_stats['mean_max_correlation']:.3f}, "
          f"random={pre_shuffle_stats['mean_random_max_correlation']:.3f}")
    print(f"  POST correlation: real={corr_post_stats['mean_max_correlation']:.3f}, "
          f"random={post_shuffle_stats['mean_random_max_correlation']:.3f}")

    # STEP 3: Identify synchronous cells from PRE
    print(f"\n{'='*80}")
    print("STEP 3: Identifying synchronous cells from PRE-gabazine")
    print(f"{'='*80}")
    
    sync_cell_indices, sync_scores = identify_top_synchronous_cells(
        corr_pre_max, n_cells=n_sync_cells
    )
    
    original_indices = np.where(final_mask)[0]
    original_sync_indices = original_indices[sync_cell_indices]
    
    print(f"\nSynchronous cells (original Suite2p ROI numbers): {original_sync_indices}")
    
    # STEP 4: Extract traces for same cells
    print(f"\n{'='*80}")
    print("STEP 4: Extracting traces for matched cells")
    print(f"{'='*80}")
    
    print("\nSynchrony comparison for selected cells:")
    for i, cell_idx in enumerate(sync_cell_indices):
        pre_score = sync_scores[i]
        
        post_corrs = np.concatenate([corr_post_max[cell_idx, :cell_idx],
                                    corr_post_max[cell_idx, cell_idx+1:]])
        post_score = np.mean(post_corrs)
        
        change_pct = ((post_score / (pre_score + 1e-10)) - 1) * 100
        
        print(f"  Cell {i+1} (ROI #{original_sync_indices[i]}):")
        print(f"    Pre:  r = {pre_score:.3f}")
        print(f"    Post: r = {post_score:.3f}")
        print(f"    Change: {change_pct:+.1f}%")
    
    # STEP 5: Create visualization
    print(f"\n{'='*80}")
    print("STEP 5: Creating matched visualization")
    print(f"{'='*80}")
    
    if 'meanImg' in ops:
        avg_projection = ops['meanImg']
    elif 'max_proj' in ops:
        avg_projection = ops['max_proj']
    else:
        avg_projection = ops['refImg']
    
    output_path = os.path.join(output_folder, f"{rec_name}_matched_pregaba_analysis.png")
    
    fig, correlation_metrics = create_matched_visualization(
        dff_pre_filtered, dff_post_filtered,
        spikes_pre_filtered, spikes_post_filtered,
        sync_cell_indices,
        stat_filtered,
        avg_projection,
        corr_pre_stats, corr_post_stats,
        frame_rate,
        output_path=output_path
    )
    
    # STEP 6: Save results with BOTH correlation metrics
    results = {
        'recording_info': {
            'recording_name': rec_name,
            'frame_rate': frame_rate,
            'n_frames_pre': dff_pre_filtered.shape[1],
            'n_frames_post': dff_post_filtered.shape[1],
            'n_cells_original': pre_data['n_cells_original'],
            'n_cells_filtered': np.sum(final_mask),
        },
        'comparison': {
            'mean_correlation_pre_global': correlation_metrics['pre_global'],
            'mean_correlation_pre_top10': correlation_metrics['pre_top10'],
            'mean_correlation_post_global': correlation_metrics['post_global'],
            'mean_correlation_post_top10': correlation_metrics['post_top10'],
            'global_correlation_change': correlation_metrics['post_global'] - correlation_metrics['pre_global'],
            'global_correlation_change_percent': ((correlation_metrics['post_global'] / (correlation_metrics['pre_global'] + 1e-10)) - 1) * 100,
            'top10_correlation_change': correlation_metrics['post_top10'] - correlation_metrics['pre_top10'],
            'top10_correlation_change_percent': ((correlation_metrics['post_top10'] / (correlation_metrics['pre_top10'] + 1e-10)) - 1) * 100,
            'pre_random_mean_correlation': pre_shuffle_stats['mean_random_max_correlation'],
            'post_random_mean_correlation': post_shuffle_stats['mean_random_max_correlation'],
        },
        'sync_cell_indices': original_sync_indices.tolist(),
        'sync_scores': sync_scores.tolist()
    }

    # Save detailed CSV
    condition_stats = pd.DataFrame([
        {
            'Recording': rec_name,
            'Condition': 'PRE',
            'N_Cells_Filtered': np.sum(final_mask),
            'N_Cells_Original': pre_data['n_cells_original'],
            'Global_Mean_Correlation': correlation_metrics['pre_global'],
            'Top10_Mean_Correlation': correlation_metrics['pre_top10'],
            'Correlation_Difference': correlation_metrics['pre_top10'] - correlation_metrics['pre_global'],
            'Random_Mean_Correlation': pre_shuffle_stats['mean_random_max_correlation'],
            'Frame_Rate_Hz': frame_rate
        },
        {
            'Recording': rec_name,
            'Condition': 'POST',
            'N_Cells_Filtered': np.sum(final_mask),
            'N_Cells_Original': pre_data['n_cells_original'],
            'Global_Mean_Correlation': correlation_metrics['post_global'],
            'Top10_Mean_Correlation': correlation_metrics['post_top10'],
            'Correlation_Difference': correlation_metrics['post_top10'] - correlation_metrics['post_global'],
            'Random_Mean_Correlation': post_shuffle_stats['mean_random_max_correlation'],
            'Frame_Rate_Hz': frame_rate
        }
    ])
    
    stats_csv_path = os.path.join(output_folder, f"{rec_name}_correlation_statistics.csv")
    condition_stats.to_csv(stats_csv_path, index=False)
    print(f"✓ Saved correlation statistics to: {stats_csv_path}")
    
    # Save cell info
    cell_info = pd.DataFrame([
        {
            'Recording': rec_name,
            'Display_Number': idx + 1,
            'ROI_Index': cell_idx,
            'Synchrony_Score_PRE': score
        }
        for idx, (cell_idx, score) in enumerate(zip(original_sync_indices, sync_scores))
    ])
    
    cell_csv_path = os.path.join(output_folder, f"{rec_name}_top10_cells.csv")
    cell_info.to_csv(cell_csv_path, index=False)
    print(f"✓ Saved top 10 cell information to: {cell_csv_path}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE - SUMMARY")
    print(f"{'='*80}")
    print(f"\nRecording: {rec_name}")
    print(f"  Cells filtered: {np.sum(final_mask)}/{pre_data['n_cells_original']}")
    print(f"\n  PRE Correlations:")
    print(f"    Global: {correlation_metrics['pre_global']:.3f}")
    print(f"    Top 10: {correlation_metrics['pre_top10']:.3f}")
    print(f"\n  POST Correlations:")
    print(f"    Global: {correlation_metrics['post_global']:.3f}")
    print(f"    Top 10: {correlation_metrics['post_top10']:.3f}")
    print(f"\n  Changes:")
    print(f"    Global: {results['comparison']['global_correlation_change']:.3f} ({results['comparison']['global_correlation_change_percent']:.1f}%)")
    print(f"    Top 10: {results['comparison']['top10_correlation_change']:.3f} ({results['comparison']['top10_correlation_change_percent']:.1f}%)")
    
    return results

def analyze_single_pregaba_pair(pre_path, post_path, comparison_name,
                                output_folder, cache, organoid_name,
                                n_sync_cells=10, max_lag=3):
    """
    Analyze one PRE vs POST comparison.

    Loading, dFF, filtering, correlation, and shuffle-baseline computation
    for each recording are handled by get_pre_recording_data /
    get_post_recording_data, which cache their results (keyed by path) in
    `cache` so the same PRE (or the same POST under the same mask) is only
    ever actually computed once, no matter how many pairings reuse it.
    """

    pre_data = get_pre_recording_data(pre_path, cache, organoid_name=organoid_name, max_lag=max_lag)
    post_data = get_post_recording_data(post_path, pre_data, cache, organoid_name=organoid_name, max_lag=max_lag)

    results = analyze_matched_pregaba_recording(
        pre_data, post_data,
        comparison_name,
        output_folder,
        n_sync_cells=n_sync_cells,
        max_lag=max_lag
    )

    return results

# ============================================================================
# ENHANCED BATCH PROCESSING
# ============================================================================

def find_organoid_folders_and_pairs_enhanced(base_folder):
    """
    Enhanced version that detects both single and multiple post recordings.
    """

    print(f"\n{'='*70}")
    print("SCANNING FOR ORGANOID FOLDERS (ENHANCED)")
    print(f"{'='*70}")
    print(f"Base folder: {base_folder}\n")

    organoid_list = []

    for item in os.listdir(base_folder):
        item_path = os.path.join(base_folder, item)

        if not os.path.isdir(item_path):
            continue

        print(f"\nChecking: {item}")

        has_multi_post, expected_count = parse_folder_for_multi_post(item)

        if has_multi_post:
            print(f"  📊 Detected multi-post pattern: expecting {expected_count} recordings")

        subfolders = [f for f in os.listdir(item_path)
                     if os.path.isdir(os.path.join(item_path, f))]

        pre_recordings = []
        post_recordings = []

        for subfolder in subfolders:
            if subfolder.lower() == 'suite2p':
                continue
            if 'analysis' in subfolder.lower():
                continue
            if 'references' in subfolder.lower():
                continue

            subfolder_path = os.path.join(item_path, subfolder)
            suite2p_path = os.path.join(subfolder_path, 'suite2p', 'plane0')

            if os.path.exists(suite2p_path):
                subfolder_upper = subfolder.upper()
                has_gz = ('_GZ_' in subfolder_upper or
                         '-GZ-' in subfolder_upper or
                         '_GZ-' in subfolder_upper or
                         '-GZ_' in subfolder_upper)

                if has_gz:
                    post_recordings.append(subfolder_path)
                    print(f"  ✓ POST: {subfolder}")
                else:
                    pre_recordings.append(subfolder_path)
                    print(f"  ✓ PRE:  {subfolder}")
            else:
                print(f"  ✗ SKIP: {subfolder} (no suite2p/plane0/)")

        if len(pre_recordings) > 0 or len(post_recordings) > 0:
            organoid_info = {
                'organoid_name': item,
                'organoid_path': item_path,
                'pre_recordings': sorted(pre_recordings),
                'post_recordings': sorted(post_recordings),
                'n_pre': len(pre_recordings),
                'n_post': len(post_recordings),
                'has_multi_post': has_multi_post,
                'expected_count': expected_count
            }
            organoid_list.append(organoid_info)

            print(f"  📊 Summary: {len(pre_recordings)} PRE, {len(post_recordings)} POST")

            if has_multi_post:
                actual_total = len(pre_recordings) + len(post_recordings)
                if actual_total != expected_count:
                    print(f"  ⚠️  WARNING: Expected {expected_count} recordings, found {actual_total}")

            if len(pre_recordings) == 0:
                print(f"  ⚠️  WARNING: No PRE recordings found!")
            if len(post_recordings) == 0:
                print(f"  ⚠️  WARNING: No POST recordings found!")
        else:
            print(f"  ⚠️  No deconcat'd recordings found")

    print(f"\n{'='*70}")
    print(f"Found {len(organoid_list)} organoid folders with recordings")
    print(f"{'='*70}\n")

    return organoid_list

def batch_process_organoids_enhanced(base_folder):
    """
    ENHANCED: Process all organoids, handling both single and multiple post recordings
    """
    global SUMMARY_CSV_PATH
    SUMMARY_CSV_PATH = os.path.join(base_folder, 'gabazine_correlation_summary.csv')

    print(f"\n{'='*80}")
    print("BATCH PROCESSING: ENHANCED ORGANOID PRE/POST GABAZINE ANALYSIS")
    print(f"{'='*80}")
    print(f"Base folder: {base_folder}")
    print(f"Correlation summary CSV: {SUMMARY_CSV_PATH}")

    organoid_list = find_organoid_folders_and_pairs_enhanced(base_folder)

    if len(organoid_list) == 0:
        print("\n❌ No organoid folders found!")
        return None

    all_results = []

    for organoid_info in organoid_list:
        organoid_name = organoid_info['organoid_name']
        organoid_path = organoid_info['organoid_path']
        pre_recs = organoid_info['pre_recordings']
        post_recs = organoid_info['post_recordings']
        has_multi_post = organoid_info['has_multi_post']

        # One cache per organoid: every unique PRE/POST recording within
        # this organoid gets loaded, filtered, correlated, and shuffled at
        # most once, then reused across every TYPE 1 pairing and TYPE 2.
        recording_cache = {}

        print(f"\n{'='*80}")
        print(f"PROCESSING ORGANOID: {organoid_name}")
        print(f"{'='*80}")
        print(f"PRE recordings:  {len(pre_recs)}")
        print(f"POST recordings: {len(post_recs)}")
        print(f"Multi-post mode: {has_multi_post}")

        if len(pre_recs) == 0 or len(post_recs) == 0:
            print(f"\n⚠️  Skipping - insufficient recordings")
            continue

        output_folder = os.path.join(organoid_path,
                                    f'analysis_{datetime.datetime.now().strftime("%Y%m%d")}')
        os.makedirs(output_folder, exist_ok=True)

        # ====================================================================
        # TYPE 1: Individual PRE vs POST comparisons
        # ====================================================================
        print(f"\n{'='*70}")
        print("TYPE 1: Individual PRE vs POST Comparisons")
        print(f"{'='*70}")

        for pre_idx, pre_path in enumerate(pre_recs, 1):
            for post_idx, post_path in enumerate(post_recs, 1):

                comparison_name = f"{organoid_name}_PRE{pre_idx}_vs_POST{post_idx}"

                print(f"\n{'-'*70}")
                print(f"Comparison: {comparison_name}")
                print(f"{'-'*70}")

                try:
                    results = analyze_single_pregaba_pair(
                        pre_path, post_path,
                        comparison_name,
                        output_folder,
                        recording_cache,
                        organoid_name,
                        n_sync_cells=10,
                        max_lag=3
                    )
                    all_results.append({
                        'organoid': organoid_name,
                        'comparison': comparison_name,
                        'pre_recording': os.path.basename(pre_path),
                        'post_recording': os.path.basename(post_path),
                        'n_cells_original': results['recording_info']['n_cells_original'],
                        'n_cells_filtered': results['recording_info']['n_cells_filtered'],
                        'pre_corr_global': results['comparison']['mean_correlation_pre_global'],
                        'pre_corr_top10': results['comparison']['mean_correlation_pre_top10'],
                        'post_corr_global': results['comparison']['mean_correlation_post_global'],
                        'post_corr_top10': results['comparison']['mean_correlation_post_top10'],
                        'global_change_%': results['comparison']['global_correlation_change_percent'],
                        'top10_change_%': results['comparison']['top10_correlation_change_percent'],
                        'pre_random_mean_corr': results['comparison']['pre_random_mean_correlation'],
                        'post_random_mean_corr': results['comparison']['post_random_mean_correlation'],
                        'status': 'SUCCESS'
                    })
                                        
                except Exception as e:
                    print(f"  ❌ ERROR: {e}")
                    all_results.append({
                        'organoid': organoid_name,
                        'comparison': comparison_name,
                        'status': 'FAILED',
                        'error': str(e)
                    })
                    
        # ====================================================================
        # TYPE 2: Mega-figure (only if multi-post)
        # ====================================================================
        if has_multi_post and len(post_recs) > 1:
            print(f"\n{'='*70}")
            print("TYPE 2: Creating Mega-Figure (All Recordings)")
            print(f"{'='*70}")
            
            try:
                pre_path = pre_recs[0]

                print("Loading data for all recordings (cached -- reuses TYPE 1 results where possible)...")

                pre_data = get_pre_recording_data(pre_path, recording_cache, organoid_name=organoid_name)

                stat_filtered = pre_data['stat_filtered']
                ops = pre_data['ops']
                frame_rate = pre_data['frame_rate']
                final_mask = pre_data['mask']
                n_cells = pre_data['n_cells_original']

                print(f"  Cells filtered: {np.sum(final_mask)}/{n_cells}")

                sync_cell_indices, sync_scores = identify_top_synchronous_cells(
                    pre_data['corr_max'], n_cells=10
                )

                all_dff_list = [pre_data['dff_filtered']]
                all_spike_list = [pre_data['spikes_filtered']]
                all_corr_stats_list = [pre_data['corr_stats']]
                recording_labels = ['PRE']

                for post_idx, post_path in enumerate(post_recs, 1):
                    print(f"  Loading POST-{post_idx:03d}...")

                    post_data = get_post_recording_data(
                        post_path, pre_data, recording_cache, organoid_name=organoid_name
                    )

                    all_dff_list.append(post_data['dff_filtered'])
                    all_spike_list.append(post_data['spikes_filtered'])
                    all_corr_stats_list.append(post_data['corr_stats'])
                    recording_labels.append(f'POST-{post_idx:03d}')

                print("\nCreating mega-figure...")
                if 'meanImg' in ops:
                    avg_projection = ops['meanImg']
                elif 'max_proj' in ops:
                    avg_projection = ops['max_proj']
                else:
                    avg_projection = ops['refImg']
                
                mega_output_path = os.path.join(output_folder, 
                                               f"{organoid_name}_mega_figure_all_recordings.png")
                
                fig, all_correlation_metrics = create_mega_figure_all_recordings(
                    all_dff_list,
                    all_spike_list,
                    sync_cell_indices,
                    stat_filtered,
                    avg_projection,
                    all_corr_stats_list,
                    frame_rate,
                    recording_labels,
                    output_path=mega_output_path
                )
                
                print(f"✓ Mega-figure created successfully")
                
                print("\nSaving mega-figure statistics...")
                mega_stats_list = []
                for metric in all_correlation_metrics:
                    mega_stats_list.append({
                        'Organoid': organoid_name,
                        'Recording_Label': metric['label'],
                        'Global_Mean_Correlation': metric['global'],
                        'Top10_Mean_Correlation': metric['top10'],
                        'Correlation_Difference': metric['top10'] - metric['global'],
                        'N_Cells_Filtered': np.sum(final_mask),
                        'N_Cells_Original': n_cells,
                        'Frame_Rate_Hz': frame_rate
                    })
                
                mega_stats_df = pd.DataFrame(mega_stats_list)
                mega_stats_path = os.path.join(output_folder, 
                                               f"{organoid_name}_mega_figure_statistics.csv")
                mega_stats_df.to_csv(mega_stats_path, index=False)
                print(f"✓ Saved mega-figure statistics to: {mega_stats_path}")
                
                cell_info_list = []
                original_indices = np.where(final_mask)[0]
                original_sync_indices = original_indices[sync_cell_indices]
                
                for idx, (cell_idx, score) in enumerate(zip(original_sync_indices, sync_scores)):
                    cell_info_list.append({
                        'Organoid': organoid_name,
                        'Display_Number': idx + 1,
                        'ROI_Index': cell_idx,
                        'Synchrony_Score': score
                    })
                
                cell_info_df = pd.DataFrame(cell_info_list)
                cell_info_path = os.path.join(output_folder, 
                                             f"{organoid_name}_mega_figure_top10_cells.csv")
                cell_info_df.to_csv(cell_info_path, index=False)
                print(f"✓ Saved top 10 cell information to: {cell_info_path}")
                
            except Exception as e:
                print(f"❌ ERROR creating mega-figure: {e}")
                import traceback
                traceback.print_exc()
    
    # ========================================================================
    # BATCH SUMMARY
    # ========================================================================
    if len(all_results) == 0:
        print("\n❌ No analyses completed!")
        return None
    
    print(f"\n{'='*80}")
    print("BATCH PROCESSING COMPLETE - SUMMARY")
    print(f"{'='*80}")
    
    summary_df = pd.DataFrame(all_results)
    
    successful = summary_df[summary_df['status'] == 'SUCCESS']
    failed = summary_df[summary_df['status'] == 'FAILED']
    
    print(f"\nTotal comparisons: {len(summary_df)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    
    if len(successful) > 0:
        print(f"\n{'='*70}")
        print("SUCCESSFUL ANALYSES:")
        print(f"{'='*70}")
        for _, row in successful.iterrows():
            print(f"\n{row['comparison']}")
            print(f"  Cells: {row['n_cells_filtered']}/{row['n_cells_original']}")
            print(f"  PRE  - Global: {row['pre_corr_global']:.3f}, Top10: {row['pre_corr_top10']:.3f}")
            print(f"  POST - Global: {row['post_corr_global']:.3f}, Top10: {row['post_corr_top10']:.3f}")
            print(f"  Global change: {row['global_change_%']:+.1f}%")
            print(f"  Top10 change:  {row['top10_change_%']:+.1f}%")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(base_folder, 
                               f'batch_summary_{timestamp}.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Batch summary saved to: {summary_path}")
    
    return summary_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def run_full_pipeline(base_folder):
    """
    Run STEP 0 (deconcat) + the main analysis pipeline for a single
    base_folder. Called once per entry in BASE_FOLDERS below.
    """

    print("="*80)
    print("ENHANCED ORGANOID PRE/POST GABAZINE ANALYSIS")
    print(f"Base folder: {base_folder}")
    print("="*80)

    # ========================================================================
    # STEP 0: DECONCAT (if needed)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 0: CHECKING FOR CONCATENATED SUITE2P DATA")
    print("="*80)
    print("\nScanning for organoid folders that need deconcatenation...")

    all_files = os.listdir(base_folder)
    for recording in all_files:
        recording_folder = os.path.join(base_folder, recording)
        if os.path.isdir(recording_folder):
            main_suite2p = os.path.join(recording_folder, 'suite2p', 'plane0')
            
            if os.path.exists(main_suite2p):
                print(f"\n{'='*70}")
                print(f"Found concatenated suite2p data in: {recording}")
                print(f"{'='*70}")
                
                subfolders = [f for f in os.listdir(recording_folder) 
                             if os.path.isdir(os.path.join(recording_folder, f)) 
                             and f != 'suite2p' and 'analysis' not in f.lower()]
                
                already_deconcat = False
                for subfolder in subfolders:
                    subfolder_suite2p = os.path.join(recording_folder, subfolder, 'suite2p', 'plane0')
                    if os.path.exists(subfolder_suite2p):
                        already_deconcat = True
                        break
                
                if already_deconcat:
                    print(f"  ✓ Already deconcatenated (found individual suite2p folders)")
                    print(f"  → Skipping deconcat for {recording}")
                else:
                    print(f"  → Running deconcat for {recording}...")
                    try:
                        frame_info = Frame_Detector_Per_Lap(recording_folder)
                        if frame_info is not None:
                            cumulative_frames = deconcat_suite2p_output(frame_info, recording_folder)
                            print(f"  ✓ Deconcat complete for {recording}")
                        else:
                            print(f"  → No per-trial XML subfolders -- falling back to equal PRE/POST split")
                            split_combined_suite2p_equal_pre_post(recording_folder, recording)
                            print(f"  ✓ Equal-split deconcat complete for {recording}")
                    except Exception as e:
                        print(f"  ❌ ERROR during deconcat: {e}")
                        import traceback
                        traceback.print_exc()
            else:
                print(f"\n  → {recording}: No main suite2p folder found (may already be split or not processed)")
    
    # ========================================================================
    # MAIN ANALYSIS PIPELINE
    # ========================================================================
    print("\n" + "="*80)
    print("MAIN ANALYSIS PIPELINE")
    print("="*80)
    print("\nThis pipeline will:")
    print("  1. Detect single vs multi-post recordings")
    print("  2. For single post: generate standard PRE vs POST figures")
    print("  3. For multi-post:")
    print("     - TYPE 1: Individual PRE vs each POST figure")
    print("     - TYPE 2: Mega-figure with all recordings side-by-side")
    print("="*80)
    
    summary_df = batch_process_organoids_enhanced(base_folder=base_folder)

    if summary_df is not None:
        print("\n" + "="*80)
        print(f"ALL DONE! ({base_folder})")
        print("="*80)

    return summary_df


if __name__ == "__main__":

    BASE_FOLDERS = [
        r'Z:\from_jasmine\3x\Negatives',
    ]

    all_summaries = {}
    for base_folder in BASE_FOLDERS:
        try:
            all_summaries[base_folder] = run_full_pipeline(base_folder)
        except Exception as e:
            print(f"\n❌ ERROR processing base folder {base_folder}: {e}")
            import traceback
            traceback.print_exc()
            all_summaries[base_folder] = None

    print("\n" + "="*80)
    print("ALL BASE FOLDERS COMPLETE")
    print("="*80)
    for base_folder, summary_df in all_summaries.items():
        status = f"{len(summary_df)} comparisons" if summary_df is not None else "FAILED / no results"
        print(f"  {base_folder}: {status}")