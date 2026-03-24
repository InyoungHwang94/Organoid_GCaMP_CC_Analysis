"""
===============================================================================
MultiDrug_Comparison.py — Sequential Multi-Drug Neural Activity Analysis
===============================================================================

Authors : Inyoung Hwang (project lead), Jasmine S. Yeo (analysis)
Script Author : Jasmine S. Yeo
Created : 2026-03-24
Last Modified : 2026-03-24

Purpose
-------
Tracks pairwise correlation coefficients across a sequential pharmacological
dissection: Pre → Gabazine → GZ_washed → AP5 → AP5_washed → CNQX → TTX.

Pipeline Position
-----------------
Runs after  : CalculateCC.py
Runs before : (downstream statistical analysis / figure generation)

Notes
-----
- Set base_folder to the directory containing condition subfolders.
- Recording order is inferred from the folder name number (e.g., 'org4_2.GZ-001').

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

# ============================================================================
# DECONCAT FUNCTIONS
# ============================================================================

def sort_key_drug_condition(folder_name):
    """
    Extract the recording number from folder name for proper ordering.
    
    Examples:
        'B4_D150_KOLF_org4_1.001' → 1
        'B4_D150_WS1_org4_2.GZ-001' → 2
        'B4_D150_KOLF_org4_7.TTX-3x-001' → 7
    
    Falls back to keyword-based sorting if number pattern not found.
    """
    # Try to extract the number after 'org' and before the dot
    # Pattern: org{digits}_{NUMBER}.{anything}
    match = re.search(r'org\d+_(\d+)\.', folder_name)
    if match:
        return int(match.group(1))
    
    # Fallback to keyword-based sorting if no number found
    folder_upper = folder_name.upper()
    
    if 'TTX' in folder_upper:
        return 1000  # Large numbers to put at end
    elif 'CNQX' in folder_upper:
        return 900
    elif 'AP5' in folder_upper and 'WASHED' in folder_upper:
        return 800
    elif 'AP5' in folder_upper:
        return 700
    elif 'GZ' in folder_upper and 'WASHED' in folder_upper:
        return 600
    elif 'GZ' in folder_upper or 'GABAZINE' in folder_upper:
        return 500
    else:
        return 0  # Pre-imaging


def Frame_Detector_Per_Lap_MultiDrug(base_folder):
    """
    Detect and sort recording folders in the correct drug condition order.
    Reads XML files from each recording folder to determine frame counts.
    """
    print(f"\n{'='*70}")
    print("AUTO-SPLIT: Multi-Drug Condition Order")
    print(f"{'='*70}")
    print(f"Base folder: {base_folder}\n")
    
    # Get all folders (excluding suite2p and CSV files)
    all_filelist = os.listdir(base_folder)
    recording_folderlist = [
        x for x in all_filelist 
        if os.path.isdir(os.path.join(base_folder, x)) 
        and x != 'suite2p' 
        and 'analysis' not in x.lower()
    ]
    
    # Sort by number in folder name
    recording_folderlist = sorted(recording_folderlist, key=sort_key_drug_condition)
    
    num_recordings = len(recording_folderlist)
    print(f"Recording folders found (sorted by number):")
    for i, folder in enumerate(recording_folderlist):
        print(f"  {i+1}. {folder}")

    # Find suite2p folder at parent level
    suite2p_folders = ""
    for item in os.scandir(base_folder):
        if item.is_dir() and item.name == 'suite2p':
            plane0_path = os.path.join(item.path, 'plane0')
            if os.path.exists(plane0_path):
                suite2p_folders = item.path
                break

    if not suite2p_folders:
        print("\n❌ No suite2p folder found at parent level")
        print("   Cannot proceed with deconcatenation")
        return None

    print(f"\n✓ Found suite2p folder: {suite2p_folders}")
    
    # Load ops to get total frame count
    ops_path = os.path.join(suite2p_folders, 'plane0', 'ops.npy')
    if not os.path.exists(ops_path):
        print(f"❌ ops.npy not found in {suite2p_folders}/plane0/")
        return None
        
    ops = np.load(ops_path, allow_pickle=True).item()
    total_frames = ops['nframes']
    print(f"✓ Total frames in concatenated data: {total_frames}")

    # Initialize arrays for frame info
    recording_frame_rate = np.zeros(num_recordings)
    num_frames = np.zeros(num_recordings)
    
    print(f"\n{'='*70}")
    print("Reading XML files from recording folders:")
    print(f"{'='*70}")
    
    # Read XML from each folder
    for idx, folder_name in enumerate(recording_folderlist):
        folder_path = os.path.join(base_folder, folder_name)
        
        print(f"\n[{idx+1}/{num_recordings}] {folder_name}")
        
        try:
            # List all files in the folder
            files_in_folder = os.listdir(folder_path)
            xml_files = [f for f in files_in_folder if f.lower().endswith('.xml')]
            
            print(f"  Total files: {len(files_in_folder)}")
            print(f"  XML files: {xml_files if xml_files else '❌ None found'}")
            
            if not xml_files:
                print(f"  ⚠️  No XML file - cannot determine frame count")
                continue
            
            # Use the first XML file (or the one matching folder name)
            xml_to_use = xml_files[0]
            # Prefer XML file that matches folder name
            for xml_file in xml_files:
                if folder_name in xml_file:
                    xml_to_use = xml_file
                    break
            
            xml_path = os.path.join(folder_path, xml_to_use)
            print(f"  → Reading: {xml_to_use}")
            
            # Parse XML
            try:
                xml_dict = files.read_xml(xml_path)
                
                # Get frame rate
                if "rel_time" in xml_dict and len(xml_dict["rel_time"]) > 1:
                    recording_frame_rate[idx] = float(1 / xml_dict["rel_time"][1])
                else:
                    recording_frame_rate[idx] = 15.0  # Default fallback
                    print(f"     ⚠️  'rel_time' not found, using default: 15.0 Hz")
                
                # Get frame count (try multiple possible keys)
                if 'num_frames' in xml_dict:
                    num_frames[idx] = xml_dict['num_frames']
                elif 'nframes' in xml_dict:
                    num_frames[idx] = xml_dict['nframes']
                elif 'frames' in xml_dict:
                    num_frames[idx] = xml_dict['frames']
                else:
                    print(f"     ❌ Frame count not found in XML")
                    print(f"     Available keys: {list(xml_dict.keys())[:10]}")
                    continue
                
                print(f"  ✓ Frames: {int(num_frames[idx])}, Rate: {recording_frame_rate[idx]:.2f} Hz")
                
            except Exception as e:
                print(f"  ❌ XML parsing error: {e}")
                print(f"     File: {xml_path}")
                
        except Exception as e:
            print(f"  ❌ Error accessing folder: {e}")

    # Verify we got frame counts
    total_frames_from_xml = int(np.sum(num_frames))
    
    print(f"\n{'='*70}")
    print("Frame Count Summary:")
    print(f"{'='*70}")
    print(f"Total frames from XMLs: {total_frames_from_xml}")
    print(f"Total frames in suite2p: {total_frames}")
    print(f"Difference: {abs(total_frames_from_xml - total_frames)}")
    print(f"\nIndividual recording frames:")
    for i, (folder, frames) in enumerate(zip(recording_folderlist, num_frames)):
        print(f"  {i+1}. {folder}: {int(frames)} frames")
    
    if total_frames_from_xml == 0:
        print(f"\n❌ ERROR: No frame counts obtained from XML files!")
        print(f"\nPossible solutions:")
        print(f"  1. Check if XML files exist in recording folders")
        print(f"  2. Verify XML files are not corrupted")
        print(f"  3. Check files.read_xml() function compatibility")
        print(f"  4. Use equal split as fallback (see Frame_Detector_Per_Lap_MultiDrug_EqualSplit)")
        return None
    
    # Check for frame count mismatch
    frame_diff = abs(total_frames_from_xml - total_frames)
    if frame_diff > num_recordings * 2:  # Allow small tolerance
        print(f"\n⚠️  WARNING: Large frame count mismatch!")
        print(f"   This may cause incorrect splitting")
        print(f"   Proceeding anyway, but verify results carefully")
    elif frame_diff > 0:
        print(f"\n⚠️  Small frame count mismatch detected ({frame_diff} frames)")
        print(f"   This is often due to rounding - should be OK")

    frame_info = {
        'recording_names': recording_folderlist,
        'num_recordings': num_recordings,
        'suite2p_folder': suite2p_folders,
        'total_frames': total_frames,
        'frame_rate': recording_frame_rate,
        'num_frames': num_frames
    }
    
    return frame_info


def Frame_Detector_Per_Lap_MultiDrug_EqualSplit(base_folder):
    """
    FALLBACK: Split frames equally when XML files are unavailable or corrupted.
    Use this only when Frame_Detector_Per_Lap_MultiDrug fails.
    
    ⚠️  WARNING: This assumes all recordings have equal length!
    """
    print(f"\n{'='*70}")
    print("FALLBACK: Equal Frame Division (No XML)")
    print(f"{'='*70}")
    print(f"⚠️  This assumes all recordings have equal duration!")
    
    all_filelist = os.listdir(base_folder)
    recording_folderlist = [
        x for x in all_filelist 
        if os.path.isdir(os.path.join(base_folder, x)) 
        and x != 'suite2p' 
        and 'analysis' not in x.lower()
    ]
    
    recording_folderlist = sorted(recording_folderlist, key=sort_key_drug_condition)
    
    num_recordings = len(recording_folderlist)
    print(f"Found {num_recordings} recording folders")
    
    # Find suite2p
    suite2p_folders = ""
    for item in os.scandir(base_folder):
        if item.is_dir() and item.name == 'suite2p':
            plane0_path = os.path.join(item.path, 'plane0')
            if os.path.exists(plane0_path):
                suite2p_folders = item.path
                break
    
    if not suite2p_folders:
        print("❌ No suite2p folder found")
        return None
    
    ops_path = os.path.join(suite2p_folders, 'plane0', 'ops.npy')
    ops = np.load(ops_path, allow_pickle=True).item()
    total_frames = ops['nframes']
    
    print(f"\nTotal frames in suite2p: {total_frames}")
    print(f"Splitting equally across {num_recordings} recordings")
    
    # Equal division
    frames_per_recording = total_frames // num_recordings
    remainder = total_frames % num_recordings
    
    num_frames = np.full(num_recordings, frames_per_recording, dtype=float)
    # Distribute remainder frames to first recordings
    num_frames[:remainder] += 1
    
    print(f"\nFrame distribution:")
    for i, (folder, frames) in enumerate(zip(recording_folderlist, num_frames)):
        print(f"  {i+1}. {folder}: {int(frames)} frames")
    
    # Assume 15 Hz frame rate (adjust if you know your actual rate)
    recording_frame_rate = np.full(num_recordings, 15.0)
    
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
    Deconcatenate suite2p output into individual drug condition recordings
    """
    
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
        
        total_frames_in_suite2p = twoP_data['F'].shape[1]
        total_frames_expected = int(np.sum(num_frames))
        
        print(f"\nVerifying frame counts:")
        print(f"  Suite2p actual: {total_frames_in_suite2p} frames")
        print(f"  XML expected: {total_frames_expected} frames")
        print(f"  Individual trial frames: {num_frames}")
        
        # CORRECTED: Calculate cumulative frame boundaries
        cumulative_frames = np.cumsum(num_frames).astype(int)
        
        print(f"\nCumulative frame boundaries: {cumulative_frames}")
        
        # Verify total matches
        if cumulative_frames[-1] != total_frames_in_suite2p:
            print(f"  ⚠️  WARNING: Frame count mismatch!")
            print(f"     Expected: {cumulative_frames[-1]}, Got: {total_frames_in_suite2p}")
            print(f"     Difference: {cumulative_frames[-1] - total_frames_in_suite2p} frames")
            
            # Auto-adjust if close
            if abs(cumulative_frames[-1] - total_frames_in_suite2p) <= num_trials:
                print(f"     → Small difference, likely due to rounding. Proceeding...")
            else:
                print(f"     → Large mismatch - check your XML files!")
                return None

        # Split and save data for each trial
        for i in range(num_trials):
            print(f"\nProcessing trial {i+1}/{num_trials}: {trial_dir[i]}")
            
            output_dir = os.path.join(recording_folder, trial_dir[i], r'suite2p/plane0/')
            os.makedirs(output_dir, exist_ok=True)
            
            # CORRECTED: Calculate start and end frames
            start_frame = 0 if i == 0 else cumulative_frames[i-1]
            end_frame = cumulative_frames[i]
            
            print(f"  Extracting frames {start_frame} to {end_frame-1} (n={end_frame-start_frame})")
            
            # CORRECTED: Use proper Python slicing (end is exclusive)
            for key in twoP_data.keys():
                if key in ['F', 'Fneu', 'spks']:
                    data_split = twoP_data[key][:, start_frame:end_frame]
                    print(f"  {key} shape: {data_split.shape}")
                    
                    # Verify we got the right number of frames
                    expected_frames = int(num_frames[i])
                    if data_split.shape[1] != expected_frames:
                        print(f"    ⚠️  WARNING: Expected {expected_frames} frames, got {data_split.shape[1]}")
                else:
                    data_split = twoP_data[key]
                
                np.save(os.path.join(output_dir, f'{key}.npy'), data_split)
            
            print(f"  ✓ Saved to {output_dir}")
        
        print("\n" + "="*70)
        print("Deconcatenation complete!")
        print(f"Total frames distributed: {cumulative_frames[-1]}")
        print("="*70)
        
        return cumulative_frames
    else:
        print("No Suite2p folder found. Exiting deconcatenation.")
        return None
# ============================================================================
# HELPER FUNCTIONS (from original script)
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

def calculate_cross_correlation_with_lags(data, max_lag=3):
    """Calculate cross-correlation with time lags"""
    
    n_cells, n_frames = data.shape
    
    print(f"\nCalculating cross-correlations with ±{max_lag} frame lags...")
    
    max_corr_matrix = np.zeros((n_cells, n_cells))
    best_lag_matrix = np.zeros((n_cells, n_cells), dtype=int)
    standard_corr_matrix = np.zeros((n_cells, n_cells))
    
    total_pairs = n_cells * (n_cells - 1) // 2
    
    with tqdm(total=total_pairs, desc="Cell pairs") as pbar:
        for i in range(n_cells):
            for j in range(i+1, n_cells):
                signal_i = data[i, :]
                signal_j = data[j, :]
                
                signal_i_norm = (signal_i - np.mean(signal_i)) / (np.std(signal_i) + 1e-10)
                signal_j_norm = (signal_j - np.mean(signal_j)) / (np.std(signal_j) + 1e-10)
                
                correlations = []
                lags = range(-max_lag, max_lag + 1)
                
                for lag in lags:
                    if lag < 0:
                        overlap_i = signal_i_norm[:lag]
                        overlap_j = signal_j_norm[-lag:]
                    elif lag > 0:
                        overlap_i = signal_i_norm[lag:]
                        overlap_j = signal_j_norm[:-lag]
                    else:
                        overlap_i = signal_i_norm
                        overlap_j = signal_j_norm
                    
                    if len(overlap_i) > 10:
                        corr = np.corrcoef(overlap_i, overlap_j)[0, 1]
                        correlations.append(corr if not np.isnan(corr) else 0.0)
                    else:
                        correlations.append(0.0)
                
                correlations = np.array(correlations)
                max_corr_idx = np.argmax(correlations)
                max_corr = correlations[max_corr_idx]
                best_lag = list(lags)[max_corr_idx]
                
                max_corr_matrix[i, j] = max_corr
                max_corr_matrix[j, i] = max_corr
                best_lag_matrix[i, j] = best_lag
                best_lag_matrix[j, i] = -best_lag
                standard_corr_matrix[i, j] = correlations[max_lag]
                standard_corr_matrix[j, i] = correlations[max_lag]
                
                pbar.update(1)
    
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
    
    print(f"  Mean correlation: {correlation_stats['mean_max_correlation']:.3f}")
    
    return max_corr_matrix, best_lag_matrix, standard_corr_matrix, correlation_stats

def identify_top_synchronous_cells(correlation_matrix, n_cells=10):
    """Identify top N most synchronous cells"""
    
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

def get_condition_label(folder_name):
    """Extract drug condition label from folder name"""
    folder_upper = folder_name.upper()
    
    if 'TTX' in folder_upper:
        return 'TTX'
    elif 'CNQX' in folder_upper:
        return 'CNQX'
    elif 'AP5' in folder_upper and 'WASHED' in folder_upper:
        return 'AP5 washed'
    elif 'AP5' in folder_upper:
        return 'AP5'
    elif 'GZ' in folder_upper and 'WASHED' in folder_upper:
        return 'GZ washed'
    elif 'GZ' in folder_upper or 'GABAZINE' in folder_upper:
        return 'Gabazine'
    else:
        return 'Pre-imaging'

# ============================================================================
# MEGA-FIGURE FOR ALL DRUG CONDITIONS
# ============================================================================
def create_mega_figure_all_conditions(
    dff_data_list,
    spike_data_list,
    sync_cell_indices,
    stat, avg_projection,
    corr_stats_list,
    frame_rate,
    condition_labels,
    output_path=None
):
    """
    Create mega-figure showing all drug conditions side-by-side
    """
    
    n_conditions = len(dff_data_list)
    n_cells = len(sync_cell_indices)
    colors = plt.cm.tab10(np.linspace(0, 1, n_cells))
    
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 9
    
    fig, axes = plt.subplots(3, n_conditions, figsize=(6*n_conditions, 18))
    
    if n_conditions == 1:
        axes = axes.reshape(-1, 1)
    
    for col_idx, (dff, spike, corr_stats, label) in enumerate(
        zip(dff_data_list, spike_data_list, corr_stats_list, condition_labels)):
        
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
        
        # Check if we have any data to plot
        if len(time_vector) == 0 or dff_subset.shape[1] == 0:
            ax_traces.text(0.5, 0.5, 'No data available', 
                          ha='center', va='center', transform=ax_traces.transAxes,
                          fontsize=14, color='red')
            ax_traces.set_title(f'{label}: No Activity Detected', 
                               fontsize=14, weight='bold', pad=8)
        else:
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
        
        # Create correlation subset for top 10 cells
        corr_subset = np.zeros((n_cells, n_cells))
        corr_matrix_full = np.corrcoef(dff)
        for i, idx_i in enumerate(sync_cell_indices):
            for j, idx_j in enumerate(sync_cell_indices):
                corr_subset[i, j] = corr_matrix_full[idx_i, idx_j]
        
        im = ax_corr.imshow(corr_subset, cmap='Reds', aspect='equal',
                           vmin=0, vmax=1, interpolation='nearest')
        
        # Calculate BOTH mean correlations
        mean_corr_global = corr_stats['mean_max_correlation']  # From ALL cells
        
        # Calculate mean correlation for only the top 10 cells
        upper_tri_indices = np.triu_indices_from(corr_subset, k=1)  # Upper triangle, exclude diagonal
        mean_corr_top10 = np.mean(corr_subset[upper_tri_indices])
        
        # Display both values in title
        ax_corr.set_title(
            f'{label}: Correlation\n'
            f'Global mean: r={mean_corr_global:.3f}\n'
            f'Top 10 mean: r={mean_corr_top10:.3f}',
            fontsize=13, weight='bold', pad=8
        )
        ax_corr.set_xlabel('Cell #', fontsize=11, weight='bold')
        ax_corr.set_ylabel('Cell #', fontsize=11, weight='bold')
        ax_corr.set_xticks(range(n_cells))
        ax_corr.set_yticks(range(n_cells))
        ax_corr.set_xticklabels(range(1, n_cells + 1))
        ax_corr.set_yticklabels(range(1, n_cells + 1))
        
        if col_idx == n_conditions - 1:
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
    
    if len(legend_elements) > 0:  # Only create legend if we have cells
        legend = ax_legend.legend(handles=legend_elements, loc='center', 
                                 ncol=min(5, n_cells),
                                 frameon=True, fontsize=11, fancybox=True, 
                                 shadow=False, framealpha=0.9,
                                 columnspacing=1.5, handlelength=2.5)
        legend.get_frame().set_linewidth(1.5)
    
    fig.text(0.5, 0.98, 
             f'Multi-Drug Condition Analysis: {n_conditions} Conditions Tracked',
             ha='center', fontsize=20, weight='bold', family='Arial')
    
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"\n✓ Saved mega-figure to: {output_path}")
    
    plt.close(fig)
    plt.rcParams.update(plt.rcParamsDefault)
    
    return fig

# ============================================================================
# MAIN PROCESSING
# ============================================================================

def process_organoid_multidrug(organoid_path, organoid_name):
    """
    Process one organoid with multiple drug conditions
    """
    
    print(f"\n{'='*80}")
    print(f"PROCESSING ORGANOID: {organoid_name}")
    print(f"{'='*80}")
    
    # Find all recording folders
    subfolders = [f for f in os.listdir(organoid_path) 
                 if os.path.isdir(os.path.join(organoid_path, f)) 
                 and f != 'suite2p' and 'analysis' not in f.lower()]
    
    # Sort by drug condition order
    def sort_key_drug_condition(folder_name):
        folder_upper = folder_name.upper()
        if 'TTX' in folder_upper:
            return 6
        elif 'CNQX' in folder_upper:
            return 5
        elif 'AP5' in folder_upper and 'WASHED' in folder_upper:
            return 4
        elif 'AP5' in folder_upper:
            return 3
        elif 'GZ' in folder_upper and 'WASHED' in folder_upper:
            return 2
        elif 'GZ' in folder_upper or 'GABAZINE' in folder_upper:
            return 1
        else:
            return 0
    
    subfolders = sorted(subfolders, key=sort_key_drug_condition)
    
    # Check if recordings have suite2p data
    recordings_with_data = []
    for subfolder in subfolders:
        suite2p_path = os.path.join(organoid_path, subfolder, 'suite2p', 'plane0')
        if os.path.exists(suite2p_path):
            recordings_with_data.append(subfolder)
    
    if len(recordings_with_data) == 0:
        print(f"  ⚠️  No recordings with suite2p data found")
        return None
    
    print(f"Found {len(recordings_with_data)} recordings with data:")
    for rec in recordings_with_data:
        label = get_condition_label(rec)
        print(f"  - {rec} ({label})")
    
    # Create output folder
    output_folder = os.path.join(organoid_path, 
                                f'analysis_{datetime.datetime.now().strftime("%Y%m%d")}')
    os.makedirs(output_folder, exist_ok=True)
    
    # Load data for all conditions
    print(f"\n{'='*70}")
    print("Loading data for all conditions...")
    print(f"{'='*70}")
    
    all_dff_list = []
    all_spike_list = []
    all_corr_stats_list = []
    condition_labels = []
    
    # Load PRE first to determine filtering
    pre_path = os.path.join(organoid_path, recordings_with_data[0])
    
    print(f"  Loading PRE for filtering...")
    F_pre = np.load(os.path.join(pre_path, 'suite2p', 'plane0', 'F.npy'))
    Fneu_pre = np.load(os.path.join(pre_path, 'suite2p', 'plane0', 'Fneu.npy'))
    
    # CONVERT TO FLOAT64 IMMEDIATELY
    F_pre = F_pre.astype(np.float64)
    Fneu_pre = Fneu_pre.astype(np.float64)

    stat = np.load(os.path.join(pre_path, 'suite2p', 'plane0', 'stat.npy'), allow_pickle=True)
    ops = np.load(os.path.join(pre_path, 'suite2p', 'plane0', 'ops.npy'), allow_pickle=True).item()
    
    # # Add diagnostic code at the start of process_organoid_multidrug:
    # print("\n=== SUITE2P DATA DIAGNOSTICS ===")
    # print(f"F.npy dtype: {F_pre.dtype}")
    # print(f"F.npy shape: {F_pre.shape}")
    # print(f"F.npy value range: [{F_pre.min():.2f}, {F_pre.max():.2f}]")
    # print(f"F.npy has NaN: {np.any(np.isnan(F_pre))}")
    # print(f"F.npy has Inf: {np.any(np.isinf(F_pre))}")

    # print(f"Fneu.npy dtype: {Fneu_pre.dtype}")

    # print(f"Suite2p version: {ops.get('version', 'unknown')}")
    # print(f"Ops keys: {list(ops.keys())[:10]}...")  # First 10 keys

    # Get frame rate
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
    
    print(f"Frame rate: {frame_rate:.2f} Hz")
    
    n_cells = F_pre.shape[0]
    
    # Process PRE
    dff_pre = (F_pre - 0.7 * Fneu_pre) / (F_pre - 0.7 * Fneu_pre).mean(axis=1, keepdims=True)
    _, spikes_pre = process_spike_data_gcamp6m(dff_pre, n_cells, F_pre.shape[1], sampling_rate=frame_rate)
    
    # Filter cells using PRE
    print(f"\n{'='*70}")
    print("Filtering cells based on PRE-imaging")
    print(f"{'='*70}")
    
    stage1_mask, _ = basic_signal_quality_filter(dff_pre, spikes_pre, use_dff_for_filtering=False)
    final_mask, _ = event_based_snr_filter(dff_pre, spikes_pre, stage1_mask, sampling_rate=frame_rate)
    
    print(f"\nFiltering complete:")
    print(f"  Original cells: {n_cells}")
    print(f"  Filtered cells: {np.sum(final_mask)}")
    print(f"  **SAME MASK APPLIED TO ALL CONDITIONS**")
    
    # Get sync cells from PRE
    corr_pre, _, _, corr_pre_stats = calculate_cross_correlation_with_lags(spikes_pre[final_mask, :])
    sync_cell_indices, sync_scores = identify_top_synchronous_cells(corr_pre, n_cells=10)  # ← CAPTURE sync_scores

    
    # Apply filtering to PRE
    dff_pre_filtered = dff_pre[final_mask, :]
    spikes_pre_filtered = spikes_pre[final_mask, :]
    stat_filtered = stat[final_mask]
    
    all_dff_list.append(dff_pre_filtered)
    all_spike_list.append(spikes_pre_filtered)
    all_corr_stats_list.append(corr_pre_stats)
    condition_labels.append(get_condition_label(recordings_with_data[0]))
    
    # Load remaining conditions
    for rec_folder in recordings_with_data[1:]:
        rec_path = os.path.join(organoid_path, rec_folder)
        label = get_condition_label(rec_folder)
        
        print(f"\n  Loading {label}...")
        
        F = np.load(os.path.join(rec_path, 'suite2p', 'plane0', 'F.npy'))
        Fneu = np.load(os.path.join(rec_path, 'suite2p', 'plane0', 'Fneu.npy'))
        
        # CONVERT TO FLOAT64
        F = F.astype(np.float64)
        Fneu = Fneu.astype(np.float64)
        
        dff = (F - 0.7 * Fneu) / (F - 0.7 * Fneu).mean(axis=1, keepdims=True)
        _, spikes = process_spike_data_gcamp6m(dff, n_cells, F.shape[1], sampling_rate=frame_rate)
        
        # Apply same filtering
        dff_filtered = dff[final_mask, :]
        spikes_filtered = spikes[final_mask, :]
        
        # Check if this condition has any activity
        if spikes_filtered.shape[1] == 0 or np.all(spikes_filtered == 0):
            print(f"    ⚠️  WARNING: No activity detected in {label} (expected for TTX)")
            print(f"    Will use DFF data for visualization")
            # For inactive conditions, correlation will be near-zero
            corr_stats = {
                'mean_max_correlation': 0.0,
                'median_max_correlation': 0.0
            }
        else:
            # Calculate correlation only if there's activity
            corr, _, _, corr_stats = calculate_cross_correlation_with_lags(spikes_filtered)
        
        all_dff_list.append(dff_filtered)
        all_spike_list.append(spikes_filtered)
        all_corr_stats_list.append(corr_stats)
        condition_labels.append(label)
    
    # Create mega-figure
    print(f"\n{'='*70}")
    print("Creating mega-figure for all conditions")
    print(f"{'='*70}")
    
    if 'meanImg' in ops:
        avg_projection = ops['meanImg']
    elif 'max_proj' in ops:
        avg_projection = ops['max_proj']
    else:
        avg_projection = ops['refImg']
    
    mega_output_path = os.path.join(output_folder, 
                                   f"{organoid_name}_all_conditions_mega_figure.png")
    
    create_mega_figure_all_conditions(
        all_dff_list,
        all_spike_list,
        sync_cell_indices,
        stat_filtered,
        avg_projection,
        all_corr_stats_list,
        frame_rate,
        condition_labels,
        output_path=mega_output_path
    )
    
    print(f"✓ Mega-figure created successfully")
    
    # After creating mega-figure, add detailed statistics saving
    print(f"\n{'='*70}")
    print("Saving detailed statistics...")
    print(f"{'='*70}")
    
    # Calculate top 10 mean correlations for each condition
    top10_mean_correlations = []
    
    for dff, sync_indices in zip(all_dff_list, [sync_cell_indices] * len(all_dff_list)):
        # Calculate correlation matrix for this condition
        corr_matrix_full = np.corrcoef(dff)
        
        # Extract subset for top 10 cells
        corr_subset = np.zeros((len(sync_indices), len(sync_indices)))
        for i, idx_i in enumerate(sync_indices):
            for j, idx_j in enumerate(sync_indices):
                corr_subset[i, j] = corr_matrix_full[idx_i, idx_j]
        
        # Calculate mean of upper triangle (excluding diagonal)
        upper_tri = np.triu_indices_from(corr_subset, k=1)
        mean_top10 = np.mean(corr_subset[upper_tri])
        top10_mean_correlations.append(mean_top10)
    
    # Save summary with both correlation metrics
    summary = {
        'organoid_name': organoid_name,
        'n_conditions': len(condition_labels),
        'conditions': condition_labels,
        'n_cells_original': n_cells,
        'n_cells_filtered': np.sum(final_mask),
        'filter_pass_rate': np.sum(final_mask) / n_cells,
        'global_mean_correlations': [stats['mean_max_correlation'] for stats in all_corr_stats_list],
        'top10_mean_correlations': top10_mean_correlations,
        'sync_cell_indices': sync_cell_indices.tolist(),
        'frame_rate': frame_rate
    }
    
    # Save detailed CSV with per-condition statistics
    condition_stats_list = []
    for idx, (label, global_corr, top10_corr) in enumerate(
        zip(condition_labels, 
            summary['global_mean_correlations'], 
            summary['top10_mean_correlations'])):
        
        condition_stats_list.append({
            'Organoid': organoid_name,
            'Condition': label,
            'Condition_Order': idx + 1,
            'N_Cells_Original': n_cells,
            'N_Cells_Filtered': np.sum(final_mask),
            'Filter_Pass_Rate_%': np.sum(final_mask) / n_cells * 100,
            'Global_Mean_Correlation': global_corr,
            'Top10_Mean_Correlation': top10_corr,
            'Correlation_Difference': top10_corr - global_corr,
            'Frame_Rate_Hz': frame_rate
        })
    
    condition_df = pd.DataFrame(condition_stats_list)
    condition_csv_path = os.path.join(output_folder, 
                                      f"{organoid_name}_condition_statistics.csv")
    condition_df.to_csv(condition_csv_path, index=False)
    print(f"✓ Saved condition statistics to: {condition_csv_path}")
    
    # Save cell-level information
    cell_info_list = []
    for idx, cell_idx in enumerate(sync_cell_indices):
        cell_info_list.append({
            'Organoid': organoid_name,
            'Display_Number': idx + 1,
            'ROI_Index': cell_idx,
            'Synchrony_Score': sync_scores[idx]
        })
    
    cell_df = pd.DataFrame(cell_info_list)
    cell_csv_path = os.path.join(output_folder, 
                                 f"{organoid_name}_top10_cells.csv")
    cell_df.to_csv(cell_csv_path, index=False)
    print(f"✓ Saved top 10 cell information to: {cell_csv_path}")
    
    print(f"✓ Mega-figure created successfully")
    
    return summary
def batch_process_organoids_multidrug(base_folder):
    """
    Process all organoids in base folder
    """
    
    print(f"\n{'='*80}")
    print("BATCH PROCESSING: MULTI-DRUG CONDITION ANALYSIS")
    print(f"{'='*80}")
    print(f"Base folder: {base_folder}")
    
    all_results = []
    
    all_items = os.listdir(base_folder)
    for item in all_items:
        item_path = os.path.join(base_folder, item)
        
        if not os.path.isdir(item_path):
            continue
        
        try:
            result = process_organoid_multidrug(item_path, item)
            if result is not None:
                all_results.append(result)
        except Exception as e:
            print(f"\n❌ ERROR processing {item}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print summary
    if len(all_results) == 0:
        print("\n❌ No organoids processed successfully!")
        return None
    
    print(f"\n{'='*80}")
    print("BATCH PROCESSING COMPLETE - SUMMARY")
    print(f"{'='*80}")
    
    for result in all_results:
        print(f"\n{result['organoid_name']}:")
        print(f"  Conditions: {', '.join(result['conditions'])}")
        print(f"  Cells: {result['n_cells_filtered']}/{result['n_cells_original']}")
        
        # FIXED: Use the correct keys for global and top10 correlations
        print(f"  Global correlations: {[f'{c:.3f}' for c in result['global_mean_correlations']]}")
        print(f"  Top10 correlations:  {[f'{c:.3f}' for c in result['top10_mean_correlations']]}")
    
    # Save comprehensive batch summary
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save all condition data (most detailed)
    all_condition_data = []
    for result in all_results:
        for idx, (condition, global_corr, top10_corr) in enumerate(
            zip(result['conditions'],
                result['global_mean_correlations'],
                result['top10_mean_correlations'])):
            
            all_condition_data.append({
                'Organoid': result['organoid_name'],
                'Condition': condition,
                'Condition_Order': idx + 1,
                'N_Cells_Filtered': result['n_cells_filtered'],
                'N_Cells_Original': result['n_cells_original'],
                'Filter_Pass_Rate_%': result['filter_pass_rate'] * 100,
                'Global_Mean_Correlation': global_corr,
                'Top10_Mean_Correlation': top10_corr,
                'Correlation_Difference': top10_corr - global_corr,
                'Frame_Rate_Hz': result.get('frame_rate', 15.0)
            })
    
    all_condition_df = pd.DataFrame(all_condition_data)
    condition_summary_path = os.path.join(base_folder, 
                                         f'batch_condition_summary_{timestamp}.csv')
    all_condition_df.to_csv(condition_summary_path, index=False)
    print(f"\n✓ Batch condition summary saved to: {condition_summary_path}")
    
    # Save organoid-level summary
    organoid_summary_list = []
    for result in all_results:
        organoid_summary_list.append({
            'Organoid': result['organoid_name'],
            'N_Conditions': result['n_conditions'],
            'Conditions': ', '.join(result['conditions']),
            'N_Cells_Original': result['n_cells_original'],
            'N_Cells_Filtered': result['n_cells_filtered'],
            'Filter_Pass_Rate_%': result['filter_pass_rate'] * 100,
            'Mean_Global_Correlation': np.mean(result['global_mean_correlations']),
            'Mean_Top10_Correlation': np.mean(result['top10_mean_correlations']),
            'Std_Global_Correlation': np.std(result['global_mean_correlations']),
            'Std_Top10_Correlation': np.std(result['top10_mean_correlations']),
            'Frame_Rate_Hz': result.get('frame_rate', 15.0)
        })
    
    organoid_df = pd.DataFrame(organoid_summary_list)
    organoid_summary_path = os.path.join(base_folder, 
                                         f'batch_organoid_summary_{timestamp}.csv')
    organoid_df.to_csv(organoid_summary_path, index=False)
    print(f"✓ Batch organoid summary saved to: {organoid_summary_path}")
    
    # Create summary statistics by condition (averaged across organoids)
    if len(all_condition_data) > 0:
        condition_df = pd.DataFrame(all_condition_data)
        
        # Group by condition and calculate mean ± std
        condition_grouped = condition_df.groupby('Condition').agg({
            'Global_Mean_Correlation': ['mean', 'std', 'count'],
            'Top10_Mean_Correlation': ['mean', 'std', 'count'],
            'Correlation_Difference': ['mean', 'std'],
            'Filter_Pass_Rate_%': ['mean', 'std'],
            'N_Cells_Filtered': ['mean', 'std']
        }).round(4)
        
        condition_grouped_path = os.path.join(base_folder, 
                                              f'batch_condition_grouped_{timestamp}.csv')
        condition_grouped.to_csv(condition_grouped_path)
        print(f"✓ Grouped condition statistics saved to: {condition_grouped_path}")
        
        # Print grouped summary
        print(f"\n{'='*80}")
        print("CONDITION-WISE SUMMARY (Averaged across organoids)")
        print(f"{'='*80}")
        
        for condition in condition_df['Condition'].unique():
            cond_data = condition_df[condition_df['Condition'] == condition]
            print(f"\n{condition}:")
            print(f"  N organoids: {len(cond_data)}")
            print(f"  Global correlation: {cond_data['Global_Mean_Correlation'].mean():.3f} ± {cond_data['Global_Mean_Correlation'].std():.3f}")
            print(f"  Top10 correlation:  {cond_data['Top10_Mean_Correlation'].mean():.3f} ± {cond_data['Top10_Mean_Correlation'].std():.3f}")
            print(f"  Difference:         {cond_data['Correlation_Difference'].mean():.3f} ± {cond_data['Correlation_Difference'].std():.3f}")
    
    print(f"\n{'='*80}")
    print("ALL BATCH PROCESSING COMPLETE!")
    print(f"{'='*80}")
    
    return all_results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    BASE_FOLDER = r'Z:\from_jasmine\3x\B4_D150_GC_3x'
    
    print("="*80)
    print("MULTI-DRUG CONDITION ANALYSIS FOR ORGANOID IMAGING")
    print("="*80)
    # ========================================================================
    # STEP 0: DECONCAT (if needed)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 0: CHECKING FOR CONCATENATED SUITE2P DATA")
    print("="*80)
    print("\nScanning for organoid folders that need deconcatenation...")

    all_files = os.listdir(BASE_FOLDER)
    for organoid in all_files:
        organoid_folder = os.path.join(BASE_FOLDER, organoid)
        if not os.path.isdir(organoid_folder):
            continue
            
        # Check if this folder has a main suite2p folder
        main_suite2p = os.path.join(organoid_folder, 'suite2p', 'plane0')
        
        if os.path.exists(main_suite2p):
            print(f"\n{'='*70}")
            print(f"Found concatenated suite2p data in: {organoid}")
            print(f"{'='*70}")
            
            # Check if subfolders already have suite2p data
            subfolders = [
                f for f in os.listdir(organoid_folder) 
                if os.path.isdir(os.path.join(organoid_folder, f)) 
                and f != 'suite2p' 
                and 'analysis' not in f.lower()
            ]
            
            already_deconcat = False
            for subfolder in subfolders:
                subfolder_suite2p = os.path.join(organoid_folder, subfolder, 'suite2p', 'plane0')
                if os.path.exists(subfolder_suite2p):
                    already_deconcat = True
                    break
            
            if already_deconcat:
                print(f"  ✓ Already deconcatenated (found individual suite2p folders)")
                print(f"  → Skipping deconcat for {organoid}")
            else:
                print(f"  → Running deconcat for {organoid}...")
                try:
                    # Try normal XML-based deconcatenation
                    frame_info = Frame_Detector_Per_Lap_MultiDrug(organoid_folder)
                    
                    if frame_info is None:
                        print(f"\n  ⚠️  XML-based split failed. Trying equal split...")
                        frame_info = Frame_Detector_Per_Lap_MultiDrug_EqualSplit(organoid_folder)
                    
                    if frame_info is not None:
                        cumulative_frames = deconcat_suite2p_output(frame_info, organoid_folder)
                        if cumulative_frames is not None:
                            print(f"  ✓ Deconcat complete for {organoid}")
                        else:
                            print(f"  ❌ Deconcat failed for {organoid}")
                    else:
                        print(f"  ❌ Could not determine frame splits for {organoid}")
                        
                except Exception as e:
                    print(f"  ❌ ERROR during deconcat: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            print(f"\n  → {organoid}: No main suite2p folder (already split or not processed)")
    # ========================================================================
    # MAIN ANALYSIS PIPELINE
    # ========================================================================
    print("\n" + "="*80)
    print("MAIN ANALYSIS PIPELINE")
    print("="*80)
    print("\nThis pipeline will:")
    print("  1. Process each organoid separately")
    print("  2. Filter cells based on PRE-imaging condition")
    print("  3. Create mega-figure showing all drug conditions side-by-side")
    print("  4. Conditions order: Pre → Gabazine → GZ_washed → AP5 → AP5_washed → CNQX → TTX")
    print("="*80)
    
    summary_df = batch_process_organoids_multidrug(base_folder=BASE_FOLDER)
    
    if summary_df is not None:
        print("\n" + "="*80)
        print("ALL DONE!")
        print("="*80)