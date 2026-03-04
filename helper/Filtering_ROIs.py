"""
Filtering_ROIs.py
This script filters regions of interest (ROIs) based on their signal quality.

JSY, 08/2025
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def basic_signal_quality_filter(dff_data, spike_data, 
                               peak_percentile=25, 
                               variance_low_percentile=15, 
                               variance_high_percentile=90,
                               use_dff_for_filtering=True):
    """
    Basic signal quality filtering using simple trace statistics.
    
    Parameters:
        dff_data: (n_cells, n_frames) DFF data
        spike_data: (n_cells, n_frames) spike data  
        peak_percentile: minimum percentile for peak amplitude (default: 25)
        variance_low_percentile: minimum percentile for variance (default: 15)
        variance_high_percentile: maximum percentile for variance (default: 90)
        use_dff_for_filtering: if True, use DFF data for filtering; if False, use spike data
    
    Returns:
        filtering_mask: boolean array indicating which ROIs pass filtering
        filtering_stats: dictionary with filtering statistics
    """
    
    n_cells, n_frames = dff_data.shape
    print(f"\n=== BASIC SIGNAL QUALITY FILTERING ===")
    print(f"Input: {n_cells} ROIs, {n_frames} frames")
    
    # Choose data for filtering
    filter_data = dff_data if use_dff_for_filtering else spike_data
    data_type = "DFF" if use_dff_for_filtering else "Spike"
    print(f"Using {data_type} data for filtering")
    
    # Calculate metrics for each ROI
    peak_amplitudes = np.zeros(n_cells)
    variances = np.zeros(n_cells)
    signal_ranges = np.zeros(n_cells)
    
    print("Calculating signal quality metrics...")
    for i in tqdm(range(n_cells)):
        roi_trace = filter_data[i, :]
        
        peak_amplitudes[i] = np.max(roi_trace)
        variances[i] = np.var(roi_trace)
        signal_ranges[i] = np.max(roi_trace) - np.min(roi_trace)
    
    # Calculate thresholds using percentiles
    peak_threshold = np.percentile(peak_amplitudes, peak_percentile)
    var_low_threshold = np.percentile(variances, variance_low_percentile)
    var_high_threshold = np.percentile(variances, variance_high_percentile)
    
    print(f"\nThresholds calculated:")
    print(f"  Peak amplitude (>{peak_percentile}th percentile): {peak_threshold:.4f}")
    print(f"  Variance range: {var_low_threshold:.6f} to {var_high_threshold:.6f}")
    
    # Apply filters
    peak_pass = peak_amplitudes >= peak_threshold
    variance_pass = (variances > var_low_threshold) & (variances < var_high_threshold)
    
    # Combined filter
    filtering_mask = peak_pass & variance_pass
    
    # Calculate statistics
    n_peak_fail = np.sum(~peak_pass)
    n_variance_fail = np.sum(~variance_pass)
    n_filtered_pass = np.sum(filtering_mask)
    
    print(f"\nFiltering results:")
    print(f"  Failed peak amplitude: {n_peak_fail}/{n_cells} ({n_peak_fail/n_cells*100:.1f}%)")
    print(f"  Failed variance bounds: {n_variance_fail}/{n_cells} ({n_variance_fail/n_cells*100:.1f}%)")
    print(f"  Passed filtering: {n_filtered_pass}/{n_cells} ({n_filtered_pass/n_cells*100:.1f}%)")
    
    # Additional statistics
    valid_peak_amplitudes = peak_amplitudes[filtering_mask]
    valid_variances = variances[filtering_mask]
    
    if len(valid_peak_amplitudes) > 0:
        print(f"\nStatistics for filtered ROIs:")
        print(f"  Peak amplitude - Mean: {np.mean(valid_peak_amplitudes):.3f}, "
              f"Range: {np.min(valid_peak_amplitudes):.3f} to {np.max(valid_peak_amplitudes):.3f}")
        print(f"  Variance - Mean: {np.mean(valid_variances):.6f}, "
              f"Range: {np.min(valid_variances):.6f} to {np.max(valid_variances):.6f}")
    
    filtering_stats = {
        'input_rois': n_cells,
        'filtered_rois': n_filtered_pass,
        'pass_rate': n_filtered_pass/n_cells,
        'peak_threshold': peak_threshold,
        'variance_low_threshold': var_low_threshold,
        'variance_high_threshold': var_high_threshold,
        'peak_failures': n_peak_fail,
        'variance_failures': n_variance_fail,
        'peak_amplitudes': peak_amplitudes,
        'variances': variances,
        'signal_ranges': signal_ranges,
        'filtering_method': 'basic_signal_quality',
        'data_type_used': data_type
    }
    
    print(f"Spike data stats: min={np.min(spike_data)}, max={np.max(spike_data)}")
    print(f"Peak amplitudes: min={np.min(peak_amplitudes)}, max={np.max(peak_amplitudes)}")
    print(f"Peak threshold ({peak_percentile}th percentile): {peak_threshold}")
    print(f"Variances: min={np.min(variances)}, max={np.max(variances)}")
    print(f"Variance thresholds: {var_low_threshold} to {var_high_threshold}")
    print(f"ROIs passing peak filter: {np.sum(peak_pass)}/{len(peak_pass)}")
    print(f"ROIs passing variance filter: {np.sum(variance_pass)}/{len(variance_pass)}")
    return filtering_mask, filtering_stats
  
def detect_events_single_roi(roi_trace, method='adaptive_threshold', 
                           threshold_factor=2.5, min_prominence=None, 
                           min_duration=2, sampling_rate=10):
    """
    Detect calcium events in a single ROI trace.
    
    Parameters:
        roi_trace: 1D array of calcium signal for one ROI
        method: 'adaptive_threshold' or 'peak_detection'
        threshold_factor: multiplier for standard deviation threshold
        min_prominence: minimum prominence for peak detection
        min_duration: minimum event duration in frames
        sampling_rate: frames per second
        
    Returns:
        event_frames: boolean array indicating event periods
        event_peaks: array of peak amplitudes
        n_events: number of detected events
    """
    
    if len(roi_trace) == 0 or np.all(np.isnan(roi_trace)) or np.var(roi_trace) < 1e-10:
        return np.zeros_like(roi_trace, dtype=bool), np.array([]), 0
    
    if method == 'adaptive_threshold':
        # Calculate baseline and threshold
        baseline_mean = np.mean(roi_trace)
        baseline_std = np.std(roi_trace)
        threshold = baseline_mean + threshold_factor * baseline_std
        
        # Find frames above threshold
        above_threshold = roi_trace > threshold
        
        # Apply minimum duration filter
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
                    # Find peak in this event
                    event_segment = roi_trace[current_event_start:i]
                    event_peaks.append(np.max(event_segment))
                current_event_start = None
        
        # Handle event that goes to end
        if current_event_start is not None:
            event_duration = len(above_threshold) - current_event_start
            if event_duration >= min_duration:
                event_frames[current_event_start:] = True
                event_segment = roi_trace[current_event_start:]
                event_peaks.append(np.max(event_segment))
        
        event_peaks = np.array(event_peaks)
        n_events = len(event_peaks)
        
    elif method == 'peak_detection':
        from scipy.signal import find_peaks
        
        # Set minimum prominence if not provided
        if min_prominence is None:
            min_prominence = np.std(roi_trace) * 1.5
        
        # Find peaks
        peaks, properties = find_peaks(roi_trace, 
                                     prominence=min_prominence,
                                     distance=min_duration)
        
        # Create event windows around peaks
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
                          snr_threshold=3.0, min_events=1,
                          event_detection_method='adaptive_threshold',
                          threshold_factor=2.5, min_duration=2, 
                          sampling_rate=10, use_dff_for_snr=True):
    """
    Stage 2: Event-based SNR filtering for ROIs that passed Stage 1.
    
    Parameters:
        dff_data: (n_cells, n_frames) DFF data
        spike_data: (n_cells, n_frames) spike data
        stage1_mask: boolean array from Stage 1 filtering
        snr_threshold: minimum SNR for keeping ROI (default: 3.0)
        min_events: minimum number of events required
        event_detection_method: 'adaptive_threshold' or 'peak_detection'
        threshold_factor: for adaptive threshold method
        min_duration: minimum event duration in frames
        sampling_rate: frames per second
        use_dff_for_snr: if True, use DFF for SNR calculation; if False, use spikes
        
    Returns:
        stage2_mask: boolean array indicating which ROIs pass both stages
        filtering_stats: dictionary with filtering statistics
    """
    
    n_cells, n_frames = dff_data.shape
    stage1_survivors = np.sum(stage1_mask)
    
    print(f"\n=== STAGE 2: Event-Based SNR Filtering ===")
    print(f"Input: {stage1_survivors} ROIs (survivors from Stage 1)")
    
    # Choose data for SNR calculation
    snr_data = dff_data if use_dff_for_snr else spike_data
    data_type = "DFF" if use_dff_for_snr else "Spike"
    print(f"Using {data_type} data for SNR calculation")
    print(f"Event detection method: {event_detection_method}")
    print(f"SNR threshold: {snr_threshold}")
    
    # Initialize results
    stage2_mask = np.zeros(n_cells, dtype=bool)
    snr_values = np.full(n_cells, np.nan)
    event_counts = np.zeros(n_cells, dtype=int)
    
    # Process only Stage 1 survivors
    stage1_indices = np.where(stage1_mask)[0]
    
    print("Processing ROIs for event-based SNR...")
    valid_snr_rois = 0
    
    for idx in tqdm(stage1_indices):
        roi_trace = snr_data[idx, :]
        
        # Detect events for this ROI
        event_frames, event_peaks, n_events = detect_events_single_roi(
            roi_trace, 
            method=event_detection_method,
            threshold_factor=threshold_factor,
            min_duration=min_duration,
            sampling_rate=sampling_rate
        )
        
        event_counts[idx] = n_events
        
        # Calculate SNR if events were detected
        if n_events >= min_events and len(event_peaks) > 0:
            # Define quiet periods (non-event frames)
            quiet_frames = ~event_frames
            
            if np.sum(quiet_frames) > 5:  # Need sufficient quiet periods
                quiet_data = roi_trace[quiet_frames]
                quiet_mean = np.mean(quiet_data)
                quiet_std = np.std(quiet_data)
                
                if quiet_std > 1e-10:  # Avoid division by zero
                    # Calculate SNR using peak events vs quiet baseline
                    peak_response = np.max(event_peaks)
                    snr = (peak_response - quiet_mean) / quiet_std
                    snr_values[idx] = snr
                    
                    if snr >= snr_threshold:
                        stage2_mask[idx] = True
                        valid_snr_rois += 1
    
    # Calculate statistics
    n_stage2_pass = np.sum(stage2_mask)
    n_no_events = np.sum((event_counts[stage1_indices] < min_events))
    n_low_snr = np.sum((stage1_mask) & (~stage2_mask) & (event_counts >= min_events))
    
    print(f"\nStage 2 filtering results:")
    print(f"  ROIs with insufficient events (<{min_events}): {n_no_events}")
    print(f"  ROIs with low SNR (<{snr_threshold}): {n_low_snr}")
    print(f"  ROIs with valid SNR calculation: {valid_snr_rois}")
    print(f"  Passed Stage 2: {n_stage2_pass}/{stage1_survivors} ({n_stage2_pass/stage1_survivors*100:.1f}% of Stage 1)")
    print(f"  Overall pass rate: {n_stage2_pass}/{n_cells} ({n_stage2_pass/n_cells*100:.1f}% of original)")
    
    # Calculate SNR statistics for valid ROIs
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

def plot_two_stage_filtering_results(dff_data, spike_data, stage1_mask, stage2_mask, 
                                   stage1_stats, stage2_stats, rec_name, save_path):
    """
    Create comprehensive plots showing two-stage filtering results.
    """
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
    axes[0,1].set_title(f'After Stage 1 - DFF\n{n_stage1} ROIs '
                       f'({stage1_stats["pass_rate"]*100:.1f}%)')
    plt.colorbar(im2, ax=axes[0,1])
    
    # After Stage 2 (Final)
    dff_stage2 = dff_data[stage2_mask, :]
    
    im3 = axes[0,2].imshow(dff_stage2, aspect='auto', cmap='hot', interpolation='nearest')
    axes[0,2].set_title(f'After Stage 2 - DFF\n{n_stage2} ROIs '
                       f'({stage2_stats["overall_pass_rate"]*100:.1f}%)')
    plt.colorbar(im3, ax=axes[0,2])
    
    # Spike data - same progression
    im4 = axes[1,0].imshow(spike_data, aspect='auto', cmap='hot', interpolation='nearest')
    axes[1,0].set_title(f'Original Spike Data\n{n_cells} ROIs')
    axes[1,0].set_ylabel('ROI Index')
    plt.colorbar(im4, ax=axes[1,0])
    
    spike_stage1 = spike_data[stage1_mask, :]
    im5 = axes[1,1].imshow(spike_stage1, aspect='auto', cmap='hot', interpolation='nearest')
    axes[1,1].set_title(f'After Stage 1 - Spikes\n{n_stage1} ROIs')
    plt.colorbar(im5, ax=axes[1,1])
    
    spike_stage2 = spike_data[stage2_mask, :]
    im6 = axes[1,2].imshow(spike_stage2, aspect='auto', cmap='hot', interpolation='nearest')
    axes[1,2].set_title(f'After Stage 2 - Spikes\n{n_stage2} ROIs')
    axes[1,2].set_xlabel('Frames')
    plt.colorbar(im6, ax=axes[1,2])
    
    plt.suptitle(f'Two-Stage Filtering Results - {rec_name}', fontsize=16)
    plt.tight_layout()
    
    filtering_plot_path = os.path.join(save_path, f"{rec_name}_two_stage_filtering.jpg")
    plt.savefig(filtering_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved two-stage filtering visualization to {filtering_plot_path}")
    
    return filtering_plot_path


def plot_two_stage_filtering_results_enhanced(dff_data, spike_data, stage1_mask, stage2_mask, 
                                   stage1_stats, stage2_stats, rec_name, save_path):
    """
    Create comprehensive plots showing two-stage filtering results with raster-based exclusion analysis.
    """
    n_cells = len(stage1_mask)
    n_stage1 = np.sum(stage1_mask)
    n_stage2 = np.sum(stage2_mask)
    
    # Create masks for excluded ROIs at each stage
    stage1_excluded_mask = ~stage1_mask  # ROIs excluded in Stage 1
    stage2_excluded_mask = stage1_mask & (~stage2_mask)  # ROIs that passed Stage 1 but failed Stage 2
    
    n_stage1_excluded = np.sum(stage1_excluded_mask)
    n_stage2_excluded = np.sum(stage2_excluded_mask)
    
    print(f"\nFiltering breakdown:")
    print(f"  Original: {n_cells} ROIs")
    print(f"  Stage 1 excluded: {n_stage1_excluded} ROIs")
    print(f"  Stage 1 survivors: {n_stage1} ROIs")
    print(f"  Stage 2 excluded: {n_stage2_excluded} ROIs") 
    print(f"  Final survivors: {n_stage2} ROIs")
    
    # Create 3x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # ========================================
    # Row 1: DFF Data Progression (Heatmaps)
    # ========================================
    
    # (0,0) Original DFF data
    im1 = axes[0,0].imshow(dff_data, aspect='auto', cmap='hot', interpolation='nearest')
    axes[0,0].set_title(f'Original DFF Data\n{n_cells} ROIs')
    axes[0,0].set_ylabel('ROI Index')
    plt.colorbar(im1, ax=axes[0,0])
    
    # (0,1) After Stage 1 - DFF
    dff_stage1 = dff_data[stage1_mask, :]
    im2 = axes[0,1].imshow(dff_stage1, aspect='auto', cmap='hot', interpolation='nearest')
    axes[0,1].set_title(f'After Stage 1 - DFF\n{n_stage1} ROIs '
                       f'({stage1_stats["pass_rate"]*100:.1f}%)')
    plt.colorbar(im2, ax=axes[0,1])
    
    # (0,2) After Stage 2 (Final) - DFF
    dff_stage2 = dff_data[stage2_mask, :]
    im3 = axes[0,2].imshow(dff_stage2, aspect='auto', cmap='hot', interpolation='nearest')
    axes[0,2].set_title(f'After Stage 2 - DFF\n{n_stage2} ROIs '
                       f'({stage2_stats["overall_pass_rate"]*100:.1f}%)')
    plt.colorbar(im3, ax=axes[0,2])
    
    # ========================================
    # Row 2: Spike Data Progression (Heatmaps)
    # ========================================
    
    # (1,0) Original Spike data
    im4 = axes[1,0].imshow(spike_data, aspect='auto', cmap='hot', interpolation='nearest')
    axes[1,0].set_title(f'Original Spike Data\n{n_cells} ROIs')
    axes[1,0].set_ylabel('ROI Index')
    plt.colorbar(im4, ax=axes[1,0])
    
    # (1,1) After Stage 1 - Spikes
    spike_stage1 = spike_data[stage1_mask, :]
    im5 = axes[1,1].imshow(spike_stage1, aspect='auto', cmap='hot', interpolation='nearest')
    axes[1,1].set_title(f'After Stage 1 - Spikes\n{n_stage1} ROIs')
    plt.colorbar(im5, ax=axes[1,1])
    
    # (1,2) After Stage 2 (Final) - Spikes
    spike_stage2 = spike_data[stage2_mask, :]
    im6 = axes[1,2].imshow(spike_stage2, aspect='auto', cmap='hot', interpolation='nearest')
    axes[1,2].set_title(f'After Stage 2 - Spikes\n{n_stage2} ROIs')
    axes[1,2].set_xlabel('Frames')
    plt.colorbar(im6, ax=axes[1,2])
    
    plt.suptitle(f'Two-Stage Filtering Results - {rec_name}', fontsize=16)
    plt.tight_layout()
    
    filtering_plot_path = os.path.join(save_path, f"{rec_name}_two_stage_filtering.jpg")
    plt.savefig(filtering_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================
    # Create Additional Detailed Raster Figure
    # ========================================
    
    # Create a more detailed raster comparison figure
    fig2, axes2 = plt.subplots(2, 3, figsize=(20, 12))
    
    # Top row: DFF rasters
    # Bottom row: Spike rasters
    
    # Column 1: Final survivors (what we kept)
    if n_stage2 > 0:
        # DFF raster for survivors
        dff_survivors = dff_data[stage2_mask, :]
        dff_survivor_raster = dff_survivors > np.percentile(dff_survivors, 75)
        
        axes2[0,0].imshow(dff_survivor_raster, aspect='auto', cmap='Greens', interpolation='nearest')
        axes2[0,0].set_title(f'Final Survivors - DFF Raster\n{n_stage2} ROIs (KEPT)')
        axes2[0,0].set_ylabel('ROI Index')
        
        # Spike raster for survivors
        spike_survivors = spike_data[stage2_mask, :]
        spike_survivor_raster = spike_survivors > np.percentile(spike_survivors, 80)
        
        axes2[1,0].imshow(spike_survivor_raster, aspect='auto', cmap='Greens', interpolation='nearest')
        axes2[1,0].set_title(f'Final Survivors - Spike Raster\n{n_stage2} ROIs')
        axes2[1,0].set_xlabel('Frames')
        axes2[1,0].set_ylabel('ROI Index')
    
    # Column 2: Stage 1 excluded (detailed raster)
    if n_stage1_excluded > 0:
        # DFF raster for Stage 1 excluded
        dff_excluded_s1 = dff_data[stage1_excluded_mask, :]
        dff_s1_raster = dff_excluded_s1 > np.percentile(dff_excluded_s1, 75)
        
        axes2[0,1].imshow(dff_s1_raster, aspect='auto', cmap='Reds', interpolation='nearest')
        axes2[0,1].set_title(f'Stage 1 Excluded - DFF Raster\n{n_stage1_excluded} ROIs'
                            f'(Poor peak amplitude or variance)')
        
        # Add green border to highlight good synchrony patterns (if visible)
        for spine in axes2[0,1].spines.values():
            spine.set_edgecolor('red')
            spine.set_linewidth(2)
            
        # Spike raster for Stage 1 excluded
        spike_excluded_s1 = spike_data[stage1_excluded_mask, :]
        spike_s1_raster = spike_excluded_s1 > np.percentile(spike_excluded_s1, 80)
        
        axes2[1,1].imshow(spike_s1_raster, aspect='auto', cmap='Reds', interpolation='nearest')
        axes2[1,1].set_title(f'Stage 1 Excluded - Spike Raster\n{n_stage1_excluded} ROIs')
        axes2[1,1].set_xlabel('Frames')
        
        for spine in axes2[1,1].spines.values():
            spine.set_edgecolor('red') 
            spine.set_linewidth(2)
    
    # Column 3: Stage 2 excluded (detailed raster)
    if n_stage2_excluded > 0:
        # DFF raster for Stage 2 excluded  
        dff_excluded_s2 = dff_data[stage2_excluded_mask, :]
        dff_s2_raster = dff_excluded_s2 > np.percentile(dff_excluded_s2, 75)
        
        axes2[0,2].imshow(dff_s2_raster, aspect='auto', cmap='Oranges', interpolation='nearest')
        axes2[0,2].set_title(f'Stage 2 Excluded - DFF Raster\n{n_stage2_excluded} ROIs'
                            f'(Passed Stage 1, failed SNR/events)')
        
        for spine in axes2[0,2].spines.values():
            spine.set_edgecolor('orange')
            spine.set_linewidth(2)
            
        # Spike raster for Stage 2 excluded
        spike_excluded_s2 = spike_data[stage2_excluded_mask, :]
        spike_s2_raster = spike_excluded_s2 > np.percentile(spike_excluded_s2, 80)
        
        axes2[1,2].imshow(spike_s2_raster, aspect='auto', cmap='Oranges', interpolation='nearest')
        axes2[1,2].set_title(f'Stage 2 Excluded - Spike Raster\n{n_stage2_excluded} ROIs')
        axes2[1,2].set_xlabel('Frames')
        
        for spine in axes2[1,2].spines.values():
            spine.set_edgecolor('orange')
            spine.set_linewidth(2)
    
    # Add explanatory text
    fig2.suptitle(f'Raster Pattern Analysis - {rec_name}\n', fontsize=16, y=0.95)
    
    plt.tight_layout()
    
    # Save both figures
    filtering_plot_path = os.path.join(save_path, f"{rec_name}_two_stage_filtering.jpg")
    fig.savefig(filtering_plot_path, dpi=300, bbox_inches='tight')
    
    raster_analysis_path = os.path.join(save_path, f"{rec_name}_raster_exclusion_analysis.jpg")
    fig2.savefig(raster_analysis_path, dpi=300, bbox_inches='tight')
    
    plt.close(fig)
    plt.close(fig2)
    
    print(f"Saved two-stage filtering visualization to {filtering_plot_path}")
    print(f"Saved raster exclusion analysis to {raster_analysis_path}")
    
    
    return [filtering_plot_path, raster_analysis_path]

def plot_filtered_vs_unfiltered_rasters(dff_data, spike_data, filtering_mask, 
                                         filtering_stats, rec_name, save_path,
                                         time_window=None, max_rois_display=None):
    """
    Create comprehensive raster plots comparing filtered vs unfiltered ROI populations.
    """
    print("debug_ HEREERE")
    def calculate_signal_noise_ratio(data):
        """Calculate signal-to-noise as (signal_std / noise_std)"""
        snrs = []
        for i in range(data.shape[0]):
            trace = data[i, :]
            signal_threshold = np.percentile(trace, 80)
            noise_threshold = np.percentile(trace, 20)
            
            signal_frames = trace >= signal_threshold
            noise_frames = trace <= noise_threshold
            
            if np.sum(signal_frames) > 5 and np.sum(noise_frames) > 5:
                signal_std = np.std(trace[signal_frames])
                noise_std = np.std(trace[noise_frames])
                if noise_std > 1e-10:
                    snrs.append(signal_std / noise_std)
        return np.array(snrs)
    
    def calculate_dff_spike_correlations(dff_data, spike_data, max_cells=50):
        """Calculate absolute correlations between DFF and spike data"""
        corrs = []
        for i in range(min(len(dff_data), max_cells)):
            if np.var(dff_data[i]) > 1e-10 and np.var(spike_data[i]) > 1e-10:
                corr = np.corrcoef(dff_data[i], spike_data[i])[0,1]
                if not np.isnan(corr):
                    corrs.append(abs(corr))
        return np.array(corrs)
    
    def calculate_active_fraction_fixed(data, method='population_threshold'):
        """
        Calculate fraction of time each ROI is active - FIXED VERSION
        """
        
        if method == 'population_threshold':
            # Use population-wide threshold
            population_threshold = np.percentile(data, 75)
            print(f"  Using population threshold: {population_threshold:.2f}")
            
            active_fractions = []
            for i in range(data.shape[0]):
                trace = data[i, :]
                active_fraction = np.mean(trace > population_threshold)
                active_fractions.append(active_fraction)
            return np.array(active_fractions)
        
        elif method == 'adaptive_threshold':
            # Use mean + 2*std for each ROI
            active_fractions = []
            for i in range(data.shape[0]):
                trace = data[i, :]
                threshold = np.mean(trace) + 2 * np.std(trace)
                active_fraction = np.mean(trace > threshold)
                active_fractions.append(active_fraction)
            return np.array(active_fractions)
        
        elif method == 'fixed_threshold':
            # Use fixed 20% DFF threshold
            fixed_threshold = 20
            print(f"  Using fixed threshold: {fixed_threshold}")
            
            active_fractions = []
            for i in range(data.shape[0]):
                trace = data[i, :]
                active_fraction = np.mean(trace > fixed_threshold)
                active_fractions.append(active_fraction)
            return np.array(active_fractions)
        
        else:
            raise ValueError("Method must be 'population_threshold', 'adaptive_threshold', or 'fixed_threshold'")
    
    # Initial setup (same as before)
    n_cells, n_frames = dff_data.shape
    n_filtered = np.sum(filtering_mask)
    
    is_two_stage = filtering_stats.get('filtering_method') == 'two_stage_snr'
    filter_type = "Two-Stage Filtered" if is_two_stage else "Filtered"
    
    if is_two_stage:
        pass_rate = filtering_stats.get('overall_pass_rate', n_filtered/n_cells)
    else:
        pass_rate = filtering_stats.get('pass_rate', n_filtered/n_cells)
    
    print(f"\nCreating {filter_type.lower()} vs unfiltered raster comparison for {rec_name}")
    print(f"Original ROIs: {n_cells}, {filter_type} ROIs: {n_filtered} ({pass_rate*100:.1f}%)")
    
    # Handle time windowing
    if time_window is not None:
        start_frame, end_frame = time_window
        start_frame = max(0, start_frame)
        end_frame = min(n_frames, end_frame)
        dff_display = dff_data[:, start_frame:end_frame]
        spike_display = spike_data[:, start_frame:end_frame]
        time_suffix = f"_frames_{start_frame}-{end_frame}"
        frame_range_text = f" (Frames {start_frame}-{end_frame})"
    else:
        dff_display = dff_data
        spike_display = spike_data
        time_suffix = ""
        frame_range_text = ""
    
    # Handle ROI subsampling for display
    if max_rois_display is not None and n_cells > max_rois_display:
        roi_indices = np.linspace(0, n_cells-1, max_rois_display, dtype=int)
        dff_display = dff_display[roi_indices, :]
        spike_display = spike_display[roi_indices, :]
        filtering_mask_display = filtering_mask[roi_indices]
        roi_suffix = f"_subset_{max_rois_display}ROIs"
        roi_text = f" (Showing {max_rois_display}/{n_cells} ROIs)"
        display_snr_values = filtering_stats.get('snr_values', np.array([]))[roi_indices] if 'snr_values' in filtering_stats else np.array([])
    else:
        filtering_mask_display = filtering_mask
        roi_suffix = ""
        roi_text = ""
        display_snr_values = filtering_stats.get('snr_values', np.array([]))
    
    # Create filtered datasets
    dff_filtered = dff_display[filtering_mask_display, :]
    spike_filtered = spike_display[filtering_mask_display, :]
    
    plot_paths = []
    
    # Calculate all metrics
    original_peaks = np.max(dff_display, axis=1)
    filtered_peaks = np.max(dff_filtered, axis=1)
    original_activity = np.mean(spike_display, axis=1)
    filtered_activity = np.mean(spike_filtered, axis=1)
    
    # Calculate improved metrics
    original_dff_snrs = calculate_signal_noise_ratio(dff_display)
    filtered_dff_snrs = calculate_signal_noise_ratio(dff_filtered)
    original_dff_spike_corr = calculate_dff_spike_correlations(dff_display, spike_display)
    filtered_dff_spike_corr = calculate_dff_spike_correlations(dff_filtered, spike_filtered)
    
    print("Calculating active fractions...")
    original_active_frac = calculate_active_fraction_fixed(dff_display, method='population_threshold')
    filtered_active_frac = calculate_active_fraction_fixed(dff_filtered, method='population_threshold')
    
    # Create the three-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Population DFF Activity Over Time
    original_pop_activity_dff = np.mean(dff_display, axis=0)
    filtered_pop_activity_dff = np.mean(dff_filtered, axis=0)
    
    time_points = np.arange(len(original_pop_activity_dff))
    axes[0].plot(time_points, original_pop_activity_dff, 'b-', alpha=0.7, label='Original', linewidth=0.8)
    axes[0].plot(time_points, filtered_pop_activity_dff, 'r-', alpha=0.7, label=filter_type, linewidth=0.8)
    axes[0].set_xlabel('Frame')
    axes[0].set_ylabel('Mean ΔF/F')
    axes[0].set_title(f'Population DFF Activity Over Time{frame_range_text}')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Panel 2: Population Spike Activity Over Time
    original_pop_activity_spike = np.mean(spike_display, axis=0)
    filtered_pop_activity_spike = np.mean(spike_filtered, axis=0)
    
    axes[1].plot(time_points, original_pop_activity_spike, 'b-', alpha=0.7, label='Original', linewidth=0.8)
    axes[1].plot(time_points, filtered_pop_activity_spike, 'r-', alpha=0.7, label=filter_type, linewidth=0.8)
    axes[1].set_xlabel('Frame')
    axes[1].set_ylabel('Mean Spike Activity')
    axes[1].set_title(f'Population Spike Activity Over Time{frame_range_text}')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    # Panel 3: Summary Statistics Table
    summary_stats = [
        ['Total ROIs', f'{len(dff_display)}', f'{len(dff_filtered)}'],
        ['Mean Peak ΔF/F', f'{np.mean(original_peaks):.2f}', f'{np.mean(filtered_peaks):.2f}'],
        ['Mean DFF-SNR', f'{np.mean(original_dff_snrs):.2f}' if len(original_dff_snrs) > 0 else 'N/A', 
                         f'{np.mean(filtered_dff_snrs):.2f}' if len(filtered_dff_snrs) > 0 else 'N/A'],
        ]
    
            
    
    # Create the table
    axes[2].axis('tight')
    axes[2].axis('off')
    
    table = axes[2].table(cellText=summary_stats, 
                          colLabels=['Metric', 'Original', filter_type],
                          cellLoc='center', 
                          loc='center',
                          colWidths=[0.4, 0.3, 0.3])
    
    # Format the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color code the header
    for i in range(3):
        table[(0, i)].set_facecolor('#E6E6FA')
        table[(0, i)].set_text_props(weight='bold')
    
    # Color code rows alternately
    for i in range(1, len(summary_stats) + 1):
        if i % 2 == 0:
            for j in range(3):
                table[(i, j)].set_facecolor('#F8F8FF')
    
    # # Highlight improvements/degradations
    # for i in range(1, len(summary_stats) + 1):
    #     original_val = summary_stats[i-1][1]
    #     filtered_val = summary_stats[i-1][2]
        
    #     if original_val != 'N/A' and filtered_val != 'N/A':
    #         orig_num = float(original_val)
    #         filt_num = float(filtered_val)
            
    #         metric_name = summary_stats[i-1][0]
    #         if 'Total ROIs' not in metric_name:
    #             if filt_num > orig_num:
    #                 table[(i, 2)].set_facecolor('#E6FFE6')  # Green
    #             elif filt_num < orig_num:
    #                 table[(i, 2)].set_facecolor('#FFE6E6')  # Red


    axes[2].set_title('Summary Statistics', fontsize=10)
    # axes[2].set_title('Summary Statistics\n(Green=Improvement, Red=Degradation)', fontsize=10)
    
    plt.suptitle(f'Activity Summary: {filter_type} vs Unfiltered - {rec_name}{roi_text}', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    filter_suffix = "_two_stage" if is_two_stage else ""
    summary_plot_path = os.path.join(save_path, f"{rec_name}_activity_summary_comparison{filter_suffix}{time_suffix}{roi_suffix}.jpg")
    plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved activity summary to {summary_plot_path}")
    plot_paths.append(summary_plot_path)
    
    # Print summary to console
    print(f"\nSummary Statistics for {rec_name}:")
    print("-" * 60)
    for stat in summary_stats:
        print(f"{stat[0]:<25}: {stat[1]:<12} → {stat[2]:<12}")
    print("-" * 60)
    
    return plot_paths