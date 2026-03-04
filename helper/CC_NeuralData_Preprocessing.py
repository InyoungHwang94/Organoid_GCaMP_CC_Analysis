"""
CC_NeuralData_Preprocessing.py
This script preprocesses neural data to calculate the correlation coefficients (for synchronicity).
1) Gaussian smoothing
2) Temporal binning
3) Active period selection based on population activity
4) Correlation calculation during active periods only

JSY, 09/2025
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from scipy import ndimage

def temporal_binning(data, bin_size=2):
    """
    Bin temporal data to reduce sampling artifacts and noise.
    Standard approach in calcium imaging analysis.
    
    Parameters:
        data: (n_cells, n_frames) neural activity data
        bin_size: number of frames to bin together (default: 2)
    
    Returns:
        binned_data: (n_cells, n_bins) temporally binned data
        binning_stats: dictionary with binning statistics
    """
    
    n_cells, n_frames = data.shape
    n_bins = n_frames // bin_size
    
    print(f"Temporal binning: {n_frames} frames → {n_bins} bins (bin size: {bin_size})")
    
    binned_data = np.zeros((n_cells, n_bins))
    
    for i in range(n_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size
        binned_data[:, i] = np.mean(data[:, start_idx:end_idx], axis=1)
    
    # Calculate statistics
    original_noise = np.mean([np.std(data[i, :]) for i in range(n_cells)])
    binned_noise = np.mean([np.std(binned_data[i, :]) for i in range(n_cells)])
    noise_reduction = (original_noise - binned_noise) / original_noise * 100
    
    binning_stats = {
        'original_frames': n_frames,
        'binned_frames': n_bins,
        'bin_size': bin_size,
        'temporal_resolution_ms': bin_size * (1000 / 15),  # Assuming 15Hz
        'noise_reduction_percent': noise_reduction,
        'original_noise_std': original_noise,
        'binned_noise_std': binned_noise
    }
    
    print(f"  Temporal resolution: {binning_stats['temporal_resolution_ms']:.1f} ms per bin")
    print(f"  Noise reduction: {noise_reduction:.1f}%")
    
    return binned_data, binning_stats

def gaussian_smoothing(data, sigma=1.0, sampling_rate=15):
    """
    Apply Gaussian smoothing to calcium traces.
    Standard preprocessing to reduce noise while preserving temporal structure.
    
    Parameters:
        data: (n_cells, n_frames) neural activity data
        sigma: Gaussian kernel standard deviation in frames
        sampling_rate: imaging frame rate (Hz)
    
    Returns:
        smoothed_data: (n_cells, n_frames) smoothed data
        smoothing_stats: dictionary with smoothing statistics
    """
    
    n_cells, n_frames = data.shape
    smoothed_data = np.zeros_like(data)
    
    # print(f"Gaussian smoothing: σ = {sigma} frames ({sigma/sampling_rate*1000:.1f} ms)")
    
    for i in tqdm(range(n_cells), desc="Smoothing cells"):
        smoothed_data[i, :] = ndimage.gaussian_filter1d(data[i, :], sigma)
    
    # Calculate smoothing statistics
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

def select_active_periods(data, activity_threshold_percentile=75, min_duration=3):
    """
    Identify periods of high population activity for correlation analysis.
    Standard approach focusing on biologically relevant periods.
    
    Parameters:
        data: (n_cells, n_frames) neural activity data
        activity_threshold_percentile: percentile for population activity threshold
        min_duration: minimum duration of active periods (frames)
    
    Returns:
        active_mask: boolean array indicating active frames
        active_stats: dictionary with activity selection statistics
    """
    
    n_cells, n_frames = data.shape
    
    # Calculate population activity as mean across all cells
    population_activity = np.mean(data, axis=0)
    
    # Set threshold based on population activity percentile
    activity_threshold = np.percentile(population_activity, activity_threshold_percentile)
    
    print(f"Active period selection:")
    print(f"  Population activity threshold: {activity_threshold:.3f} ({activity_threshold_percentile}th percentile)")
    
    # Find frames above threshold
    above_threshold = population_activity > activity_threshold
    
    # Apply minimum duration filter
    active_mask = np.zeros_like(above_threshold, dtype=bool)
    current_period_start = None
    
    for i, is_active in enumerate(above_threshold):
        if is_active and current_period_start is None:
            current_period_start = i
        elif not is_active and current_period_start is not None:
            period_duration = i - current_period_start
            if period_duration >= min_duration:
                active_mask[current_period_start:i] = True
            current_period_start = None
    
    # Handle period that goes to end
    if current_period_start is not None:
        period_duration = len(above_threshold) - current_period_start
        if period_duration >= min_duration:
            active_mask[current_period_start:] = True
    
    # Calculate statistics
    n_active_frames = np.sum(active_mask)
    active_percentage = n_active_frames / n_frames * 100
    mean_activity_during_active = np.mean(population_activity[active_mask]) if n_active_frames > 0 else 0
    mean_activity_during_quiet = np.mean(population_activity[~active_mask]) if n_active_frames < n_frames else 0
    
    active_stats = {
        'total_frames': n_frames,
        'active_frames': n_active_frames,
        'active_percentage': active_percentage,
        'activity_threshold': activity_threshold,
        'activity_threshold_percentile': activity_threshold_percentile,
        'min_duration': min_duration,
        'mean_activity_active_periods': mean_activity_during_active,
        'mean_activity_quiet_periods': mean_activity_during_quiet,
        'activity_contrast': mean_activity_during_active / mean_activity_during_quiet if mean_activity_during_quiet > 0 else np.inf
    }
    
    print(f"  Active frames: {n_active_frames}/{n_frames} ({active_percentage:.1f}%)")
    print(f"  Activity contrast: {active_stats['activity_contrast']:.2f}x")
    print(f"  Mean activity during active periods: {mean_activity_during_active:.3f}")
    
    return active_mask, active_stats

def conservative_preprocessing_pipeline(data, 
                                      temporal_bin_size=2,
                                      gaussian_sigma=1.0,
                                      activity_threshold_percentile=75,
                                      min_active_duration=3,
                                      sampling_rate=15,
                                      apply_temporal_binning=True,
                                      apply_gaussian_smoothing=True):
    """
    Complete conservative preprocessing pipeline using standard methods.
    
    Parameters:
        data: (n_cells, n_frames) neural activity data
        temporal_bin_size: frames to bin together
        gaussian_sigma: standard deviation for Gaussian smoothing
        activity_threshold_percentile: percentile for active period selection
        min_active_duration: minimum duration for active periods
        sampling_rate: imaging frame rate
        apply_temporal_binning: whether to apply temporal binning
        apply_gaussian_smoothing: whether to apply Gaussian smoothing
    
    Returns:
        processed_data: preprocessed neural data
        active_mask: boolean mask for active periods
        preprocessing_stats: comprehensive statistics
    """
    
    
    processed_data = data.copy()
    preprocessing_stats = {'original_shape': data.shape}
    
    # Step 1: Gaussian smoothing (if enabled)
    if apply_gaussian_smoothing:
        processed_data, smoothing_stats = gaussian_smoothing(
            processed_data, sigma=gaussian_sigma, sampling_rate=sampling_rate
        )
        preprocessing_stats['smoothing'] = smoothing_stats
    else:
        print(f"  Gaussian smoothing skipped")
    
    # Step 2: Temporal binning (if enabled)
    if apply_temporal_binning:
        processed_data, binning_stats = temporal_binning(
            processed_data, bin_size=temporal_bin_size
        )
        preprocessing_stats['binning'] = binning_stats
    else:
        print(f"  Temporal binning skipped")
    
    # Step 3: Active period selection
    active_mask, active_stats = select_active_periods(
        processed_data, 
        activity_threshold_percentile=activity_threshold_percentile,
        min_duration=min_active_duration
    )
    preprocessing_stats['active_selection'] = active_stats
    
    preprocessing_stats['final_shape'] = processed_data.shape
    preprocessing_stats['methods_applied'] = {
        'gaussian_smoothing': apply_gaussian_smoothing,
        'temporal_binning': apply_temporal_binning,
        'active_period_selection': True
    }
    
    print(f"\nPreprocessing complete!")
    print(f"  Final data shape: {processed_data.shape}")
    print(f"  Active frames for correlation: {np.sum(active_mask)}/{len(active_mask)} ({np.sum(active_mask)/len(active_mask)*100:.1f}%)")
    
    return processed_data, active_mask, preprocessing_stats

def calculate_correlation_during_active_periods(data, active_mask):
    """
    Calculate standard Pearson correlation matrix during active periods only.
    
    Parameters:
        data: (n_cells, n_frames) preprocessed neural data
        active_mask: boolean array indicating active frames
    
    Returns:
        correlation_matrix: (n_cells, n_cells) correlation matrix
        correlation_stats: statistics about correlations
    """
    
    n_cells = data.shape[0]
    
    if np.sum(active_mask) < 10:
        print("Warning: Very few active frames for correlation calculation")
        return np.eye(n_cells), {'warning': 'insufficient_active_frames'}
    
    # Extract data during active periods
    active_data = data[:, active_mask]
    
    print(f"Calculating correlations using {active_data.shape[1]} active frames...")
    
    # Remove cells with no variance during active periods
    valid_cells = []
    for i in range(n_cells):
        if np.var(active_data[i, :]) > 1e-10:
            valid_cells.append(i)
    
    if len(valid_cells) < 2:
        print("Warning: Too few cells with variance during active periods")
        return np.eye(n_cells), {'warning': 'insufficient_valid_cells'}
    
    print(f"Using {len(valid_cells)}/{n_cells} cells with sufficient variance")
    
    # Calculate correlation matrix
    correlation_matrix = np.eye(n_cells)
    
    if len(valid_cells) >= 2:
        valid_active_data = active_data[valid_cells, :]
        valid_corr_matrix = np.corrcoef(valid_active_data)
        
        # Handle edge cases
        if valid_corr_matrix.ndim == 0:
            valid_corr_matrix = np.array([[1.0]])
        elif valid_corr_matrix.ndim == 1:
            valid_corr_matrix = valid_corr_matrix.reshape(1, 1)
        
        # Fill in correlation matrix for valid cells
        for idx_i, cell_i in enumerate(valid_cells):
            for idx_j, cell_j in enumerate(valid_cells):
                correlation_matrix[cell_i, cell_j] = valid_corr_matrix[idx_i, idx_j]
    
    # Clean up matrix
    correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
    np.fill_diagonal(correlation_matrix, 1.0)
    
    # Calculate statistics
    upper_tri = np.triu_indices_from(correlation_matrix, k=1)
    correlations = correlation_matrix[upper_tri]
    valid_correlations = correlations[~np.isnan(correlations)]
    
    correlation_stats = {
        'n_cells_total': n_cells,
        'n_cells_valid': len(valid_cells),
        'n_active_frames': np.sum(active_mask),
        'mean_correlation': np.mean(valid_correlations) if len(valid_correlations) > 0 else 0,
        'std_correlation': np.std(valid_correlations) if len(valid_correlations) > 0 else 0,
        'min_correlation': np.min(valid_correlations) if len(valid_correlations) > 0 else 0,
        'max_correlation': np.max(valid_correlations) if len(valid_correlations) > 0 else 0,
        'n_correlations': len(valid_correlations)
    }
    
    print(f"Correlation statistics:")
    print(f"  Mean correlation: {correlation_stats['mean_correlation']:.3f}")
    print(f"  Range: {correlation_stats['min_correlation']:.3f} to {correlation_stats['max_correlation']:.3f}")
    print(f"  Number of cell pairs: {correlation_stats['n_correlations']}")
    
    return correlation_matrix, correlation_stats

def plot_conservative_preprocessing_comparison(original_data, processed_data, active_mask,
                                             preprocessing_stats, rec_name, save_path):
    """
    Create comprehensive plots comparing original vs conservatively preprocessed data.
    """
    
    n_cells = original_data.shape[0]
    
    # Select subset of cells for visualization
    max_cells_display = 15
    if n_cells > max_cells_display:
        cell_indices = np.linspace(0, n_cells-1, max_cells_display, dtype=int)
    else:
        cell_indices = np.arange(n_cells)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Row 1: Data comparison
    # Original data
    im1 = axes[0,0].imshow(original_data[cell_indices, :], aspect='auto', cmap='hot', interpolation='nearest')
    axes[0,0].set_title(f'Original Data\n{len(cell_indices)} cells, {original_data.shape[1]} frames')
    axes[0,0].set_ylabel('Cell Index')
    plt.colorbar(im1, ax=axes[0,0])
    
    # Processed data
    im2 = axes[0,1].imshow(processed_data[cell_indices, :], aspect='auto', cmap='hot', interpolation='nearest')
    title_parts = []
    if preprocessing_stats['methods_applied']['gaussian_smoothing']:
        title_parts.append("Smoothed")
    if preprocessing_stats['methods_applied']['temporal_binning']:
        bin_size = preprocessing_stats['binning']['bin_size']
        title_parts.append(f"Binned")
    
    axes[0,1].set_title(f'Processed Data\n{", ".join(title_parts)}\n{processed_data.shape[1]} frames')
    plt.colorbar(im2, ax=axes[0,1])
    
    # Active periods overlay
    if processed_data.shape[1] == len(active_mask):
        # Create activity mask visualization
        activity_display = np.zeros((len(cell_indices), len(active_mask)))
        for i, cell_idx in enumerate(cell_indices):
            activity_display[i, :] = processed_data[cell_idx, :] * active_mask
        
        im3 = axes[0,2].imshow(activity_display, aspect='auto', cmap='hot', interpolation='nearest')
        active_pct = preprocessing_stats['active_selection']['active_percentage']
        axes[0,2].set_title(f'Active Periods Only\n{active_pct:.1f}% of frames')
        plt.colorbar(im3, ax=axes[0,2])
    else:
        axes[0,2].text(0.5, 0.5, 'Active period overlay\nnot available\n(different frame counts)', 
                      ha='center', va='center', transform=axes[0,2].transAxes)
        axes[0,2].set_title('Active Periods')
    
    # Row 2: Analysis plots
    # Population activity comparison
    original_pop = np.mean(original_data, axis=0)
    processed_pop = np.mean(processed_data, axis=0)
    
    if original_data.shape[1] == processed_data.shape[1]:
        # Same length - can compare directly
        time_orig = np.arange(len(original_pop))
        time_proc = np.arange(len(processed_pop))
        axes[1,0].plot(time_orig, original_pop, 'b-', alpha=0.7, label='Original', linewidth=0.8)
        axes[1,0].plot(time_proc, processed_pop, 'r-', alpha=0.7, label='Processed', linewidth=0.8)
    else:
        # Different lengths - plot separately
        time_orig = np.arange(len(original_pop))
        time_proc = np.arange(len(processed_pop)) * (len(original_pop) / len(processed_pop))
        axes[1,0].plot(time_orig, original_pop, 'b-', alpha=0.7, label='Original', linewidth=0.8)
        axes[1,0].plot(time_proc, processed_pop, 'r-', alpha=0.7, label='Processed', linewidth=0.8)
    
    axes[1,0].set_xlabel('Time (frames)')
    axes[1,0].set_ylabel('Mean Population Activity')
    axes[1,0].set_title('Population Activity Comparison')
    axes[1,0].legend(loc='upper right')  # FIXED: Removed extra parentheses
    axes[1,0].grid(True, alpha=0.3)
    
    # Active period threshold visualization
    if processed_data.shape[1] == len(active_mask):
        axes[1,1].plot(processed_pop, 'k-', alpha=0.7, linewidth=0.8)
        threshold = preprocessing_stats['active_selection']['activity_threshold']
        axes[1,1].axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.3f})')
        axes[1,1].fill_between(range(len(active_mask)), 0, np.max(processed_pop), 
                              where=active_mask, alpha=0.3, color='green', label='Active periods')
        axes[1,1].set_xlabel('Time (frames)')
        axes[1,1].set_ylabel('Population Activity')
        axes[1,1].set_title('Active Period Selection')
        axes[1,1].legend(loc='upper right')  # FIXED: Removed extra parentheses
        axes[1,1].grid(True, alpha=0.3)
    else:
        axes[1,1].text(0.5, 0.5, 'Active period\nvisualization\nnot available', 
                      ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Active Period Selection')
    
    # Statistics summary
    axes[1,2].axis('off')
    
    stats_text = f"Conservative Preprocessing Summary\n\n"
    
    if preprocessing_stats['methods_applied']['gaussian_smoothing']:
        smooth_stats = preprocessing_stats['smoothing']
        stats_text += f"Gaussian Smoothing:\n"
        stats_text += f"  σ = {smooth_stats['sigma_ms']:.1f} ms\n"
        stats_text += f"  Noise reduction: {smooth_stats['noise_reduction_percent']:.1f}%\n\n"
    
    if preprocessing_stats['methods_applied']['temporal_binning']:
        bin_stats = preprocessing_stats['binning']
        stats_text += f"Temporal Binning:\n"
        stats_text += f"  Bin size: {bin_stats['bin_size']} frames\n"
        stats_text += f"  Temporal resolution: {bin_stats['temporal_resolution_ms']:.1f} ms\n"
        stats_text += f"  Noise reduction: {bin_stats['noise_reduction_percent']:.1f}%\n\n"
    
    active_stats = preprocessing_stats['active_selection']
    stats_text += f"Active Period Selection:\n"
    stats_text += f"  Threshold: {active_stats['activity_threshold_percentile']}th percentile\n"
    stats_text += f"  Active frames: {active_stats['active_percentage']:.1f}%\n"
    stats_text += f"  Activity contrast: {active_stats['activity_contrast']:.2f}x\n"
    
    axes[1,2].text(0.05, 0.95, stats_text, transform=axes[1,2].transAxes, 
                  fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle(f'NeuralData Preprocessing Results - {rec_name}', fontsize=16)
    plt.tight_layout()
    
    # Save the plot
    preprocessing_plot_path = os.path.join(save_path, f"{rec_name}_neuraldata_preprocessing.jpg")
    plt.savefig(preprocessing_plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # ADDED: Close the figure to free memory

    print(f"Saved neuraldata preprocessing comparison to {preprocessing_plot_path}")

    return [preprocessing_plot_path]