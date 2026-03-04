"""
Detect_spike.py
This script detects synchronous spike peaks across neural populations
and analyzes network coordination patterns.

JSY, 09/2025
"""

import numpy as np
from scipy.signal import find_peaks

def detect_synchronous_spike_peaks(spike_data, dff_data, active_mask, sampling_rate=15):
    """
    Detect synchronous spike peaks and return frame numbers.
    
    Parameters:
    -----------
    spike_data : array (n_cells, n_frames)
        Processed spike data (normalized)
    dff_data : array (n_cells, n_frames)  
        ΔF/F data for the same cells
    active_mask : array (n_frames,)
        Boolean mask indicating active/relevant frames
    sampling_rate : float
        Imaging frame rate in Hz
        
    Returns:
    --------
    cell_spike_data : dict
        Dictionary containing spike timing data for each cell
    synchrony_stats : dict
        Population-level synchrony statistics
    """
    
    n_cells, n_frames = spike_data.shape
    
    # Detect individual spike peaks for each cell
    cell_spike_data = {}
    
    print(f"Detecting spike peaks for {n_cells} cells...")
    
    for cell_idx in range(n_cells):
        spike_trace = spike_data[cell_idx, :]
        dff_trace = dff_data[cell_idx, :]
        
        # Use adaptive threshold for spike detection
        baseline_mean = np.mean(spike_trace)
        baseline_std = np.std(spike_trace)
        spike_threshold = baseline_mean + 1.5 * baseline_std
        
        if baseline_std > 1e-10:  # Only if cell has variance
            peaks, properties = find_peaks(
                spike_trace, 
                height=spike_threshold,
                distance=int(sampling_rate * 0.2),  # Minimum 200ms between peaks
                prominence=baseline_std * 0.5
            )
            
            # Get peak amplitudes
            peak_amplitudes = spike_trace[peaks]
            
            cell_spike_data[f'cell_{cell_idx}'] = {
                'all_peak_frames': peaks,
                'all_peak_amplitudes': peak_amplitudes,
                'all_peak_dff_values': dff_trace[peaks] if len(peaks) > 0 else np.array([])
            }
        else:
            cell_spike_data[f'cell_{cell_idx}'] = {
                'all_peak_frames': np.array([]),
                'all_peak_amplitudes': np.array([]),
                'all_peak_dff_values': np.array([])
            }
    
    # Detect synchronous events (population bursts)
    print("Detecting synchronous events...")
    
    # Method 1: Population activity-based synchrony
    population_activity = np.mean(spike_data, axis=0)
    pop_threshold = np.percentile(population_activity, 80)
    synchronous_frames = population_activity > pop_threshold
    
    # Apply active mask if provided
    if active_mask is not None:
        synchronous_frames = synchronous_frames & active_mask
    
    # Method 2: Multi-cell coincidence detection
    spike_binary = spike_data > (np.mean(spike_data, axis=1, keepdims=True) + 
                               1.5 * np.std(spike_data, axis=1, keepdims=True))
    cells_active_per_frame = np.sum(spike_binary, axis=0)
    min_coincident_cells = max(2, int(n_cells * 0.1))  # At least 10% of cells or 2 cells
    coincident_frames = cells_active_per_frame >= min_coincident_cells
    
    # Combine methods
    synchronous_frames = synchronous_frames | coincident_frames
    
    # Find synchronous spike peaks for each cell
    print("Identifying synchronous spike peaks...")
    
    for cell_key in cell_spike_data.keys():
        cell_idx = int(cell_key.split('_')[1])
        all_peaks = cell_spike_data[cell_key]['all_peak_frames']
        
        # Find which peaks occur during synchronous periods
        synchronous_peak_frames = []
        synchronous_peak_amplitudes = []
        
        for peak_frame in all_peaks:
            if peak_frame < len(synchronous_frames) and synchronous_frames[peak_frame]:
                synchronous_peak_frames.append(peak_frame)
                synchronous_peak_amplitudes.append(spike_data[cell_idx, peak_frame])
        
        # Add synchronous data to cell info
        cell_spike_data[cell_key].update({
            'synchronous_peak_frames': np.array(synchronous_peak_frames),
            'synchronous_peak_amplitudes': np.array(synchronous_peak_amplitudes),
            'n_total_spikes': len(all_peaks),
            'n_synchronous_spikes': len(synchronous_peak_frames),
            'synchrony_ratio': len(synchronous_peak_frames) / max(1, len(all_peaks))
        })
    
    # Population synchrony statistics
    synchrony_stats = {
        'total_frames': n_frames,
        'synchronous_frames': np.sum(synchronous_frames),
        'synchrony_percentage': np.sum(synchronous_frames) / n_frames * 100,
        'synchronous_frame_indices': np.where(synchronous_frames)[0],
        'population_threshold': pop_threshold,
        'min_coincident_cells': min_coincident_cells,
        'method': 'population_activity_and_coincidence_detection'
    }
    
    print(f"Synchrony detection complete:")
    print(f"  Synchronous frames: {synchrony_stats['synchronous_frames']}/{n_frames} ({synchrony_stats['synchrony_percentage']:.1f}%)")
    print(f"  Average synchronous spikes per cell: {np.mean([data['n_synchronous_spikes'] for data in cell_spike_data.values()]):.1f}")
    
    return cell_spike_data, synchrony_stats


def analyze_synchrony_patterns(cell_spike_data, synchrony_stats, sampling_rate=15):
    """
    Additional analysis of synchrony patterns.
    
    Parameters:
    -----------
    cell_spike_data : dict
        Output from detect_synchronous_spike_peaks
    synchrony_stats : dict
        Population synchrony statistics
    sampling_rate : float
        Imaging frame rate in Hz
        
    Returns:
    --------
    pattern_analysis : dict
        Advanced synchrony pattern analysis
    """
    
    n_cells = len(cell_spike_data)
    
    # Calculate synchrony metrics
    synchrony_ratios = [data['synchrony_ratio'] for data in cell_spike_data.values()]
    total_sync_spikes = sum([data['n_synchronous_spikes'] for data in cell_spike_data.values()])
    
    # Identify highly synchronous cells
    high_sync_threshold = np.percentile(synchrony_ratios, 75)
    highly_sync_cells = [cell for cell, data in cell_spike_data.items() 
                        if data['synchrony_ratio'] > high_sync_threshold]
    
    # Calculate temporal clustering of synchronous events
    sync_frames = synchrony_stats['synchronous_frame_indices']
    if len(sync_frames) > 1:
        inter_sync_intervals = np.diff(sync_frames) / sampling_rate  # Convert to seconds
        mean_interval = np.mean(inter_sync_intervals)
        interval_variability = np.std(inter_sync_intervals) / mean_interval if mean_interval > 0 else 0
    else:
        mean_interval = 0
        interval_variability = 0
    
    pattern_analysis = {
        'synchrony_metrics': {
            'mean_synchrony_ratio': np.mean(synchrony_ratios),
            'std_synchrony_ratio': np.std(synchrony_ratios),
            'total_synchronous_spikes': total_sync_spikes,
            'highly_synchronous_cells': len(highly_sync_cells),
            'high_sync_threshold': high_sync_threshold
        },
        'temporal_patterns': {
            'mean_inter_sync_interval_sec': mean_interval,
            'interval_variability_cv': interval_variability,
            'sync_event_frequency_hz': len(sync_frames) / (synchrony_stats['total_frames'] / sampling_rate)
        },
        'network_properties': {
            'sync_participation_rate': len([c for c in cell_spike_data.values() if c['n_synchronous_spikes'] > 0]) / n_cells,
            'avg_spikes_per_sync_event': total_sync_spikes / max(1, synchrony_stats['synchronous_frames'])
        }
    }
    
    return pattern_analysis


def plot_synchrony_analysis(spike_data, cell_spike_data, synchrony_stats, 
                          save_path=None, rec_name="Recording"):
    """
    Create visualization plots for synchrony analysis.
    
    Parameters:
    -----------
    spike_data : array (n_cells, n_frames)
        Processed spike data
    cell_spike_data : dict
        Individual cell spike data
    synchrony_stats : dict
        Population synchrony statistics  
    save_path : str, optional
        Path to save the plot
    rec_name : str
        Recording name for titles
    """
    
    import matplotlib.pyplot as plt
    
    n_cells, n_frames = spike_data.shape
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Spike raster with synchronous events highlighted
    ax1 = axes[0, 0]
    
    # Create raster plot
    for cell_idx in range(min(50, n_cells)):  # Show max 50 cells
        spike_times = cell_spike_data[f'cell_{cell_idx}']['all_peak_frames']
        sync_spike_times = cell_spike_data[f'cell_{cell_idx}']['synchronous_peak_frames']
        
        # Plot all spikes
        ax1.scatter(spike_times, [cell_idx] * len(spike_times), 
                   c='black', s=1, alpha=0.6)
        
        # Highlight synchronous spikes
        ax1.scatter(sync_spike_times, [cell_idx] * len(sync_spike_times), 
                   c='red', s=2, alpha=0.8)
    
    # Highlight synchronous frames as background
    sync_frames = synchrony_stats['synchronous_frame_indices']
    for frame in sync_frames[::10]:  # Every 10th frame to avoid clutter
        ax1.axvline(frame, color='red', alpha=0.1, linewidth=0.5)
    
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Cell Index')
    ax1.set_title(f'Spike Raster with Synchronous Events\n{len(sync_frames)} sync frames')
    
    # Plot 2: Population activity and synchrony threshold
    ax2 = axes[0, 1]
    
    population_activity = np.mean(spike_data, axis=0)
    ax2.plot(population_activity, 'k-', linewidth=0.5, alpha=0.7)
    ax2.axhline(synchrony_stats['population_threshold'], 
               color='red', linestyle='--', label=f"Threshold ({synchrony_stats['population_threshold']:.3f})")
    
    # Highlight synchronous periods
    sync_mask = np.zeros(n_frames, dtype=bool)
    sync_mask[sync_frames] = True
    ax2.fill_between(range(n_frames), 0, population_activity, 
                    where=sync_mask, alpha=0.3, color='red', label='Synchronous')
    
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Population Activity')
    ax2.set_title('Population Activity & Synchrony Detection')
    ax2.legend()
    
    # Plot 3: Synchrony ratio distribution
    ax3 = axes[1, 0]
    
    synchrony_ratios = [data['synchrony_ratio'] for data in cell_spike_data.values()]
    ax3.hist(synchrony_ratios, bins=20, alpha=0.7, color='blue', edgecolor='black')
    ax3.axvline(np.mean(synchrony_ratios), color='red', linestyle='--', 
               label=f'Mean: {np.mean(synchrony_ratios):.3f}')
    ax3.set_xlabel('Synchrony Ratio')
    ax3.set_ylabel('Number of Cells')
    ax3.set_title('Distribution of Cell Synchrony Ratios')
    ax3.legend()
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    total_sync_spikes = sum([data['n_synchronous_spikes'] for data in cell_spike_data.values()])
    mean_sync_ratio = np.mean(synchrony_ratios)
    
    summary_text = f"""
Synchrony Analysis Summary - {rec_name}

Population Statistics:
• Total frames: {n_frames}
• Synchronous frames: {synchrony_stats['synchronous_frames']} ({synchrony_stats['synchrony_percentage']:.1f}%)
• Population threshold: {synchrony_stats['population_threshold']:.3f}

Cell-Level Statistics:
• Total cells: {n_cells}
• Mean synchrony ratio: {mean_sync_ratio:.3f}
• Cells with sync spikes: {len([c for c in cell_spike_data.values() if c['n_synchronous_spikes'] > 0])}/{n_cells}

Network Properties:
• Avg spikes per sync event: {total_sync_spikes / max(1, synchrony_stats['synchronous_frames']):.1f}
• Sync event frequency: {len(sync_frames) / (n_frames / 15):.2f} events/sec
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle(f'Synchronous Spike Analysis - {rec_name}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Synchrony analysis plot saved: {save_path}")
    
    return fig


# Example usage and testing
if __name__ == "__main__":
    # Create synthetic test data
    print("Testing synchronous spike detection...")
    
    n_cells = 50
    n_frames = 1000
    sampling_rate = 15
    
    # Generate synthetic spike data with some synchronous events
    np.random.seed(42)
    spike_data = np.random.exponential(0.1, (n_cells, n_frames))
    
    # Add synchronized bursts
    burst_frames = [200, 400, 600, 800]
    for frame in burst_frames:
        # Make 70% of cells spike together
        burst_cells = np.random.choice(n_cells, int(0.7 * n_cells), replace=False)
        spike_data[burst_cells, frame:frame+5] += 1.0
    
    # Create corresponding DFF data
    dff_data = spike_data * 100 + np.random.normal(0, 10, (n_cells, n_frames))
    
    # Create active mask (all frames active for test)
    active_mask = np.ones(n_frames, dtype=bool)
    
    # Run detection
    cell_spike_data, synchrony_stats = detect_synchronous_spike_peaks(
        spike_data, dff_data, active_mask, sampling_rate
    )
    
    # Additional analysis
    pattern_analysis = analyze_synchrony_patterns(cell_spike_data, synchrony_stats, sampling_rate)
    
    print("\nTest completed successfully!")
    print(f"Detected {synchrony_stats['synchronous_frames']} synchronous frames")
    print(f"Mean synchrony ratio: {pattern_analysis['synchrony_metrics']['mean_synchrony_ratio']:.3f}")