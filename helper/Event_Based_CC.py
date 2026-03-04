"""
Event_Based_CC.py
This script calculates cross-correlation using event-based sampling
Updated to support binary/preprocessed neural data using Pearson correlation

JSY, 08/28/2025
"""

import numpy as np
from tqdm import tqdm

def threshold_based_events(data, method='percentile', threshold=80, min_duration=2):
    """
    Define active periods using threshold-based event detection.
    
    Parameters:
        data: 1D array of calcium signal
        method: 'percentile', 'std', 'absolute'
        threshold: threshold value (meaning depends on method)
        min_duration: minimum frames for an event
    
    Returns:
        active_frames: boolean array indicating active periods
        threshold_value: actual threshold used
    """
    
    if method == 'percentile':
        threshold_value = np.percentile(data, threshold)
    elif method == 'std':
        threshold_value = np.mean(data) + threshold * np.std(data)
    elif method == 'absolute':
        threshold_value = threshold
    else:
        raise ValueError("Method must be 'percentile', 'std', or 'absolute'")
    
    # Find frames above threshold
    above_threshold = data > threshold_value
    
    # Apply minimum duration filter
    if min_duration > 1:
        active_frames = np.zeros_like(above_threshold)
        current_event_start = None
        
        for i, is_active in enumerate(above_threshold):
            if is_active and current_event_start is None:
                current_event_start = i
            elif not is_active and current_event_start is not None:
                event_duration = i - current_event_start
                if event_duration >= min_duration:
                    active_frames[current_event_start:i] = True
                current_event_start = None
        
        # Handle event that goes to end
        if current_event_start is not None:
            event_duration = len(above_threshold) - current_event_start
            if event_duration >= min_duration:
                active_frames[current_event_start:] = True
    else:
        active_frames = above_threshold
    
    return active_frames, threshold_value

def find_population_active_frames_binary(data_matrix, min_active_cells_percent=5, min_duration=2):
    """
    Find frames where sufficient population activity occurs - optimized for binary data.
    
    Parameters:
        data_matrix: (n_cells, n_frames) binary data matrix
        min_active_cells_percent: minimum percentage of cells that must be active
        min_duration: minimum duration of population events
    
    Returns:
        population_active_frames: boolean array of frames with population activity
        stats: dictionary with statistics about active periods
    """
    n_cells, n_frames = data_matrix.shape
    
    # For binary data, we can directly count active cells per frame
    active_cells_per_frame = np.sum(data_matrix, axis=0)
    
    # Define population active frames
    min_active_cells = max(1, int(n_cells * min_active_cells_percent / 100))
    population_active = active_cells_per_frame >= min_active_cells
    
    # Apply minimum duration filter to population events
    if min_duration > 1:
        filtered_active = np.zeros_like(population_active)
        current_event_start = None
        
        for i, is_active in enumerate(population_active):
            if is_active and current_event_start is None:
                current_event_start = i
            elif not is_active and current_event_start is not None:
                event_duration = i - current_event_start
                if event_duration >= min_duration:
                    filtered_active[current_event_start:i] = True
                current_event_start = None
        
        # Handle event that goes to end
        if current_event_start is not None:
            event_duration = len(population_active) - current_event_start
            if event_duration >= min_duration:
                filtered_active[current_event_start:] = True
        
        population_active_frames = filtered_active
    else:
        population_active_frames = population_active
    
    # Calculate statistics
    stats = {
        'total_frames': n_frames,
        'active_frames': np.sum(population_active_frames),
        'active_percentage': np.sum(population_active_frames) / n_frames * 100,
        'mean_active_cells_per_frame': np.mean(active_cells_per_frame),
        'max_active_cells_per_frame': np.max(active_cells_per_frame),
        'min_active_cells_threshold': min_active_cells
    }
    
    return population_active_frames, stats

def find_population_active_frames(data_matrix, activity_threshold_percentile=80, 
                                min_active_cells_percent=5, min_duration=2):
    """
    Find frames where sufficient population activity occurs.
    
    Parameters:
        data_matrix: (n_cells, n_frames) array
        activity_threshold_percentile: percentile threshold for each cell
        min_active_cells_percent: minimum percentage of cells that must be active
        min_duration: minimum duration of population events
    
    Returns:
        population_active_frames: boolean array of frames with population activity
        stats: dictionary with statistics about active periods
    """
    n_cells, n_frames = data_matrix.shape
    
    # Find active frames for each cell
    cell_active_frames = np.zeros((n_cells, n_frames), dtype=bool)
    
    for i in range(n_cells):
        cell_active_frames[i], _ = threshold_based_events(
            data_matrix[i], 'percentile', activity_threshold_percentile, min_duration=1
        )
    
    # Count active cells per frame
    active_cells_per_frame = np.sum(cell_active_frames, axis=0)
    
    # Define population active frames
    min_active_cells = max(1, int(n_cells * min_active_cells_percent / 100))
    population_active = active_cells_per_frame >= min_active_cells
    
    # Apply minimum duration filter to population events
    if min_duration > 1:
        filtered_active = np.zeros_like(population_active)
        current_event_start = None
        
        for i, is_active in enumerate(population_active):
            if is_active and current_event_start is None:
                current_event_start = i
            elif not is_active and current_event_start is not None:
                event_duration = i - current_event_start
                if event_duration >= min_duration:
                    filtered_active[current_event_start:i] = True
                current_event_start = None
        
        # Handle event that goes to end
        if current_event_start is not None:
            event_duration = len(population_active) - current_event_start
            if event_duration >= min_duration:
                filtered_active[current_event_start:] = True
        
        population_active_frames = filtered_active
    else:
        population_active_frames = population_active
    
    # Calculate statistics
    stats = {
        'total_frames': n_frames,
        'active_frames': np.sum(population_active_frames),
        'active_percentage': np.sum(population_active_frames) / n_frames * 100,
        'mean_active_cells_per_frame': np.mean(active_cells_per_frame),
        'max_active_cells_per_frame': np.max(active_cells_per_frame),
        'min_active_cells_threshold': min_active_cells
    }
    
    return population_active_frames, stats

def calculate_binary_correlation(data1, data2):
    """
    Calculate correlation optimized for binary data.
    Uses Jaccard index for binary data, Pearson for continuous.
    
    Parameters:
        data1: 1D array of neural activity
        data2: 1D array of neural activity
    
    Returns:
        correlation: correlation coefficient
    """
    
    # Check if data is binary (only 0s and 1s)
    is_binary1 = np.all(np.isin(data1, [0, 1]))
    is_binary2 = np.all(np.isin(data2, [0, 1]))
    
    if is_binary1 and is_binary2:
        # Use Jaccard index for binary data
        intersection = np.sum((data1 == 1) & (data2 == 1))
        union = np.sum((data1 == 1) | (data2 == 1))
        
        if union == 0:
            return 0.0  # Both cells never active
        else:
            jaccard = intersection / union
            # Convert Jaccard to correlation-like scale [-1, 1]
            # This is a heuristic conversion
            return jaccard  # Keep as 0-1 scale for interpretability
    else:
        # Use standard Pearson correlation for continuous data
        if np.var(data1) < 1e-10 or np.var(data2) < 1e-10:
            return 0.0
        correlation = np.corrcoef(data1, data2)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0

def event_based_correlation_multiple_iterations(data, n_iterations=1000, 
                                               activity_threshold_percentile=80,
                                               min_active_cells_percent=5,
                                               min_duration=2,
                                               subsample_ratio=0.8,
                                               is_binary_data=False):
    """
    Calculate cross-correlation using event-based sampling across multiple iterations.
    Updated to handle binary/preprocessed data.
    
    Parameters:
        data: (n_cells, n_frames) neural activity data
        n_iterations: number of iterations for correlation calculation
        activity_threshold_percentile: percentile threshold for defining active cells
        min_active_cells_percent: minimum percentage of cells that must be active
        min_duration: minimum duration for population events
        subsample_ratio: fraction of active frames to use in each iteration
        is_binary_data: if True, use binary-optimized methods
    
    Returns:
        mean_correlation: mean cross-correlation matrix
        std_correlation: standard deviation of correlations
        sampling_stats: statistics about the sampling process
    """
    n_cells, n_frames = data.shape
    
    print(f"Starting EVENT-BASED correlation calculation: {n_cells} cells, {n_frames} frames")
    print(f"Will run {n_iterations} iterations with event-based sampling")
    print(f"Binary data mode: {is_binary_data}")
    
    # Find population active frames using appropriate method
    if is_binary_data:
        population_active, pop_stats = find_population_active_frames_binary(
            data, min_active_cells_percent, min_duration
        )
    else:
        population_active, pop_stats = find_population_active_frames(
            data, activity_threshold_percentile, min_active_cells_percent, min_duration
        )
    
    active_frame_indices = np.where(population_active)[0]
    
    print(f"Population activity statistics:")
    print(f"  Active frames: {pop_stats['active_frames']}/{pop_stats['total_frames']} "
          f"({pop_stats['active_percentage']:.1f}%)")
    print(f"  Mean active cells per frame: {pop_stats['mean_active_cells_per_frame']:.1f}")
    print(f"  Max active cells per frame: {pop_stats['max_active_cells_per_frame']}")
    
    if len(active_frame_indices) < 20:
        print("Warning: Very few active frames found. Results may be unreliable.")
        return np.eye(n_cells), np.zeros((n_cells, n_cells)), pop_stats
    
    # Remove cells with no activity during active periods
    valid_cells = []
    for i in range(n_cells):
        cell_data_active = data[i, active_frame_indices]
        if is_binary_data:
            # For binary data, check if cell is ever active
            if np.any(cell_data_active > 0):
                valid_cells.append(i)
        else:
            # For continuous data, check variance
            if np.var(cell_data_active) > 1e-10:
                valid_cells.append(i)
            else:
                print(f"Cell {i} excluded: no variance during active periods")
    
    if len(valid_cells) < 2:
        print(f"Warning: Only {len(valid_cells)} valid cells during active periods.")
        return np.eye(n_cells), np.zeros((n_cells, n_cells)), pop_stats
    
    print(f"Using {len(valid_cells)} out of {n_cells} cells for event-based correlation")
    print(f"Sampling from {len(active_frame_indices)} active frames")
    
    all_correlation_matrices = np.zeros((n_iterations, n_cells, n_cells))
    successful_iterations = 0
    
    import time
    start_time = time.time()
    
    for i in tqdm(range(n_iterations), desc="Event-based correlations"):
        try:
            # Subsample from active frames only
            n_subsample = max(int(len(active_frame_indices) * subsample_ratio), 10)
            if n_subsample >= len(active_frame_indices):
                selected_active_indices = active_frame_indices
            else:
                selected_active_indices = np.random.choice(
                    active_frame_indices, n_subsample, replace=False
                )
            
            # Debug info for first few iterations
            if i < 3:
                print(f"Iteration {i}: sampling {len(selected_active_indices)} "
                      f"out of {len(active_frame_indices)} active frames")
            
            # Extract data for selected active frames
            subsampled_data = data[:, selected_active_indices]
            
            # Initialize correlation matrix
            correlation_matrix = np.eye(n_cells)
            
            # Calculate correlations only for valid cells
            if len(valid_cells) >= 2:
                valid_data = subsampled_data[valid_cells, :]
                
                corr_start = time.time()
                
                if is_binary_data:
                    # Use optimized correlation for binary data
                    valid_corr = np.eye(len(valid_cells))
                    for idx_i in range(len(valid_cells)):
                        for idx_j in range(idx_i + 1, len(valid_cells)):
                            corr = calculate_binary_correlation(
                                valid_data[idx_i, :], valid_data[idx_j, :]
                            )
                            valid_corr[idx_i, idx_j] = corr
                            valid_corr[idx_j, idx_i] = corr  # Symmetric
                else:
                    # Use standard Pearson correlation
                    valid_corr = np.corrcoef(valid_data)
                
                corr_time = time.time() - corr_start
                
                if i < 3:
                    correlation_method = "Binary correlation" if is_binary_data else "Pearson correlation"
                    print(f"  {correlation_method} took {corr_time:.4f} seconds for {len(valid_cells)} cells")
                
                # Handle edge cases
                if valid_corr.ndim == 0:
                    valid_corr = np.array([[1.0]])
                elif valid_corr.ndim == 1:
                    valid_corr = valid_corr.reshape(1, 1)
                
                # Fill correlation matrix
                for idx_i, cell_i in enumerate(valid_cells):
                    for idx_j, cell_j in enumerate(valid_cells):
                        correlation_matrix[cell_i, cell_j] = valid_corr[idx_i, idx_j]
            
            # Clean up matrix
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
            np.fill_diagonal(correlation_matrix, 1.0)
            
            all_correlation_matrices[i] = correlation_matrix
            successful_iterations += 1
            
        except Exception as e:
            print(f"Error in iteration {i}: {e}")
            all_correlation_matrices[i] = np.eye(n_cells)
    
    total_time = time.time() - start_time
    print(f"Event-based correlation calculation complete:")
    print(f"  Total time: {total_time:.2f} seconds")
    print(f"  Successful iterations: {successful_iterations}/{n_iterations}")
    print(f"  Average time per iteration: {total_time/n_iterations:.4f} seconds")
    
    # Calculate statistics
    mean_correlation = np.mean(all_correlation_matrices, axis=0)
    std_correlation = np.std(all_correlation_matrices, axis=0)
    
    sampling_stats = {
        **pop_stats,
        'successful_iterations': successful_iterations,
        'valid_cells': len(valid_cells),
        'frames_per_iteration': len(active_frame_indices) * subsample_ratio,
        'is_binary_data': is_binary_data
    }
    
    return mean_correlation, std_correlation, sampling_stats

def print_correlation_statistics_event_based(mean_corr_dff, mean_corr_spikes, rec_name, 
                                            dff_stats, spike_stats):
    """
    Print detailed correlation statistics for event-based analysis.
    Updated to handle binary data statistics.
    """
    print(f"\n=== EVENT-BASED CORRELATION STATISTICS FOR {rec_name} ===")
    
    # DFF correlations (excluding diagonal)
    dff_corr_values = mean_corr_dff[np.triu_indices_from(mean_corr_dff, k=1)]
    print(f"DFF correlations - Min: {np.min(dff_corr_values):.3f}, "
          f"Max: {np.max(dff_corr_values):.3f}, Mean: {np.mean(dff_corr_values):.3f}")
    
    # Spike correlations (excluding diagonal)  
    spike_corr_values = mean_corr_spikes[np.triu_indices_from(mean_corr_spikes, k=1)]
    print(f"Spike correlations - Min: {np.min(spike_corr_values):.3f}, "
          f"Max: {np.max(spike_corr_values):.3f}, Mean: {np.mean(spike_corr_values):.3f}")
    
    corr_diff = abs(np.mean(dff_corr_values) - np.mean(spike_corr_values))
    print(f"Mean correlation difference: {corr_diff:.3f}")
    
    print(f"\nSampling Statistics:")
    print(f"DFF - Active frames used: {dff_stats['active_frames']}/{dff_stats['total_frames']} "
          f"({dff_stats['active_percentage']:.1f}%)")
    print(f"Spike - Active frames used: {spike_stats['active_frames']}/{spike_stats['total_frames']} "
          f"({spike_stats['active_percentage']:.1f}%)")
    
    # Add binary data specific statistics
    if dff_stats.get('is_binary_data', False):
        print(f"Note: Using binary correlation methods (Jaccard index)")
    
    return np.mean(dff_corr_values), np.mean(spike_corr_values)

# """
# Event_Based_CC.py
# This script calculates cross-correlation using event-based sampling

# JSY, 08/28/2025
# """

# import numpy as np
# from tqdm import tqdm

# def threshold_based_events(data, method='percentile', threshold=80, min_duration=2):
#     """
#     Define active periods using threshold-based event detection.
    
#     Parameters:
#         data: 1D array of calcium signal
#         method: 'percentile', 'std', 'absolute'
#         threshold: threshold value (meaning depends on method)
#         min_duration: minimum frames for an event
    
#     Returns:
#         active_frames: boolean array indicating active periods
#         threshold_value: actual threshold used
#     """
    
#     if method == 'percentile':
#         threshold_value = np.percentile(data, threshold)
#     elif method == 'std':
#         threshold_value = np.mean(data) + threshold * np.std(data)
#     elif method == 'absolute':
#         threshold_value = threshold
#     else:
#         raise ValueError("Method must be 'percentile', 'std', or 'absolute'")
    
#     # Find frames above threshold
#     above_threshold = data > threshold_value
    
#     # Apply minimum duration filter
#     if min_duration > 1:
#         active_frames = np.zeros_like(above_threshold)
#         current_event_start = None
        
#         for i, is_active in enumerate(above_threshold):
#             if is_active and current_event_start is None:
#                 current_event_start = i
#             elif not is_active and current_event_start is not None:
#                 event_duration = i - current_event_start
#                 if event_duration >= min_duration:
#                     active_frames[current_event_start:i] = True
#                 current_event_start = None
        
#         # Handle event that goes to end
#         if current_event_start is not None:
#             event_duration = len(above_threshold) - current_event_start
#             if event_duration >= min_duration:
#                 active_frames[current_event_start:] = True
#     else:
#         active_frames = above_threshold
    
#     return active_frames, threshold_value

# def find_population_active_frames(data_matrix, activity_threshold_percentile=80, 
#                                 min_active_cells_percent=5, min_duration=2):
#     """
#     Find frames where sufficient population activity occurs.
    
#     Parameters:
#         data_matrix: (n_cells, n_frames) array
#         activity_threshold_percentile: percentile threshold for each cell
#         min_active_cells_percent: minimum percentage of cells that must be active
#         min_duration: minimum duration of population events
    
#     Returns:
#         population_active_frames: boolean array of frames with population activity
#         stats: dictionary with statistics about active periods
#     """
#     n_cells, n_frames = data_matrix.shape
    
#     # Find active frames for each cell
#     cell_active_frames = np.zeros((n_cells, n_frames), dtype=bool)
    
#     for i in range(n_cells):
#         cell_active_frames[i], _ = threshold_based_events(
#             data_matrix[i], 'percentile', activity_threshold_percentile, min_duration=1
#         )
    
#     # Count active cells per frame
#     active_cells_per_frame = np.sum(cell_active_frames, axis=0)
    
#     # Define population active frames
#     min_active_cells = max(1, int(n_cells * min_active_cells_percent / 100))
#     population_active = active_cells_per_frame >= min_active_cells
    
#     # Apply minimum duration filter to population events
#     if min_duration > 1:
#         filtered_active = np.zeros_like(population_active)
#         current_event_start = None
        
#         for i, is_active in enumerate(population_active):
#             if is_active and current_event_start is None:
#                 current_event_start = i
#             elif not is_active and current_event_start is not None:
#                 event_duration = i - current_event_start
#                 if event_duration >= min_duration:
#                     filtered_active[current_event_start:i] = True
#                 current_event_start = None
        
#         # Handle event that goes to end
#         if current_event_start is not None:
#             event_duration = len(population_active) - current_event_start
#             if event_duration >= min_duration:
#                 filtered_active[current_event_start:] = True
        
#         population_active_frames = filtered_active
#     else:
#         population_active_frames = population_active
    
#     # Calculate statistics
#     stats = {
#         'total_frames': n_frames,
#         'active_frames': np.sum(population_active_frames),
#         'active_percentage': np.sum(population_active_frames) / n_frames * 100,
#         'mean_active_cells_per_frame': np.mean(active_cells_per_frame),
#         'max_active_cells_per_frame': np.max(active_cells_per_frame),
#         'min_active_cells_threshold': min_active_cells
#     }
    
#     return population_active_frames, stats

# def cross_correlation_with_lags(signal1, signal2, max_lag=5):
#     """Calculate correlation allowing for temporal offsets"""
#     correlations = []
#     for lag in range(-max_lag, max_lag+1):
#         if lag < 0:
#             corr = np.corrcoef(signal1[:lag], signal2[-lag:])[0,1]
#         elif lag > 0:
#             corr = np.corrcoef(signal1[lag:], signal2[:-lag])[0,1]
#         else:
#             corr = np.corrcoef(signal1, signal2)[0,1]
#         correlations.append(corr)
#     return max(correlations)

# def event_based_correlation_multiple_iterations(data, n_iterations=1000, 
#                                                activity_threshold_percentile=80,
#                                                min_active_cells_percent=5,
#                                                min_duration=2,
#                                                subsample_ratio=0.8):
#     """
#     Calculate cross-correlation using event-based sampling across multiple iterations.
    
#     Parameters:
#         data: (n_cells, n_frames) neural activity data
#         n_iterations: number of iterations for correlation calculation
#         activity_threshold_percentile: percentile threshold for defining active cells
#         min_active_cells_percent: minimum percentage of cells that must be active
#         min_duration: minimum duration for population events
#         subsample_ratio: fraction of active frames to use in each iteration
    
#     Returns:
#         mean_correlation: mean cross-correlation matrix
#         std_correlation: standard deviation of correlations
#         sampling_stats: statistics about the sampling process
#     """
#     n_cells, n_frames = data.shape
    
#     print(f"Starting EVENT-BASED correlation calculation: {n_cells} cells, {n_frames} frames")
#     print(f"Will run {n_iterations} iterations with event-based sampling")
    
#     # Find population active frames
#     population_active, pop_stats = find_population_active_frames(
#         data, activity_threshold_percentile, min_active_cells_percent, min_duration
#     )
    
#     active_frame_indices = np.where(population_active)[0]
    
#     print(f"Population activity statistics:")
#     print(f"  Active frames: {pop_stats['active_frames']}/{pop_stats['total_frames']} "
#           f"({pop_stats['active_percentage']:.1f}%)")
#     print(f"  Mean active cells per frame: {pop_stats['mean_active_cells_per_frame']:.1f}")
#     print(f"  Max active cells per frame: {pop_stats['max_active_cells_per_frame']}")
    
#     if len(active_frame_indices) < 20:
#         print("Warning: Very few active frames found. Results may be unreliable.")
#         return np.eye(n_cells), np.zeros((n_cells, n_cells)), pop_stats
    
#     # Remove cells with no activity during active periods
#     valid_cells = []
#     for i in range(n_cells):
#         cell_data_active = data[i, active_frame_indices]
#         if np.var(cell_data_active) > 1e-10:
#             valid_cells.append(i)
#         else:
#             print(f"Cell {i} excluded: no variance during active periods")
    
#     if len(valid_cells) < 2:
#         print(f"Warning: Only {len(valid_cells)} valid cells during active periods.")
#         return np.eye(n_cells), np.zeros((n_cells, n_cells)), pop_stats
    
#     print(f"Using {len(valid_cells)} out of {n_cells} cells for event-based correlation")
#     print(f"Sampling from {len(active_frame_indices)} active frames")
    
#     all_correlation_matrices = np.zeros((n_iterations, n_cells, n_cells))
#     successful_iterations = 0
    
#     import time
#     start_time = time.time()
    
#     for i in tqdm(range(n_iterations), desc="Event-based correlations"):
#         try:
#             # Subsample from active frames only
#             n_subsample = max(int(len(active_frame_indices) * subsample_ratio), 10)
#             if n_subsample >= len(active_frame_indices):
#                 selected_active_indices = active_frame_indices
#             else:
#                 selected_active_indices = np.random.choice(
#                     active_frame_indices, n_subsample, replace=False
#                 )
            
#             # Debug info for first few iterations
#             if i < 3:
#                 print(f"Iteration {i}: sampling {len(selected_active_indices)} "
#                       f"out of {len(active_frame_indices)} active frames")
            
#             # Extract data for selected active frames
#             subsampled_data = data[:, selected_active_indices]
            
#             # Initialize correlation matrix
#             correlation_matrix = np.eye(n_cells)
            
#             # Calculate correlations only for valid cells
#             if len(valid_cells) >= 2:
#                 valid_data = subsampled_data[valid_cells, :]
                
#                 corr_start = time.time()
#                 # valid_corr = cross_correlation_with_lags(valid_data)
#                 valid_corr = np.corrcoef(valid_data)
#                 corr_time = time.time() - corr_start
                
#                 if i < 3:
#                     print(f"  np.corrcoef took {corr_time:.4f} seconds for {len(valid_cells)} cells")
                
#                 # Handle edge cases
#                 if valid_corr.ndim == 0:
#                     valid_corr = np.array([[1.0]])
#                 elif valid_corr.ndim == 1:
#                     valid_corr = valid_corr.reshape(1, 1)
                
#                 # Fill correlation matrix
#                 for idx_i, cell_i in enumerate(valid_cells):
#                     for idx_j, cell_j in enumerate(valid_cells):
#                         correlation_matrix[cell_i, cell_j] = valid_corr[idx_i, idx_j]
            
#             # Clean up matrix
#             correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
#             np.fill_diagonal(correlation_matrix, 1.0)
            
#             all_correlation_matrices[i] = correlation_matrix
#             successful_iterations += 1
            
#         except Exception as e:
#             print(f"Error in iteration {i}: {e}")
#             all_correlation_matrices[i] = np.eye(n_cells)
    
#     total_time = time.time() - start_time
#     print(f"Event-based correlation calculation complete:")
#     print(f"  Total time: {total_time:.2f} seconds")
#     print(f"  Successful iterations: {successful_iterations}/{n_iterations}")
#     print(f"  Average time per iteration: {total_time/n_iterations:.4f} seconds")
    
#     # Calculate statistics
#     mean_correlation = np.mean(all_correlation_matrices, axis=0)
#     std_correlation = np.std(all_correlation_matrices, axis=0)
    
#     sampling_stats = {
#         **pop_stats,
#         'successful_iterations': successful_iterations,
#         'valid_cells': len(valid_cells),
#         'frames_per_iteration': len(active_frame_indices) * subsample_ratio
#     }
    
#     return mean_correlation, std_correlation, sampling_stats

# def print_correlation_statistics_event_based(mean_corr_dff, mean_corr_spikes, rec_name, 
    #                                         dff_stats, spike_stats):
    # """
    # Print detailed correlation statistics for event-based analysis.
    # """
    # print(f"\n=== EVENT-BASED CORRELATION STATISTICS FOR {rec_name} ===")
    
    # # DFF correlations (excluding diagonal)
    # dff_corr_values = mean_corr_dff[np.triu_indices_from(mean_corr_dff, k=1)]
    # print(f"DFF correlations - Min: {np.min(dff_corr_values):.3f}, "
    #       f"Max: {np.max(dff_corr_values):.3f}, Mean: {np.mean(dff_corr_values):.3f}")
    
    # # Spike correlations (excluding diagonal)  
    # spike_corr_values = mean_corr_spikes[np.triu_indices_from(mean_corr_spikes, k=1)]
    # print(f"Spike correlations - Min: {np.min(spike_corr_values):.3f}, "
    #       f"Max: {np.max(spike_corr_values):.3f}, Mean: {np.mean(spike_corr_values):.3f}")
    
    # corr_diff = abs(np.mean(dff_corr_values) - np.mean(spike_corr_values))
    # print(f"Mean correlation difference: {corr_diff:.3f}")
    
    # print(f"\nSampling Statistics:")
    # print(f"DFF - Active frames used: {dff_stats['active_frames']}/{dff_stats['total_frames']} "
    #       f"({dff_stats['active_percentage']:.1f}%)")
    # print(f"Spike - Active frames used: {spike_stats['active_frames']}/{spike_stats['total_frames']} "
    #       f"({spike_stats['active_percentage']:.1f}%)")
    
    # return np.mean(dff_corr_values), np.mean(spike_corr_values)