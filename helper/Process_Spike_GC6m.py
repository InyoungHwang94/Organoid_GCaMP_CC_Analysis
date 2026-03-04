"""
Process_Spike_GC6m.py 
This script processes spike data specifically for GCaMP6m kinetics

JSY, 08/28/2025
"""

import numpy as np
import oasis
from tqdm import tqdm
from scipy import signal

def process_spike_data_gcamp6m(DFF_final, n_cells, numFrames, sampling_rate=10, 
                               debug_level='summary', debug_cells=3):
    """
    Process spike data optimized for GCaMP6m kinetics.
    
    Parameters:
        DFF_final: DFF data array
        n_cells: Number of cells
        numFrames: Number of frames
        sampling_rate: Imaging frame rate in Hz (default 10)
        debug_level: 'none', 'summary', 'detailed' (default 'summary')
        debug_cells: Number of cells to show detailed debug for (default 3)
    
    Returns:
        spikes: Raw processed spike data
        norm_spikes: Normalized spike data
    """
    
    # Initialize arrays
    spikes = np.zeros([n_cells, numFrames])
    norm_spikes = np.zeros([n_cells, numFrames])    
    
    print("Calculating spikes for GCaMP6m...")
    successful_cells = 0
    failed_cells = 0
    failed_cell_ids = []
    g_estimation_failures = 0
    
    # GCaMP6m specific parameters
    expected_tau = 0.245  # seconds (245ms decay time constant)
    expected_g = np.exp(-1/(sampling_rate * expected_tau))
    
    # Validation check
    if DFF_final.shape[0] != n_cells or DFF_final.shape[1] != numFrames:
        raise ValueError(f"Data shape mismatch: expected ({n_cells}, {numFrames}), got {DFF_final.shape}")

    for c in tqdm(range(n_cells)):
        try:
            # Get cell data and check if valid
            cell_dff = DFF_final[c, :].copy()
            
            # Skip if cell has no variation or is all zeros/NaN
            if (np.all(cell_dff == 0) or 
                np.all(np.isnan(cell_dff)) or 
                np.var(cell_dff) < 1e-10 or
                len(cell_dff) == 0):
                
                if debug_level == 'detailed':
                    print(f"Skipping cell {c}: no valid data (var={np.var(cell_dff):.2e})")
                failed_cells += 1
                failed_cell_ids.append(c)
                continue
            
            # Light pre-processing for GCaMP6m 
            if sampling_rate > 15:
                nyquist = sampling_rate / 2
                cutoff = min(10, nyquist * 0.8)
                b, a = signal.butter(1, cutoff/nyquist, btype='low')
                cell_dff_filtered = signal.filtfilt(b, a, cell_dff)
            else:
                cell_dff_filtered = cell_dff
                
            # Debug output for first few cells only
            if c < debug_cells and debug_level == 'detailed':
                print(f"\n=== DEBUG CELL {c} ===")
                print(f"Raw data stats:")
                print(f"  Shape: {cell_dff.shape}")
                print(f"  Min: {np.min(cell_dff):.6f}, Max: {np.max(cell_dff):.6f}")
                print(f"  Mean: {np.mean(cell_dff):.6f}, Std: {np.std(cell_dff):.6f}")
                print(f"  Non-zero values: {np.sum(cell_dff != 0)}/{len(cell_dff)}")
                
                print(f"Filtered data stats:")
                print(f"  Min: {np.min(cell_dff_filtered):.6f}, Max: {np.max(cell_dff_filtered):.6f}")
                print(f"  Mean: {np.mean(cell_dff_filtered):.6f}, Std: {np.std(cell_dff_filtered):.6f}")
                print(f"  Sampling rate: {sampling_rate}")
                
                print(f"Testing OASIS estimation...")
            
            # Estimate time constant
            try:
                g_auto = oasis.functions.estimate_time_constant(cell_dff_filtered, 1)
                
                # Extract scalar from array if needed
                if isinstance(g_auto, np.ndarray):
                    g_auto = g_auto.item()
                
                # Constraint check
                g_min = np.exp(-1/(sampling_rate * 0.40))  # 400ms max
                g_max = np.exp(-1/(sampling_rate * 0.05))  # 50ms min

                if g_min <= g_auto <= g_max:
                    g = g_auto
                    if c < debug_cells and debug_level == 'detailed':
                        tau_ms = -1/(sampling_rate * np.log(g))*1000
                        print(f"  SUCCESS: g_auto = {g_auto:.6f}")
                        print(f"  Estimated tau: {tau_ms:.1f} ms")
                        print(f"Cell {c}: Using auto g={g:.3f} (tau={tau_ms:.1f}ms)")
                else:
                    g = expected_g
                    if c < debug_cells and debug_level == 'detailed':
                        print(f"  SUCCESS: g_auto = {g_auto:.6f}")
                        tau_ms = -1/(sampling_rate * np.log(g_auto))*1000
                        print(f"  Estimated tau: {tau_ms:.1f} ms")
                        print(f"Cell {c}: Using expected g={g:.3f} (auto={g_auto:.3f} out of range)")
                        
            except Exception as e:
                g = expected_g
                g_estimation_failures += 1
                if c < debug_cells and debug_level == 'detailed':
                    print(f"  FAILED: {type(e).__name__}")
                    print(f"  Error message: {str(e)}")
                    print(f"Cell {c}: Using expected g={g:.3f} (estimation failed: {e})")
            
            # Deconvolution with GCaMP6m parameters
            try:
                _, cell_spikes = oasis.oasisAR1(
                    cell_dff_filtered, 
                    g,
                    lam=0.03,
                    optimize_g=False
                )
            except:
                _, cell_spikes = oasis.oasisAR1(cell_dff_filtered, g)
            
            # Check if spike detection produced valid results
            if np.all(cell_spikes == 0) or np.all(np.isnan(cell_spikes)):
                failed_cells += 1
                failed_cell_ids.append(c)
                continue
            
            # Light post-processing threshold
            spike_threshold = np.std(cell_spikes) * 0.03
            cell_spikes[cell_spikes < spike_threshold] = 0
            
            spikes[c, :] = cell_spikes
            successful_cells += 1
            
            # Normalize spikes
            spikes_min = np.min(spikes[c])
            spikes_max = np.max(spikes[c])
            if spikes_max > spikes_min:
                norm_spikes[c] = (spikes[c] - spikes_min) / (spikes_max - spikes_min)
                
        except Exception as e:
            if debug_level in ['detailed', 'summary']:
                print(f"Error processing cell {c}: {e}")
            failed_cells += 1
            failed_cell_ids.append(c)
            continue
    
    # Summary output
    failure_rate = failed_cells / n_cells * 100
    print(f"GCaMP6m spike processing complete: {successful_cells} successful, {failed_cells} failed")
    
    if debug_level in ['summary', 'detailed']:
        print(f"  Success rate: {successful_cells/n_cells*100:.1f}%")
        if g_estimation_failures > 0:
            print(f"  g-estimation failures: {g_estimation_failures}")
        
        if failure_rate > 50:
            print(f"  WARNING: High failure rate ({failure_rate:.1f}%) - check data quality")
        elif failure_rate > 20:
            print(f"  NOTICE: Moderate failure rate ({failure_rate:.1f}%) - may indicate noisy data")
    
    # Only show failed cell IDs if requested and manageable number
    if debug_level == 'detailed' and failed_cells > 0 and failed_cells <= 20:
        print(f"  Failed cells: {failed_cell_ids}")
    elif debug_level == 'detailed' and failed_cells > 20:
        print(f"  Failed cells: {failed_cell_ids[:10]}... (showing first 10 of {failed_cells})")
    
    return spikes, norm_spikes