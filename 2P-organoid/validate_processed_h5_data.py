"""
validate_h5_data.py
Script to test reading and validate accessibility of all variables in processed.h5 files

JSY, 09/2025
"""

import sys
sys.path.insert(0, r"C:\Users\jasmineyeo\Documents\GitHub\WSDup")

import h5py
import numpy as np
import os
import glob
from helper import files

def read_h5(filename, aslist=False):
    """ Read an .h5 file in as a dictionary.

    Modified from https://codereview.stackexchange.com/a/121308

    Parameters
    ----------
    filename : str
        Path to the .h5 file.
    aslist : bool
        If True, the dictionary will be read in as a list (on the first
        layer). Keys must have been convertable to integers when the file
        was written.
    """
    
    with h5py.File(filename, 'r') as h5file:
        out = files.recursively_load_dict_contents_from_group(h5file, '/')

        if aslist:
            outl = [None for l in range(len(out.keys()))]
            for key, item in out.items():
                outl[int(key)] = item
            out = outl

        return out

def validate_data_structure(data, file_path):
    """
    Comprehensive validation of the loaded data structure
    """
    print(f"\n{'='*80}")
    print(f"VALIDATING: {os.path.basename(file_path)}")
    print(f"{'='*80}")
    
    validation_results = {
        'file_readable': True,
        'sections_present': {},
        'data_integrity': {},
        'errors': []
    }
    
    # Expected main sections based on the processing script
    expected_sections = [
        'recording_info',
        'filtering_results', 
        'preprocessing_results',
        'correlation_analysis',
        'synchronous_spike_analysis',
        'processed_data'
    ]
    
    print(f"File loaded successfully. Main sections found:")
    for section in data.keys():
        print(f"  ✓ {section}")
    
    # Check each expected section
    for section in expected_sections:
        if section in data:
            validation_results['sections_present'][section] = True
            print(f"\n--- {section.upper()} SECTION ---")
            try:
                validate_section(data[section], section, validation_results)
            except Exception as e:
                validation_results['errors'].append(f"Error validating {section}: {e}")
                print(f"  ❌ Error: {e}")
        else:
            validation_results['sections_present'][section] = False
            validation_results['errors'].append(f"Missing section: {section}")
            print(f"  ❌ Missing section: {section}")
    
    return validation_results

def validate_section(section_data, section_name, validation_results):
    """
    Validate individual sections of the data
    """
    if section_name == 'recording_info':
        validate_recording_info(section_data, validation_results)
    elif section_name == 'filtering_results':
        validate_filtering_results(section_data, validation_results)
    elif section_name == 'preprocessing_results':
        validate_preprocessing_results(section_data, validation_results)
    elif section_name == 'correlation_analysis':
        validate_correlation_analysis(section_data, validation_results)
    elif section_name == 'synchronous_spike_analysis':
        validate_synchronous_spike_analysis(section_data, validation_results)
    elif section_name == 'processed_data':
        validate_processed_data(section_data, validation_results)

def validate_recording_info(data, validation_results):
    """Validate recording_info section"""
    expected_keys = ['recording_name', 'basepath', 'output_folder', 'frame_rate', 
                    'total_frames', 'original_cell_count', 'processing_date']
    
    for key in expected_keys:
        if key in data:
            value = data[key]
            print(f"  ✓ {key}: {value}")
            validation_results['data_integrity'][f'recording_info_{key}'] = True
        else:
            print(f"  ❌ Missing: {key}")
            validation_results['data_integrity'][f'recording_info_{key}'] = False

def validate_filtering_results(data, validation_results):
    """Validate filtering_results section"""
    expected_keys = ['filtering_applied', 'stage1_mask', 'stage2_mask', 
                    'stage1_survivors', 'stage2_survivors', 'final_cell_count']
    
    for key in expected_keys:
        if key in data:
            value = data[key]
            if key.endswith('_mask'):
                print(f"  ✓ {key}: array shape {np.array(value).shape}, type {type(value)}")
            else:
                print(f"  ✓ {key}: {value}")
            validation_results['data_integrity'][f'filtering_{key}'] = True
        else:
            print(f"  ❌ Missing: {key}")
            validation_results['data_integrity'][f'filtering_{key}'] = False
    
    # Validate mask consistency
    if 'stage1_mask' in data and 'stage2_mask' in data:
        stage1_mask = np.array(data['stage1_mask'])
        stage2_mask = np.array(data['stage2_mask'])
        stage1_count = np.sum(stage1_mask)
        stage2_count = np.sum(stage2_mask)
        
        print(f"  ✓ Stage 1 survivors: {stage1_count}")
        print(f"  ✓ Stage 2 survivors: {stage2_count}")
        print(f"  ✓ Overall survival rate: {stage2_count/len(stage1_mask)*100:.1f}%")

def validate_preprocessing_results(data, validation_results):
    """Validate preprocessing_results section"""
    expected_keys = ['preprocessing_applied', 'neural_smoothing_params']
    
    for key in expected_keys:
        if key in data:
            print(f"  ✓ {key}: present")
            validation_results['data_integrity'][f'preprocessing_{key}'] = True
        else:
            print(f"  ❌ Missing: {key}")
            validation_results['data_integrity'][f'preprocessing_{key}'] = False

def validate_correlation_analysis(data, validation_results):
    """Validate correlation_analysis section"""
    expected_keys = ['dff_correlation_matrix', 'spikes_correlation_matrix', 
                    'dff_correlation_stats', 'spikes_correlation_stats']
    
    for key in expected_keys:
        if key in data:
            if 'matrix' in key:
                matrix = np.array(data[key])
                print(f"  ✓ {key}: shape {matrix.shape}, mean = {np.mean(matrix):.3f}")
                
                # Validate matrix properties
                if matrix.shape[0] == matrix.shape[1]:
                    print(f"    ✓ Square matrix: {matrix.shape}")
                else:
                    print(f"    ❌ Not square: {matrix.shape}")
                
                # Check for reasonable correlation values
                if np.all(np.abs(matrix) <= 1.1):  # Allow small numerical errors
                    print(f"    ✓ Values in valid range: [{np.min(matrix):.3f}, {np.max(matrix):.3f}]")
                else:
                    print(f"    ❌ Values out of range: [{np.min(matrix):.3f}, {np.max(matrix):.3f}]")
                
            elif 'stats' in key:
                stats = data[key]
                print(f"  ✓ {key}: {type(stats)}")
                if isinstance(stats, dict):
                    for stat_key, stat_val in stats.items():
                        print(f"    - {stat_key}: {stat_val}")
            
            validation_results['data_integrity'][f'correlation_{key}'] = True
        else:
            print(f"  ❌ Missing: {key}")
            validation_results['data_integrity'][f'correlation_{key}'] = False

def validate_synchronous_spike_analysis(data, validation_results):
    """Validate synchronous_spike_analysis section"""
    expected_keys = ['synchronous_spike_data', 'synchrony_stats']
    
    for key in expected_keys:
        if key in data:
            if key == 'synchronous_spike_data':
                spike_data = data[key]
                print(f"  ✓ {key}: {type(spike_data)}")
                if isinstance(spike_data, dict):
                    print(f"    - Number of cells: {len(spike_data)}")
                    if len(spike_data) > 0:
                        first_cell = list(spike_data.keys())[0]
                        cell_data = spike_data[first_cell]
                        print(f"    - Example cell ({first_cell}) keys: {list(cell_data.keys())}")
            else:
                stats = data[key]
                print(f"  ✓ {key}: {type(stats)}")
                if isinstance(stats, dict):
                    for stat_key, stat_val in stats.items():
                        print(f"    - {stat_key}: {stat_val}")
            
            validation_results['data_integrity'][f'synchrony_{key}'] = True
        else:
            print(f"  ❌ Missing: {key}")
            validation_results['data_integrity'][f'synchrony_{key}'] = False

def validate_processed_data(data, validation_results):
    """Validate processed_data section"""
    expected_keys = ['dff_processed', 'spikes_processed', 'data_shape']
    
    for key in expected_keys:
        if key in data:
            if key in ['dff_processed', 'spikes_processed']:
                array_data = np.array(data[key])
                print(f"  ✓ {key}: shape {array_data.shape}, dtype {array_data.dtype}")
                print(f"    - Range: [{np.min(array_data):.3f}, {np.max(array_data):.3f}]")
                print(f"    - Mean: {np.mean(array_data):.3f}, Std: {np.std(array_data):.3f}")
            else:
                print(f"  ✓ {key}: {data[key]}")
            
            validation_results['data_integrity'][f'processed_{key}'] = True
        else:
            print(f"  ❌ Missing: {key}")
            validation_results['data_integrity'][f'processed_{key}'] = False

def print_summary(validation_results, file_path):
    """Print validation summary"""
    print(f"\n{'='*80}")
    print(f"VALIDATION SUMMARY FOR {os.path.basename(file_path)}")
    print(f"{'='*80}")
    
    total_checks = len(validation_results['data_integrity'])
    passed_checks = sum(validation_results['data_integrity'].values())
    
    print(f"Overall Status: {passed_checks}/{total_checks} checks passed")
    
    if validation_results['errors']:
        print(f"\nErrors found ({len(validation_results['errors'])}):")
        for error in validation_results['errors']:
            print(f"  ❌ {error}")
    else:
        print("\n✅ No errors found!")
    
    print(f"\nSection Completeness:")
    for section, present in validation_results['sections_present'].items():
        status = "✅" if present else "❌"
        print(f"  {status} {section}")

def test_data_accessibility(data):
    """Test that we can actually access and manipulate the data"""
    print(f"\n{'='*50}")
    print("TESTING DATA ACCESSIBILITY")
    print(f"{'='*50}")
    
    try:
        # Test correlation matrices
        if 'correlation_analysis' in data:
            dff_matrix = np.array(data['correlation_analysis']['dff_correlation_matrix'])
            spike_matrix = np.array(data['correlation_analysis']['spikes_correlation_matrix'])
            
            print(f"✅ Can convert correlation matrices to numpy arrays")
            print(f"  DFF matrix: {dff_matrix.shape}")
            print(f"  Spike matrix: {spike_matrix.shape}")
            
            # Test basic operations
            dff_mean = np.mean(dff_matrix)
            spike_mean = np.mean(spike_matrix)
            print(f"✅ Can calculate statistics: DFF mean = {dff_mean:.3f}, Spike mean = {spike_mean:.3f}")
        
        # Test processed data
        if 'processed_data' in data:
            dff_processed = np.array(data['processed_data']['dff_processed'])
            spikes_processed = np.array(data['processed_data']['spikes_processed'])
            
            print(f"✅ Can access processed neural data")
            print(f"  DFF processed: {dff_processed.shape}")
            print(f"  Spikes processed: {spikes_processed.shape}")
            
            # Test indexing
            if dff_processed.size > 0:
                first_cell_dff = dff_processed[0, :]
                print(f"✅ Can index individual cells: first cell has {len(first_cell_dff)} time points")
        
        # Test filtering masks
        if 'filtering_results' in data:
            if 'stage2_mask' in data['filtering_results']:
                final_mask = np.array(data['filtering_results']['stage2_mask'])
                n_survivors = np.sum(final_mask)
                print(f"✅ Can access filtering masks: {n_survivors} cells survived filtering")
        
        print(f"✅ All accessibility tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Accessibility test failed: {e}")
        return False

def main():
    """Main validation function"""
    # Define base path where processed h5 files are located
    base_path = r'\\kosik-nas1\Kosik_Lab\Inyoung Hwang\WS_Dup\suite2p\B1\B1_D150_Dup_org2_004\20250907_B1_D150_Dup_org2_004_correlation_data'
    
    # Find all processed.h5 files
    search_pattern = os.path.join(base_path, "**", "*_processed.h5")
    h5_files = glob.glob(search_pattern, recursive=True)
    
    if not h5_files:
        print(f"No *_processed.h5 files found in {base_path}")
        print("Please check the path and file naming pattern.")
        return
    
    print(f"Found {len(h5_files)} processed.h5 files:")
    for file in h5_files:
        print(f"  - {file}")
    
    # Test each file
    all_results = {}
    
    for h5_file in h5_files:
        try:
            print(f"\n{'#'*100}")
            print(f"TESTING FILE: {h5_file}")
            print(f"{'#'*100}")
            
            # Load the data
            print("Loading data...")
            data = read_h5(h5_file)
            
            # Validate structure
            validation_results = validate_data_structure(data, h5_file)
            
            # Test accessibility
            accessibility_ok = test_data_accessibility(data)
            validation_results['accessibility_passed'] = accessibility_ok
            
            # Print summary
            print_summary(validation_results, h5_file)
            
            # Store results
            all_results[os.path.basename(h5_file)] = validation_results
            
        except Exception as e:
            print(f"❌ FAILED to process {h5_file}: {e}")
            all_results[os.path.basename(h5_file)] = {'error': str(e)}
    
    # Final summary
    print(f"\n{'#'*100}")
    print("FINAL SUMMARY")
    print(f"{'#'*100}")
    
    for filename, results in all_results.items():
        if 'error' in results:
            print(f"❌ {filename}: FAILED - {results['error']}")
        else:
            total_checks = len(results['data_integrity'])
            passed_checks = sum(results['data_integrity'].values())
            accessibility = "✅" if results.get('accessibility_passed', False) else "❌"
            quality = "✅" if results.get('quality_passed', True) else "⚠️ "
            print(f"{accessibility}{quality} {filename}: {passed_checks}/{total_checks} checks passed")
            if not results.get('quality_passed', True):
                print(f"    └── Quality issues detected - see detailed report above")

if __name__ == "__main__":
    main()