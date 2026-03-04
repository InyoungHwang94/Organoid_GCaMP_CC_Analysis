"""
Functions helping calculating cross-correlation in neural data
JSY, 08/2025
"""

# Import functions
from .twop import (
    TwoP)

from .files import(
    read_xml,
    write_h5,
    read_h5,
    recursively_save_dict_contents_to_group,
    recursively_load_dict_contents_from_group
)

from .time import(
    time2float,
    time2str
)

from .Filtering_ROIs import(
    basic_signal_quality_filter,
    detect_events_single_roi,
    event_based_snr_filter,
    plot_two_stage_filtering_results,
    plot_two_stage_filtering_results_enhanced,
    plot_filtered_vs_unfiltered_rasters
)

from .Process_Spike_GC6m import(
    process_spike_data_gcamp6m
)

from .CC_NeuralData_Preprocessing import(
    temporal_binning,
    gaussian_smoothing,
    select_active_periods,
    conservative_preprocessing_pipeline,
    calculate_correlation_during_active_periods,
    plot_conservative_preprocessing_comparison
)

from .Detect_spike import(
    detect_synchronous_spike_peaks
)

# Specify what is available when you import the package
__all__ = [
    "TwoP",
    "read_xml","write_h5","read_h5","recursively_save_dict_contents_to_group","recursively_load_dict_contents_from_group",
    "time2float","time2str",
    "basic_signal_quality_filter", "detect_events_single_roi", "event_based_snr_filter", 
    "plot_two_stage_filtering_results", "plot_two_stage_filtering_results_enhanced", "plot_filtered_vs_unfiltered_rasters",
    "process_spike_data_gcamp6m",
    "temporal_binning", "gaussian_smoothing", "select_active_periods", "conservative_preprocessing_pipeline", 
    "calculate_correlation_during_active_periods", "plot_conservative_preprocessing_comparison",
    "detect_synchronous_spike_peaks"
]