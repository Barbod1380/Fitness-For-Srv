"""
Failure prediction analysis for pipeline joints over time.
Predicts when joints will fail based on ERF < 1.0 or depth > 80% criteria.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import warnings

def predict_joint_failures_over_time(
    defects_df: pd.DataFrame,
    joints_df: pd.DataFrame,
    pipe_diameter_mm: float,
    smys_mpa: float,
    operating_pressure_mpa: float,
    assessment_method: str = 'b31g',
    window_years: int = 15,
    safety_factor: float = 1.39,
    growth_rates_dict: Dict = None, # type: ignore
    pipe_creation_year: int = None, # type: ignore
    current_year: int = None # type: ignore
) -> Dict:
    """
    Predict joint failures over time based on ERF and depth criteria.
    
    Parameters:
    - defects_df: DataFrame with current defects
    - joints_df: DataFrame with joint information
    - pipe_diameter_mm: Pipe diameter in mm
    - smys_mpa: SMYS in MPa
    - operating_pressure_mpa: Operating pressure for ERF calculation
    - assessment_method: 'b31g', 'modified_b31g', or 'simplified_eff_area'
    - window_years: Number of years to predict forward
    - safety_factor: Safety factor for failure pressure calculation
    - growth_rates_dict: Growth rates from multi-year analysis (if available)
    - pipe_creation_year: Year pipe was created (for single file scenario)
    - current_year: Current inspection year (for single file scenario)
    
    Returns:
    - Dictionary with failure predictions by year
    """
    
    # Validate inputs
    if operating_pressure_mpa <= 0:
        raise ValueError("Operating pressure must be positive")
    
    if assessment_method not in ['b31g', 'modified_b31g', 'simplified_eff_area']:
        raise ValueError(f"Unknown assessment method: {assessment_method}")
    
    # Determine if we're in single-file or multi-year mode
    if growth_rates_dict is None:
        if pipe_creation_year is None or current_year is None:
            raise ValueError("For single file analysis, both pipe_creation_year and current_year are required")
        
        # Estimate growth rates based on pipe age
        growth_rates_dict = estimate_single_file_growth_rates(
            defects_df, joints_df, pipe_creation_year, current_year
        )
    
    # Create wall thickness lookup
    wt_lookup = dict(zip(joints_df['joint number'], joints_df['wt nom [mm]']))
    
    # Initialize results structure
    results = {
        'years': list(range(1, window_years + 1)),
        'erf_failures_by_year': [],
        'depth_failures_by_year': [],
        'total_joints': len(joints_df),
        'joints_with_defects': len(defects_df['joint number'].unique()),
        'assessment_method': assessment_method,
        'operating_pressure_mpa': operating_pressure_mpa,
        'failure_details': []
    }
    
    # Get unique joints that have defects
    joints_with_defects = defects_df['joint number'].unique()
    
    # For each year in the prediction window
    for year in results['years']:
        # Project defects to this future year
        projected_defects = project_defects_to_year(defects_df, growth_rates_dict, year)
        
        # Calculate failures for this year
        erf_failed_joints, depth_failed_joints, year_details = calculate_joint_failures_for_year(
            projected_defects, joints_df, wt_lookup, pipe_diameter_mm, smys_mpa,
            operating_pressure_mpa, assessment_method, safety_factor, year
        )
        
        results['erf_failures_by_year'].append(len(erf_failed_joints))
        results['depth_failures_by_year'].append(len(depth_failed_joints))
        results['failure_details'].append(year_details)
    
    # Calculate cumulative failures
    results['cumulative_erf_failures'] = np.cumsum(results['erf_failures_by_year']).tolist()
    results['cumulative_depth_failures'] = np.cumsum(results['depth_failures_by_year']).tolist()
    
    # Add summary statistics
    results['summary'] = generate_failure_summary(results)
    
    return results


def estimate_single_file_growth_rates(
    defects_df: pd.DataFrame,
    joints_df: pd.DataFrame,
    pipe_creation_year: int,
    current_year: int
) -> Dict:
    """
    Estimate growth rates for defects based on pipe age and current defect sizes.
    
    Parameters:
    - defects_df: DataFrame with current defects
    - joints_df: DataFrame with joint information
    - pipe_creation_year: Year the pipe was created
    - current_year: Year of current inspection
    
    Returns:
    - Dictionary mapping defect indices to growth rates
    """
    
    pipe_age = current_year - pipe_creation_year
    if pipe_age <= 0:
        raise ValueError("Current year must be after pipe creation year")
    
    growth_rates_dict = {}
    wt_lookup = dict(zip(joints_df['joint number'], joints_df['wt nom [mm]']))
    
    for idx, defect in defects_df.iterrows():
        # Get wall thickness for depth calculation
        joint_num = defect['joint number']
        wall_thickness = wt_lookup.get(joint_num, 10.0)  # Default 10mm if missing
        
        # Estimate depth growth rate
        current_depth_pct = defect.get('depth [%]', 0)
        if current_depth_pct > 0:
            # Assume defect started at minimal detectable depth (1%)
            initial_depth_pct = 1.0
            depth_growth_pct_per_year = (current_depth_pct - initial_depth_pct) / pipe_age
        else:
            depth_growth_pct_per_year = 1.0  # Conservative default
        
        # Estimate length growth rate
        current_length_mm = defect.get('length [mm]', 0)
        if current_length_mm > 0:
            # Assume defect started at minimal detectable length (5mm)
            initial_length_mm = 5.0
            length_growth_mm_per_year = (current_length_mm - initial_length_mm) / pipe_age
        else:
            length_growth_mm_per_year = 2.0  # Conservative default
        
        # Estimate width growth rate
        current_width_mm = defect.get('width [mm]', 0)
        if current_width_mm > 0:
            # Assume defect started at minimal detectable width (5mm)
            initial_width_mm = 5.0
            width_growth_mm_per_year = (current_width_mm - initial_width_mm) / pipe_age
        else:
            width_growth_mm_per_year = 1.5  # Conservative default
        
        # Ensure positive growth rates (no shrinking defects)
        growth_rates_dict[idx] = {
            'depth_growth_pct_per_year': max(0.1, depth_growth_pct_per_year),
            'length_growth_mm_per_year': max(0.5, length_growth_mm_per_year),
            'width_growth_mm_per_year': max(0.3, width_growth_mm_per_year)
        }
    
    return growth_rates_dict


def project_defects_to_year(
    defects_df: pd.DataFrame,
    growth_rates_dict: Dict,
    target_year: int
) -> pd.DataFrame:
    """
    Project all defects to a future year based on their growth rates.
    
    Parameters:
    - defects_df: Current defects DataFrame
    - growth_rates_dict: Growth rates for each defect
    - target_year: Years from now to project to
    
    Returns:
    - DataFrame with projected defect dimensions
    """
    
    projected_df = defects_df.copy()
    
    for idx, defect in projected_df.iterrows():
        if idx in growth_rates_dict:
            growth_rates = growth_rates_dict[idx]
            
            # Project depth
            if 'depth_growth_pct_per_year' in growth_rates:
                new_depth = defect['depth [%]'] + (growth_rates['depth_growth_pct_per_year'] * target_year)
                projected_df.loc[idx, 'depth [%]'] = min(100.0, max(0.0, new_depth)) # type: ignore
            
            # Project length
            if 'length_growth_mm_per_year' in growth_rates:
                new_length = defect['length [mm]'] + (growth_rates['length_growth_mm_per_year'] * target_year)
                projected_df.loc[idx, 'length [mm]'] = max(defect['length [mm]'], new_length) # type: ignore
            
            # Project width
            if 'width_growth_mm_per_year' in growth_rates:
                new_width = defect['width [mm]'] + (growth_rates['width_growth_mm_per_year'] * target_year)
                projected_df.loc[idx, 'width [mm]'] = max(defect['width [mm]'], new_width) # type: ignore
    
    return projected_df


def calculate_joint_failures_for_year(
    projected_defects: pd.DataFrame,
    joints_df: pd.DataFrame,
    wt_lookup: Dict,
    pipe_diameter_mm: float,
    smys_mpa: float,
    operating_pressure_mpa: float,
    assessment_method: str,
    safety_factor: float,
    year: int
) -> Tuple[List, List, Dict]:
    """
    Calculate which joints fail in a specific year based on ERF and depth criteria.
    
    Returns:
    - Tuple of (erf_failed_joints, depth_failed_joints, year_details)
    """
    
    # Import calculation functions
    if assessment_method == 'b31g':
        from app.views.corrosion import calculate_b31g as calc_func
    elif assessment_method == 'modified_b31g':
        from app.views.corrosion import calculate_modified_b31g as calc_func
    elif assessment_method == 'simplified_eff_area':
        from app.views.corrosion import calculate_simplified_effective_area_method as calc_func
    
    erf_failed_joints = set()
    depth_failed_joints = set()
    joint_details = {}
    
    # Group defects by joint
    for joint_num in projected_defects['joint number'].unique():
        joint_defects = projected_defects[projected_defects['joint number'] == joint_num]
        wall_thickness = wt_lookup.get(joint_num, 10.0)
        
        joint_erf_failed = False
        joint_depth_failed = False
        defect_failures = []
        
        for idx, defect in joint_defects.iterrows():
            depth_pct = defect['depth [%]']
            length_mm = defect['length [mm]']
            width_mm = defect.get('width [mm]', defect['length [mm]'] * 0.5)  # Estimate if missing
            
            # Check depth failure (>80%)
            if depth_pct > 80.0:
                joint_depth_failed = True
                defect_failures.append({
                    'defect_idx': idx,
                    'failure_type': 'depth',
                    'depth_pct': depth_pct,
                    'location_m': defect['log dist. [m]']
                })
            
            # Check ERF failure (ERF < 1.0)
            try:
                if assessment_method == 'simplified_eff_area':
                    calc_result = calc_func(
                        depth_pct, length_mm, width_mm, pipe_diameter_mm, 
                        wall_thickness, smys_mpa, safety_factor # type: ignore
                    )
                else:
                    calc_result = calc_func(
                        depth_pct, length_mm, pipe_diameter_mm, 
                        wall_thickness, smys_mpa, safety_factor
                    )
                
                if calc_result['safe'] and calc_result['failure_pressure_mpa'] > 0:
                    erf = calc_result['failure_pressure_mpa'] / operating_pressure_mpa
                    
                    if erf < 1.0:
                        joint_erf_failed = True
                        defect_failures.append({
                            'defect_idx': idx,
                            'failure_type': 'erf',
                            'erf': erf,
                            'failure_pressure_mpa': calc_result['failure_pressure_mpa'],
                            'location_m': defect['log dist. [m]']
                        })
                        
            except Exception as e:
                warnings.warn(f"Failed to calculate failure pressure for defect {idx}: {e}")
        
        # Record joint failure status
        if joint_erf_failed:
            erf_failed_joints.add(joint_num)
        if joint_depth_failed:
            depth_failed_joints.add(joint_num)
        
        if defect_failures:
            joint_details[joint_num] = defect_failures
    
    year_details = {
        'year': year,
        'erf_failed_joints': list(erf_failed_joints),
        'depth_failed_joints': list(depth_failed_joints),
        'joint_failure_details': joint_details
    }
    
    return list(erf_failed_joints), list(depth_failed_joints), year_details


def generate_failure_summary(results: Dict) -> Dict:
    """Generate summary statistics for the failure prediction."""
    
    total_joints = results['total_joints']
    max_erf_failures = max(results['cumulative_erf_failures']) if results['cumulative_erf_failures'] else 0
    max_depth_failures = max(results['cumulative_depth_failures']) if results['cumulative_depth_failures'] else 0
    
    # Find first year with failures
    first_erf_failure_year = None
    first_depth_failure_year = None
    
    for i, (erf_count, depth_count) in enumerate(zip(results['erf_failures_by_year'], results['depth_failures_by_year'])):
        if erf_count > 0 and first_erf_failure_year is None:
            first_erf_failure_year = results['years'][i]
        if depth_count > 0 and first_depth_failure_year is None:
            first_depth_failure_year = results['years'][i]
    
    summary = {
        'total_joints_analyzed': total_joints,
        'joints_with_defects': results['joints_with_defects'],
        'max_erf_failures': max_erf_failures,
        'max_depth_failures': max_depth_failures,
        'pct_erf_failures': (max_erf_failures / total_joints * 100) if total_joints > 0 else 0,
        'pct_depth_failures': (max_depth_failures / total_joints * 100) if total_joints > 0 else 0,
        'first_erf_failure_year': first_erf_failure_year,
        'first_depth_failure_year': first_depth_failure_year,
        'prediction_window_years': results['years'][-1] if results['years'] else 0
    }
    
    return summary