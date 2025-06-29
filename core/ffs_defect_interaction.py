# core/ffs_defect_interaction_enhanced.py
"""
Enhanced FFS defect interaction with proper vector summation per API 579-1.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import warnings
import logging

logger = logging.getLogger(__name__)

class FFSDefectInteraction:
    """
    Enhanced FFS defect interaction implementing proper vector summation
    and geometric considerations per API 579-1/ASME FFS-1 Part 4.
    """
    
    def __init__(self, 
                 axial_interaction_distance_mm: float = 25.4,
                 circumferential_interaction_method: str = 'sqrt_dt',
                 custom_circ_distance_mm: float = None): # type: ignore
        self.axial_interaction_distance = axial_interaction_distance_mm
        self.circ_method = circumferential_interaction_method
        self.custom_circ_distance = custom_circ_distance_mm
    
    def combine_interacting_defects_enhanced(self, 
                                           defects_df: pd.DataFrame, 
                                           joints_df: pd.DataFrame, 
                                           pipe_diameter_mm: float) -> pd.DataFrame:
        """
        Enhanced defect combination using proper vector summation geometry.
        
        Implements API 579-1 Part 4 Section 4.3.4 for interacting defect assessment.
        """
        # Find interacting groups using existing logic
        groups = self.find_interacting_defects(defects_df, joints_df, pipe_diameter_mm) # type: ignore
        combined_defects = []
        
        for group in groups:
            if len(group) == 1:
                # Single defect - keep as is
                combined_defects.append(defects_df.iloc[group[0]].to_dict())
            else:
                # Multiple interacting defects - use enhanced combination
                group_defects = defects_df.iloc[group]
                combined = self._combine_defects_with_vector_summation(
                    group_defects, pipe_diameter_mm, joints_df
                )
                combined_defects.append(combined)
        
        return pd.DataFrame(combined_defects)
    
    def _combine_defects_with_vector_summation(self, 
                                             group_defects: pd.DataFrame,
                                             pipe_diameter_mm: float,
                                             joints_df: pd.DataFrame) -> Dict:
        """
        Combine multiple defects using proper vector summation per API 579.
        
        This implements the industry-standard geometric approach for
        calculating equivalent defect dimensions from interacting flaws.
        """
        
        # Convert defects to 3D coordinate system
        defect_vectors = self._convert_to_defect_vectors(group_defects, pipe_diameter_mm)
        
        # Calculate combined dimensions using vector geometry
        combined_length = self._calculate_vector_summed_length(defect_vectors)
        combined_width = self._calculate_vector_summed_width(defect_vectors, pipe_diameter_mm)
        
        # Use maximum depth (conservative per ASME B31G §5.2)
        max_depth_pct = group_defects['depth [%]'].max()
        
        # Calculate weighted center position
        total_area = (group_defects['length [mm]'] * group_defects['width [mm]']).sum()
        center_location = np.average(
            group_defects['log dist. [m]'], 
            weights=group_defects['length [mm]'] * group_defects['width [mm]']
        )
        
        # Calculate combined clock position (vector average)
        combined_clock = self._calculate_combined_clock_position(group_defects)
        
        # Create combined defect record
        combined = {
            # Combined dimensions using vector summation
            'depth [%]': max_depth_pct,
            'length [mm]': combined_length,
            'width [mm]': combined_width,
            
            # Position data
            'log dist. [m]': center_location,
            'clock': combined_clock,
            
            # Preserve joint information (use mode)
            'joint number': group_defects['joint number'].mode().iloc[0],
            
            # Metadata about combination
            'is_combined': True,
            'num_original_defects': len(group_defects),
            'original_indices': group_defects.index.tolist(),
            'combination_method': 'vector_summation_api579',
            'combination_note': f"Combined {len(group_defects)} defects using API 579-1 vector summation",
            
            # Preserve other important fields from dominant defect
            'component / anomaly identification': 'Combined_Defect',
            'surface location': group_defects['surface location'].mode().iloc[0] if 'surface location' in group_defects.columns else None
        }
        
        # Copy additional columns from the largest defect
        largest_defect_idx = group_defects['length [mm]'].idxmax()
        largest_defect = group_defects.loc[largest_defect_idx]
        
        for col in group_defects.columns:
            if col not in combined and col not in ['depth [%]', 'length [mm]', 'width [mm]', 'log dist. [m]', 'clock', 'joint number']:
                combined[col] = largest_defect[col]
        
        return combined
    
    def _convert_to_defect_vectors(self, group_defects: pd.DataFrame, pipe_diameter_mm: float) -> List[Dict]:
        """
        Convert defects to 3D vector representation for geometric calculations.
        
        Returns list of defect vectors with spatial components and orientations.
        """
        vectors = []
        
        for idx, defect in group_defects.iterrows():
            # Parse clock position
            clock_hours = self.parse_clock_to_decimal_hours(defect['clock'])
            
            # Convert to 3D coordinates
            # Axial position (along pipe)
            x_center = defect['log dist. [m]'] * 1000  # Convert to mm
            
            # Circumferential position (around pipe)
            angle_rad = (clock_hours % 12) * (np.pi / 6)  # Convert to radians
            radius = pipe_diameter_mm / 2
            y_circumferential = radius * np.sin(angle_rad)
            z_circumferential = radius * np.cos(angle_rad)
            
            # Defect orientation vectors
            # Axial component (along x-axis)
            axial_vector = np.array([defect['length [mm]'], 0, 0])
            
            # Circumferential component (tangent to pipe at clock position)
            circ_vector = np.array([0, 
                                   defect['width [mm]'] * np.cos(angle_rad),
                                   -defect['width [mm]'] * np.sin(angle_rad)])
            
            vector_data = {
                'defect_idx': idx,
                'center_position': np.array([x_center, y_circumferential, z_circumferential]),
                'axial_vector': axial_vector,
                'circumferential_vector': circ_vector,
                'length_mm': defect['length [mm]'],
                'width_mm': defect['width [mm]'],
                'depth_pct': defect['depth [%]'],
                'clock_angle_rad': angle_rad
            }
            
            vectors.append(vector_data)
        
        return vectors
    
    def _calculate_vector_summed_length(self, defect_vectors: List[Dict]) -> float:
        """
        Calculate combined axial length using proper vector summation.
        
        Implements API 579-1 geometric interaction methodology.
        """
        if len(defect_vectors) == 1:
            return defect_vectors[0]['length_mm']
        
        # Method 1: Axial span calculation (conservative)
        min_x = float('inf')
        max_x = float('-inf')
        
        for vector in defect_vectors:
            center_x = vector['center_position'][0]
            half_length = vector['length_mm'] / 2
            
            defect_min_x = center_x - half_length
            defect_max_x = center_x + half_length
            
            min_x = min(min_x, defect_min_x)
            max_x = max(max_x, defect_max_x)
        
        span_length = max_x - min_x
        
        # Method 2: Vector summation approach
        # Calculate interaction factor based on relative positioning
        total_vector_length = 0
        
        for i, vector_i in enumerate(defect_vectors):
            for j, vector_j in enumerate(defect_vectors):
                if i >= j:
                    continue
                
                # Calculate spatial separation
                distance = np.linalg.norm(vector_i['center_position'] - vector_j['center_position'])
                
                # Calculate interaction factor (decreases with separation)
                interaction_factor = self._calculate_interaction_factor(
                    distance, vector_i['length_mm'], vector_j['length_mm'] # type: ignore
                )
                
                # Vector summation with interaction factor
                L1, L2 = vector_i['length_mm'], vector_j['length_mm']
                
                # Simplified assumption: defects are roughly aligned axially
                # cos(θ) ≈ 1 for axially aligned defects
                # cos(θ) ≈ 0 for circumferentially separated defects
                cos_theta = interaction_factor
                
                vector_summed_length = np.sqrt(L1**2 + L2**2 + 2*L1*L2*cos_theta)
                total_vector_length = max(total_vector_length, vector_summed_length)
        
        # Use the more conservative of the two methods
        combined_length = max(span_length, total_vector_length)
        
        # Apply engineering judgment limits
        max_individual_length = max(v['length_mm'] for v in defect_vectors)
        combined_length = max(combined_length, max_individual_length)
        
        return combined_length
    
    def _calculate_vector_summed_width(self, defect_vectors: List[Dict], pipe_diameter_mm: float) -> float:
        """
        Calculate combined circumferential width using angular vector summation.
        
        Properly accounts for pipe curvature and circumferential extent.
        """
        if len(defect_vectors) == 1:
            return defect_vectors[0]['width_mm']
        
        # Convert all defects to angular extents
        angular_extents = []
        
        for vector in defect_vectors:
            # Convert width to angular extent
            width_mm = vector['width_mm']
            radius = pipe_diameter_mm / 2
            angular_width = width_mm / radius  # radians
            
            center_angle = vector['clock_angle_rad']
            start_angle = center_angle - angular_width / 2
            end_angle = center_angle + angular_width / 2
            
            # Normalize to [0, 2π] range
            start_angle = start_angle % (2 * np.pi)
            end_angle = end_angle % (2 * np.pi)
            
            angular_extents.append((start_angle, end_angle))
        
        # Merge overlapping angular extents using existing method
        merged_extents = self._merge_angular_extents(angular_extents)
        
        # Calculate total angular span
        total_angular_span = 0
        for start, end in merged_extents:
            if start <= end:
                total_angular_span += end - start
            else:
                # Wraps around 0
                total_angular_span += (2 * np.pi - start) + end
        
        # Convert back to arc length
        radius = pipe_diameter_mm / 2
        combined_width = total_angular_span * radius
        
        # Limit to pipe circumference
        max_width = np.pi * pipe_diameter_mm
        combined_width = min(combined_width, max_width)
        
        return combined_width
    
    def _calculate_interaction_factor(self, distance: float, L1: float, L2: float) -> float:
        """
        Calculate interaction factor based on defect separation and sizes.
        
        Used to determine the cos(θ) term in vector summation formula.
        Returns value between 0 (no interaction) and 1 (full interaction).
        """
        # Characteristic length scale for interaction
        char_length = (L1 + L2) / 2
        
        # Interaction decreases exponentially with relative distance
        # Factor of 2 means interaction drops to ~37% at 2x characteristic length
        interaction_factor = np.exp(-2 * distance / char_length)
        
        # Ensure bounds [0, 1]
        return np.clip(interaction_factor, 0.0, 1.0)
    
    def _calculate_combined_clock_position(self, group_defects: pd.DataFrame) -> str:
        """
        Calculate combined clock position using vector averaging.
        
        Properly handles wrap-around at 12 o'clock position.
        """
        # Convert clock positions to unit vectors
        vectors_x = []
        vectors_y = []
        weights = []
        
        for _, defect in group_defects.iterrows():
            clock_hours = self.parse_clock_to_decimal_hours(defect['clock'])
            angle_rad = (clock_hours % 12) * (np.pi / 6)
            
            # Unit vector on clock face
            x = np.cos(angle_rad)
            y = np.sin(angle_rad)
            
            # Weight by defect area
            weight = defect['length [mm]'] * defect['width [mm]']
            
            vectors_x.append(x * weight)
            vectors_y.append(y * weight)
            weights.append(weight)
        
        # Calculate weighted average vector
        total_weight = sum(weights)
        avg_x = sum(vectors_x) / total_weight
        avg_y = sum(vectors_y) / total_weight
        
        # Convert back to clock position
        avg_angle_rad = np.arctan2(avg_y, avg_x)
        avg_clock_hours = avg_angle_rad * 6 / np.pi
        
        # Normalize to 1-12 range
        if avg_clock_hours <= 0:
            avg_clock_hours += 12
        elif avg_clock_hours > 12:
            avg_clock_hours -= 12
        
        # Convert to clock string
        hours = int(avg_clock_hours)
        minutes = int((avg_clock_hours - hours) * 60)
        
        return f"{hours}:{minutes:02d}"
    
    def parse_clock_to_decimal_hours(self, clock_str):
        """Parse clock string to decimal hours (existing method)."""
        try:
            parts = clock_str.strip().split(':')
            hours = int(parts[0])
            minutes = int(parts[1]) if len(parts) > 1 else 0
            return hours + minutes / 60.0
        except:
            return 12.0
    
    def _merge_angular_extents(self, extents: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Merge overlapping angular extents (existing method from original implementation)."""
        if not extents:
            return []
        
        # Convert extents that wrap around into two separate extents
        unwrapped = []
        for start, end in extents:
            if start <= end:
                unwrapped.append((start, end))
            else:
                # Splits into two parts
                unwrapped.append((start, 2 * np.pi))
                unwrapped.append((0, end))
        
        # Sort by start angle
        unwrapped.sort(key=lambda x: x[0])
        
        # Merge overlapping extents
        merged = []
        current_start, current_end = unwrapped[0]
        
        for start, end in unwrapped[1:]:
            if start <= current_end:
                # Overlapping, extend current
                current_end = max(current_end, end)
            else:
                # Non-overlapping, save current and start new
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        merged.append((current_start, current_end))
        
        # Check if first and last extents should be merged (wrap-around case)
        if len(merged) > 1:
            first_start, first_end = merged[0]
            last_start, last_end = merged[-1]
            
            if last_end >= 2 * np.pi and first_start == 0:
                # They connect at the wrap-around point
                merged[0] = (last_start, first_end)
                merged.pop()
        
        return merged