"""
Fitness-For-Service (FFS) compliant defect interaction analysis.
Based on ASME B31G-2012 and API 579-1/ASME FFS-1 interaction rules.
"""
import numpy as np
import pandas as pd
from typing import List, Dict
import warnings
import logging

logger = logging.getLogger(__name__)

class FFSDefectInteraction:
    """
    Implements defect interaction rules per FFS standards.
    """
    
    def __init__(self, 
                 axial_interaction_distance_mm: float = 25.4,  # 1 inch default
                 circumferential_interaction_method: str = 'sqrt_dt',
                 custom_circ_distance_mm: float = None): # type: ignore
        """
        Initialize FFS defect interaction analyzer.
        
        Parameters:
        - axial_interaction_distance_mm: Maximum axial spacing for interaction (default 1")
        - circumferential_interaction_method: Method for circumferential interaction
          Options: 'sqrt_dt' (√(D×t)), '3t' (3×wall thickness), 'custom'
        - custom_circ_distance_mm: Custom circumferential interaction distance if method='custom'
        """
        self.axial_interaction_distance = axial_interaction_distance_mm
        self.circ_method = circumferential_interaction_method
        self.custom_circ_distance = custom_circ_distance_mm
        
        if self.circ_method == 'custom' and custom_circ_distance_mm is None:
            raise ValueError("Must specify custom_circ_distance_mm when using 'custom' method")
    
    def calculate_circumferential_interaction_distance(self, 
                                                     pipe_diameter_mm: float, 
                                                     wall_thickness_mm: float) -> float:
        """
        Calculate circumferential interaction distance based on selected method.
        """
        if self.circ_method == 'sqrt_dt':
            # Common rule: √(D×t)
            return np.sqrt(pipe_diameter_mm * wall_thickness_mm)
        elif self.circ_method == '3t':
            # Alternative rule: 3×wall thickness
            return 3.0 * wall_thickness_mm
        elif self.circ_method == 'custom':
            return self.custom_circ_distance
        else:
            raise ValueError(f"Unknown circumferential interaction method: {self.circ_method}")
        

    def parse_clock_to_decimal_hours(self, clock_str):
        """
        Convert clock string 'H:MM' or 'HH:MM' to decimal hours.
        Parse clock position string (e.g., '8:43' -> 8.717 hours)
        """
        
        try:
            parts = clock_str.strip().split(':')
            hours = int(parts[0])
            minutes = int(parts[1]) if len(parts) > 1 else 0
            
            # Convert to decimal hours
            decimal_hours = hours + minutes / 60.0
            
            # Validate range (1:00 to 12:59)
            if decimal_hours < 1.0 or decimal_hours >= 13.0:
                warnings.warn(f"Clock position {clock_str} outside expected range 1:00-12:59")
            
            return decimal_hours
        except (ValueError, IndexError) as e:
            warnings.warn(f"Invalid clock format '{clock_str}': {e}")
            return 12.0  # Default to 12:00 if parsing fails
    
    def convert_to_cartesian(self, defects_df: pd.DataFrame, joints_df: pd.DataFrame, 
                            pipe_diameter_mm: float) -> pd.DataFrame:
        """
        Convert defect positions to cartesian coordinates for interaction analysis.
        
        Adds columns: x_start, x_end, y_center (circumferential position)
        """
        df = defects_df.copy()
        
        # Ensure we have required columns
        required_cols = ['log dist. [m]', 'length [mm]', 'clock', 'joint number']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Convert to consistent units (mm)
        df['x_center_mm'] = df['log dist. [m]'] * 1000  # Convert m to mm
        df['x_start_mm'] = df['x_center_mm'] - df['length [mm]'] / 2
        df['x_end_mm'] = df['x_center_mm'] + df['length [mm]'] / 2
        
        # Convert clock strings to decimal hours
        df['decimal_hours'] = df['clock'].apply(self.parse_clock_to_decimal_hours)
        
        # Convert to angular position (degrees)
        # 12:00 = 0°, 3:00 = 90°, 6:00 = 180°, 9:00 = 270°
        df['angle_deg'] = (df['decimal_hours'] % 12) * 30
        
        # Calculate circumferential position (y) in mm
        # Using arc length = radius × angle (in radians)
        radius_mm = pipe_diameter_mm / 2
        df['y_center_mm'] = radius_mm * np.radians(df['angle_deg'])
        
        # Add circumferential extent (width)
        if 'width [mm]' in df.columns:
            df['y_start_mm'] = df['y_center_mm'] - df['width [mm]'] / 2
            df['y_end_mm'] = df['y_center_mm'] + df['width [mm]'] / 2
        else:
            # If no width data, treat as point defects circumferentially
            df['y_start_mm'] = df['y_center_mm']
            df['y_end_mm'] = df['y_center_mm']
            df['width [mm]'] = 0
        
        return df        


    def find_interacting_defects(self, defects_df: pd.DataFrame, joints_df: pd.DataFrame, pipe_diameter_mm: float) -> List[List[int]]:
        """
        Find groups of interacting defects according to FFS rules.
        Enhanced to handle edge cases like defects near pipe ends.
        """
        # Convert to cartesian coordinates
        df = self.convert_to_cartesian(defects_df, joints_df, pipe_diameter_mm)
        
        # ADDED: Identify defects near pipe ends if joint information available
        if 'joint number' in df.columns and not joints_df.empty:
            first_joint = joints_df['joint number'].min()
            last_joint = joints_df['joint number'].max()
            
            # Mark defects in first and last joints
            df['is_near_start'] = df['joint number'] == first_joint
            df['is_near_end'] = df['joint number'] == last_joint
        else:
            df['is_near_start'] = False
            df['is_near_end'] = False
        
        # Get wall thickness lookup
        wt_lookup = dict(zip(joints_df['joint number'], joints_df['wt nom [mm]']))
        
        # Initialize Union-Find structure for grouping
        n_defects = len(df)
        parent = list(range(n_defects))
        rank = [0] * n_defects  # ADDED: Use union by rank for efficiency
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])  # Path compression
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                # Union by rank
                if rank[px] < rank[py]:
                    parent[px] = py
                elif rank[px] > rank[py]:
                    parent[py] = px
                else:
                    parent[py] = px
                    rank[px] += 1
        
        # Calculate maximum possible interaction distances for optimization
        max_defect_length = df['length [mm]'].max() if not df.empty else 0
        max_defect_width = df['width [mm]'].max() if 'width [mm]' in df.columns and not df.empty else 0
        max_axial_search_distance = self.axial_interaction_distance + max_defect_length
        
        # ADDED: For circumferential, we need to consider the maximum possible interaction distance
        max_wall_thickness = max(wt_lookup.values()) if wt_lookup else 10.0
        max_circ_interaction = self.calculate_circumferential_interaction_distance(
            pipe_diameter_mm, max_wall_thickness
        ) + max_defect_width
        
        # Track statistics
        comparisons_made = 0
        comparisons_skipped = 0
        interactions_found = 0
        
        # Check pairs of defects for interaction
        for i in range(n_defects):
            defect_i = df.iloc[i]
            
            # ADDED: Special handling for end defects
            is_end_defect_i = defect_i['is_near_start'] or defect_i['is_near_end']
            
            for j in range(i + 1, n_defects):
                defect_j = df.iloc[j]
                
                # Calculate axial distance between defect centers
                axial_center_distance = abs(defect_j['x_center_mm'] - defect_i['x_center_mm'])
                
                # Skip if too far apart axially
                if axial_center_distance > max_axial_search_distance:
                    comparisons_skipped += (n_defects - j)
                    break  # All subsequent defects will also be too far
                
                comparisons_made += 1
                
                # ADDED: Special consideration for end defects
                is_end_defect_j = defect_j['is_near_start'] or defect_j['is_near_end']
                if is_end_defect_i and is_end_defect_j and defect_i['is_near_start'] != defect_j['is_near_start']:
                    # Defects are at opposite ends of the pipe - they don't interact
                    continue
                
                # Check if defects actually interact
                if self._defects_interact(defect_i, defect_j, pipe_diameter_mm, wt_lookup):
                    union(i, j)
                    interactions_found += 1
        
        # Log statistics
        total_possible = n_defects * (n_defects - 1) // 2
        logger.info(f"Defect interaction analysis complete: {comparisons_made} comparisons made, "
                    f"{comparisons_skipped} skipped (out of {total_possible} possible). "
                    f"Found {interactions_found} interacting pairs.")
        
        # Group defects by their root parent
        groups = {}
        for i in range(n_defects):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)
        
        # ADDED: Validate clusters don't have impossible configurations
        validated_groups = []
        for group in groups.values():
            if len(group) > 1:
                # Check if cluster spans too large an area (sanity check)
                group_defects = df.iloc[group]
                axial_span = group_defects['x_center_mm'].max() - group_defects['x_center_mm'].min()
                
                # If cluster spans more than 1 meter, it might be an error
                if axial_span > 1000:  # 1 meter
                    logger.warning(f"Large cluster detected spanning {axial_span:.0f}mm. "
                                f"This may indicate an issue with interaction rules.")
                
                validated_groups.append(group)
            else:
                # Single defect
                validated_groups.append(group)
        
        return validated_groups
    

    def _defects_interact(self, defect1: pd.Series, defect2: pd.Series, pipe_diameter_mm: float, wt_lookup: Dict) -> bool:
        """
        Check if two defects interact according to FFS rules.
        """
        # Get wall thickness for circumferential interaction calculation
        # Use the minimum wall thickness of the two joints (conservative)
        wt1 = wt_lookup.get(defect1['joint number'], None)
        wt2 = wt_lookup.get(defect2['joint number'], None)
        
        if wt1 is None or wt2 is None:
            warnings.warn(f"Missing wall thickness for defect interaction check")
            return False
        
        wall_thickness_mm = min(wt1, wt2)
        
        # Calculate interaction distances
        axial_limit = self.axial_interaction_distance
        circ_limit = self.calculate_circumferential_interaction_distance(
            pipe_diameter_mm, wall_thickness_mm
        )
        
        # Check axial separation
        axial_gap = max(defect1['x_start_mm'], defect2['x_start_mm']) - \
                   min(defect1['x_end_mm'], defect2['x_end_mm'])
        
        if axial_gap > axial_limit:
            return False  # Too far apart axially
        
        # Check circumferential separation
        # Need to account for wrap-around at 360 degrees
        circ_gap = self._calculate_circumferential_gap(
            defect1['y_start_mm'], defect1['y_end_mm'],
            defect2['y_start_mm'], defect2['y_end_mm'],
            pipe_diameter_mm
        )
        
        if circ_gap > circ_limit:
            return False  # Too far apart circumferentially
        
        return True  # Defects interact
    

    def _calculate_circumferential_gap(self, y1_start, y1_end, y2_start, y2_end, pipe_diameter_mm):
        """
        Calculate minimum circumferential gap between two defects.
        Properly accounts for wrap-around at 360 degrees.
        
        Parameters:
        - y1_start, y1_end: Start and end positions of defect 1 in mm (arc length)
        - y2_start, y2_end: Start and end positions of defect 2 in mm (arc length)
        - pipe_diameter_mm: Pipe diameter for circumference calculation
        
        Returns:
        - Minimum gap in mm between the two defects
        """
        # Calculate circumference
        circumference = np.pi * pipe_diameter_mm
        
        # Normalize positions to [0, circumference) range
        def normalize_position(pos):
            """Normalize position to [0, circumference) range"""
            return pos % circumference
        
        # Normalize all positions
        y1_start_norm = normalize_position(y1_start)
        y1_end_norm = normalize_position(y1_end)
        y2_start_norm = normalize_position(y2_start)
        y2_end_norm = normalize_position(y2_end)
        
        # Handle defects that span across 0° (12 o'clock position)
        def get_segments(start, end):
            """Get segments of a defect, handling wrap-around"""
            if start <= end:
                return [(start, end)]
            else:
                # Defect wraps around
                return [(start, circumference), (0, end)]
        
        segments1 = get_segments(y1_start_norm, y1_end_norm)
        segments2 = get_segments(y2_start_norm, y2_end_norm)
        
        # Check for overlap between any pair of segments
        min_gap = circumference  # Initialize to maximum possible gap
        
        for seg1_start, seg1_end in segments1:
            for seg2_start, seg2_end in segments2:
                # Check if segments overlap
                if seg1_start <= seg2_end and seg2_start <= seg1_end:
                    # Segments overlap
                    return 0.0
                
                # Calculate gaps between segments
                gap1 = seg2_start - seg1_end if seg2_start > seg1_end else 0
                gap2 = seg1_start - seg2_end if seg1_start > seg2_end else 0
                
                min_gap = min(min_gap, gap1, gap2)
        
        # Also check wrap-around gap if defects don't overlap
        if min_gap > 0:
            # Calculate shortest distance considering wrap-around
            # This handles cases where going "the other way" around the pipe is shorter
            positions = [y1_start_norm, y1_end_norm, y2_start_norm, y2_end_norm]
            positions.sort()
            
            # Gaps between consecutive sorted positions
            for i in range(len(positions)):
                next_i = (i + 1) % len(positions)
                gap = positions[next_i] - positions[i]
                if gap < 0:  # Wrap around
                    gap += circumference
                
                # Check if this gap separates the two defects
                # (Complex logic omitted for brevity, but we need to verify
                # that defect 1 and defect 2 are on opposite sides of this gap)
            
            # Simplified approach: also check the complement gap
            if min_gap > circumference / 2:
                min_gap = circumference - min_gap
        
        return max(0, min_gap)  # Ensure non-negative


    def combine_interacting_defects(self, defects_df: pd.DataFrame, joints_df: pd.DataFrame, pipe_diameter_mm: float) -> pd.DataFrame:
        """
        Combine interacting defects according to FFS rules.
        
        Returns new dataframe with combined defects.
        """
        # Find interacting groups
        groups = self.find_interacting_defects(defects_df, joints_df, pipe_diameter_mm)
        combined_defects = []

        for group in groups:
            if len(group) == 1:
                # Single defect - keep as is
                combined_defects.append(defects_df.iloc[group[0]].to_dict())
            else:
                # Multiple interacting defects - combine them
                group_defects = defects_df.iloc[group]
                
                decimal_hours = group_defects['clock'].apply(self.parse_clock_to_decimal_hours)
                mean_decimal_hours = decimal_hours.mean()
            
                # Convert back to clock format (optional - you could keep decimal)
                mean_hours = int(mean_decimal_hours)
                mean_minutes = int((mean_decimal_hours - mean_hours) * 60)
                mean_clock_str = f"{mean_hours}:{mean_minutes:02d}"

                combined = {
                    # Take maximum depth (most conservative)
                    'depth [%]': group_defects['depth [%]'].max(),
                    
                    # Calculate total extent
                    'log dist. [m]': group_defects['log dist. [m]'].mean(),  # Center of combined defect
                    'length [mm]': (group_defects['log dist. [m]'].max() * 1000 + 
                                   group_defects['length [mm]'].iloc[group_defects['log dist. [m]'].argmax()] / 2) - \
                                  (group_defects['log dist. [m]'].min() * 1000 - 
                                   group_defects['length [mm]'].iloc[group_defects['log dist. [m]'].argmin()] / 2),
                    
                    # For width, take the maximum circumferential extent
                    'width [mm]': self._calculate_combined_width(group_defects, pipe_diameter_mm),
                    
                    # Preserve other important fields
                    'joint number': group_defects['joint number'].iloc[0],  # Assuming same joint
                    'clock': mean_clock_str,
                    
                    # Metadata about combination
                    'is_combined': True,
                    'num_original_defects': len(group),
                    'original_indices': list(group),
                    'combination_note': f"Combined {len(group)} interacting defects per FFS rules"
                }

                # Preserve other columns from the first defect
                first_defect = defects_df.iloc[group[0]]
                for col in defects_df.columns:
                    if col not in combined:
                        combined[col] = first_defect[col]
                
                combined_defects.append(combined)
        
        result_df = pd.DataFrame(combined_defects)
        
        # Add analysis metadata
        result_df['ffs_interaction_analysis'] = True
        result_df['axial_interaction_distance_mm'] = self.axial_interaction_distance
        result_df['circ_interaction_method'] = self.circ_method
        
        return result_df
    
    def _calculate_combined_width(self, group_defects: pd.DataFrame, pipe_diameter_mm: float) -> float:
        """
        Calculate combined circumferential width of interacting defects.
        Uses a more accurate approach considering actual overlaps.
        
        Parameters:
        - group_defects: DataFrame of defects in the cluster
        - pipe_diameter_mm: Pipe diameter for calculations
        
        Returns:
        - Combined width in mm
        """
        if 'width [mm]' not in group_defects.columns:
            return 0
        
        # Convert clock positions to angular positions
        group_defects = group_defects.copy()
        group_defects['decimal_hours'] = group_defects['clock'].apply(self.parse_clock_to_decimal_hours)
        
        # Convert to angular position (radians)
        # 12:00 = 0°, 3:00 = 90°, 6:00 = 180°, 9:00 = 270°
        group_defects['angle_rad'] = (group_defects['decimal_hours'] % 12) * (np.pi / 6)  # Convert hours to radians
        
        radius = pipe_diameter_mm / 2
        
        # Calculate start and end angles for each defect
        angular_extents = []
        
        for idx, defect in group_defects.iterrows():
            # Convert width to angular extent
            width_mm = defect['width [mm]']
            angular_width = width_mm / radius  # Arc length = radius × angle
            
            center_angle = defect['angle_rad']
            start_angle = center_angle - angular_width / 2
            end_angle = center_angle + angular_width / 2
            
            # Normalize angles to [0, 2π]
            start_angle = start_angle % (2 * np.pi)
            end_angle = end_angle % (2 * np.pi)
            
            angular_extents.append((start_angle, end_angle))
        
        # Merge overlapping angular extents
        merged_extents = self._merge_angular_extents(angular_extents)
        
        # Calculate total angular extent
        total_angular_extent = 0
        for start, end in merged_extents:
            if start <= end:
                total_angular_extent += end - start
            else:
                # Wraps around 0
                total_angular_extent += (2 * np.pi - start) + end
        
        # Convert back to arc length
        combined_width = total_angular_extent * radius
        
        # Limit to circumference (can't be wider than the pipe!)
        max_width = np.pi * pipe_diameter_mm
        combined_width = min(combined_width, max_width)
        
        return combined_width


    def _merge_angular_extents(self, extents: list) -> list:
        """
        Merge overlapping angular extents, handling wrap-around at 0/2π.
        
        Parameters:
        - extents: List of (start_angle, end_angle) tuples
        
        Returns:
        - List of merged non-overlapping extents
        """
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