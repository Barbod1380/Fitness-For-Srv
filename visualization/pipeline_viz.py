import numpy as np
import pandas as pd
import plotly.graph_objects as go

def create_unwrapped_pipeline_visualization(defects_df, pipe_diameter=None):
    """
    Create an enhanced unwrapped cylinder visualization of pipeline defects,
    with Y-axis showing clock positions for intuitive interpretation.
    
    Parameters:
    - defects_df: DataFrame containing defect information
    - pipe_diameter: Pipeline diameter in meters (REQUIRED)
    
    Returns:
    - Plotly figure object
    """
    
    if pipe_diameter is None:
        raise ValueError("pipe_diameter is required. Pass the actual pipe diameter from your dataset.")
    
    if pipe_diameter <= 0:
        raise ValueError(f"pipe_diameter must be positive, got {pipe_diameter}")

    max_points = 7000  # WebGL performance threshold
    
    if len(defects_df) > max_points:
        # Priority-based sampling: keep critical defects + representative sample
        
        # Always keep high-severity defects
        critical_mask = defects_df['depth [%]'] > 70
        critical_defects = defects_df[critical_mask]
        
        # Sample remaining defects spatially distributed
        remaining_defects = defects_df[~critical_mask]
        
        if len(remaining_defects) > 0:
            # Spatial binning for representative sampling
            n_bins = min(50, len(remaining_defects) // 10)
            remaining_defects['spatial_bin'] = pd.cut(remaining_defects['log dist. [m]'], 
                                                    bins=n_bins, labels=False)
            
            # Sample from each bin
            samples_per_bin = max(1, (max_points - len(critical_defects)) // n_bins)
            sampled_remaining = (remaining_defects.groupby('spatial_bin', group_keys=False)
                               .apply(lambda x: x.sample(min(len(x), samples_per_bin))))
            
            plot_data = pd.concat([critical_defects, sampled_remaining], ignore_index=True)
        else:
            plot_data = critical_defects
    else:
        plot_data = defects_df
    
    # Extract axial values (X-axis) - unchanged
    x_vals = plot_data["log dist. [m]"].values
    
    # Convert clock positions to actual circumferential distances (for proper spacing)
    clock_hours = plot_data["clock_float"].values
    angles_radians = (clock_hours / 12.0) * 2 * np.pi # type: ignore
    circumferential_distance_m = (pipe_diameter / 2) * angles_radians
    
    # Y-axis uses circumferential distances for proportional spacing
    y_vals = circumferential_distance_m
    
    # === Efficient marker properties ===
    if "depth [%]" in plot_data.columns:
        depth_values = plot_data["depth [%]"].values
        marker_props = dict(
            size=6,  # Slightly smaller for better performance
            color=depth_values,
            colorscale="Turbo",
            cmin=0,
            cmax=depth_values.max(), # type: ignore
            colorbar=dict(title="Depth (%)", thickness=15, len=0.6),
            opacity=0.8,
        )
    else:
        marker_props = dict(size=6, color="blue", opacity=0.8)
    
    # Minimize data in customdata to reduce memory usage
    has_component = 'component / anomaly identification' in plot_data.columns
    
    if has_component:
        custom_data = np.column_stack([
            plot_data["joint number"].astype(str).values,
            plot_data["component / anomaly identification"].values,
            plot_data["depth [%]"].fillna(0).values,
            clock_hours,  # Include original clock position for hover
        ]) # type: ignore
    else:
        custom_data = np.column_stack([
            plot_data["joint number"].astype(str).values,
            plot_data["depth [%]"].fillna(0).values,
            clock_hours,  # Include original clock position for hover
        ]) # type: ignore
    
    # === Use WebGL for large datasets ===
    use_webgl = len(plot_data) > 1000
    scatter_class = go.Scattergl if use_webgl else go.Scatter
    
    # === UPDATED: Enhanced hover template emphasizing clock positions ===
    if has_component:
        hover_template = (
            "<b>Distance:</b> %{x:.2f} m<br>"
            "<b>Clock Position:</b> %{customdata[3]:.1f}:00<br>"
            "<b>Depth:</b> %{customdata[2]:.1f}%<br>"
            "<b>Type:</b> %{customdata[1]}<br>"
            "<b>Joint:</b> %{customdata[0]}<extra></extra>"
        )
    else:
        hover_template = (
            "<b>Distance:</b> %{x:.2f} m<br>"
            "<b>Clock Position:</b> %{customdata[2]:.1f}:00<br>"
            "<b>Depth:</b> %{customdata[1]:.1f}%<br>"
            "<b>Joint:</b> %{customdata[0]}<extra></extra>"
        )
    
    fig = go.Figure()
    fig.add_trace(
        scatter_class(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker=marker_props,
            customdata=custom_data,
            hovertemplate=hover_template,
            name="Defects",
        )
    )
    
    # === NEW: Calculate major clock positions and their circumferential equivalents ===
    major_clock_positions = [12, 3, 6, 9, 12]  # 12:00, 3:00, 6:00, 9:00, 12:00
    tick_positions = []
    tick_labels = []
    
    for clock_pos in major_clock_positions:
        # Convert clock position to circumferential distance
        angle_rad = (clock_pos / 12.0) * 2 * np.pi
        circ_distance = (pipe_diameter / 2) * angle_rad
        tick_positions.append(circ_distance)
        tick_labels.append(f"{clock_pos}:00")
    
    # === Enhanced grid lines at major clock positions ===
    x_range = [x_vals.min() - 1, x_vals.max() + 1] # type: ignore
    
    for clock_pos, circ_distance in zip(major_clock_positions, tick_positions):
        fig.add_shape(
            type="line",
            x0=x_range[0], x1=x_range[1],
            y0=circ_distance, y1=circ_distance,
            line=dict(color="lightgray", width=1, dash="dot"),
            layer="below",
        )
    
    # Calculate total circumference for Y-axis range
    total_circumference = np.pi * pipe_diameter
    
    # === NEW: Clock-based Y-axis configuration ===
    fig.update_layout(
        title=f"Unwrapped Pipeline Defect Map (Ø{pipe_diameter:.2f}m)" + 
              (f" - Showing {len(plot_data):,} of {len(defects_df):,} defects" if len(plot_data) != len(defects_df) else ""),
        xaxis=dict(
            title="Axial Distance Along Pipeline (m)",
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.3)",
            range=x_range
        ),
        yaxis=dict(
            title="Clock Position Around Pipe<br><sub>(Pipeline circumferential position)</sub>",
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.3)",
            range=[0, total_circumference],
            # NEW: Custom tick positions and labels for clock positions
            tickmode='array',
            tickvals=tick_positions,
            ticktext=tick_labels,
            # Add minor ticks for intermediate positions (1:00, 2:00, etc.)
            dtick=None  # Disable automatic ticks
        ),
        height=600,
        plot_bgcolor="white",
        hovermode="closest",
        uirevision="constant",  # Prevent unnecessary re-renders
        dragmode="pan",  # Better for large datasets
        # Add annotation explaining the clock system
        annotations=[
            dict(
                x=1, y=1,
                xref="paper", yref="paper",
                text="12:00 = top of pipe<br>6:00 = bottom of pipe<br>3:00 = right side, 9:00 = left side",
                showarrow=False,
                xanchor="right", yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1,
                font=dict(size=10)
            )
        ]
    )
    
    return fig