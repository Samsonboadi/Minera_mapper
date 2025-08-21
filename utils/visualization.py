"""
Visualization utilities for mineral prospectivity analysis
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
import rasterio
from rasterio.plot import show
import pandas as pd
from scipy import stats
from qgis.PyQt.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QLabel
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.core import QgsRasterLayer, QgsVectorLayer, QgsProject, QgsMessageLog, Qgis
import json

class Visualizer:
    """Advanced visualization tools for geological and mineral data"""
    
    def __init__(self):
        self.color_schemes = {
            'geological': ['#8B4513', '#A0522D', '#CD853F', '#DEB887', '#F5DEB3'],
            'mineral': ['#FF0000', '#FF8C00', '#FFD700', '#ADFF2F', '#00FF00'],
            'prospectivity': ['#000080', '#0000FF', '#00FFFF', '#FFFF00', '#FF0000'],
            'alteration': ['#800080', '#9932CC', '#BA55D3', '#DA70D6', '#EE82EE'],
            'magnetic': ['#191970', '#4169E1', '#87CEEB', '#ADD8E6', '#F0F8FF'],
            'elevation': ['#2F4F4F', '#696969', '#A9A9A9', '#D3D3D3', '#F5F5F5']
        }
        
        self.figure_size = (12, 8)
        self.dpi = 100
    
    def create_mineral_map_visualization(self, mineral_maps, output_path=None):
        """Create comprehensive mineral map visualization"""
        n_maps = len(mineral_maps)
        if n_maps == 0:
            return None
        
        # Calculate grid dimensions
        cols = min(3, n_maps)
        rows = (n_maps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
        if n_maps == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Mineral Abundance Maps', fontsize=16, weight='bold')
        
        for i, (mineral_name, map_path) in enumerate(mineral_maps.items()):
            row = i // cols
            col = i % cols
            
            if rows > 1:
                ax = axes[row, col]
            else:
                ax = axes[col]
            
            try:
                with rasterio.open(map_path) as src:
                    data = src.read(1)
                    
                    # Create colormap
                    cmap = plt.cm.get_cmap('viridis')
                    
                    # Plot data
                    im = ax.imshow(data, cmap=cmap, aspect='auto')
                    ax.set_title(f'{mineral_name}', fontsize=12, weight='bold')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    
                    # Add colorbar
                    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                    cbar.set_label('Abundance', rotation=270, labelpad=15)
                    
            except Exception as e:
                ax.text(0.5, 0.5, f'Error loading\n{mineral_name}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{mineral_name} (Error)', fontsize=12)
        
        # Hide unused subplots
        for i in range(n_maps, rows * cols):
            row = i // cols
            col = i % cols
            if rows > 1:
                axes[row, col].set_visible(False)
            else:
                axes[col].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            QgsMessageLog.logMessage(f"Mineral map visualization saved: {output_path}", 
                                    'Mineral Prospectivity', Qgis.Info)
        
        return fig
    
    def create_prospectivity_visualization(self, prospectivity_map, confidence_map=None, output_path=None):
        """Create prospectivity map with uncertainty visualization"""
        try:
            with rasterio.open(prospectivity_map) as src:
                prospect_data = src.read(1)
                transform = src.transform
                bounds = src.bounds
        except Exception as e:
            QgsMessageLog.logMessage(f"Error loading prospectivity map: {str(e)}", 
                                    'Mineral Prospectivity', Qgis.Critical)
            return None
        
        if confidence_map:
            try:
                with rasterio.open(confidence_map) as src:
                    confidence_data = src.read(1)
            except:
                confidence_data = None
        else:
            confidence_data = None
        
        # Create figure
        if confidence_data is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes = [axes[0], axes[1], None]
        
        fig.suptitle('Mineral Prospectivity Analysis', fontsize=16, weight='bold')
        
        # Main prospectivity map
        cmap_prospect = plt.cm.get_cmap('RdYlGn')
        im1 = axes[0].imshow(prospect_data, cmap=cmap_prospect, aspect='auto', 
                            extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
        axes[0].set_title('Prospectivity Map', fontsize=14, weight='bold')
        axes[0].set_xlabel('Easting')
        axes[0].set_ylabel('Northing')
        
        # Colorbar for prospectivity
        cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8)
        cbar1.set_label('Prospectivity Score', rotation=270, labelpad=15)
        
        # Histogram of prospectivity values
        valid_data = prospect_data[np.isfinite(prospect_data)]
        if len(valid_data) > 0:
            axes[1].hist(valid_data, bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[1].set_title('Prospectivity Distribution', fontsize=14, weight='bold')
            axes[1].set_xlabel('Prospectivity Score')
            axes[1].set_ylabel('Frequency')
            axes[1].grid(True, alpha=0.3)
            
            # Add statistics
            mean_val = np.mean(valid_data)
            std_val = np.std(valid_data)
            axes[1].axvline(mean_val, color='red', linestyle='--', 
                           label=f'Mean: {mean_val:.3f}')
            axes[1].axvline(mean_val + std_val, color='orange', linestyle='--', 
                           label=f'+1σ: {mean_val + std_val:.3f}')
            axes[1].legend()
        
        # Confidence map if available
        if confidence_data is not None and axes[2] is not None:
            cmap_conf = plt.cm.get_cmap('Blues')
            im3 = axes[2].imshow(confidence_data, cmap=cmap_conf, aspect='auto',
                               extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
            axes[2].set_title('Confidence Map', fontsize=14, weight='bold')
            axes[2].set_xlabel('Easting')
            axes[2].set_ylabel('Northing')
            
            cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.8)
            cbar3.set_label('Confidence Level', rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            QgsMessageLog.logMessage(f"Prospectivity visualization saved: {output_path}", 
                                    'Mineral Prospectivity', Qgis.Info)
        
        return fig
    
    def create_correlation_heatmap(self, correlation_matrix, layer_names, output_path=None):
        """Create correlation matrix heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                   xticklabels=layer_names, yticklabels=layer_names, ax=ax)
        
        ax.set_title('Layer Correlation Matrix', fontsize=16, weight='bold', pad=20)
        
        # Add correlation values
        for i in range(len(layer_names)):
            for j in range(i):
                value = correlation_matrix[i, j]
                ax.text(j + 0.5, i + 0.5, f'{value:.2f}', 
                       ha='center', va='center', color='white' if abs(value) > 0.5 else 'black')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            QgsMessageLog.logMessage(f"Correlation heatmap saved: {output_path}", 
                                    'Mineral Prospectivity', Qgis.Info)
        
        return fig
    
    def create_spectral_signature_plot(self, mineral_signatures, wavelengths, output_path=None):
        """Create spectral signature plot for minerals"""
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(mineral_signatures)))
        
        for i, (mineral_name, signature) in enumerate(mineral_signatures.items()):
            if isinstance(signature, dict) and 'signature' in signature:
                sig_data = signature['signature']
                sig_wavelengths = signature.get('wavelengths', wavelengths)
            else:
                sig_data = signature
                sig_wavelengths = wavelengths
            
            ax.plot(sig_wavelengths, sig_data, color=colors[i], 
                   linewidth=2, label=mineral_name, marker='o', markersize=4)
        
        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('Reflectance', fontsize=12)
        ax.set_title('Mineral Spectral Signatures', fontsize=16, weight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            QgsMessageLog.logMessage(f"Spectral signature plot saved: {output_path}", 
                                    'Mineral Prospectivity', Qgis.Info)
        
        return fig
    
    def create_geological_cross_section(self, data_arrays, labels, output_path=None):
        """Create geological cross-section visualization"""
        fig, axes = plt.subplots(len(data_arrays), 1, figsize=(12, 3 * len(data_arrays)))
        if len(data_arrays) == 1:
            axes = [axes]
        
        fig.suptitle('Geological Cross-Section Analysis', fontsize=16, weight='bold')
        
        for i, (data, label) in enumerate(zip(data_arrays, labels)):
            # Assume data is a 2D cross-section
            if data.ndim == 2:
                im = axes[i].imshow(data, aspect='auto', cmap='terrain')
                axes[i].set_title(label, fontsize=12, weight='bold')
                axes[i].set_ylabel('Depth/Elevation')
                
                if i == len(data_arrays) - 1:
                    axes[i].set_xlabel('Distance')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=axes[i], shrink=0.8)
                cbar.set_label('Value', rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            QgsMessageLog.logMessage(f"Cross-section visualization saved: {output_path}", 
                                    'Mineral Prospectivity', Qgis.Info)
        
        return fig
    
    def create_statistical_summary_plot(self, layer_statistics, output_path=None):
        """Create statistical summary visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Statistical Summary of Layers', fontsize=16, weight='bold')
        
        # Extract statistics
        layer_names = list(layer_statistics.keys())
        means = [stats.get('mean', 0) for stats in layer_statistics.values()]
        stds = [stats.get('std', 0) for stats in layer_statistics.values()]
        mins = [stats.get('min', 0) for stats in layer_statistics.values()]
        maxs = [stats.get('max', 1) for stats in layer_statistics.values()]
        
        # Truncate long layer names for better display
        display_names = [name[:15] + '...' if len(name) > 15 else name for name in layer_names]
        
        # Mean values
        axes[0, 0].bar(range(len(means)), means, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Mean Values', fontsize=14, weight='bold')
        axes[0, 0].set_xticks(range(len(display_names)))
        axes[0, 0].set_xticklabels(display_names, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Mean Value')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Standard deviations
        axes[0, 1].bar(range(len(stds)), stds, color='lightcoral', edgecolor='black')
        axes[0, 1].set_title('Standard Deviations', fontsize=14, weight='bold')
        axes[0, 1].set_xticks(range(len(display_names)))
        axes[0, 1].set_xticklabels(display_names, rotation=45, ha='right')
        axes[0, 1].set_ylabel('Standard Deviation')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Value ranges
        ranges = [max_val - min_val for min_val, max_val in zip(mins, maxs)]
        axes[1, 0].bar(range(len(ranges)), ranges, color='lightgreen', edgecolor='black')
        axes[1, 0].set_title('Value Ranges', fontsize=14, weight='bold')
        axes[1, 0].set_xticks(range(len(display_names)))
        axes[1, 0].set_xticklabels(display_names, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Range (Max - Min)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Coefficient of variation
        cv = [std / mean if mean != 0 else 0 for mean, std in zip(means, stds)]
        axes[1, 1].bar(range(len(cv)), cv, color='gold', edgecolor='black')
        axes[1, 1].set_title('Coefficient of Variation', fontsize=14, weight='bold')
        axes[1, 1].set_xticks(range(len(display_names)))
        axes[1, 1].set_xticklabels(display_names, rotation=45, ha='right')
        axes[1, 1].set_ylabel('CV (σ/μ)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            QgsMessageLog.logMessage(f"Statistical summary plot saved: {output_path}", 
                                    'Mineral Prospectivity', Qgis.Info)
        
        return fig
    
    def create_rgb_composite(self, red_band, green_band, blue_band, output_path=None, stretch_type='percent'):
        """Create RGB composite visualization"""
        try:
            # Load band data
            with rasterio.open(red_band) as src:
                red_data = src.read(1).astype(np.float32)
                profile = src.profile
                bounds = src.bounds
            
            with rasterio.open(green_band) as src:
                green_data = src.read(1).astype(np.float32)
            
            with rasterio.open(blue_band) as src:
                blue_data = src.read(1).astype(np.float32)
            
            # Apply stretch
            if stretch_type == 'percent':
                red_stretched = self._percent_stretch(red_data, 2, 98)
                green_stretched = self._percent_stretch(green_data, 2, 98)
                blue_stretched = self._percent_stretch(blue_data, 2, 98)
            elif stretch_type == 'minmax':
                red_stretched = self._minmax_stretch(red_data)
                green_stretched = self._minmax_stretch(green_data)
                blue_stretched = self._minmax_stretch(blue_data)
            else:
                red_stretched = red_data
                green_stretched = green_data
                blue_stretched = blue_data
            
            # Stack RGB
            rgb = np.stack([red_stretched, green_stretched, blue_stretched], axis=2)
            rgb = np.clip(rgb, 0, 1)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            ax.imshow(rgb, extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
            ax.set_title('RGB Composite', fontsize=16, weight='bold')
            ax.set_xlabel('Easting')
            ax.set_ylabel('Northing')
            
            # Add band information
            band_info = f"R: {os.path.basename(red_band)}\nG: {os.path.basename(green_band)}\nB: {os.path.basename(blue_band)}"
            ax.text(0.02, 0.98, band_info, transform=ax.transAxes, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   verticalalignment='top', fontsize=10)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                QgsMessageLog.logMessage(f"RGB composite saved: {output_path}", 
                                        'Mineral Prospectivity', Qgis.Info)
            
            return fig
            
        except Exception as e:
            QgsMessageLog.logMessage(f"Error creating RGB composite: {str(e)}", 
                                    'Mineral Prospectivity', Qgis.Critical)
            return None
    
    def _percent_stretch(self, data, lower_percent, upper_percent):
        """Apply percentile stretch to data"""
        valid_data = data[np.isfinite(data)]
        if len(valid_data) == 0:
            return data
        
        lower_val = np.percentile(valid_data, lower_percent)
        upper_val = np.percentile(valid_data, upper_percent)
        
        stretched = (data - lower_val) / (upper_val - lower_val)
        return np.clip(stretched, 0, 1)
    
    def _minmax_stretch(self, data):
        """Apply min-max stretch to data"""
        valid_data = data[np.isfinite(data)]
        if len(valid_data) == 0:
            return data
        
        min_val = np.min(valid_data)
        max_val = np.max(valid_data)
        
        if max_val > min_val:
            stretched = (data - min_val) / (max_val - min_val)
        else:
            stretched = np.zeros_like(data)
        
        return np.clip(stretched, 0, 1)
    
    def create_scatter_plot_matrix(self, data_dict, output_path=None):
        """Create scatter plot matrix for correlation analysis"""
        variables = list(data_dict.keys())
        n_vars = len(variables)
        
        if n_vars < 2:
            QgsMessageLog.logMessage("Need at least 2 variables for scatter plot matrix", 
                                    'Mineral Prospectivity', Qgis.Warning)
            return None
        
        fig, axes = plt.subplots(n_vars, n_vars, figsize=(3 * n_vars, 3 * n_vars))
        fig.suptitle('Scatter Plot Matrix', fontsize=16, weight='bold')
        
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                ax = axes[i, j] if n_vars > 1 else axes
                
                if i == j:
                    # Diagonal: histogram
                    data = data_dict[var1]
                    valid_data = data[np.isfinite(data)]
                    if len(valid_data) > 0:
                        ax.hist(valid_data, bins=30, alpha=0.7, density=True, color='skyblue')
                        ax.set_xlabel(var1)
                        ax.set_ylabel('Density')
                else:
                    # Off-diagonal: scatter plot
                    data1 = data_dict[var1]
                    data2 = data_dict[var2]
                    
                    # Find common valid points
                    valid_mask = np.isfinite(data1) & np.isfinite(data2)
                    if valid_mask.sum() > 0:
                        x_data = data1[valid_mask]
                        y_data = data2[valid_mask]
                        
                        # Subsample if too many points
                        if len(x_data) > 10000:
                            indices = np.random.choice(len(x_data), 10000, replace=False)
                            x_data = x_data[indices]
                            y_data = y_data[indices]
                        
                        ax.scatter(x_data, y_data, alpha=0.5, s=1)
                        
                        # Calculate and display correlation
                        if len(x_data) > 1:
                            corr, p_val = stats.pearsonr(x_data, y_data)
                            ax.text(0.05, 0.95, f'r={corr:.3f}\np={p_val:.3f}', 
                                   transform=ax.transAxes, 
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                                   verticalalignment='top', fontsize=8)
                    
                    if i == n_vars - 1:
                        ax.set_xlabel(var2)
                    if j == 0:
                        ax.set_ylabel(var1)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            QgsMessageLog.logMessage(f"Scatter plot matrix saved: {output_path}", 
                                    'Mineral Prospectivity', Qgis.Info)
        
        return fig
    
    def create_elevation_profile(self, elevation_data, profile_coordinates, output_path=None):
        """Create elevation profile along specified coordinates"""
        try:
            # Extract elevation values along profile
            profile_values = []
            distances = []
            cumulative_distance = 0
            
            for i, coord in enumerate(profile_coordinates):
                if i > 0:
                    prev_coord = profile_coordinates[i-1]
                    distance = np.sqrt((coord[0] - prev_coord[0])**2 + (coord[1] - prev_coord[1])**2)
                    cumulative_distance += distance
                
                distances.append(cumulative_distance)
                
                # Extract elevation value (simplified - assumes coordinate indexing)
                try:
                    elevation = elevation_data[int(coord[1]), int(coord[0])]
                    profile_values.append(elevation)
                except:
                    profile_values.append(np.nan)
            
            # Create profile plot
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            valid_mask = ~np.isnan(profile_values)
            ax.plot(np.array(distances)[valid_mask], np.array(profile_values)[valid_mask], 
                   linewidth=2, color='brown', marker='o', markersize=3)
            
            ax.fill_between(np.array(distances)[valid_mask], 
                           np.array(profile_values)[valid_mask], 
                           alpha=0.3, color='brown')
            
            ax.set_xlabel('Distance along profile')
            ax.set_ylabel('Elevation')
            ax.set_title('Elevation Profile', fontsize=16, weight='bold')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                QgsMessageLog.logMessage(f"Elevation profile saved: {output_path}", 
                                        'Mineral Prospectivity', Qgis.Info)
            
            return fig
            
        except Exception as e:
            QgsMessageLog.logMessage(f"Error creating elevation profile: {str(e)}", 
                                    'Mineral Prospectivity', Qgis.Critical)
            return None
    
    def create_uncertainty_visualization(self, data, uncertainty, output_path=None):
        """Create uncertainty visualization with confidence intervals"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Uncertainty Analysis', fontsize=16, weight='bold')
        
        # Main data plot
        im1 = axes[0].imshow(data, cmap='viridis', aspect='auto')
        axes[0].set_title('Primary Data', fontsize=14, weight='bold')
        cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8)
        cbar1.set_label('Value', rotation=270, labelpad=15)
        
        # Uncertainty plot
        im2 = axes[1].imshow(uncertainty, cmap='Reds', aspect='auto')
        axes[1].set_title('Uncertainty', fontsize=14, weight='bold')
        cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8)
        cbar2.set_label('Uncertainty Level', rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            QgsMessageLog.logMessage(f"Uncertainty visualization saved: {output_path}", 
                                    'Mineral Prospectivity', Qgis.Info)
        
        return fig
    
    def save_visualization_config(self, config_dict, output_path):
        """Save visualization configuration for reproducibility"""
        config = {
            'visualization_settings': config_dict,
            'color_schemes': self.color_schemes,
            'figure_size': self.figure_size,
            'dpi': self.dpi,
            'creation_date': pd.Timestamp.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        QgsMessageLog.logMessage(f"Visualization config saved: {output_path}", 
                                'Mineral Prospectivity', Qgis.Info)

class InteractiveVisualizationWidget(QWidget):
    """Interactive visualization widget for QGIS integration"""
    
    layer_selected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.visualizer = Visualizer()
        self.setup_ui()
        
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout()
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.layer_combo = QComboBox()
        self.layer_combo.currentTextChanged.connect(self.on_layer_changed)
        controls_layout.addWidget(QLabel("Layer:"))
        controls_layout.addWidget(self.layer_combo)
        
        self.viz_type_combo = QComboBox()
        self.viz_type_combo.addItems([
            'Histogram', 'Scatter Plot', 'Heat Map', 'Profile'
        ])
        controls_layout.addWidget(QLabel("Visualization:"))
        controls_layout.addWidget(self.viz_type_combo)
        
        self.update_btn = QPushButton("Update Plot")
        self.update_btn.clicked.connect(self.update_plot)
        controls_layout.addWidget(self.update_btn)
        
        layout.addLayout(controls_layout)
        
        # Matplotlib canvas
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        
        # Populate layer combo
        self.populate_layers()
    
    def populate_layers(self):
        """Populate layer combo box"""
        self.layer_combo.clear()
        project = QgsProject.instance()
        
        for layer in project.mapLayers().values():
            if isinstance(layer, (QgsRasterLayer, QgsVectorLayer)):
                self.layer_combo.addItem(layer.name())
    
    def on_layer_changed(self, layer_name):
        """Handle layer selection change"""
        self.layer_selected.emit(layer_name)
    
    def update_plot(self):
        """Update the visualization plot"""
        layer_name = self.layer_combo.currentText()
        viz_type = self.viz_type_combo.currentText()
        
        if not layer_name:
            return
        
        # Find the layer
        project = QgsProject.instance()
        layer = None
        for l in project.mapLayers().values():
            if l.name() == layer_name:
                layer = l
                break
        
        if not layer:
            return
        
        self.figure.clear()
        
        try:
            if isinstance(layer, QgsRasterLayer):
                self.plot_raster_data(layer, viz_type)
            else:
                self.plot_vector_data(layer, viz_type)
        except Exception as e:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center')
            ax.set_title(f'Error plotting {layer_name}')
        
        self.canvas.draw()
    
    def plot_raster_data(self, layer, viz_type):
        """Plot raster layer data"""
        source_path = layer.source()
        
        if viz_type == 'Histogram':
            with rasterio.open(source_path) as src:
                data = src.read(1)
                valid_data = data[np.isfinite(data) & (data != src.nodata)]
                
                ax = self.figure.add_subplot(111)
                ax.hist(valid_data, bins=50, alpha=0.7)
                ax.set_title(f'Histogram - {layer.name()}')
                ax.set_xlabel('Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
        
        elif viz_type == 'Heat Map':
            with rasterio.open(source_path) as src:
                data = src.read(1)
                
                ax = self.figure.add_subplot(111)
                im = ax.imshow(data, cmap='viridis', aspect='auto')
                ax.set_title(f'Heat Map - {layer.name()}')
                plt.colorbar(im, ax=ax)
    
    def plot_vector_data(self, layer, viz_type):
        """Plot vector layer data"""
        # Simplified vector plotting
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, f'Vector visualization\nfor {layer.name()}\nnot implemented', 
               ha='center', va='center')
        ax.set_title(f'{viz_type} - {layer.name()}')
