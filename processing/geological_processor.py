"""
Geological data processor for structural analysis and geological mapping
"""

import os
import numpy as np
import rasterio
import geopandas as gpd
from rasterio.features import rasterize, shapes
from rasterio.transform import from_bounds
from scipy import ndimage
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from skimage import measure, morphology, filters
from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox, QProgressDialog
from qgis.PyQt.QtCore import Qt
from qgis.core import (QgsRasterLayer, QgsVectorLayer, QgsProject, QgsMessageLog, 
                       Qgis, QgsProcessingFeedback, QgsWkbTypes)
import json
import tempfile

class GeologicalProcessor:
    """Process geological and structural data for mineral exploration"""
    
    def __init__(self, iface):
        self.iface = iface
        self.geological_units = {}
        self.structural_features = {}
        self.processing_results = {}
        
    def process_geological_map(self, vector_path=None, raster_path=None):
        """Process geological map data"""
        if not vector_path and not raster_path:
            # Let user select input
            file_path, file_type = self.select_geological_data()
            if not file_path:
                return
            
            if file_type == 'vector':
                vector_path = file_path
            else:
                raster_path = file_path
        
        try:
            progress = QProgressDialog("Processing geological data...", "Cancel", 0, 100)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            progress.setValue(10)
            
            if vector_path:
                result = self.process_vector_geology(vector_path, progress)
            else:
                result = self.process_raster_geology(raster_path, progress)
            
            progress.setValue(100)
            progress.close()
            
            QMessageBox.information(
                self.iface.mainWindow(),
                "Success",
                "Geological data processed successfully!"
            )
            
            return result
            
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            QMessageBox.critical(
                self.iface.mainWindow(),
                "Error",
                f"Failed to process geological data: {str(e)}"
            )
            QgsMessageLog.logMessage(f"Geological processing error: {str(e)}", 'Mineral Prospectivity', Qgis.Critical)
    
    def select_geological_data(self):
        """Select geological data file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.iface.mainWindow(),
            "Select Geological Data",
            "",
            "Vector files (*.shp *.gpkg *.geojson);;Raster files (*.tif *.tiff);;All files (*)"
        )
        
        if not file_path:
            return None, None
        
        # Determine file type
        if file_path.lower().endswith(('.shp', '.gpkg', '.geojson')):
            return file_path, 'vector'
        else:
            return file_path, 'raster'
    
    def process_vector_geology(self, vector_path, progress=None):
        """Process vector geological data"""
        if progress:
            progress.setLabelText("Loading vector geological data...")
            progress.setValue(20)
        
        # Load vector data
        gdf = gpd.read_file(vector_path)
        
        # Analyze geological units
        self.analyze_geological_units(gdf)
        
        if progress:
            progress.setLabelText("Creating geological unit raster...")
            progress.setValue(40)
        
        # Rasterize geological units
        geology_raster = self.rasterize_geological_units(gdf)
        
        if progress:
            progress.setLabelText("Calculating geological favorability...")
            progress.setValue(60)
        
        # Calculate geological favorability
        favorability_map = self.calculate_geological_favorability(geology_raster)
        
        if progress:
            progress.setLabelText("Analyzing structural features...")
            progress.setValue(80)
        
        # Extract structural features if present
        structural_features = self.extract_structural_features(gdf)
        
        results = {
            'geological_raster': geology_raster,
            'favorability_map': favorability_map,
            'structural_features': structural_features,
            'geological_units': self.geological_units
        }
        
        # Add to QGIS
        self.add_geological_layers_to_qgis(results)
        
        return results
    
    def process_raster_geology(self, raster_path, progress=None):
        """Process raster geological data"""
        if progress:
            progress.setLabelText("Loading raster geological data...")
            progress.setValue(20)
        
        with rasterio.open(raster_path) as src:
            geology_data = src.read(1)
            profile = src.profile
        
        if progress:
            progress.setLabelText("Analyzing geological units...")
            progress.setValue(40)
        
        # Analyze unique geological units
        unique_units = np.unique(geology_data[geology_data != src.nodata])
        self.geological_units = {int(unit): f"Unit_{int(unit)}" for unit in unique_units}
        
        if progress:
            progress.setLabelText("Calculating geological favorability...")
            progress.setValue(60)
        
        # Calculate favorability based on unit values
        favorability_map = self.calculate_raster_favorability(geology_data, profile)
        
        if progress:
            progress.setLabelText("Detecting structural patterns...")
            progress.setValue(80)
        
        # Detect structural patterns
        structural_analysis = self.analyze_raster_structure(geology_data, profile)
        
        results = {
            'geological_raster': raster_path,
            'favorability_map': favorability_map,
            'structural_analysis': structural_analysis,
            'geological_units': self.geological_units
        }
        
        # Add to QGIS
        self.add_geological_layers_to_qgis(results)
        
        return results
    
    def analyze_geological_units(self, gdf):
        """Analyze geological units and assign favorability scores"""
        # Define favorability scores for different rock types
        rock_type_favorability = {
            # High favorability
            'granite': 0.9,
            'granodiorite': 0.85,
            'quartz': 0.8,
            'greenstone': 0.9,
            'schist': 0.7,
            'gneiss': 0.65,
            
            # Medium favorability
            'volcanic': 0.6,
            'andesite': 0.55,
            'basalt': 0.4,
            'rhyolite': 0.7,
            'tuff': 0.5,
            
            # Lower favorability
            'limestone': 0.3,
            'sandstone': 0.2,
            'shale': 0.15,
            'mudstone': 0.1,
            'conglomerate': 0.25,
            
            # Very low favorability
            'alluvium': 0.05,
            'clay': 0.05
        }
        
        # Try to find geological unit column
        unit_columns = [col for col in gdf.columns if any(keyword in col.lower() 
                       for keyword in ['geol', 'lithol', 'rock', 'unit', 'formation'])]
        
        if not unit_columns:
            # Use first string column
            string_columns = gdf.select_dtypes(include=['object']).columns
            if len(string_columns) > 0:
                unit_columns = [string_columns[0]]
        
        if unit_columns:
            unit_column = unit_columns[0]
            unique_units = gdf[unit_column].unique()
            
            for i, unit in enumerate(unique_units):
                # Assign favorability based on rock type keywords
                favorability = 0.5  # Default
                unit_lower = str(unit).lower()
                
                for rock_type, score in rock_type_favorability.items():
                    if rock_type in unit_lower:
                        favorability = score
                        break
                
                self.geological_units[i] = {
                    'name': str(unit),
                    'favorability': favorability,
                    'id': i
                }
            
            # Add unit ID to geodataframe
            unit_map = {unit: i for i, unit in enumerate(unique_units)}
            gdf['unit_id'] = gdf[unit_column].map(unit_map)
        else:
            # No unit information, assign default
            self.geological_units[0] = {
                'name': 'Unknown',
                'favorability': 0.5,
                'id': 0
            }
            gdf['unit_id'] = 0
    
    def rasterize_geological_units(self, gdf):
        """Rasterize geological units"""
        # Get bounds and create grid
        bounds = gdf.total_bounds
        width, height = 1000, 1000  # Default grid size
        
        transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], width, height)
        
        # Rasterize using unit IDs
        shapes_gen = ((geom, unit_id) for geom, unit_id in zip(gdf.geometry, gdf['unit_id']))
        
        geology_raster = rasterize(
            shapes_gen,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype=np.uint8
        )
        
        # Save raster
        temp_dir = tempfile.mkdtemp(prefix='geology_')
        raster_path = os.path.join(temp_dir, 'geological_units.tif')
        
        profile = {
            'driver': 'GTiff',
            'dtype': 'uint8',
            'nodata': 0,
            'width': width,
            'height': height,
            'count': 1,
            'crs': gdf.crs,
            'transform': transform,
            'compress': 'lzw'
        }
        
        with rasterio.open(raster_path, 'w', **profile) as dst:
            dst.write(geology_raster, 1)
        
        return raster_path
    
    def calculate_geological_favorability(self, geology_raster_path):
        """Calculate geological favorability map"""
        temp_dir = tempfile.mkdtemp(prefix='favorability_')
        output_path = os.path.join(temp_dir, 'geological_favorability.tif')
        
        with rasterio.open(geology_raster_path) as src:
            geology_data = src.read(1)
            profile = src.profile
        
        # Create favorability map
        favorability = np.zeros_like(geology_data, dtype=np.float32)
        
        for unit_id, unit_info in self.geological_units.items():
            mask = geology_data == unit_id
            favorability[mask] = unit_info['favorability']
        
        # Save favorability map
        profile.update({
            'dtype': 'float32',
            'nodata': -9999
        })
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(favorability, 1)
        
        return output_path
    
    def calculate_raster_favorability(self, geology_data, profile):
        """Calculate favorability for raster geological data"""
        temp_dir = tempfile.mkdtemp(prefix='favorability_')
        output_path = os.path.join(temp_dir, 'geological_favorability.tif')
        
        # Simple favorability based on unit values
        # Higher values generally indicate more competent rocks
        favorability = np.zeros_like(geology_data, dtype=np.float32)
        
        valid_mask = geology_data != profile.get('nodata', 0)
        if valid_mask.any():
            valid_data = geology_data[valid_mask]
            
            # Normalize to 0-1 range
            min_val, max_val = np.min(valid_data), np.max(valid_data)
            if max_val > min_val:
                normalized = (geology_data.astype(np.float32) - min_val) / (max_val - min_val)
                favorability[valid_mask] = normalized[valid_mask]
            else:
                favorability[valid_mask] = 0.5
        
        # Save favorability map
        output_profile = profile.copy()
        output_profile.update({
            'dtype': 'float32',
            'nodata': -9999
        })
        
        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(favorability, 1)
        
        return output_path
    
    def extract_structural_features(self, gdf):
        """Extract structural features from vector data"""
        structural_features = {}
        
        # Look for linear features (faults, fractures, lineaments)
        linear_features = gdf[gdf.geometry.type.isin(['LineString', 'MultiLineString'])]
        
        if not linear_features.empty:
            # Calculate lineament density
            density_map = self.calculate_lineament_density(linear_features)
            structural_features['lineament_density'] = density_map
            
            # Analyze lineament orientations
            orientations = self.analyze_lineament_orientations(linear_features)
            structural_features['orientations'] = orientations
        
        return structural_features
    
    def analyze_raster_structure(self, geology_data, profile):
        """Analyze structural patterns in raster geological data"""
        structural_analysis = {}
        
        # Edge detection for geological boundaries
        edges = self.detect_geological_boundaries(geology_data)
        
        # Save edge map
        temp_dir = tempfile.mkdtemp(prefix='structure_')
        edge_path = os.path.join(temp_dir, 'geological_boundaries.tif')
        
        edge_profile = profile.copy()
        edge_profile.update({
            'dtype': 'uint8',
            'nodata': 0
        })
        
        with rasterio.open(edge_path, 'w', **edge_profile) as dst:
            dst.write(edges.astype(np.uint8), 1)
        
        structural_analysis['boundaries'] = edge_path
        
        # Calculate structural complexity
        complexity_map = self.calculate_structural_complexity(geology_data, profile)
        structural_analysis['complexity'] = complexity_map
        
        return structural_analysis
    
    def detect_geological_boundaries(self, geology_data):
        """Detect geological unit boundaries"""
        # Use Sobel edge detection
        sobel_x = ndimage.sobel(geology_data, axis=1)
        sobel_y = ndimage.sobel(geology_data, axis=0)
        edges = np.hypot(sobel_x, sobel_y)
        
        # Threshold to binary
        threshold = np.percentile(edges[edges > 0], 75)
        binary_edges = edges > threshold
        
        # Clean up edges
        binary_edges = morphology.binary_closing(binary_edges, morphology.disk(2))
        binary_edges = morphology.skeletonize(binary_edges)
        
        return binary_edges.astype(np.uint8) * 255
    
    def calculate_structural_complexity(self, geology_data, profile):
        """Calculate structural complexity map"""
        temp_dir = tempfile.mkdtemp(prefix='complexity_')
        output_path = os.path.join(temp_dir, 'structural_complexity.tif')
        
        # Calculate local variation using a moving window
        window_size = 5
        complexity = np.zeros_like(geology_data, dtype=np.float32)
        
        for i in range(window_size//2, geology_data.shape[0] - window_size//2):
            for j in range(window_size//2, geology_data.shape[1] - window_size//2):
                window = geology_data[i-window_size//2:i+window_size//2+1, 
                                   j-window_size//2:j+window_size//2+1]
                
                # Count unique units in window
                unique_units = len(np.unique(window))
                complexity[i, j] = unique_units / (window_size * window_size)
        
        # Save complexity map
        output_profile = profile.copy()
        output_profile.update({
            'dtype': 'float32',
            'nodata': -9999
        })
        
        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(complexity, 1)
        
        return output_path
    
    def calculate_lineament_density(self, linear_features):
        """Calculate lineament density map"""
        # Create a grid for density calculation
        bounds = linear_features.total_bounds
        grid_size = 100  # 100x100 grid
        
        # Create density grid
        x_edges = np.linspace(bounds[0], bounds[2], grid_size + 1)
        y_edges = np.linspace(bounds[1], bounds[3], grid_size + 1)
        
        density_grid = np.zeros((grid_size, grid_size))
        
        # Calculate total length of lineaments in each grid cell
        for i in range(grid_size):
            for j in range(grid_size):
                cell_bounds = [x_edges[j], y_edges[i], x_edges[j+1], y_edges[i+1]]
                
                # Find lineaments intersecting this cell
                cell_geom = gpd.GeoDataFrame({'geometry': [
                    gpd.geometry.box(*cell_bounds)
                ]}, crs=linear_features.crs)
                
                intersecting = gpd.overlay(linear_features, cell_geom, how='intersection')
                
                if not intersecting.empty:
                    total_length = intersecting.geometry.length.sum()
                    cell_area = (x_edges[j+1] - x_edges[j]) * (y_edges[i+1] - y_edges[i])
                    density_grid[i, j] = total_length / cell_area
        
        # Convert to raster
        temp_dir = tempfile.mkdtemp(prefix='lineament_')
        density_path = os.path.join(temp_dir, 'lineament_density.tif')
        
        transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], 
                              grid_size, grid_size)
        
        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': -9999,
            'width': grid_size,
            'height': grid_size,
            'count': 1,
            'crs': linear_features.crs,
            'transform': transform,
            'compress': 'lzw'
        }
        
        with rasterio.open(density_path, 'w', **profile) as dst:
            dst.write(density_grid.astype(np.float32), 1)
        
        return density_path
    
    def analyze_lineament_orientations(self, linear_features):
        """Analyze lineament orientations"""
        orientations = []
        
        for _, feature in linear_features.iterrows():
            geom = feature.geometry
            
            if geom.geom_type == 'LineString':
                coords = list(geom.coords)
                if len(coords) >= 2:
                    # Calculate azimuth
                    dx = coords[-1][0] - coords[0][0]
                    dy = coords[-1][1] - coords[0][1]
                    azimuth = np.degrees(np.arctan2(dx, dy))
                    if azimuth < 0:
                        azimuth += 360
                    orientations.append(azimuth)
            
            elif geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    coords = list(line.coords)
                    if len(coords) >= 2:
                        dx = coords[-1][0] - coords[0][0]
                        dy = coords[-1][1] - coords[0][1]
                        azimuth = np.degrees(np.arctan2(dx, dy))
                        if azimuth < 0:
                            azimuth += 360
                        orientations.append(azimuth)
        
        # Analyze orientation distribution
        orientation_stats = {
            'orientations': orientations,
            'mean_orientation': np.mean(orientations) if orientations else 0,
            'dominant_directions': self.find_dominant_directions(orientations)
        }
        
        return orientation_stats
    
    def find_dominant_directions(self, orientations, bin_size=10):
        """Find dominant structural directions"""
        if not orientations:
            return []
        
        # Create orientation histogram
        bins = np.arange(0, 360 + bin_size, bin_size)
        hist, _ = np.histogram(orientations, bins=bins)
        
        # Find peaks
        peak_indices = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1]:
                peak_indices.append(i)
        
        # Convert back to angles
        dominant_directions = []
        for idx in peak_indices:
            angle = bins[idx] + bin_size / 2
            strength = hist[idx] / len(orientations)
            if strength > 0.1:  # Only significant peaks
                dominant_directions.append({
                    'angle': angle,
                    'strength': strength
                })
        
        return sorted(dominant_directions, key=lambda x: x['strength'], reverse=True)
    
    def add_geological_layers_to_qgis(self, results):
        """Add geological processing results to QGIS"""
        project = QgsProject.instance()
        
        # Create geology group
        geology_group = project.layerTreeRoot().addGroup("Geological Analysis")
        
        # Add geological raster
        if 'geological_raster' in results and results['geological_raster']:
            layer = QgsRasterLayer(results['geological_raster'], "Geological Units")
            if layer.isValid():
                project.addMapLayer(layer, False)
                geology_group.addLayer(layer)
        
        # Add favorability map
        if 'favorability_map' in results:
            layer = QgsRasterLayer(results['favorability_map'], "Geological Favorability")
            if layer.isValid():
                project.addMapLayer(layer, False)
                geology_group.addLayer(layer)
        
        # Add structural analysis results
        if 'structural_analysis' in results:
            struct_group = geology_group.addGroup("Structural Analysis")
            
            for name, path in results['structural_analysis'].items():
                if os.path.exists(path):
                    layer = QgsRasterLayer(path, f"Structural {name.title()}")
                    if layer.isValid():
                        project.addMapLayer(layer, False)
                        struct_group.addLayer(layer)
        
        # Add lineament density if available
        if 'structural_features' in results and 'lineament_density' in results['structural_features']:
            layer = QgsRasterLayer(results['structural_features']['lineament_density'], 
                                 "Lineament Density")
            if layer.isValid():
                project.addMapLayer(layer, False)
                geology_group.addLayer(layer)
        
        # Refresh canvas
        self.iface.mapCanvas().refresh()
    
    def process_magnetic_data(self, magnetic_raster_path):
        """Process magnetic anomaly data"""
        temp_dir = tempfile.mkdtemp(prefix='magnetic_')
        
        with rasterio.open(magnetic_raster_path) as src:
            magnetic_data = src.read(1)
            profile = src.profile
        
        # Calculate magnetic derivatives
        results = {}
        
        # First vertical derivative
        vertical_derivative = self.calculate_vertical_derivative(magnetic_data)
        vd_path = os.path.join(temp_dir, 'magnetic_vertical_derivative.tif')
        
        with rasterio.open(vd_path, 'w', **profile) as dst:
            dst.write(vertical_derivative.astype(np.float32), 1)
        results['vertical_derivative'] = vd_path
        
        # Total horizontal derivative
        horizontal_derivative = self.calculate_horizontal_derivative(magnetic_data)
        hd_path = os.path.join(temp_dir, 'magnetic_horizontal_derivative.tif')
        
        with rasterio.open(hd_path, 'w', **profile) as dst:
            dst.write(horizontal_derivative.astype(np.float32), 1)
        results['horizontal_derivative'] = hd_path
        
        # Analytic signal
        analytic_signal = np.sqrt(vertical_derivative**2 + horizontal_derivative**2)
        as_path = os.path.join(temp_dir, 'magnetic_analytic_signal.tif')
        
        with rasterio.open(as_path, 'w', **profile) as dst:
            dst.write(analytic_signal.astype(np.float32), 1)
        results['analytic_signal'] = as_path
        
        return results
    
    def calculate_vertical_derivative(self, data):
        """Calculate first vertical derivative of magnetic data"""
        return ndimage.sobel(data, axis=0)
    
    def calculate_horizontal_derivative(self, data):
        """Calculate total horizontal derivative"""
        dx = ndimage.sobel(data, axis=1)
        dy = ndimage.sobel(data, axis=0)
        return np.sqrt(dx**2 + dy**2)
    
    def save_geological_metadata(self, output_dir, results):
        """Save geological processing metadata"""
        metadata = {
            'processing_date': str(np.datetime64('now')),
            'geological_units': self.geological_units,
            'processing_steps': [
                'Geological unit analysis',
                'Favorability calculation',
                'Structural pattern analysis',
                'Boundary detection'
            ],
            'results': {key: path for key, path in results.items() if isinstance(path, str)}
        }
        
        metadata_path = os.path.join(output_dir, 'geological_processing_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return metadata_path
