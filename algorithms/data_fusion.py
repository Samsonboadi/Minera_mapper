"""
Multi-source data fusion algorithms for geological analysis
"""

import numpy as np
import json
import os

# Optional imports with fallbacks
try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPRegressor
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import skfuzzy as fuzz
    HAS_SKFUZZY = True
except ImportError:
    HAS_SKFUZZY = False

try:
    from scipy import ndimage
    from scipy.spatial.distance import pdist, squareform
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import geopandas as gpd
    from shapely.geometry import box
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

class DataFusionProcessor:
    """Advanced multi-source geospatial data fusion"""
    
    def __init__(self):
        self.layers = {}
        self.weights = {}
        self.reference_transform = None
        self.reference_crs = None
        self.reference_shape = None
        self.fused_data = None
        self.normalization_params = {}
        
    def load_layers(self, layer_ids, weights):
        """Load and align multiple data layers"""
        from qgis.core import QgsProject
        
        if len(layer_ids) != len(weights):
            raise ValueError("Number of layers must match number of weights")
        
        project = QgsProject.instance()
        loaded_layers = {}
        layer_data = {}
        
        # Load all layers
        for i, layer_id in enumerate(layer_ids):
            layer = project.mapLayer(layer_id)
            if layer is None:
                continue
                
            layer_path = layer.source()
            weight = weights[i]
            
            try:
                with rasterio.open(layer_path) as src:
                    data = src.read(1)  # Read first band
                    profile = src.profile
                    
                    loaded_layers[layer_id] = {
                        'data': data,
                        'profile': profile,
                        'transform': src.transform,
                        'crs': src.crs,
                        'weight': weight,
                        'name': layer.name()
                    }
                    
            except Exception as e:
                print(f"Error loading layer {layer_id}: {str(e)}")
                continue
        
        if not loaded_layers:
            raise ValueError("No layers could be loaded")
        
        # Set reference layer (first layer)
        reference_layer = list(loaded_layers.values())[0]
        self.reference_transform = reference_layer['transform']
        self.reference_crs = reference_layer['crs']
        self.reference_shape = reference_layer['data'].shape
        
        # Reproject and align all layers to reference
        for layer_id, layer_info in loaded_layers.items():
            aligned_data = self._align_layer_to_reference(layer_info)
            layer_data[layer_id] = {
                'data': aligned_data,
                'weight': layer_info['weight'],
                'name': layer_info['name']
            }
        
        self.layers = layer_data
        self.weights = {layer_id: info['weight'] for layer_id, info in layer_data.items()}
        
        return True
    
    def _align_layer_to_reference(self, layer_info):
        """Align a layer to the reference coordinate system and grid"""
        if (layer_info['crs'] == self.reference_crs and 
            layer_info['transform'] == self.reference_transform and
            layer_info['data'].shape == self.reference_shape):
            return layer_info['data']
        
        # Create destination array
        aligned_data = np.empty(self.reference_shape, dtype=np.float32)
        
        # Reproject
        reproject(
            source=layer_info['data'],
            destination=aligned_data,
            src_transform=layer_info['transform'],
            src_crs=layer_info['crs'],
            dst_transform=self.reference_transform,
            dst_crs=self.reference_crs,
            resampling=Resampling.bilinear
        )
        
        return aligned_data
    
    def normalize_layers(self, method='min_max'):
        """Normalize all layers using specified method"""
        normalized_layers = {}
        
        for layer_id, layer_info in self.layers.items():
            data = layer_info['data']
            valid_mask = np.isfinite(data) & (data != 0)
            
            if not valid_mask.any():
                normalized_layers[layer_id] = layer_info
                continue
            
            valid_data = data[valid_mask]
            normalized_data = data.copy()
            
            if method == 'min_max':
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(valid_data.reshape(-1, 1)).flatten()
                normalized_data[valid_mask] = scaled_data
                
                self.normalization_params[layer_id] = {
                    'method': 'min_max',
                    'min': float(scaler.data_min_[0]),
                    'max': float(scaler.data_max_[0])
                }
                
            elif method == 'z_score':
                scaler = StandardScaler()
                scaled_data = scaler.fit_transform(valid_data.reshape(-1, 1)).flatten()
                normalized_data[valid_mask] = scaled_data
                
                self.normalization_params[layer_id] = {
                    'method': 'z_score',
                    'mean': float(scaler.mean_[0]),
                    'std': float(scaler.scale_[0])
                }
                
            elif method == 'robust':
                scaler = RobustScaler()
                scaled_data = scaler.fit_transform(valid_data.reshape(-1, 1)).flatten()
                normalized_data[valid_mask] = scaled_data
                
                self.normalization_params[layer_id] = {
                    'method': 'robust',
                    'median': float(scaler.center_[0]),
                    'scale': float(scaler.scale_[0])
                }
            
            normalized_layers[layer_id] = {
                'data': normalized_data,
                'weight': layer_info['weight'],
                'name': layer_info['name']
            }
        
        self.layers = normalized_layers
        return True
    
    def weighted_average_fusion(self):
        """Weighted average fusion of normalized layers"""
        if not self.layers:
            raise ValueError("No layers loaded for fusion")
        
        # Initialize fusion result
        fused_data = np.zeros(self.reference_shape, dtype=np.float32)
        weight_sum = np.zeros(self.reference_shape, dtype=np.float32)
        
        # Weighted sum
        for layer_id, layer_info in self.layers.items():
            data = layer_info['data']
            weight = layer_info['weight']
            
            # Create weight mask for valid pixels
            valid_mask = np.isfinite(data) & (data != 0)
            
            fused_data[valid_mask] += data[valid_mask] * weight
            weight_sum[valid_mask] += weight
        
        # Normalize by weight sum
        valid_fusion_mask = weight_sum > 0
        fused_data[valid_fusion_mask] /= weight_sum[valid_fusion_mask]
        fused_data[~valid_fusion_mask] = np.nan
        
        self.fused_data = fused_data
        return fused_data
    
    def principal_component_fusion(self, n_components=None):
        """Principal Component Analysis fusion"""
        if not self.layers:
            raise ValueError("No layers loaded for fusion")
        
        # Stack all layers
        layer_stack = []
        layer_names = []
        
        for layer_id, layer_info in self.layers.items():
            layer_stack.append(layer_info['data'].flatten())
            layer_names.append(layer_info['name'])
        
        layer_matrix = np.column_stack(layer_stack)
        
        # Remove invalid pixels
        valid_mask = np.all(np.isfinite(layer_matrix), axis=1)
        valid_data = layer_matrix[valid_mask]
        
        if len(valid_data) == 0:
            raise ValueError("No valid pixels for PCA fusion")
        
        # Apply PCA
        if n_components is None:
            n_components = min(len(self.layers), valid_data.shape[0])
        
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(valid_data)
        
        # Use first principal component as fusion result
        fused_values = pca_result[:, 0]
        
        # Map back to spatial grid
        fused_data = np.full(layer_matrix.shape[0], np.nan)
        fused_data[valid_mask] = fused_values
        fused_data = fused_data.reshape(self.reference_shape)
        
        self.fused_data = fused_data
        return fused_data
    
    def fuzzy_logic_fusion(self):
        """Fuzzy logic-based data fusion"""
        if not self.layers:
            raise ValueError("No layers loaded for fusion")
        
        # Initialize fuzzy membership functions
        fused_data = np.zeros(self.reference_shape, dtype=np.float32)
        
        # For each pixel, calculate fuzzy membership
        for i in range(self.reference_shape[0]):
            for j in range(self.reference_shape[1]):
                pixel_values = []
                pixel_weights = []
                
                for layer_id, layer_info in self.layers.items():
                    value = layer_info['data'][i, j]
                    weight = layer_info['weight']
                    
                    if np.isfinite(value):
                        pixel_values.append(value)
                        pixel_weights.append(weight)
                
                if pixel_values:
                    # Calculate fuzzy membership based on proximity to ideal values
                    pixel_values = np.array(pixel_values)
                    pixel_weights = np.array(pixel_weights)
                    
                    # Simple fuzzy aggregation using weighted mean
                    fuzzy_result = np.sum(pixel_values * pixel_weights) / np.sum(pixel_weights)
                    fused_data[i, j] = fuzzy_result
                else:
                    fused_data[i, j] = np.nan
        
        self.fused_data = fused_data
        return fused_data
    
    def neural_network_fusion(self, hidden_layers=(100, 50)):
        """Neural network-based fusion"""
        if not self.layers:
            raise ValueError("No layers loaded for fusion")
        
        # Prepare training data
        layer_stack = []
        weights_stack = []
        
        for layer_id, layer_info in self.layers.items():
            layer_stack.append(layer_info['data'].flatten())
            weights_stack.append(layer_info['weight'])
        
        X = np.column_stack(layer_stack)
        weights_array = np.array(weights_stack)
        
        # Remove invalid pixels
        valid_mask = np.all(np.isfinite(X), axis=1)
        X_valid = X[valid_mask]
        
        if len(X_valid) == 0:
            raise ValueError("No valid pixels for neural network fusion")
        
        # Create target values (weighted average as ground truth)
        y_valid = np.average(X_valid, axis=1, weights=weights_array)
        
        # Train neural network
        nn = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        nn.fit(X_valid, y_valid)
        
        # Predict for all pixels
        fused_values = np.full(X.shape[0], np.nan)
        fused_values[valid_mask] = nn.predict(X_valid)
        
        fused_data = fused_values.reshape(self.reference_shape)
        
        self.fused_data = fused_data
        return fused_data
    
    def bayesian_fusion(self):
        """Bayesian data fusion"""
        if not self.layers:
            raise ValueError("No layers loaded for fusion")
        
        # Simple Bayesian fusion using weighted posterior
        fused_data = np.zeros(self.reference_shape, dtype=np.float32)
        
        # Calculate prior (uniform)
        prior = 1.0 / len(self.layers)
        
        for i in range(self.reference_shape[0]):
            for j in range(self.reference_shape[1]):
                posteriors = []
                
                for layer_id, layer_info in self.layers.items():
                    value = layer_info['data'][i, j]
                    weight = layer_info['weight']
                    
                    if np.isfinite(value):
                        # Simple likelihood based on value and weight
                        likelihood = weight * np.exp(-0.5 * (value - 0.5)**2)
                        posterior = likelihood * prior
                        posteriors.append(posterior * value)
                
                if posteriors:
                    fused_data[i, j] = np.sum(posteriors) / len(posteriors)
                else:
                    fused_data[i, j] = np.nan
        
        self.fused_data = fused_data
        return fused_data
    
    def create_preview(self, method='weighted_average'):
        """Create a quick preview of fusion result"""
        if method == 'weighted_average':
            return self.weighted_average_fusion()
        elif method == 'principal_component':
            return self.principal_component_fusion()
        elif method == 'fuzzy_logic':
            return self.fuzzy_logic_fusion()
        elif method == 'neural_network':
            return self.neural_network_fusion()
        elif method == 'bayesian':
            return self.bayesian_fusion()
        else:
            return self.weighted_average_fusion()
    
    def calculate_fusion_quality_metrics(self, fused_data):
        """Calculate quality metrics for fusion result"""
        if not self.layers or fused_data is None:
            return {}
        
        metrics = {}
        
        # Data coverage
        valid_pixels = np.isfinite(fused_data).sum()
        total_pixels = fused_data.size
        metrics['coverage_percent'] = (valid_pixels / total_pixels) * 100
        
        # Value statistics
        valid_data = fused_data[np.isfinite(fused_data)]
        if len(valid_data) > 0:
            metrics['mean'] = float(np.mean(valid_data))
            metrics['std'] = float(np.std(valid_data))
            metrics['min'] = float(np.min(valid_data))
            metrics['max'] = float(np.max(valid_data))
            metrics['range'] = metrics['max'] - metrics['min']
        
        # Correlation with individual layers
        correlations = {}
        for layer_id, layer_info in self.layers.items():
            layer_data = layer_info['data']
            
            # Find common valid pixels
            common_mask = np.isfinite(fused_data) & np.isfinite(layer_data)
            
            if common_mask.sum() > 10:  # Need minimum pixels for correlation
                corr = np.corrcoef(
                    fused_data[common_mask], 
                    layer_data[common_mask]
                )[0, 1]
                if np.isfinite(corr):
                    correlations[layer_info['name']] = float(corr)
        
        metrics['layer_correlations'] = correlations
        
        return metrics
    
    def save_result(self, fused_data, output_path):
        """Save fusion result to GeoTIFF"""
        if self.reference_transform is None or self.reference_crs is None:
            raise ValueError("No reference spatial information available")
        
        # Create output profile
        profile = {
            'driver': 'GTiff',
            'dtype': 'float32',
            'nodata': np.nan,
            'width': self.reference_shape[1],
            'height': self.reference_shape[0],
            'count': 1,
            'crs': self.reference_crs,
            'transform': self.reference_transform,
            'compress': 'lzw'
        }
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(fused_data.astype(np.float32), 1)
        
        # Save metadata
        metadata = {
            'fusion_parameters': {
                'layers': [info['name'] for info in self.layers.values()],
                'weights': list(self.weights.values()),
                'normalization_params': self.normalization_params
            },
            'quality_metrics': self.calculate_fusion_quality_metrics(fused_data),
            'spatial_reference': {
                'crs': str(self.reference_crs),
                'transform': list(self.reference_transform)[:6],
                'shape': self.reference_shape
            }
        }
        
        metadata_path = os.path.splitext(output_path)[0] + '_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return output_path
