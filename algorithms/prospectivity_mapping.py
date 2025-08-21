"""
Prospectivity mapping algorithms for mineral exploration
"""

import numpy as np
import rasterio
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy import ndimage
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import json
import os

class ProspectivityMapper:
    """Advanced prospectivity mapping for mineral exploration"""
    
    def __init__(self):
        self.input_layers = {}
        self.weights = {}
        self.prospectivity_map = None
        self.confidence_map = None
        self.reference_profile = None
        self.fuzzy_system = None
        
    def load_layers(self, layer_ids, weights):
        """Load and prepare input layers for prospectivity mapping"""
        from qgis.core import QgsProject
        
        if len(layer_ids) != len(weights):
            raise ValueError("Number of layers must match number of weights")
        
        project = QgsProject.instance()
        loaded_layers = {}
        
        # Reference properties from first layer
        reference_layer = None
        reference_transform = None
        reference_crs = None
        reference_shape = None
        
        for i, layer_id in enumerate(layer_ids):
            layer = project.mapLayer(layer_id)
            if layer is None:
                continue
                
            layer_path = layer.source()
            weight = weights[i]
            
            try:
                with rasterio.open(layer_path) as src:
                    data = src.read(1)  # Read first band
                    
                    if reference_layer is None:
                        reference_layer = layer
                        reference_transform = src.transform
                        reference_crs = src.crs
                        reference_shape = data.shape
                        self.reference_profile = src.profile
                    
                    # Ensure all layers have the same dimensions
                    if data.shape != reference_shape:
                        # Resample to reference shape if needed
                        from rasterio.warp import reproject, Resampling
                        aligned_data = np.empty(reference_shape, dtype=np.float32)
                        reproject(
                            source=data,
                            destination=aligned_data,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=reference_transform,
                            dst_crs=reference_crs,
                            resampling=Resampling.bilinear
                        )
                        data = aligned_data
                    
                    loaded_layers[layer_id] = {
                        'data': data,
                        'weight': weight,
                        'name': layer.name(),
                        'layer_type': self._identify_layer_type(layer.name())
                    }
                    
            except Exception as e:
                print(f"Error loading layer {layer_id}: {str(e)}")
                continue
        
        if not loaded_layers:
            raise ValueError("No layers could be loaded")
        
        self.input_layers = loaded_layers
        self.weights = {layer_id: info['weight'] for layer_id, info in loaded_layers.items()}
        
        return True
    
    def _identify_layer_type(self, layer_name):
        """Identify the type of geological layer"""
        name_lower = layer_name.lower()
        
        if any(word in name_lower for word in ['geology', 'geological', 'lithology']):
            return 'geological'
        elif any(word in name_lower for word in ['magnetic', 'aeromagnetic', 'mag']):
            return 'magnetic'
        elif any(word in name_lower for word in ['radiometric', 'gamma', 'uranium', 'thorium', 'potassium']):
            return 'radiometric'
        elif any(word in name_lower for word in ['mineral', 'alteration', 'spectral']):
            return 'mineral'
        elif any(word in name_lower for word in ['topographic', 'elevation', 'dem', 'slope']):
            return 'topographic'
        elif any(word in name_lower for word in ['structure', 'lineament', 'fault']):
            return 'structural'
        else:
            return 'unknown'
    
    def normalize_layers(self, method='percentile'):
        """Normalize input layers for consistent scaling"""
        normalized_layers = {}
        
        for layer_id, layer_info in self.input_layers.items():
            data = layer_info['data'].copy()
            valid_mask = np.isfinite(data) & (data != 0)
            
            if not valid_mask.any():
                normalized_layers[layer_id] = layer_info
                continue
            
            valid_data = data[valid_mask]
            
            if method == 'percentile':
                # Normalize to 0-1 using 1st and 99th percentiles
                p1, p99 = np.percentile(valid_data, [1, 99])
                if p99 > p1:
                    data[valid_mask] = np.clip((valid_data - p1) / (p99 - p1), 0, 1)
                else:
                    data[valid_mask] = 0.5
                    
            elif method == 'minmax':
                # Standard min-max normalization
                min_val, max_val = np.min(valid_data), np.max(valid_data)
                if max_val > min_val:
                    data[valid_mask] = (valid_data - min_val) / (max_val - min_val)
                else:
                    data[valid_mask] = 0.5
                    
            elif method == 'zscore':
                # Z-score normalization
                mean_val, std_val = np.mean(valid_data), np.std(valid_data)
                if std_val > 0:
                    data[valid_mask] = (valid_data - mean_val) / std_val
                    # Convert to 0-1 range using sigmoid
                    data[valid_mask] = 1 / (1 + np.exp(-data[valid_mask]))
                else:
                    data[valid_mask] = 0.5
            
            normalized_layers[layer_id] = {
                'data': data,
                'weight': layer_info['weight'],
                'name': layer_info['name'],
                'layer_type': layer_info['layer_type']
            }
        
        self.input_layers = normalized_layers
        return True
    
    def weighted_overlay_analysis(self):
        """Traditional weighted overlay analysis"""
        if not self.input_layers:
            raise ValueError("No input layers loaded")
        
        # Initialize prospectivity map
        shape = list(self.input_layers.values())[0]['data'].shape
        prospectivity = np.zeros(shape, dtype=np.float32)
        weight_sum = np.zeros(shape, dtype=np.float32)
        
        # Weighted sum
        for layer_id, layer_info in self.input_layers.items():
            data = layer_info['data']
            weight = layer_info['weight']
            
            valid_mask = np.isfinite(data)
            prospectivity[valid_mask] += data[valid_mask] * weight
            weight_sum[valid_mask] += weight
        
        # Normalize by total weights
        valid_mask = weight_sum > 0
        prospectivity[valid_mask] /= weight_sum[valid_mask]
        prospectivity[~valid_mask] = np.nan
        
        self.prospectivity_map = prospectivity
        return prospectivity
    
    def fuzzy_logic_analysis(self):
        """Fuzzy logic-based prospectivity analysis"""
        if not self.input_layers:
            raise ValueError("No input layers loaded")
        
        # Create fuzzy system
        self._setup_fuzzy_system()
        
        shape = list(self.input_layers.values())[0]['data'].shape
        prospectivity = np.zeros(shape, dtype=np.float32)
        
        # Process each pixel through fuzzy system
        layer_names = list(self.input_layers.keys())
        layer_data = [self.input_layers[lid]['data'] for lid in layer_names]
        
        for i in range(shape[0]):
            for j in range(shape[1]):
                pixel_values = []
                valid_pixel = True
                
                for data in layer_data:
                    value = data[i, j]
                    if np.isfinite(value):
                        pixel_values.append(value)
                    else:
                        valid_pixel = False
                        break
                
                if valid_pixel and len(pixel_values) == len(layer_data):
                    fuzzy_result = self._evaluate_fuzzy_pixel(pixel_values)
                    prospectivity[i, j] = fuzzy_result
                else:
                    prospectivity[i, j] = np.nan
        
        self.prospectivity_map = prospectivity
        return prospectivity
    
    def _setup_fuzzy_system(self):
        """Setup fuzzy inference system"""
        # Create antecedents and consequent
        geological = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'geological')
        magnetic = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'magnetic')
        radiometric = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'radiometric')
        prospectivity = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'prospectivity')
        
        # Auto-membership functions
        geological.automf(3, names=['low', 'medium', 'high'])
        magnetic.automf(3, names=['low', 'medium', 'high'])
        radiometric.automf(3, names=['low', 'medium', 'high'])
        prospectivity.automf(5, names=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        # Define rules
        rule1 = ctrl.Rule(geological['high'] & magnetic['high'] & radiometric['high'], 
                         prospectivity['very_high'])
        rule2 = ctrl.Rule(geological['high'] & magnetic['high'], 
                         prospectivity['high'])
        rule3 = ctrl.Rule(geological['medium'] & magnetic['medium'], 
                         prospectivity['medium'])
        rule4 = ctrl.Rule(geological['low'] | magnetic['low'], 
                         prospectivity['low'])
        rule5 = ctrl.Rule(geological['low'] & magnetic['low'] & radiometric['low'], 
                         prospectivity['very_low'])
        
        # Control system
        prospectivity_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
        self.fuzzy_system = ctrl.ControlSystemSimulation(prospectivity_ctrl)
    
    def _evaluate_fuzzy_pixel(self, pixel_values):
        """Evaluate a single pixel through fuzzy system"""
        try:
            if len(pixel_values) >= 3:
                self.fuzzy_system.input['geological'] = pixel_values[0]
                self.fuzzy_system.input['magnetic'] = pixel_values[1]
                self.fuzzy_system.input['radiometric'] = pixel_values[2]
            else:
                # Handle cases with fewer inputs
                self.fuzzy_system.input['geological'] = pixel_values[0] if len(pixel_values) > 0 else 0.5
                self.fuzzy_system.input['magnetic'] = pixel_values[1] if len(pixel_values) > 1 else 0.5
                self.fuzzy_system.input['radiometric'] = 0.5
            
            self.fuzzy_system.compute()
            return self.fuzzy_system.output['prospectivity']
        except:
            return 0.5  # Default neutral value
    
    def analytic_hierarchy_process(self, pairwise_matrix=None):
        """Analytic Hierarchy Process for weight determination"""
        if pairwise_matrix is None:
            # Default pairwise comparison matrix
            n = len(self.input_layers)
            pairwise_matrix = np.eye(n)
            
            # Simple default: geological > magnetic > radiometric > others
            layer_types = [info['layer_type'] for info in self.input_layers.values()]
            priorities = {'geological': 3, 'magnetic': 2, 'radiometric': 1.5, 'mineral': 2.5}
            
            for i, type1 in enumerate(layer_types):
                for j, type2 in enumerate(layer_types):
                    if i != j:
                        priority1 = priorities.get(type1, 1)
                        priority2 = priorities.get(type2, 1)
                        pairwise_matrix[i, j] = priority1 / priority2
        
        # Calculate eigenvector (priority weights)
        eigenvals, eigenvecs = np.linalg.eig(pairwise_matrix)
        max_eigenval_idx = np.argmax(eigenvals.real)
        priority_vector = eigenvecs[:, max_eigenval_idx].real
        priority_vector = priority_vector / np.sum(priority_vector)
        
        # Update weights
        layer_ids = list(self.input_layers.keys())
        for i, layer_id in enumerate(layer_ids):
            self.weights[layer_id] = priority_vector[i]
            self.input_layers[layer_id]['weight'] = priority_vector[i]
        
        # Calculate consistency ratio
        lambda_max = eigenvals[max_eigenval_idx].real
        n = len(pairwise_matrix)
        ci = (lambda_max - n) / (n - 1) if n > 1 else 0
        ri_values = {3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
        ri = ri_values.get(n, 1.0)
        cr = ci / ri if ri > 0 else 0
        
        return priority_vector, cr
    
    def neural_network_prospectivity(self, training_data=None):
        """Neural network-based prospectivity mapping"""
        if not self.input_layers:
            raise ValueError("No input layers loaded")
        
        # Prepare input data
        layer_stack = []
        for layer_info in self.input_layers.values():
            layer_stack.append(layer_info['data'].flatten())
        
        X = np.column_stack(layer_stack)
        valid_mask = np.all(np.isfinite(X), axis=1)
        X_valid = X[valid_mask]
        
        if len(X_valid) == 0:
            raise ValueError("No valid pixels for neural network analysis")
        
        # Create synthetic training data if none provided
        if training_data is None:
            # Use weighted overlay as ground truth for training
            weighted_result = self.weighted_overlay_analysis()
            y_valid = weighted_result.flatten()[valid_mask]
        else:
            y_valid = training_data[valid_mask]
        
        # Train neural network
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)
        
        nn = MLPRegressor(
            hidden_layer_sizes=(100, 50, 25),
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.2
        )
        
        nn.fit(X_scaled, y_valid)
        
        # Predict prospectivity
        prospectivity_values = np.full(X.shape[0], np.nan)
        X_all_scaled = scaler.transform(X[valid_mask])
        prospectivity_values[valid_mask] = nn.predict(X_all_scaled)
        
        shape = list(self.input_layers.values())[0]['data'].shape
        prospectivity = prospectivity_values.reshape(shape)
        
        self.prospectivity_map = prospectivity
        return prospectivity
    
    def evidence_weights_modeling(self, training_points=None):
        """Weights of Evidence modeling"""
        if not self.input_layers:
            raise ValueError("No input layers loaded")
        
        shape = list(self.input_layers.values())[0]['data'].shape
        
        # If no training points, use high-value areas as proxy
        if training_points is None:
            # Create synthetic training points from high-value areas
            weighted_result = self.weighted_overlay_analysis()
            threshold = np.nanpercentile(weighted_result, 90)
            training_mask = weighted_result > threshold
        else:
            training_mask = training_points
        
        # Calculate weights of evidence for each layer
        evidence_maps = {}
        
        for layer_id, layer_info in self.input_layers.items():
            data = layer_info['data']
            
            # Binarize layer (above/below median)
            valid_mask = np.isfinite(data)
            if valid_mask.sum() == 0:
                continue
                
            median_val = np.nanmedian(data[valid_mask])
            binary_layer = data > median_val
            
            # Calculate conditional probabilities
            n_total = valid_mask.sum()
            n_deposits = training_mask[valid_mask].sum()
            
            if n_deposits == 0:
                continue
            
            # P(B|D) - probability of pattern given deposit
            n_pattern_deposit = (binary_layer & training_mask & valid_mask).sum()
            p_b_given_d = n_pattern_deposit / n_deposits if n_deposits > 0 else 0
            
            # P(B|~D) - probability of pattern given no deposit
            n_no_deposit = n_total - n_deposits
            n_pattern_no_deposit = (binary_layer & ~training_mask & valid_mask).sum()
            p_b_given_not_d = n_pattern_no_deposit / n_no_deposit if n_no_deposit > 0 else 0
            
            # Calculate weight
            if p_b_given_not_d > 0 and p_b_given_d > 0:
                weight_positive = np.log(p_b_given_d / p_b_given_not_d)
            else:
                weight_positive = 0
            
            # Apply weight to create evidence map
            evidence_map = np.zeros_like(data)
            evidence_map[binary_layer & valid_mask] = weight_positive
            evidence_map[~binary_layer & valid_mask] = -weight_positive
            
            evidence_maps[layer_id] = evidence_map
        
        # Combine evidence maps
        if evidence_maps:
            combined_evidence = np.zeros(shape)
            for evidence_map in evidence_maps.values():
                combined_evidence += evidence_map
            
            # Convert to probability
            prospectivity = 1 / (1 + np.exp(-combined_evidence))
            prospectivity[~np.isfinite(prospectivity)] = np.nan
        else:
            prospectivity = np.full(shape, np.nan)
        
        self.prospectivity_map = prospectivity
        return prospectivity
    
    def compute_prospectivity(self, method='weighted_overlay', fuzzy_logic=False, **kwargs):
        """Main prospectivity computation function"""
        # Normalize layers first
        self.normalize_layers()
        
        if method == 'weighted_overlay':
            prospectivity = self.weighted_overlay_analysis()
        elif method == 'fuzzy_logic':
            prospectivity = self.fuzzy_logic_analysis()
        elif method == 'ahp':
            self.analytic_hierarchy_process()
            prospectivity = self.weighted_overlay_analysis()
        elif method == 'neural_network':
            prospectivity = self.neural_network_prospectivity()
        elif method == 'evidence_weights':
            prospectivity = self.evidence_weights_modeling()
        else:
            prospectivity = self.weighted_overlay_analysis()
        
        # Calculate confidence map
        self.confidence_map = self._calculate_confidence_map(prospectivity)
        
        return prospectivity
    
    def _calculate_confidence_map(self, prospectivity):
        """Calculate confidence/uncertainty map"""
        shape = prospectivity.shape
        confidence = np.zeros(shape, dtype=np.float32)
        
        # Number of contributing layers per pixel
        layer_count = np.zeros(shape)
        for layer_info in self.input_layers.values():
            valid_mask = np.isfinite(layer_info['data'])
            layer_count[valid_mask] += 1
        
        # Confidence based on layer agreement and coverage
        max_layers = len(self.input_layers)
        confidence = layer_count / max_layers
        
        # Additional confidence based on data variability
        for i in range(shape[0]):
            for j in range(shape[1]):
                pixel_values = []
                for layer_info in self.input_layers.values():
                    value = layer_info['data'][i, j]
                    if np.isfinite(value):
                        pixel_values.append(value)
                
                if len(pixel_values) > 1:
                    # Lower confidence for high variability
                    variability = np.std(pixel_values)
                    confidence[i, j] *= (1 - min(variability, 1))
        
        confidence[~np.isfinite(prospectivity)] = np.nan
        return confidence
    
    def save_prospectivity_map(self, prospectivity_map, output_path):
        """Save prospectivity map to GeoTIFF"""
        if self.reference_profile is None:
            raise ValueError("No reference profile available")
        
        # Create output profile
        output_profile = self.reference_profile.copy()
        output_profile.update({
            'count': 1,
            'dtype': 'float32',
            'nodata': np.nan,
            'compress': 'lzw'
        })
        
        with rasterio.open(output_path, 'w', **output_profile) as dst:
            dst.write(prospectivity_map.astype(np.float32), 1)
        
        # Save confidence map if available
        if self.confidence_map is not None:
            confidence_path = output_path.replace('.tif', '_confidence.tif')
            with rasterio.open(confidence_path, 'w', **output_profile) as dst:
                dst.write(self.confidence_map.astype(np.float32), 1)
        
        # Save metadata
        metadata = {
            'prospectivity_mapping': {
                'input_layers': [info['name'] for info in self.input_layers.values()],
                'layer_weights': {info['name']: info['weight'] for info in self.input_layers.values()},
                'layer_types': {info['name']: info['layer_type'] for info in self.input_layers.values()}
            },
            'statistics': {
                'valid_pixels': int(np.isfinite(prospectivity_map).sum()),
                'total_pixels': int(prospectivity_map.size),
                'coverage_percent': float(np.isfinite(prospectivity_map).sum() / prospectivity_map.size * 100)
            }
        }
        
        if np.isfinite(prospectivity_map).any():
            valid_data = prospectivity_map[np.isfinite(prospectivity_map)]
            metadata['statistics'].update({
                'min_prospectivity': float(np.min(valid_data)),
                'max_prospectivity': float(np.max(valid_data)),
                'mean_prospectivity': float(np.mean(valid_data)),
                'std_prospectivity': float(np.std(valid_data))
            })
        
        metadata_path = os.path.splitext(output_path)[0] + '_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return output_path
