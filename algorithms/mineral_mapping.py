"""
Advanced mineral mapping algorithms for geological exploration
"""

import numpy as np
import json
import os

# Optional imports with fallbacks
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import FastICA, NMF
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from scipy.optimize import nnls
    from scipy.spatial.distance import cdist
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

class MineralMapper:
    """Advanced mineral mapping using multiple algorithms"""
    
    def __init__(self):
        self.data = None
        self.mineral_signatures = {}
        self.wavelengths = None
        self.spatial_dims = None
        self.profile = None
        self.mineral_abundance_maps = {}
        
    def load_data(self, raster_path):
        """Load hyperspectral/multispectral data"""
        try:
            with rasterio.open(raster_path) as src:
                self.data = src.read()
                self.profile = src.profile
                n_bands, height, width = self.data.shape
                
                # Reshape for analysis
                self.spectral_data = self.data.reshape(n_bands, -1).T
                self.spatial_dims = (height, width)
                
                # Set default wavelengths
                self.wavelengths = np.linspace(400, 2500, n_bands)
                
                # Mask invalid pixels
                self.valid_mask = np.all(self.spectral_data > 0, axis=1)
                self.valid_mask &= np.all(np.isfinite(self.spectral_data), axis=1)
                
                print(f"Loaded data: {n_bands} bands, {height}x{width} pixels")
                return True
                
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def set_mineral_signatures(self, mineral_dict):
        """Set mineral spectral signatures"""
        self.mineral_signatures = mineral_dict
    
    def spectral_unmixing_nnls(self, mineral_list):
        """Non-negative least squares spectral unmixing"""
        if not self.mineral_signatures:
            raise ValueError("No mineral signatures loaded")
        
        # Prepare endmember matrix
        endmembers = []
        mineral_names = []
        
        for mineral in mineral_list:
            if mineral in self.mineral_signatures:
                signature = self.mineral_signatures[mineral]['signature']
                # Interpolate to match data wavelengths if needed
                if len(signature) != self.spectral_data.shape[1]:
                    signature = np.interp(self.wavelengths, 
                                        self.mineral_signatures[mineral]['wavelengths'], 
                                        signature)
                endmembers.append(signature)
                mineral_names.append(mineral)
        
        if not endmembers:
            raise ValueError("No valid endmembers found")
        
        endmember_matrix = np.array(endmembers).T
        n_pixels = self.spectral_data.shape[0]
        n_endmembers = len(endmembers)
        
        # Initialize abundance maps
        abundances = np.zeros((n_pixels, n_endmembers))
        
        # Perform NNLS for each pixel
        for i in range(n_pixels):
            if self.valid_mask[i]:
                pixel_spectrum = self.spectral_data[i, :]
                try:
                    # Non-negative least squares
                    abundance, residual = nnls(endmember_matrix, pixel_spectrum)
                    # Normalize to sum to 1
                    if abundance.sum() > 0:
                        abundance = abundance / abundance.sum()
                    abundances[i, :] = abundance
                except:
                    abundances[i, :] = 0
        
        # Convert to spatial format
        mineral_maps = {}
        for i, mineral_name in enumerate(mineral_names):
            abundance_map = np.full(self.spatial_dims[0] * self.spatial_dims[1], np.nan)
            abundance_map[self.valid_mask] = abundances[:, i]
            mineral_maps[mineral_name] = abundance_map.reshape(self.spatial_dims)
        
        return mineral_maps
    
    def independent_component_analysis(self, n_components=10):
        """Independent Component Analysis for blind source separation"""
        if self.spectral_data is None:
            raise ValueError("No data loaded")
        
        # Prepare clean data
        valid_data = self.spectral_data[self.valid_mask]
        
        # Apply ICA
        ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
        ica_components = ica.fit_transform(valid_data)
        
        # Convert to spatial format
        ica_maps = {}
        for i in range(n_components):
            component_map = np.full(self.spatial_dims[0] * self.spatial_dims[1], np.nan)
            component_map[self.valid_mask] = ica_components[:, i]
            ica_maps[f'ICA_Component_{i+1}'] = component_map.reshape(self.spatial_dims)
        
        # Get mixing matrix (endmembers)
        endmember_spectra = ica.components_
        
        return ica_maps, endmember_spectra
    
    def non_negative_matrix_factorization(self, n_components=10):
        """Non-negative Matrix Factorization for mineral unmixing"""
        if self.spectral_data is None:
            raise ValueError("No data loaded")
        
        valid_data = self.spectral_data[self.valid_mask]
        
        # Ensure non-negative data
        valid_data = np.maximum(valid_data, 0)
        
        # Apply NMF
        nmf = NMF(n_components=n_components, init='random', random_state=42, max_iter=1000)
        nmf_abundances = nmf.fit_transform(valid_data)
        nmf_endmembers = nmf.components_
        
        # Convert to spatial format
        nmf_maps = {}
        for i in range(n_components):
            abundance_map = np.full(self.spatial_dims[0] * self.spatial_dims[1], np.nan)
            abundance_map[self.valid_mask] = nmf_abundances[:, i]
            nmf_maps[f'NMF_Component_{i+1}'] = abundance_map.reshape(self.spatial_dims)
        
        return nmf_maps, nmf_endmembers
    
    def gaussian_mixture_clustering(self, n_clusters=10):
        """Gaussian Mixture Model clustering for mineral identification"""
        if self.spectral_data is None:
            raise ValueError("No data loaded")
        
        valid_data = self.spectral_data[self.valid_mask]
        
        # Apply Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_clusters, random_state=42, max_iter=300)
        cluster_labels = gmm.fit_predict(valid_data)
        cluster_probs = gmm.predict_proba(valid_data)
        
        # Convert labels to spatial format
        label_map = np.full(self.spatial_dims[0] * self.spatial_dims[1], -1)
        label_map[self.valid_mask] = cluster_labels
        label_map = label_map.reshape(self.spatial_dims)
        
        # Convert probabilities to spatial format
        prob_maps = {}
        for i in range(n_clusters):
            prob_map = np.full(self.spatial_dims[0] * self.spatial_dims[1], np.nan)
            prob_map[self.valid_mask] = cluster_probs[:, i]
            prob_maps[f'Cluster_{i+1}_Probability'] = prob_map.reshape(self.spatial_dims)
        
        # Get cluster centers (representative spectra)
        cluster_centers = gmm.means_
        
        return label_map, prob_maps, cluster_centers
    
    def k_means_clustering(self, n_clusters=10):
        """K-means clustering for mineral identification"""
        if self.spectral_data is None:
            raise ValueError("No data loaded")
        
        valid_data = self.spectral_data[self.valid_mask]
        
        # Apply K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(valid_data)
        
        # Convert to spatial format
        label_map = np.full(self.spatial_dims[0] * self.spatial_dims[1], -1)
        label_map[self.valid_mask] = cluster_labels
        label_map = label_map.reshape(self.spatial_dims)
        
        # Calculate distances to cluster centers
        distances = kmeans.transform(valid_data)
        
        # Convert distances to spatial format
        distance_maps = {}
        for i in range(n_clusters):
            distance_map = np.full(self.spatial_dims[0] * self.spatial_dims[1], np.nan)
            distance_map[self.valid_mask] = distances[:, i]
            distance_maps[f'Cluster_{i+1}_Distance'] = distance_map.reshape(self.spatial_dims)
        
        return label_map, distance_maps, kmeans.cluster_centers_
    
    def create_mineral_maps(self, minerals, method='spectral_unmixing'):
        """Create mineral abundance maps using specified method"""
        results = {}
        
        if method == 'spectral_unmixing':
            mineral_maps = self.spectral_unmixing_nnls(minerals)
            results.update(mineral_maps)
            
        elif method == 'ica':
            ica_maps, endmembers = self.independent_component_analysis()
            results.update(ica_maps)
            results['endmember_spectra'] = endmembers
            
        elif method == 'nmf':
            nmf_maps, endmembers = self.non_negative_matrix_factorization()
            results.update(nmf_maps)
            results['endmember_spectra'] = endmembers
            
        elif method == 'gmm_clustering':
            label_map, prob_maps, centers = self.gaussian_mixture_clustering()
            results['cluster_labels'] = label_map
            results.update(prob_maps)
            results['cluster_centers'] = centers
            
        elif method == 'kmeans_clustering':
            label_map, distance_maps, centers = self.k_means_clustering()
            results['cluster_labels'] = label_map
            results.update(distance_maps)
            results['cluster_centers'] = centers
        
        self.mineral_abundance_maps = results
        return results
    
    def apply_spatial_filtering(self, mineral_maps, filter_type='gaussian', sigma=1.0):
        """Apply spatial filtering to mineral maps"""
        filtered_maps = {}
        
        for mineral_name, mineral_map in mineral_maps.items():
            if isinstance(mineral_map, np.ndarray) and mineral_map.ndim == 2:
                if filter_type == 'gaussian':
                    # Handle NaN values
                    valid_mask = ~np.isnan(mineral_map)
                    filtered_map = mineral_map.copy()
                    if valid_mask.any():
                        filtered_map[valid_mask] = gaussian_filter(
                            mineral_map[valid_mask].reshape(mineral_map.shape)[valid_mask], 
                            sigma=sigma
                        )
                    filtered_maps[f'{mineral_name}_filtered'] = filtered_map
                else:
                    filtered_maps[mineral_name] = mineral_map
            else:
                filtered_maps[mineral_name] = mineral_map
        
        return filtered_maps
    
    def calculate_mineral_indices(self, mineral_maps):
        """Calculate mineral-specific spectral indices"""
        indices = {}
        
        for mineral_name, abundance_map in mineral_maps.items():
            if isinstance(abundance_map, np.ndarray) and abundance_map.ndim == 2:
                # Calculate basic statistics
                valid_data = abundance_map[~np.isnan(abundance_map)]
                if len(valid_data) > 0:
                    indices[f'{mineral_name}_mean'] = np.mean(valid_data)
                    indices[f'{mineral_name}_std'] = np.std(valid_data)
                    indices[f'{mineral_name}_max'] = np.max(valid_data)
                    indices[f'{mineral_name}_coverage'] = len(valid_data) / abundance_map.size
                    
                    # Calculate percentiles
                    indices[f'{mineral_name}_p95'] = np.percentile(valid_data, 95)
                    indices[f'{mineral_name}_p90'] = np.percentile(valid_data, 90)
                    indices[f'{mineral_name}_p75'] = np.percentile(valid_data, 75)
        
        return indices
    
    def save_mineral_maps(self, mineral_maps, output_dir):
        """Save mineral maps to GeoTIFF files"""
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = []
        
        for mineral_name, mineral_map in mineral_maps.items():
            if isinstance(mineral_map, np.ndarray) and mineral_map.ndim == 2:
                output_path = os.path.join(output_dir, f"{mineral_name}_map.tif")
                
                # Create output profile
                output_profile = self.profile.copy()
                output_profile.update({
                    'count': 1,
                    'dtype': 'float32',
                    'nodata': np.nan
                })
                
                with rasterio.open(output_path, 'w', **output_profile) as dst:
                    dst.write(mineral_map.astype(np.float32), 1)
                
                saved_files.append(output_path)
        
        # Save metadata
        metadata = {
            'mineral_maps': list(mineral_maps.keys()),
            'spatial_dimensions': self.spatial_dims,
            'wavelengths': self.wavelengths.tolist() if self.wavelengths is not None else None,
            'saved_files': saved_files
        }
        
        metadata_path = os.path.join(output_dir, "mineral_mapping_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return saved_files
    
    def validate_mineral_maps(self, mineral_maps, reference_data=None):
        """Validate mineral mapping results"""
        validation_results = {}
        
        for mineral_name, mineral_map in mineral_maps.items():
            if isinstance(mineral_map, np.ndarray) and mineral_map.ndim == 2:
                valid_data = mineral_map[~np.isnan(mineral_map)]
                
                validation_results[mineral_name] = {
                    'valid_pixels': len(valid_data),
                    'coverage_percent': (len(valid_data) / mineral_map.size) * 100,
                    'value_range': [float(np.min(valid_data)), float(np.max(valid_data))] if len(valid_data) > 0 else [0, 0],
                    'mean_abundance': float(np.mean(valid_data)) if len(valid_data) > 0 else 0,
                    'std_abundance': float(np.std(valid_data)) if len(valid_data) > 0 else 0
                }
        
        return validation_results
