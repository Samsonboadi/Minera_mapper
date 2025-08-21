"""
Spectral analysis algorithms for mineral identification
"""

import numpy as np
import os
import json

# Optional imports with fallbacks
try:
    from osgeo import gdal
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False
    
try:
    import spectral as spy
    HAS_SPECTRAL = True
except ImportError:
    HAS_SPECTRAL = False
    
try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    
try:
    from scipy import ndimage
    from scipy.spatial.distance import cdist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

class SpectralAnalyzer:
    """Advanced spectral analysis for mineral identification"""
    
    def __init__(self):
        self.data = None
        self.wavelengths = None
        self.mineral_libraries = self.load_mineral_libraries()
        self.processed_bands = None
        self.spectral_data = None
        self.clean_data = None
        self.valid_pixels = None
    
    def load_mineral_libraries(self):
        """Load spectral libraries for different minerals"""
        # Standard mineral spectral signatures
        libraries = {
            'Gold': {
                'signature': np.array([0.1, 0.12, 0.15, 0.18, 0.22, 0.25, 0.28, 0.32, 0.35]),
                'wavelengths': np.array([450, 500, 550, 600, 650, 700, 750, 800, 850]),
                'description': 'Gold spectral signature'
            },
            'Iron Oxide': {
                'signature': np.array([0.08, 0.10, 0.15, 0.25, 0.35, 0.40, 0.38, 0.32, 0.28]),
                'wavelengths': np.array([450, 500, 550, 600, 650, 700, 750, 800, 850]),
                'description': 'Hematite and goethite'
            },
            'Iron Hydroxide': {
                'signature': np.array([0.06, 0.08, 0.12, 0.20, 0.30, 0.42, 0.40, 0.35, 0.30]),
                'wavelengths': np.array([450, 500, 550, 600, 650, 700, 750, 800, 850]),
                'description': 'Limonite and other iron hydroxides'
            },
            'Clay Minerals': {
                'signature': np.array([0.15, 0.18, 0.22, 0.28, 0.32, 0.30, 0.25, 0.20, 0.18]),
                'wavelengths': np.array([450, 500, 550, 600, 650, 700, 750, 800, 850]),
                'description': 'Kaolinite, montmorillonite, illite'
            },
            'Carbonate': {
                'signature': np.array([0.35, 0.40, 0.42, 0.38, 0.32, 0.28, 0.25, 0.22, 0.20]),
                'wavelengths': np.array([450, 500, 550, 600, 650, 700, 750, 800, 850]),
                'description': 'Calcite, dolomite'
            },
            'Silica': {
                'signature': np.array([0.25, 0.28, 0.32, 0.35, 0.38, 0.35, 0.32, 0.28, 0.25]),
                'wavelengths': np.array([450, 500, 550, 600, 650, 700, 750, 800, 850]),
                'description': 'Quartz, chalcedony'
            },
            'Lithium': {
                'signature': np.array([0.12, 0.15, 0.20, 0.25, 0.30, 0.28, 0.24, 0.20, 0.16]),
                'wavelengths': np.array([450, 500, 550, 600, 650, 700, 750, 800, 850]),
                'description': 'Spodumene, petalite'
            },
            'Diamond Indicator Minerals': {
                'signature': np.array([0.18, 0.22, 0.28, 0.32, 0.35, 0.38, 0.35, 0.30, 0.25]),
                'wavelengths': np.array([450, 500, 550, 600, 650, 700, 750, 800, 850]),
                'description': 'Kimberlite indicators'
            },
            'Alteration Minerals': {
                'signature': np.array([0.10, 0.12, 0.16, 0.22, 0.28, 0.35, 0.32, 0.28, 0.24]),
                'wavelengths': np.array([450, 500, 550, 600, 650, 700, 750, 800, 850]),
                'description': 'Sericite, chlorite, epidote'
            },
            'Gossans': {
                'signature': np.array([0.05, 0.08, 0.15, 0.28, 0.42, 0.45, 0.40, 0.32, 0.25]),
                'wavelengths': np.array([450, 500, 550, 600, 650, 700, 750, 800, 850]),
                'description': 'Iron-rich oxidized zones'
            }
        }
        
        return libraries
    
    def load_raster(self, raster_path):
        """Load raster data for analysis"""
        print(f"Attempting to load raster: {raster_path}")
        
        # Handle GDAL virtual file paths (especially HDF4 EOS)
        actual_file_path = raster_path
        is_hdf_virtual = False
        
        if raster_path.startswith('HDF4_EOS') or raster_path.startswith('HDF5'):
            print("Detected HDF virtual path from QGIS")
            is_hdf_virtual = True
            
            # Extract actual file path from GDAL virtual path
            import re
            # Pattern for HDF4_EOS:EOS_SWATH:"filepath":subdataset
            match = re.search(r'HDF[45]?[^:]*:"([^"]+)"', raster_path)
            if match:
                actual_file_path = match.group(1)
                print(f"Extracted HDF file path: {actual_file_path}")
            else:
                print(f"Could not extract file path from: {raster_path}")
                return False
        
        # Check if file exists
        import os
        if not os.path.exists(actual_file_path):
            print(f"Error: File does not exist: {actual_file_path}")
            return False
            
        try:
            # For HDF files, always use GDAL as it handles subdatasets better
            if is_hdf_virtual or actual_file_path.lower().endswith('.hdf'):
                print("Using GDAL for HDF file...")
                return self._load_with_gdal(raster_path, actual_file_path, is_hdf_virtual)
            
            # Try rasterio first for regular files
            elif HAS_RASTERIO:
                print("Using rasterio to load data...")
                with rasterio.open(actual_file_path) as src:
                    # Read all bands
                    self.data = src.read()
                    self.profile = src.profile
                    self.transform = src.transform
                    self.crs = src.crs
                    
                    # Get band count and spatial dimensions
                    n_bands, height, width = self.data.shape
                    
                    # Reshape for spectral analysis (pixels x bands)
                    self.spectral_data = self.data.reshape(n_bands, -1).T
                    self.spatial_dims = (height, width)
                    
                    # Set default wavelengths if not available
                    if self.wavelengths is None:
                        self.wavelengths = np.linspace(400, 2500, n_bands)
                    
                    print(f"Successfully loaded raster: {n_bands} bands, {height}x{width} pixels")
                    return True
            
            # Fall back to GDAL
            else:
                print("Falling back to GDAL...")
                return self._load_with_gdal(actual_file_path, actual_file_path, False)
                
        except Exception as e:
            print(f"Error loading raster: {str(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def _load_with_gdal(self, original_path, file_path, is_virtual):
        """Load raster using GDAL, handling both regular and virtual paths"""
        if not HAS_GDAL:
            print("Error: GDAL not available")
            return False
            
        from osgeo import gdal
        
        try:
            # For virtual paths, use the original GDAL path
            path_to_open = original_path if is_virtual else file_path
            print(f"Opening with GDAL: {path_to_open}")
            
            dataset = gdal.Open(path_to_open)
            if dataset is None:
                print(f"GDAL could not open: {path_to_open}")
                
                # If virtual path failed, try to open the base HDF file and get subdatasets
                if is_virtual:
                    print("Trying to open base HDF file and get subdatasets...")
                    base_dataset = gdal.Open(file_path)
                    if base_dataset:
                        subdatasets = base_dataset.GetSubDatasets()
                        print(f"Found {len(subdatasets)} subdatasets")
                        
                        # Try to find a suitable subdataset (prefer VNIR data)
                        for subdataset in subdatasets:
                            sub_name, sub_desc = subdataset
                            print(f"Subdataset: {sub_desc} -> {sub_name}")
                            if 'VNIR' in sub_desc or 'SurfaceReflectance' in sub_desc:
                                print(f"Using subdataset: {sub_name}")
                                dataset = gdal.Open(sub_name)
                                break
                        
                        # If no VNIR found, use the first subdataset
                        if dataset is None and subdatasets:
                            sub_name, sub_desc = subdatasets[0]
                            print(f"Using first subdataset: {sub_name}")
                            dataset = gdal.Open(sub_name)
                
                if dataset is None:
                    return False
            
            n_bands = dataset.RasterCount
            width = dataset.RasterXSize
            height = dataset.RasterYSize
            
            print(f"Dataset info: {n_bands} bands, {height}x{width} pixels")
            
            if n_bands == 0:
                print("Error: No bands found in dataset")
                return False
            
            # Read all bands
            data = []
            for i in range(1, n_bands + 1):
                band = dataset.GetRasterBand(i)
                if band is None:
                    print(f"Warning: Could not get band {i}")
                    continue
                    
                band_data = band.ReadAsArray()
                if band_data is not None:
                    data.append(band_data)
                else:
                    print(f"Warning: Could not read data from band {i}")
            
            if not data:
                print("Error: No band data could be read")
                return False
            
            self.data = np.array(data)
            self.spectral_data = self.data.reshape(len(data), -1).T
            self.spatial_dims = (height, width)
            
            # Set ASTER-specific wavelengths if this is ASTER data
            if self.wavelengths is None:
                if 'AST_07' in file_path:  # ASTER L2 Surface Reflectance
                    # ASTER VNIR wavelengths (in nanometers)
                    aster_wavelengths = [560, 660, 820]  # Band 1, 2, 3N
                    if len(data) == len(aster_wavelengths):
                        self.wavelengths = np.array(aster_wavelengths)
                    else:
                        self.wavelengths = np.linspace(500, 900, len(data))
                else:
                    self.wavelengths = np.linspace(400, 2500, len(data))
            
            print(f"Successfully loaded with GDAL: {len(data)} bands, {height}x{width} pixels")
            print(f"Wavelengths: {self.wavelengths}")
            return True
            
        except Exception as e:
            print(f"Error in GDAL loading: {str(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def preprocess_data(self, method='standard'):
        """Preprocess spectral data"""
        if not hasattr(self, 'spectral_data') or self.spectral_data is None:
            raise ValueError("No spectral data available. Call load_raster() first.")
        
        if method == 'standard':
            # Remove invalid pixels (zeros, NaN)
            valid_mask = np.all(self.spectral_data > 0, axis=1)
            valid_mask &= np.all(np.isfinite(self.spectral_data), axis=1)
            
            self.valid_pixels = valid_mask
            self.clean_data = self.spectral_data[valid_mask]
            
            # Normalize to reflectance (0-1)
            self.clean_data = np.clip(self.clean_data / 10000.0, 0, 1)
            
        elif method == 'continuum_removal':
            self.clean_data = self.continuum_removal(self.spectral_data)
            
        elif method == 'derivative':
            self.clean_data = self.spectral_derivative(self.spectral_data)
            
        return self.clean_data
    
    def continuum_removal(self, spectra):
        """Apply continuum removal to spectral data"""
        processed_spectra = np.zeros_like(spectra)
        
        for i in range(spectra.shape[0]):
            spectrum = spectra[i, :]
            
            # Find convex hull
            from scipy.spatial import ConvexHull
            points = np.column_stack((self.wavelengths, spectrum))
            
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                
                # Interpolate continuum
                continuum = np.interp(self.wavelengths, 
                                    hull_points[:, 0], 
                                    hull_points[:, 1])
                
                # Remove continuum
                processed_spectra[i, :] = spectrum / continuum
                
            except:
                processed_spectra[i, :] = spectrum
        
        return processed_spectra
    
    def spectral_derivative(self, spectra):
        """Calculate spectral derivatives"""
        return np.gradient(spectra, axis=1)
    
    def spectral_angle_mapper(self, target_spectra):
        """Spectral Angle Mapper (SAM) classification"""
        if not hasattr(self, 'clean_data') or self.clean_data is None:
            self.preprocess_data()
        
        sam_results = {}
        
        for mineral_name, mineral_data in self.mineral_libraries.items():
            if mineral_name not in target_spectra:
                continue
                
            reference_spectrum = mineral_data['signature']
            
            # Interpolate reference to match data wavelengths
            if len(reference_spectrum) != self.clean_data.shape[1]:
                reference_spectrum = np.interp(
                    self.wavelengths, 
                    mineral_data['wavelengths'], 
                    reference_spectrum
                )
            
            # Calculate spectral angles
            angles = np.zeros(self.clean_data.shape[0])
            
            for i in range(self.clean_data.shape[0]):
                pixel_spectrum = self.clean_data[i, :]
                
                # Calculate angle between vectors
                dot_product = np.dot(pixel_spectrum, reference_spectrum)
                norm_pixel = np.linalg.norm(pixel_spectrum)
                norm_ref = np.linalg.norm(reference_spectrum)
                
                if norm_pixel > 0 and norm_ref > 0:
                    cos_angle = dot_product / (norm_pixel * norm_ref)
                    angle = np.arccos(np.clip(cos_angle, -1, 1))
                    angles[i] = angle
                else:
                    angles[i] = np.pi / 2  # Maximum angle
            
            # Create full image array
            full_angles = np.full(self.spatial_dims[0] * self.spatial_dims[1], np.nan)
            full_angles[self.valid_pixels] = angles
            
            sam_results[mineral_name] = full_angles.reshape(self.spatial_dims)
        
        return sam_results
    
    def spectral_unmixing(self, target_minerals):
        """Linear spectral unmixing"""
        if not hasattr(self, 'clean_data') or self.clean_data is None:
            self.preprocess_data()
        
        # Collect endmember spectra
        endmembers = []
        mineral_names = []
        
        for mineral_name in target_minerals:
            if mineral_name in self.mineral_libraries:
                mineral_data = self.mineral_libraries[mineral_name]
                reference_spectrum = mineral_data['signature']
                
                # Interpolate to match data wavelengths
                if len(reference_spectrum) != self.clean_data.shape[1]:
                    reference_spectrum = np.interp(
                        self.wavelengths, 
                        mineral_data['wavelengths'], 
                        reference_spectrum
                    )
                
                endmembers.append(reference_spectrum)
                mineral_names.append(mineral_name)
        
        if not endmembers:
            raise ValueError("No valid endmembers found")
        
        endmember_matrix = np.array(endmembers).T
        
        # Perform unmixing using least squares
        abundances = np.zeros((self.clean_data.shape[0], len(endmembers)))
        
        for i in range(self.clean_data.shape[0]):
            pixel_spectrum = self.clean_data[i, :]
            
            # Solve: pixel_spectrum = endmember_matrix * abundances
            try:
                abundance, _, _, _ = np.linalg.lstsq(endmember_matrix, pixel_spectrum, rcond=None)
                # Constrain abundances to be non-negative and sum to 1
                abundance = np.maximum(abundance, 0)
                if abundance.sum() > 0:
                    abundance = abundance / abundance.sum()
                abundances[i, :] = abundance
            except:
                abundances[i, :] = 0
        
        # Convert to image format
        unmixing_results = {}
        for i, mineral_name in enumerate(mineral_names):
            full_abundance = np.full(self.spatial_dims[0] * self.spatial_dims[1], np.nan)
            full_abundance[self.valid_pixels] = abundances[:, i]
            unmixing_results[mineral_name] = full_abundance.reshape(self.spatial_dims)
        
        return unmixing_results
    
    def minimum_noise_fraction(self, n_components=10):
        """Minimum Noise Fraction (MNF) transformation"""
        if not hasattr(self, 'clean_data') or self.clean_data is None:
            self.preprocess_data()
        
        # Estimate noise covariance
        noise_data = self.clean_data + np.random.normal(0, 0.01, self.clean_data.shape)
        noise_cov = np.cov((self.clean_data - noise_data).T)
        
        # Signal covariance
        signal_cov = np.cov(self.clean_data.T)
        
        # Generalized eigenvalue problem
        eigenvals, eigenvecs = np.linalg.eig(np.linalg.inv(noise_cov) @ signal_cov)
        
        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Transform data
        mnf_data = self.clean_data @ eigenvecs[:, :n_components]
        
        # Convert to image format
        mnf_images = []
        for i in range(n_components):
            full_mnf = np.full(self.spatial_dims[0] * self.spatial_dims[1], np.nan)
            full_mnf[self.valid_pixels] = mnf_data[:, i]
            mnf_images.append(full_mnf.reshape(self.spatial_dims))
        
        return mnf_images, eigenvals[:n_components]
    
    def principal_component_analysis(self, n_components=10):
        """Principal Component Analysis"""
        if not hasattr(self, 'clean_data') or self.clean_data is None:
            self.preprocess_data()
        
        pca = PCA(n_components=n_components)
        pca_data = pca.fit_transform(self.clean_data)
        
        # Convert to image format
        pca_images = []
        for i in range(n_components):
            full_pca = np.full(self.spatial_dims[0] * self.spatial_dims[1], np.nan)
            full_pca[self.valid_pixels] = pca_data[:, i]
            pca_images.append(full_pca.reshape(self.spatial_dims))
        
        return pca_images, pca.explained_variance_ratio_
    
    def matched_filtering(self, target_minerals):
        """Matched filtering for target detection"""
        if not hasattr(self, 'clean_data') or self.clean_data is None:
            self.preprocess_data()
        
        # Calculate background covariance
        background_cov = np.cov(self.clean_data.T)
        background_cov_inv = np.linalg.pinv(background_cov)
        
        mf_results = {}
        
        for mineral_name in target_minerals:
            if mineral_name not in self.mineral_libraries:
                continue
                
            mineral_data = self.mineral_libraries[mineral_name]
            target_spectrum = mineral_data['signature']
            
            # Interpolate to match data wavelengths
            if len(target_spectrum) != self.clean_data.shape[1]:
                target_spectrum = np.interp(
                    self.wavelengths, 
                    mineral_data['wavelengths'], 
                    target_spectrum
                )
            
            # Matched filter
            mf_scores = np.zeros(self.clean_data.shape[0])
            
            for i in range(self.clean_data.shape[0]):
                pixel_spectrum = self.clean_data[i, :]
                
                # MF = s^T * C^-1 * x / sqrt(s^T * C^-1 * s)
                numerator = target_spectrum.T @ background_cov_inv @ pixel_spectrum
                denominator = np.sqrt(target_spectrum.T @ background_cov_inv @ target_spectrum)
                
                if denominator > 0:
                    mf_scores[i] = numerator / denominator
            
            # Convert to image format
            full_mf = np.full(self.spatial_dims[0] * self.spatial_dims[1], np.nan)
            full_mf[self.valid_pixels] = mf_scores
            mf_results[mineral_name] = full_mf.reshape(self.spatial_dims)
        
        return mf_results
    
    def analyze_spectra(self, method='SAM', target_minerals=None):
        """Main spectral analysis function"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_raster() first.")
        
        # Ensure data is preprocessed
        if not hasattr(self, 'clean_data') or self.clean_data is None:
            self.preprocess_data()
        
        if target_minerals is None:
            target_minerals = list(self.mineral_libraries.keys())
        
        results = {}
        
        if method == 'SAM' or method == 'Spectral Angle Mapper (SAM)':
            results = self.spectral_angle_mapper(target_minerals)
            
        elif method == 'Spectral Unmixing':
            results = self.spectral_unmixing(target_minerals)
            
        elif method == 'Minimum Noise Fraction (MNF)':
            mnf_images, eigenvals = self.minimum_noise_fraction()
            results = {f'MNF_Component_{i+1}': mnf_images[i] for i in range(len(mnf_images))}
            results['eigenvalues'] = eigenvals
            
        elif method == 'Principal Component Analysis (PCA)':
            pca_images, variance_ratios = self.principal_component_analysis()
            results = {f'PC_{i+1}': pca_images[i] for i in range(len(pca_images))}
            results['variance_ratios'] = variance_ratios
            
        elif method == 'Matched Filtering':
            results = self.matched_filtering(target_minerals)
        
        return results
    
    def save_results(self, results, output_path):
        """Save analysis results to GeoTIFF files"""
        base_path = os.path.splitext(output_path)[0]
        
        for result_name, result_data in results.items():
            if isinstance(result_data, np.ndarray):
                output_file = f"{base_path}_{result_name}.tif"
                
                # Create output profile
                output_profile = self.profile.copy()
                output_profile.update({
                    'count': 1,
                    'dtype': 'float32',
                    'nodata': np.nan
                })
                
                with rasterio.open(output_file, 'w', **output_profile) as dst:
                    dst.write(result_data.astype(np.float32), 1)
        
        # Save metadata
        metadata = {
            'analysis_results': list(results.keys()),
            'mineral_libraries': list(self.mineral_libraries.keys()),
            'wavelengths': self.wavelengths.tolist() if self.wavelengths is not None else None,
            'spatial_dimensions': self.spatial_dims
        }
        
        metadata_file = f"{base_path}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
