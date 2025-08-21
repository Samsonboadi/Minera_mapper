import numpy as np
import json
import os

# Safe imports with fallbacks
try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from sklearn.cluster import KMeans
    from sklearn.mixture import GaussianMixture
    from sklearn.decomposition import FastICA, NMF
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
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
    """Enhanced mineral mapping with proper ASTER processing - CORE FIXES"""
    
    def __init__(self):
        self.data = None
        self.normalized_data = None
        self.mineral_signatures = {}
        self.wavelengths = None
        self.spatial_dims = None
        self.profile = None
        self.valid_mask = None
        self.mineral_abundance_maps = {}
        self.normalization_params = {}
        # CRITICAL FIX: Add resampling properties
        self.target_resolution = 15.0
        self.resampled_data = None
        
    def load_data(self, raster_path):
        """FIXED: Load data with proper ZIP and HDF handling"""
        try:
            import os
            import zipfile
            import tempfile
            
            # Check if input is a ZIP file (common for ASTER data)
            if raster_path.lower().endswith('.zip'):
                # Cannot directly load ZIP with rasterio - need to extract first
                print(f"ZIP file detected: {raster_path}")
                
                # Create temporary directory for extraction
                temp_dir = tempfile.mkdtemp(prefix='aster_zip_')
                
                try:
                    # Extract ZIP file
                    with zipfile.ZipFile(raster_path, 'r') as zip_ref:
                        zip_ref.extractall(temp_dir)
                    
                    # Find HDF files in extracted content
                    hdf_files = []
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if file.lower().endswith(('.hdf', '.h5')):
                                hdf_files.append(os.path.join(root, file))
                    
                    if not hdf_files:
                        print("No HDF files found in ZIP archive")
                        return False
                    
                    print(f"Found {len(hdf_files)} HDF files in ZIP")
                    
                    # For now, try to process the first HDF file
                    # In a full implementation, you'd combine VNIR and SWIR data
                    first_hdf = hdf_files[0]
                    
                    # Try to read HDF with GDAL (since rasterio may not handle HDF subdatasets well)
                    try:
                        from osgeo import gdal
                        dataset = gdal.Open(first_hdf, gdal.GA_ReadOnly)
                        
                        if dataset:
                            subdatasets = dataset.GetSubDatasets()
                            if subdatasets:
                                # Use first subdataset that looks like spectral data
                                for subdataset_path, subdataset_desc in subdatasets:
                                    if 'VNIR' in subdataset_desc or 'SWIR' in subdataset_desc:
                                        print(f"Loading subdataset: {subdataset_desc}")
                                        
                                        # Try to open with rasterio
                                        with rasterio.open(subdataset_path) as src:
                                            self.data = src.read()
                                            self.profile = src.profile
                                            
                                            n_bands, height, width = self.data.shape
                                            print(f"Loaded data: {n_bands} bands, {height}x{width} pixels")
                                            
                                            # Reshape for analysis
                                            self.spectral_data = self.data.reshape(n_bands, -1).T
                                            self.spatial_dims = (height, width)
                                            
                                            # Set appropriate wavelengths based on band count
                                            if n_bands == 9:  # ASTER VNIR+SWIR
                                                self.wavelengths = np.array([560, 660, 810, 1650, 2165, 2205, 2260, 2330, 2395])
                                            elif n_bands == 3:  # ASTER VNIR only
                                                self.wavelengths = np.array([560, 660, 810])
                                            elif n_bands == 6:  # ASTER SWIR only
                                                self.wavelengths = np.array([1650, 2165, 2205, 2260, 2330, 2395])
                                            else:
                                                # Generic wavelengths
                                                self.wavelengths = np.linspace(400, 2500, n_bands)
                                            
                                            # Create valid mask (non-zero, non-NaN pixels)
                                            self.valid_mask = np.all(
                                                (self.spectral_data > 0) & 
                                                np.isfinite(self.spectral_data), 
                                                axis=1
                                            )
                                            
                                            valid_pixels = np.sum(self.valid_mask)
                                            total_pixels = self.valid_mask.size
                                            print(f"Valid pixels: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")
                                            
                                            return True
                                    
                                print("No suitable VNIR/SWIR subdatasets found")
                                return False
                            else:
                                print("HDF file contains no subdatasets")
                                return False
                        else:
                            print("Could not open HDF file with GDAL")
                            return False
                            
                    except ImportError:
                        print("GDAL not available - cannot process HDF files")
                        return False
                        
                except Exception as e:
                    print(f"Error processing ZIP file: {str(e)}")
                    return False
                finally:
                    # Clean up temporary directory
                    try:
                        import shutil
                        shutil.rmtree(temp_dir)
                    except:
                        pass  # Ignore cleanup errors
            
            else:
                # Try to load as regular raster file
                if not HAS_RASTERIO:
                    print("Rasterio not available")
                    return False
                    
                try:
                    with rasterio.open(raster_path) as src:
                        self.data = src.read()
                        self.profile = src.profile
                        n_bands, height, width = self.data.shape
                        
                        print(f"Loaded raster: {n_bands} bands, {height}x{width} pixels")
                        
                        # Reshape for analysis
                        self.spectral_data = self.data.reshape(n_bands, -1).T
                        self.spatial_dims = (height, width)
                        
                        # Set appropriate wavelengths
                        if n_bands == 9:  # ASTER VNIR+SWIR
                            self.wavelengths = np.array([560, 660, 810, 1650, 2165, 2205, 2260, 2330, 2395])
                        elif n_bands == 3:  # ASTER VNIR
                            self.wavelengths = np.array([560, 660, 810])
                        elif n_bands == 6:  # ASTER SWIR
                            self.wavelengths = np.array([1650, 2165, 2205, 2260, 2330, 2395])
                        else:
                            # Generic wavelengths
                            self.wavelengths = np.linspace(400, 2500, n_bands)
                        
                        # Create valid mask
                        self.valid_mask = np.all(
                            (self.spectral_data > 0) & 
                            np.isfinite(self.spectral_data), 
                            axis=1
                        )
                        
                        valid_pixels = np.sum(self.valid_mask)
                        total_pixels = self.valid_mask.size
                        print(f"Valid pixels: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)")
                        
                        return True
                        
                except Exception as e:
                    print(f"Error loading raster file: {str(e)}")
                    return False
            
        except Exception as e:
            print(f"Load data error: {str(e)}")
            return False
    
    def create_enhanced_valid_mask(self):
        """Create comprehensive valid pixel mask - CRITICAL FIX"""
        mask = np.ones(self.spectral_data.shape[0], dtype=bool)
        
        # Remove pixels with zero values
        mask &= np.all(self.spectral_data > 0, axis=1)
        
        # Remove pixels with NaN or infinite values
        mask &= np.all(np.isfinite(self.spectral_data), axis=1)
        
        # Remove saturated pixels (16-bit data)
        mask &= np.all(self.spectral_data < 65535, axis=1)
        
        # CRITICAL FIX: Remove unrealistic reflectance values
        mean_reflectance = np.mean(self.spectral_data, axis=1)
        mask &= (mean_reflectance > 50) & (mean_reflectance < 30000)
        
        # CRITICAL FIX: Remove pixels with flat spectra (likely bad data)
        spectral_std = np.std(self.spectral_data, axis=1)
        mask &= spectral_std > 10  # Minimum spectral variation
        
        return mask
    
    def normalize_data_enhanced(self, method='min_max', per_band=True):
        """Enhanced data normalization - CRITICAL FIX"""
        if self.spectral_data is None:
            raise ValueError("No data loaded")
        
        valid_data = self.spectral_data[self.valid_mask]
        normalized_data = self.spectral_data.copy().astype(np.float32)
        
        if method == 'min_max':
            if per_band:
                # Normalize each band separately (RECOMMENDED for ASTER)
                for band_idx in range(self.spectral_data.shape[1]):
                    band_data = valid_data[:, band_idx]
                    if len(band_data) > 0:
                        min_val, max_val = np.min(band_data), np.max(band_data)
                        if max_val > min_val:
                            normalized_data[self.valid_mask, band_idx] = (
                                (band_data - min_val) / (max_val - min_val)
                            )
            else:
                # Global normalization
                min_val, max_val = np.min(valid_data), np.max(valid_data)
                if max_val > min_val:
                    normalized_data[self.valid_mask] = (
                        (valid_data - min_val) / (max_val - min_val)
                    )
        
        elif method == 'percentile':
            # CRITICAL FIX: Percentile normalization (robust to outliers)
            if per_band:
                for band_idx in range(self.spectral_data.shape[1]):
                    band_data = valid_data[:, band_idx]
                    if len(band_data) > 0:
                        p2, p98 = np.percentile(band_data, [2, 98])
                        if p98 > p2:
                            normalized_data[self.valid_mask, band_idx] = np.clip(
                                (band_data - p2) / (p98 - p2), 0, 1
                            )
            else:
                p2, p98 = np.percentile(valid_data, [2, 98])
                if p98 > p2:
                    normalized_data[self.valid_mask] = np.clip(
                        (valid_data - p2) / (p98 - p2), 0, 1
                    )
        
        self.normalized_data = normalized_data
        self.normalization_params = {
            'method': method,
            'per_band': per_band,
            'valid_pixels': np.sum(self.valid_mask)
        }
        
        print(f"Data normalized using {method} method (per_band={per_band})")
        return True
    
    def resample_to_target_resolution(self, bands_data, transforms, target_res=15.0):
        """Resample bands to target resolution - CRITICAL FIX"""
        if not HAS_RASTERIO:
            print("Warning: rasterio not available, skipping resampling")
            return bands_data
        
        resampled_bands = []
        
        # Use first band as reference for bounds
        reference_transform = transforms[0]
        reference_bounds = rasterio.transform.array_bounds(
            bands_data[0].shape[0], bands_data[0].shape[1], reference_transform
        )
        
        # Calculate target dimensions
        target_width = int((reference_bounds[2] - reference_bounds[0]) / target_res)
        target_height = int((reference_bounds[3] - reference_bounds[1]) / target_res)
        
        target_transform = from_bounds(
            reference_bounds[0], reference_bounds[1], 
            reference_bounds[2], reference_bounds[3],
            target_width, target_height
        )
        
        for i, (band_data, transform) in enumerate(zip(bands_data, transforms)):
            try:
                # Create output array
                resampled_data = np.empty((target_height, target_width), dtype=np.float32)
                
                # Resample
                reproject(
                    source=band_data,
                    destination=resampled_data,
                    src_transform=transform,
                    dst_transform=target_transform,
                    resampling=Resampling.bilinear,
                    src_nodata=0,
                    dst_nodata=np.nan
                )
                
                resampled_bands.append(resampled_data)
                print(f"Resampled band {i+1} to {target_res}m resolution")
                
            except Exception as e:
                print(f"Failed to resample band {i}: {str(e)}")
                # Fallback: use original data
                resampled_bands.append(band_data)
        
        return resampled_bands
    
    def spectral_unmixing_nnls(self, mineral_list):
        """Enhanced spectral unmixing - FIXED"""
        if not self.mineral_signatures:
            raise ValueError("No mineral signatures loaded")
        
        # Use normalized data if available
        spectral_data = self.normalized_data if self.normalized_data is not None else self.spectral_data
        
        # Prepare endmember matrix
        endmembers = []
        mineral_names = []
        
        for mineral in mineral_list:
            if mineral in self.mineral_signatures:
                signature = self.mineral_signatures[mineral]['signature']
                
                # CRITICAL FIX: Interpolate signature to match wavelengths
                if len(signature) != spectral_data.shape[1]:
                    signature = np.interp(
                        self.wavelengths, 
                        self.mineral_signatures[mineral]['wavelengths'], 
                        signature
                    )
                
                # CRITICAL FIX: Normalize signature to [0,1] if data is normalized
                if self.normalized_data is not None:
                    signature = (signature - np.min(signature)) / (np.max(signature) - np.min(signature))
                
                endmembers.append(signature)
                mineral_names.append(mineral)
        
        if not endmembers:
            raise ValueError("No valid endmembers found")
        
        endmember_matrix = np.array(endmembers).T
        n_pixels = spectral_data.shape[0]
        n_endmembers = len(endmembers)
        
        # Initialize abundance maps
        abundances = np.zeros((n_pixels, n_endmembers))
        rmse_values = np.zeros(n_pixels)
        
        # CRITICAL FIX: Perform NNLS only for valid pixels
        valid_count = 0
        for i in range(n_pixels):
            if self.valid_mask[i]:
                pixel_spectrum = spectral_data[i, :]
                try:
                    if HAS_SCIPY:
                        # Use NNLS from scipy
                        abundance, residual = nnls(endmember_matrix, pixel_spectrum)
                    else:
                        # Fallback: simple least squares
                        abundance = np.linalg.lstsq(endmember_matrix, pixel_spectrum, rcond=None)[0]
                        abundance = np.maximum(abundance, 0)  # Force non-negative
                    
                    # Calculate RMSE
                    predicted = endmember_matrix @ abundance
                    rmse_values[i] = np.sqrt(np.mean((pixel_spectrum - predicted)**2))
                    
                    # Normalize abundances to sum to 1
                    if abundance.sum() > 0:
                        abundance = abundance / abundance.sum()
                    
                    abundances[i, :] = abundance
                    valid_count += 1
                    
                except Exception as e:
                    abundances[i, :] = 0
                    rmse_values[i] = np.inf
        
        print(f"Spectral unmixing completed for {valid_count} pixels")
        
        # Convert to spatial format
        mineral_maps = {}
        for i, mineral_name in enumerate(mineral_names):
            abundance_map = np.full(self.spatial_dims[0] * self.spatial_dims[1], np.nan)
            valid_indices = np.where(self.valid_mask)[0]
            abundance_map[valid_indices] = abundances[valid_indices, i]
            mineral_maps[mineral_name] = abundance_map.reshape(self.spatial_dims)
        
        # Add RMSE map
        rmse_map = np.full(self.spatial_dims[0] * self.spatial_dims[1], np.nan)
        rmse_map[valid_indices] = rmse_values[valid_indices]
        mineral_maps['unmixing_rmse'] = rmse_map.reshape(self.spatial_dims)
        
        return mineral_maps
    
    def calculate_spectral_indices(self):
        """Calculate common spectral indices for mineral detection"""
        if self.normalized_data is None:
            print("Warning: Using original data for indices. Consider normalizing first.")
            data = self.spectral_data
        else:
            data = self.normalized_data
        
        indices = {}
        
        # Assuming ASTER band ordering: [560, 660, 810, 1650, 2165, 2205, 2260, 2330, 2395]
        if data.shape[1] >= 9:
            # Clay mineral indices
            indices['clay_index'] = (data[:, 4] + data[:, 6]) / (2 * data[:, 5])  # (B5+B7)/(2*B6)
            indices['kaolinite_index'] = data[:, 5] / data[:, 6]  # B6/B7
            indices['illite_index'] = data[:, 4] / data[:, 5]  # B5/B6
            
            # Iron oxide indices
            indices['iron_oxide'] = data[:, 1] / data[:, 0]  # Red/Green
            
            # Carbonate index
            indices['carbonate_index'] = (data[:, 6] + data[:, 8]) / (2 * data[:, 7])  # (B7+B9)/(2*B8)
            
            # Vegetation indices
            indices['ndvi'] = (data[:, 2] - data[:, 1]) / (data[:, 2] + data[:, 1])  # (NIR-Red)/(NIR+Red)
        
        # Convert to spatial format
        spatial_indices = {}
        for index_name, index_values in indices.items():
            index_map = np.full(self.spatial_dims[0] * self.spatial_dims[1], np.nan)
            valid_indices = np.where(self.valid_mask)[0]
            index_map[valid_indices] = index_values[valid_indices]
            spatial_indices[index_name] = index_map.reshape(self.spatial_dims)
        
        return spatial_indices
    
    def save_results(self, results, output_dir):
        """Save all results to files"""
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []
        
        # Update profile for output
        output_profile = self.profile.copy()
        output_profile.update({
            'count': 1,
            'dtype': 'float32',
            'nodata': np.nan
        })
        
        for result_name, result_data in results.items():
            if isinstance(result_data, np.ndarray) and result_data.ndim == 2:
                output_path = os.path.join(output_dir, f"{result_name}.tif")
                
                try:
                    with rasterio.open(output_path, 'w', **output_profile) as dst:
                        dst.write(result_data.astype(np.float32), 1)
                    saved_files.append(output_path)
                    print(f"Saved {result_name} to {output_path}")
                except Exception as e:
                    print(f"Failed to save {result_name}: {str(e)}")
        
        # Save metadata
        metadata = {
            'processing_date': str(np.datetime64('now')),
            'normalization_params': self.normalization_params,
            'spatial_dimensions': self.spatial_dims,
            'wavelengths': self.wavelengths.tolist() if self.wavelengths is not None else None,
            'mineral_signatures': self.mineral_signatures,
            'valid_pixel_count': int(np.sum(self.valid_mask)) if self.valid_mask is not None else 0,
            'saved_files': saved_files
        }
        
        metadata_path = os.path.join(output_dir, "processing_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return saved_files