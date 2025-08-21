"""
Complete Fixed ASTER Processor - HDF-EOS Compatible
Replace your entire aster_processor.py file with this version
"""

import os
import numpy as np
import tempfile
import zipfile
import shutil
import datetime
from qgis.PyQt.QtCore import QThread, pyqtSignal
from qgis.core import QgsRasterLayer, QgsProject, QgsMessageLog, Qgis

try:
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_bounds
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from osgeo import gdal, osr
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False

class AsterProcessingThread(QThread):
    """Enhanced ASTER processing thread - HDF-EOS COMPATIBLE"""
    
    progress_updated = pyqtSignal(int, str)
    log_message = pyqtSignal(str)
    processing_finished = pyqtSignal(bool, dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, file_path, processor=None):
        super().__init__()
        self.file_path = file_path
        self.processor = processor
        self.should_stop = False
        self.target_resolution = 15.0
        self.temp_dirs = []
    
    def stop(self):
        """Stop processing"""
        self.should_stop = True
    
    def run(self):
        """Main processing with HDF-EOS support"""
        try:
            self.log_message.emit("Starting ASTER processing with HDF-EOS support...")
            self.progress_updated.emit(5, "Initializing...")
            
            # Handle ZIP extraction
            if self.file_path.lower().endswith('.zip'):
                hdf_files = self.extract_zip_file()
                if not hdf_files:
                    self.error_occurred.emit("No HDF files found in ZIP")
                    return
            else:
                hdf_files = [self.file_path]
            
            self.progress_updated.emit(15, "Loading ASTER bands using GDAL...")
            
            # CRITICAL FIX: Use GDAL-based HDF reading
            all_band_data = []
            geotransforms = []
            
            for i, hdf_file in enumerate(hdf_files):
                self.log_message.emit(f"Processing HDF file {i+1}/{len(hdf_files)}: {os.path.basename(hdf_file)}")
                
                # Use GDAL method for HDF-EOS
                extracted_data = self.read_hdf_eos_file(hdf_file)
                if extracted_data:
                    all_band_data.append(extracted_data)
                    # Store geotransforms for later use
                    if 'vnir_geotransform' in extracted_data:
                        geotransforms.append(extracted_data['vnir_geotransform'])
                    elif 'swir_geotransform' in extracted_data:
                        geotransforms.append(extracted_data['swir_geotransform'])
            
            if not all_band_data:
                self.error_occurred.emit("No valid data could be extracted from HDF files")
                return
            
            self.progress_updated.emit(40, "Processing and resampling bands...")
            
            # Process the extracted data
            combined_result = self.process_combined_data(all_band_data, geotransforms)
            
            if combined_result:
                self.progress_updated.emit(90, "Adding to QGIS...")
                layer = self.create_qgis_layer(combined_result['output_path'])
                
                self.progress_updated.emit(100, "Processing complete!")
                
                results = {
                    'layer': layer,
                    'file_path': combined_result['output_path'],
                    'band_count': combined_result.get('band_count', 0),
                    'resolution': self.target_resolution
                }
                
                self.processing_finished.emit(True, results)
            else:
                self.error_occurred.emit("Failed to process combined data")
            
        except Exception as e:
            import traceback
            error_msg = f"Processing failed: {str(e)}\n{traceback.format_exc()}"
            self.log_message.emit(error_msg)
            self.error_occurred.emit(str(e))
        finally:
            self.cleanup_temp_dirs()
    
    def extract_zip_file(self):
        """Extract ZIP file and find HDF files"""
        try:
            temp_dir = tempfile.mkdtemp(prefix='aster_processing_')
            self.temp_dirs.append(temp_dir)
            
            self.log_message.emit(f"Extracting ZIP to: {temp_dir}")
            
            with zipfile.ZipFile(self.file_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find HDF files recursively
            hdf_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith(('.hdf', '.h5')):
                        full_path = os.path.join(root, file)
                        hdf_files.append(full_path)
                        self.log_message.emit(f"Found HDF file: {file}")
            
            self.log_message.emit(f"Total HDF files found: {len(hdf_files)}")
            return hdf_files
            
        except Exception as e:
            self.log_message.emit(f"ZIP extraction failed: {str(e)}")
            return []
    
    def read_hdf_eos_file(self, hdf_path):
        """Corrected HDF-EOS reader for ASTER Surface Reflectance data"""
        try:
            self.log_message.emit(f"Reading HDF-EOS file: {os.path.basename(hdf_path)}")
            
            if not HAS_GDAL:
                raise Exception("GDAL not available for HDF-EOS reading")
            
            gdal.UseExceptions()
            
            # Open the HDF file
            dataset = gdal.Open(hdf_path, gdal.GA_ReadOnly)
            if dataset is None:
                raise Exception(f"GDAL could not open HDF file: {hdf_path}")
            
            # Get subdatasets
            subdatasets = dataset.GetSubDatasets()
            self.log_message.emit(f"Found {len(subdatasets)} subdatasets")
            
            if not subdatasets:
                raise Exception("No subdatasets found in HDF-EOS file")
            
            extracted_data = {
                'vnir_bands': {},
                'swir_bands': {}
            }
            
            for i, (subdataset_path, subdataset_desc) in enumerate(subdatasets):
                try:
                    self.log_message.emit(f"Processing subdataset {i+1}: {subdataset_desc}")
                    
                    # Skip QA and metadata subdatasets
                    desc_upper = subdataset_desc.upper()
                    if any(skip_term in desc_upper for skip_term in 
                        ['QA', 'QUALITY', 'DATAPLANE', 'FLAG', 'METADATA']):
                        self.log_message.emit(f"Skipping: {subdataset_desc}")
                        continue
                    
                    # CRITICAL FIX: Build proper HDF4_EOS subdataset path
                    # The subdataset_path from GetSubDatasets() may not be complete
                    
                    # Extract the actual subdataset name from description
                    if 'SurfaceReflectanceVNIR' in subdataset_desc:
                        # VNIR subdatasets
                        if 'Band1' in subdataset_desc:
                            eos_path = f'HDF4_EOS:EOS_SWATH:"{hdf_path}":SurfaceReflectanceVNIR:Band1'
                            band_key = 'band1'
                            band_type = 'VNIR'
                        elif 'Band2' in subdataset_desc:
                            eos_path = f'HDF4_EOS:EOS_SWATH:"{hdf_path}":SurfaceReflectanceVNIR:Band2'
                            band_key = 'band2'
                            band_type = 'VNIR'
                        elif 'Band3N' in subdataset_desc:
                            eos_path = f'HDF4_EOS:EOS_SWATH:"{hdf_path}":SurfaceReflectanceVNIR:Band3N'
                            band_key = 'band3n'
                            band_type = 'VNIR'
                        else:
                            continue
                            
                    elif 'SurfaceReflectanceSWIR' in subdataset_desc:
                        # SWIR subdatasets
                        if 'Band4' in subdataset_desc:
                            eos_path = f'HDF4_EOS:EOS_SWATH:"{hdf_path}":SurfaceReflectanceSWIR:Band4'
                            band_key = 'band4'
                            band_type = 'SWIR'
                        elif 'Band5' in subdataset_desc:
                            eos_path = f'HDF4_EOS:EOS_SWATH:"{hdf_path}":SurfaceReflectanceSWIR:Band5'
                            band_key = 'band5'
                            band_type = 'SWIR'
                        elif 'Band6' in subdataset_desc:
                            eos_path = f'HDF4_EOS:EOS_SWATH:"{hdf_path}":SurfaceReflectanceSWIR:Band6'
                            band_key = 'band6'
                            band_type = 'SWIR'
                        elif 'Band7' in subdataset_desc:
                            eos_path = f'HDF4_EOS:EOS_SWATH:"{hdf_path}":SurfaceReflectanceSWIR:Band7'
                            band_key = 'band7'
                            band_type = 'SWIR'
                        elif 'Band8' in subdataset_desc:
                            eos_path = f'HDF4_EOS:EOS_SWATH:"{hdf_path}":SurfaceReflectanceSWIR:Band8'
                            band_key = 'band8'
                            band_type = 'SWIR'
                        elif 'Band9' in subdataset_desc:
                            eos_path = f'HDF4_EOS:EOS_SWATH:"{hdf_path}":SurfaceReflectanceSWIR:Band9'
                            band_key = 'band9'
                            band_type = 'SWIR'
                        else:
                            continue
                    else:
                        # Try using the original subdataset path
                        eos_path = subdataset_path
                        band_key = f'band_{i}'
                        band_type = 'UNKNOWN'
                    
                    self.log_message.emit(f"Trying EOS path: {eos_path}")
                    
                    # Open the subdataset using the proper EOS path
                    sub_ds = gdal.Open(eos_path, gdal.GA_ReadOnly)
                    if sub_ds is None:
                        self.log_message.emit(f"⚠️ Could not open EOS path, trying original...")
                        # Fallback to original path
                        sub_ds = gdal.Open(subdataset_path, gdal.GA_ReadOnly)
                        if sub_ds is None:
                            self.log_message.emit(f"⚠️ Could not open subdataset: {subdataset_desc}")
                            continue
                    
                    # Read the data
                    band_array = None
                    
                    try:
                        # Get dimensions first
                        xsize = sub_ds.RasterXSize
                        ysize = sub_ds.RasterYSize
                        band_count = sub_ds.RasterCount
                        
                        self.log_message.emit(f"Subdataset info: {xsize}x{ysize}, {band_count} bands")
                        
                        if band_count > 0:
                            band = sub_ds.GetRasterBand(1)
                            if band is not None:
                                # Try reading with explicit parameters
                                band_array = band.ReadAsArray(0, 0, xsize, ysize)
                                
                                if band_array is None:
                                    self.log_message.emit(f"⚠️ ReadAsArray failed, trying ReadRaster...")
                                    
                                    # Try ReadRaster as backup
                                    raw_data = sub_ds.ReadRaster(0, 0, xsize, ysize, xsize, ysize, gdal.GDT_UInt16)
                                    if raw_data:
                                        band_array = np.frombuffer(raw_data, dtype=np.uint16)
                                        band_array = band_array.reshape(ysize, xsize)
                        
                        # Validate the result
                        if band_array is not None and isinstance(band_array, np.ndarray):
                            if band_array.size > 0 and len(band_array.shape) == 2:
                                # Get geospatial information
                                geotransform = sub_ds.GetGeoTransform()
                                projection = sub_ds.GetProjection()
                                
                                self.log_message.emit(f"✅ Successfully read: {band_array.shape}, dtype: {band_array.dtype}")
                                self.log_message.emit(f"   Data range: {np.min(band_array)} to {np.max(band_array)}")
                                
                                # Store the band data
                                if band_type == 'VNIR':
                                    extracted_data['vnir_bands'][band_key] = band_array
                                    if 'vnir_geotransform' not in extracted_data:
                                        extracted_data['vnir_geotransform'] = geotransform
                                        extracted_data['vnir_projection'] = projection
                                    self.log_message.emit(f"✅ Stored VNIR {band_key}")
                                    
                                elif band_type == 'SWIR':
                                    extracted_data['swir_bands'][band_key] = band_array
                                    if 'swir_geotransform' not in extracted_data:
                                        extracted_data['swir_geotransform'] = geotransform
                                        extracted_data['swir_projection'] = projection
                                    self.log_message.emit(f"✅ Stored SWIR {band_key}")
                            else:
                                self.log_message.emit(f"⚠️ Invalid array: shape={band_array.shape}, size={band_array.size}")
                        else:
                            self.log_message.emit(f"⚠️ No valid data read from subdataset")
                    
                    except Exception as read_error:
                        self.log_message.emit(f"❌ Read error: {str(read_error)}")
                        continue
                    
                    finally:
                        if sub_ds is not None:
                            sub_ds = None
                    
                except Exception as e:
                    self.log_message.emit(f"⚠️ Subdataset {i+1} failed: {str(e)}")
                    continue
            
            # Clean up main dataset
            dataset = None
            
            # Validate results
            vnir_count = len(extracted_data.get('vnir_bands', {}))
            swir_count = len(extracted_data.get('swir_bands', {}))
            total_bands = vnir_count + swir_count
            
            if total_bands == 0:
                raise Exception("No valid band data extracted from any subdatasets")
            
            self.log_message.emit(f"✅ Extraction complete: {vnir_count} VNIR, {swir_count} SWIR bands")
            self.log_message.emit(f"   VNIR bands: {list(extracted_data.get('vnir_bands', {}).keys())}")
            self.log_message.emit(f"   SWIR bands: {list(extracted_data.get('swir_bands', {}).keys())}")
            
            return extracted_data
            
        except Exception as e:
            self.log_message.emit(f"❌ HDF-EOS read failed: {str(e)}")
            return None
    
    def process_combined_data(self, all_band_data, geotransforms):
        """Process and resample combined ASTER data"""
        try:
            self.log_message.emit("Processing combined ASTER data...")
            
            # Collect all bands for processing
            all_bands = []
            band_info = []
            reference_geotransform = None
            
            for data_dict in all_band_data:
                # Process VNIR bands (15m native)
                vnir_bands = data_dict.get('vnir_bands', {})
                for band_name, band_data in vnir_bands.items():
                    all_bands.append(band_data)
                    band_info.append({
                        'name': f'VNIR_{band_name}',
                        'resolution': 15,
                        'type': 'VNIR'
                    })
                    if reference_geotransform is None:
                        reference_geotransform = data_dict.get('vnir_geotransform')
                
                # Process SWIR bands (30m native)
                swir_bands = data_dict.get('swir_bands', {})
                for band_name, band_data in swir_bands.items():
                    all_bands.append(band_data)
                    band_info.append({
                        'name': f'SWIR_{band_name}',
                        'resolution': 30,
                        'type': 'SWIR'
                    })
                    if reference_geotransform is None:
                        reference_geotransform = data_dict.get('swir_geotransform')
            
            if not all_bands:
                raise Exception("No bands available for processing")
            
            self.log_message.emit(f"Processing {len(all_bands)} bands...")
            
            # Resample all bands to target resolution (15m)
            resampled_bands = []
            
            for i, (band_data, info) in enumerate(zip(all_bands, band_info)):
                try:
                    if info['resolution'] == self.target_resolution:
                        # Already at target resolution
                        resampled_bands.append(band_data.astype(np.float32))
                        self.log_message.emit(f"✅ {info['name']}: Using original {info['resolution']}m data")
                    else:
                        # Resample to target resolution
                        self.log_message.emit(f"🔄 {info['name']}: Resampling from {info['resolution']}m to {self.target_resolution}m...")
                        
                        # Use scipy zoom for resampling
                        try:
                            from scipy.ndimage import zoom
                            zoom_factor = info['resolution'] / self.target_resolution
                            resampled_data = zoom(band_data, zoom_factor, order=1).astype(np.float32)
                            resampled_bands.append(resampled_data)
                            self.log_message.emit(f"✅ {info['name']}: Resampled to {resampled_data.shape}")
                        except ImportError:
                            # Fallback: use original data
                            self.log_message.emit(f"⚠️ SciPy not available, using original data for {info['name']}")
                            resampled_bands.append(band_data.astype(np.float32))
                    
                    # Update progress
                    progress = 40 + (i / len(all_bands)) * 40
                    self.progress_updated.emit(int(progress), f"Processed {info['name']}")
                    
                except Exception as e:
                    self.log_message.emit(f"⚠️ Failed to process {info['name']}: {str(e)}")
                    # Use original data as fallback
                    resampled_bands.append(band_data.astype(np.float32))
            
            if not resampled_bands:
                raise Exception("No bands were successfully processed")
            
            # Apply normalization
            self.progress_updated.emit(80, "Applying normalization...")
            normalized_bands = self.apply_normalization(resampled_bands)
            
            # Create combined dataset
            self.progress_updated.emit(85, "Creating combined dataset...")
            output_path = self.save_combined_dataset(normalized_bands, reference_geotransform, band_info)
            
            return {
                'output_path': output_path,
                'band_count': len(normalized_bands),
                'band_info': band_info
            }
            
        except Exception as e:
            self.log_message.emit(f"❌ Combined data processing failed: {str(e)}")
            return None
    
    def apply_normalization(self, bands, method='percentile'):
        """Apply pixel normalization to bands"""
        try:
            self.log_message.emit(f"Applying {method} normalization...")
            normalized_bands = []
            
            for i, band_data in enumerate(bands):
                try:
                    # Create valid pixel mask
                    valid_mask = (band_data > 0) & (band_data < 65535) & np.isfinite(band_data)
                    valid_pixels = band_data[valid_mask]
                    
                    if len(valid_pixels) > 100:
                        if method == 'percentile':
                            # Robust percentile normalization (2-98%)
                            p2, p98 = np.percentile(valid_pixels, [2, 98])
                            if p98 > p2:
                                normalized = np.clip((band_data - p2) / (p98 - p2), 0, 1)
                                normalized_bands.append(normalized.astype(np.float32))
                                self.log_message.emit(f"✅ Band {i+1}: Normalized using percentiles")
                            else:
                                normalized_bands.append(band_data.astype(np.float32))
                        
                        elif method == 'min_max':
                            min_val, max_val = np.min(valid_pixels), np.max(valid_pixels)
                            if max_val > min_val:
                                normalized = (band_data - min_val) / (max_val - min_val)
                                normalized_bands.append(normalized.astype(np.float32))
                                self.log_message.emit(f"✅ Band {i+1}: Min-max normalized")
                            else:
                                normalized_bands.append(band_data.astype(np.float32))
                    else:
                        self.log_message.emit(f"⚠️ Band {i+1}: Insufficient valid pixels, using original")
                        normalized_bands.append(band_data.astype(np.float32))
                
                except Exception as e:
                    self.log_message.emit(f"⚠️ Normalization failed for band {i+1}: {str(e)}")
                    normalized_bands.append(band_data.astype(np.float32))
            
            return normalized_bands
            
        except Exception as e:
            self.log_message.emit(f"⚠️ Normalization process failed: {str(e)}")
            return bands
    
    def save_combined_dataset(self, bands, geotransform, band_info):
        """Save combined dataset as GeoTIFF"""
        try:
            # Create temporary output file
            temp_dir = tempfile.mkdtemp(prefix='aster_combined_')
            self.temp_dirs.append(temp_dir)
            output_path = os.path.join(temp_dir, 'aster_combined_15m.tif')
            
            # Stack bands
            combined_data = np.stack(bands, axis=0)
            n_bands, height, width = combined_data.shape
            
            self.log_message.emit(f"Saving combined dataset: {n_bands} bands, {height}x{width} pixels")
            
            if HAS_GDAL:
                # Use GDAL for saving
                driver = gdal.GetDriverByName('GTiff')
                out_ds = driver.Create(
                    output_path, width, height, n_bands, gdal.GDT_Float32,
                    ['COMPRESS=LZW', 'TILED=YES', 'BLOCKXSIZE=512', 'BLOCKYSIZE=512']
                )
                
                if geotransform:
                    out_ds.SetGeoTransform(geotransform)
                
                # Set projection (default to WGS84)
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(4326)
                out_ds.SetProjection(srs.ExportToWkt())
                
                # Write bands
                for i in range(n_bands):
                    out_band = out_ds.GetRasterBand(i + 1)
                    out_band.WriteArray(combined_data[i])
                    out_band.SetNoDataValue(-9999)
                    
                    # Set band description
                    if i < len(band_info):
                        out_band.SetDescription(band_info[i]['name'])
                
                # Write metadata
                metadata = {
                    'PROCESSING_DATE': str(datetime.datetime.now()),
                    'TARGET_RESOLUTION': str(self.target_resolution),
                    'BAND_COUNT': str(n_bands)
                }
                out_ds.SetMetadata(metadata)
                
                out_ds = None  # Close file
                
            elif HAS_RASTERIO:
                # Fallback to rasterio
                profile = {
                    'driver': 'GTiff',
                    'count': n_bands,
                    'height': height,
                    'width': width,
                    'dtype': 'float32',
                    'nodata': -9999,
                    'compress': 'lzw'
                }
                
                if geotransform:
                    from rasterio.transform import Affine
                    profile['transform'] = Affine.from_gdal(*geotransform)
                
                with rasterio.open(output_path, 'w', **profile) as dst:
                    for i in range(n_bands):
                        dst.write(combined_data[i], i + 1)
            
            else:
                raise Exception("Neither GDAL nor rasterio available for saving")
            
            self.log_message.emit(f"✅ Saved combined dataset: {output_path}")
            return output_path
            
        except Exception as e:
            raise Exception(f"Failed to save combined dataset: {str(e)}")
    
    def create_qgis_layer(self, file_path):
        """Create QGIS layer from processed file"""
        try:
            layer_name = f"ASTER_Combined_15m_{datetime.datetime.now().strftime('%H%M%S')}"
            layer = QgsRasterLayer(file_path, layer_name)
            
            if not layer.isValid():
                raise Exception(f"Invalid QGIS layer: {file_path}")
            
            # Set layer properties
            layer.setCustomProperty('layer_type', 'spectral')
            layer.setCustomProperty('aster_type', 'COMBINED_RESAMPLED')
            layer.setCustomProperty('target_resolution', self.target_resolution)
            layer.setCustomProperty('processing_date', str(datetime.datetime.now()))
            
            # Add to QGIS project
            QgsProject.instance().addMapLayer(layer)
            
            self.log_message.emit(f"✅ Added layer to project: {layer.name()}")
            return layer
            
        except Exception as e:
            raise Exception(f"Failed to create QGIS layer: {str(e)}")
    
    def cleanup_temp_dirs(self):
        """Clean up temporary directories"""
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    self.log_message.emit(f"Cleaned up: {temp_dir}")
            except Exception as e:
                self.log_message.emit(f"Cleanup warning: {str(e)}")

# For compatibility with existing code
class AsterProcessor:
    """Compatibility wrapper for existing code"""
    
    def __init__(self, iface):
        self.iface = iface
    
    def process_specific_file(self, file_path, processing_options=None):
        """Process ASTER file using the thread"""
        try:
            # Create and run processing thread synchronously
            thread = AsterProcessingThread(file_path)
            
            # Store results
            self.results = None
            self.success = False
            
            def on_finished(success, results):
                self.success = success
                self.results = results
            
            def on_error(error):
                self.success = False
                self.error_message = error
            
            thread.processing_finished.connect(on_finished)
            thread.error_occurred.connect(on_error)
            
            # Run in current thread for compatibility
            thread.run()
            
            return self.success
            
        except Exception as e:
            print(f"AsterProcessor failed: {str(e)}")
            return False