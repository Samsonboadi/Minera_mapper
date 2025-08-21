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
import traceback
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
        """Main processing function that runs in separate thread - FIXED VERSION"""
        try:
            # Create callback functions for the processor
            def log_callback(message):
                """Thread-safe log callback"""
                self.log_message.emit(message)
            
            def progress_callback(value, message):
                """Thread-safe progress callback"""
                self.progress_updated.emit(value, message)
            
            def should_stop_callback():
                """Thread-safe stop check callback"""
                return self.should_stop
            
            # Start processing
            log_callback("🚀 Starting ASTER data processing...")
            progress_callback(5, "Initializing processing...")
            
            if self.should_stop:
                return
            
            # Validate file
            log_callback(f"📁 Validating file: {os.path.basename(self.file_path)}")
            if not self.processor.validate_aster_file(self.file_path):
                self.error_occurred.emit("File validation failed")
                return
            
            progress_callback(10, "File validated successfully")
            
            if self.should_stop:
                return
            
            # Process the file with proper callbacks
            result = self.processor.process_aster_file_threaded(
                self.file_path, 
                progress_callback,
                log_callback,
                should_stop_callback
            )
            
            if result and not self.should_stop:
                log_callback("✅ ASTER processing completed successfully!")
                self.processing_finished.emit(True, result)
            elif self.should_stop:
                log_callback("⏹️ Processing cancelled by user")
                self.processing_finished.emit(False, {})
            else:
                self.error_occurred.emit("Processing failed")
                
        except Exception as e:
            error_msg = f"Processing error: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)
    
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
    
    def read_hdf_with_gdal(self, file_path, log_callback):
        """Read HDF file using GDAL with improved error handling - FIXED VERSION"""
        gdal.UseExceptions()
        
        try:
            log_callback(f"🔧 Opening HDF file with GDAL...")
            dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
            if dataset is None:
                raise Exception("GDAL could not open the file")
            
            data = {}
            subdatasets = dataset.GetSubDatasets()
            
            if subdatasets:
                log_callback(f"📊 Found {len(subdatasets)} subdatasets")
                
                for i, subdataset in enumerate(subdatasets):
                    name = subdataset[1].upper()
                    path = subdataset[0]
                    
                    log_callback(f"📋 Processing subdataset {i+1}: {name}")
                    
                    if 'QA_DATAPLANE' in name:
                        log_callback(f"⏭️ Skipping QA band: {name}")
                        continue
                    
                    try:
                        sub_ds = gdal.Open(path, gdal.GA_ReadOnly)
                        if sub_ds is None:
                            log_callback(f"⚠️ Could not open subdataset: {name}")
                            continue
                        
                        log_callback(f"📏 Subdataset dimensions: {sub_ds.RasterXSize}x{sub_ds.RasterYSize}, {sub_ds.RasterCount} bands")
                        
                        bands = []
                        for j in range(1, sub_ds.RasterCount + 1):
                            band = sub_ds.GetRasterBand(j)
                            
                            try:
                                band_type = band.DataType
                                band_xsize = band.XSize
                                band_ysize = band.YSize
                                
                                # === FIXED SECTION: Multiple reading strategies ===
                                band_data = None
                                
                                # Strategy 1: Standard ReadAsArray
                                try:
                                    band_data = band.ReadAsArray()
                                    if band_data is not None and isinstance(band_data, np.ndarray) and band_data.size > 0:
                                        log_callback(f"✅ Strategy 1 (ReadAsArray) successful for band {j}")
                                    else:
                                        band_data = None
                                except Exception as e:
                                    log_callback(f"⚠️ Strategy 1 failed: {str(e)}")
                                    band_data = None
                                
                                # Strategy 2: ReadAsArray with explicit parameters
                                if band_data is None:
                                    try:
                                        log_callback(f"🔄 Trying Strategy 2: ReadAsArray with explicit parameters for band {j}...")
                                        band_data = band.ReadAsArray(0, 0, band_xsize, band_ysize)
                                        if band_data is not None and isinstance(band_data, np.ndarray) and band_data.size > 0:
                                            log_callback(f"✅ Strategy 2 successful for band {j}")
                                        else:
                                            band_data = None
                                    except Exception as e:
                                        log_callback(f"⚠️ Strategy 2 failed: {str(e)}")
                                        band_data = None
                                
                                # Strategy 3: Chunked reading test first
                                if band_data is None:
                                    try:
                                        log_callback(f"🔄 Trying Strategy 3: Chunked reading test for band {j}...")
                                        chunk_size = min(100, band_xsize, band_ysize)
                                        test_chunk = band.ReadAsArray(0, 0, chunk_size, chunk_size)
                                        
                                        if test_chunk is not None and isinstance(test_chunk, np.ndarray) and test_chunk.size > 0:
                                            log_callback(f"✅ Test chunk successful, reading full band {j}...")
                                            band_data = band.ReadAsArray(0, 0, band_xsize, band_ysize)
                                            
                                            if band_data is not None and isinstance(band_data, np.ndarray) and band_data.size > 0:
                                                log_callback(f"✅ Strategy 3 successful for band {j}")
                                            else:
                                                band_data = None
                                        else:
                                            log_callback(f"⚠️ Test chunk failed for band {j}")
                                            band_data = None
                                    except Exception as e:
                                        log_callback(f"⚠️ Strategy 3 failed: {str(e)}")
                                        band_data = None
                                
                                # Strategy 4: Pre-allocated numpy array with ReadAsArray
                                if band_data is None:
                                    try:
                                        log_callback(f"🔄 Trying Strategy 4: Pre-allocated array for band {j}...")
                                        dtype_map = {
                                            gdal.GDT_Byte: np.uint8,
                                            gdal.GDT_UInt16: np.uint16,
                                            gdal.GDT_Int16: np.int16,
                                            gdal.GDT_UInt32: np.uint32,
                                            gdal.GDT_Int32: np.int32,
                                            gdal.GDT_Float32: np.float32,
                                            gdal.GDT_Float64: np.float64
                                        }
                                        numpy_dtype = dtype_map.get(band_type, np.int16)
                                        
                                        # Pre-allocate array
                                        band_data = np.zeros((band_ysize, band_xsize), dtype=numpy_dtype)
                                        result = band.ReadAsArray(buf_obj=band_data)
                                        
                                        if result is not None and np.any(band_data != 0):
                                            log_callback(f"✅ Strategy 4 successful for band {j}")
                                        else:
                                            log_callback(f"⚠️ Strategy 4 returned empty data for band {j}")
                                            band_data = None
                                    except Exception as e:
                                        log_callback(f"⚠️ Strategy 4 failed: {str(e)}")
                                        band_data = None
                                
                                # Strategy 5: GDAL ReadRaster with manual conversion
                                if band_data is None:
                                    try:
                                        log_callback(f"🔄 Trying Strategy 5: ReadRaster for band {j}...")
                                        raw_data = band.ReadRaster(0, 0, band_xsize, band_ysize)
                                        
                                        if raw_data:
                                            dtype_map = {
                                                gdal.GDT_Byte: np.uint8,
                                                gdal.GDT_UInt16: np.uint16,
                                                gdal.GDT_Int16: np.int16,
                                                gdal.GDT_UInt32: np.uint32,
                                                gdal.GDT_Int32: np.int32,
                                                gdal.GDT_Float32: np.float32,
                                                gdal.GDT_Float64: np.float64
                                            }
                                            numpy_dtype = dtype_map.get(band_type, np.int16)
                                            
                                            try:
                                                band_data = np.frombuffer(raw_data, dtype=numpy_dtype).reshape(band_ysize, band_xsize)
                                                if band_data is not None and band_data.size > 0:
                                                    log_callback(f"✅ Strategy 5 successful for band {j}")
                                                else:
                                                    band_data = None
                                            except Exception as reshape_error:
                                                log_callback(f"⚠️ Strategy 5 reshape failed: {str(reshape_error)}")
                                                band_data = None
                                        else:
                                            log_callback(f"⚠️ Strategy 5: ReadRaster returned no data for band {j}")
                                            band_data = None
                                    except Exception as e:
                                        log_callback(f"⚠️ Strategy 5 failed: {str(e)}")
                                        band_data = None
                                
                                # Strategy 6: Block-by-block reading
                                if band_data is None:
                                    try:
                                        log_callback(f"🔄 Trying Strategy 6: Block-by-block reading for band {j}...")
                                        
                                        # Initialize array
                                        dtype_map = {
                                            gdal.GDT_Byte: np.uint8,
                                            gdal.GDT_UInt16: np.uint16,
                                            gdal.GDT_Int16: np.int16,
                                            gdal.GDT_UInt32: np.uint32,
                                            gdal.GDT_Int32: np.int32,
                                            gdal.GDT_Float32: np.float32,
                                            gdal.GDT_Float64: np.float64
                                        }
                                        numpy_dtype = dtype_map.get(band_type, np.int16)
                                        band_data = np.zeros((band_ysize, band_xsize), dtype=numpy_dtype)
                                        
                                        # Read in blocks
                                        block_size = min(512, band_xsize, band_ysize)
                                        successful_blocks = 0
                                        total_blocks = 0
                                        
                                        for y in range(0, band_ysize, block_size):
                                            for x in range(0, band_xsize, block_size):
                                                x_size = min(block_size, band_xsize - x)
                                                y_size = min(block_size, band_ysize - y)
                                                total_blocks += 1
                                                
                                                try:
                                                    block_data = band.ReadAsArray(x, y, x_size, y_size)
                                                    if block_data is not None and isinstance(block_data, np.ndarray):
                                                        band_data[y:y+y_size, x:x+x_size] = block_data
                                                        successful_blocks += 1
                                                except Exception as block_error:
                                                    log_callback(f"⚠️ Block read failed at ({x},{y}): {str(block_error)}")
                                        
                                        if successful_blocks > 0:
                                            success_rate = successful_blocks / total_blocks
                                            log_callback(f"✅ Strategy 6: Read {successful_blocks}/{total_blocks} blocks ({success_rate:.1%}) for band {j}")
                                            if success_rate > 0.5:  # Accept if we got more than 50% of blocks
                                                log_callback(f"✅ Strategy 6 successful for band {j}")
                                            else:
                                                log_callback(f"⚠️ Strategy 6: Insufficient data ({success_rate:.1%}) for band {j}")
                                                band_data = None
                                        else:
                                            log_callback(f"⚠️ Strategy 6: No blocks successfully read for band {j}")
                                            band_data = None
                                            
                                    except Exception as e:
                                        log_callback(f"⚠️ Strategy 6 failed: {str(e)}")
                                        band_data = None
                                
                                # Final validation
                                if band_data is None:
                                    log_callback(f"❌ All read strategies failed for band {j} in subdataset: {name}")
                                    continue
                                
                                # Verify we have a valid numpy array
                                if not isinstance(band_data, np.ndarray):
                                    log_callback(f"❌ Read data is not a numpy array for band {j}: {type(band_data)}")
                                    continue
                                
                                if band_data.size == 0:
                                    log_callback(f"❌ Read data is empty for band {j}")
                                    continue
                                
                                # Check for valid data range (ASTER L2 should be 0-10000 range typically)
                                data_min, data_max = np.nanmin(band_data), np.nanmax(band_data)
                                log_callback(f"📊 Band {j} data range: {data_min} to {data_max}")
                                
                                # Convert to reflectance if needed (ASTER L2 uses scale factor 0.001)
                                if data_max > 10:  # Likely needs scaling
                                    band_data = band_data.astype(np.float32) * 0.001
                                    band_data = np.clip(band_data, 0, 1)
                                    log_callback(f"🔄 Applied reflectance scaling to band {j}")
                                
                                bands.append(band_data)
                                log_callback(f"✅ Successfully read band {j}: {band_data.shape}, dtype: {band_data.dtype}")
                                
                            except Exception as band_error:
                                log_callback(f"❌ Error processing band {j}: {str(band_error)}")
                                continue
                        
                        # Store the bands if we got any
                        if bands:
                            # Determine band type from subdataset name
                            if 'VNIR' in name.upper():
                                if 'vnir_bands' not in data:
                                    data['vnir_bands'] = {}
                                    data['vnir_geotransform'] = sub_ds.GetGeoTransform()
                                    data['vnir_projection'] = sub_ds.GetProjection()
                                
                                # Map to specific VNIR bands
                                for idx, band_data in enumerate(bands):
                                    band_names = ['BAND1', 'BAND2', 'BAND3N']
                                    if idx < len(band_names):
                                        data['vnir_bands'][band_names[idx]] = band_data
                                        log_callback(f"✅ Stored {band_names[idx]} from VNIR subdataset")
                            
                            elif 'SWIR' in name.upper():
                                if 'swir_bands' not in data:
                                    data['swir_bands'] = {}
                                    data['swir_geotransform'] = sub_ds.GetGeoTransform()
                                    data['swir_projection'] = sub_ds.GetProjection()
                                
                                # Map to specific SWIR bands
                                for idx, band_data in enumerate(bands):
                                    band_names = ['BAND4', 'BAND5', 'BAND6', 'BAND7', 'BAND8', 'BAND9']
                                    if idx < len(band_names):
                                        data['swir_bands'][band_names[idx]] = band_data
                                        log_callback(f"✅ Stored {band_names[idx]} from SWIR subdataset")
                            
                            else:
                                # Generic storage for unknown band types
                                log_callback(f"⚠️ Unknown subdataset type: {name}, storing generically")
                                data[f'subdataset_{i}'] = bands
                        
                        else:
                            log_callback(f"❌ No bands successfully read from subdataset: {name}")
                    
                    except Exception as subdataset_error:
                        log_callback(f"❌ Error processing subdataset {name}: {str(subdataset_error)}")
                        continue
            
            else:
                log_callback("❌ No subdatasets found in HDF file")
                return {}
            
            # Calculate total bands extracted
            total_vnir = len(data.get('vnir_bands', {}))
            total_swir = len(data.get('swir_bands', {}))
            total_bands = total_vnir + total_swir
            
            if total_bands == 0:
                raise Exception("No valid band data could be extracted from HDF file")
            
            log_callback(f"✅ Successfully extracted {total_bands} bands from HDF file ({total_vnir} VNIR, {total_swir} SWIR)")
            return data
            
        except Exception as e:
            log_callback(f"❌ GDAL read error for {file_path}: {str(e)}")
            raise




    def validate_extracted_data(self, data, log_callback):
        """Validate that extracted data is usable"""
        try:
            log_callback("🔍 Validating extracted data...")
            
            if not data:
                log_callback("❌ No data to validate")
                return False
            
            total_bands = 0
            
            # Check VNIR bands
            if 'vnir_bands' in data:
                vnir_bands = data['vnir_bands']
                log_callback(f"📊 VNIR bands found: {list(vnir_bands.keys())}")
                
                for band_name, band_data in vnir_bands.items():
                    if isinstance(band_data, np.ndarray) and band_data.size > 0:
                        total_bands += 1
                        valid_pixels = np.sum((band_data > 0) & (~np.isnan(band_data)))
                        total_pixels = band_data.size
                        log_callback(f"✅ {band_name}: {band_data.shape}, {valid_pixels}/{total_pixels} valid pixels")
                    else:
                        log_callback(f"❌ {band_name}: Invalid data")
            
            # Check SWIR bands
            if 'swir_bands' in data:
                swir_bands = data['swir_bands']
                log_callback(f"📊 SWIR bands found: {list(swir_bands.keys())}")
                
                for band_name, band_data in swir_bands.items():
                    if isinstance(band_data, np.ndarray) and band_data.size > 0:
                        total_bands += 1
                        valid_pixels = np.sum((band_data > 0) & (~np.isnan(band_data)))
                        total_pixels = band_data.size
                        log_callback(f"✅ {band_name}: {band_data.shape}, {valid_pixels}/{total_pixels} valid pixels")
                    else:
                        log_callback(f"❌ {band_name}: Invalid data")
            
            if total_bands >= 3:  # Need at least 3 bands for basic processing
                log_callback(f"✅ Data validation successful: {total_bands} valid bands")
                return True
            else:
                log_callback(f"❌ Data validation failed: Only {total_bands} valid bands (need at least 3)")
                return False
                
        except Exception as e:
            log_callback(f"❌ Data validation error: {str(e)}")
            return False



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