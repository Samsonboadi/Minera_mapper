"""
Complete Fixed ASTER Processor - HDF-EOS Compatible
Replace your entire processing/aster_processor.py file with this version
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

class AsterProcessor:
    """Main ASTER processor class with all required methods"""
    
    def __init__(self, iface=None):
        self.iface = iface
        self.temp_dirs = []
        self.target_resolution = 15.0
        
    def validate_aster_file(self, file_path):
        """Validate ASTER file - CRITICAL MISSING METHOD"""
        try:
            if not os.path.exists(file_path):
                return False
            
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext == '.zip':
                # Validate ZIP contains HDF files
                try:
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        file_list = zip_ref.namelist()
                        hdf_files = [f for f in file_list if f.lower().endswith(('.hdf', '.h5'))]
                        
                        if not hdf_files:
                            return False
                        
                        # Check for VNIR and SWIR files
                        vnir_files = [f for f in hdf_files if 'VNIR' in f.upper()]
                        swir_files = [f for f in hdf_files if 'SWIR' in f.upper()]
                        
                        # Valid if we have at least one HDF file
                        return len(hdf_files) > 0
                        
                except Exception:
                    return False
                    
            elif file_ext in ['.hdf', '.h5']:
                # Validate single HDF file
                try:
                    if HAS_GDAL:
                        dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
                        if dataset:
                            subdatasets = dataset.GetSubDatasets()
                            return len(subdatasets) > 0
                    return True  # Basic validation passed
                except Exception:
                    return False
            else:
                return False
                
        except Exception:
            return False
    
    def process_aster_file_threaded(self, file_path, progress_callback, log_callback, should_stop_callback):
        """Process ASTER file - CRITICAL MISSING METHOD"""
        try:
            log_callback("🚀 Starting ASTER file processing...")
            progress_callback(10, "Initializing...")
            
            if should_stop_callback():
                return False
            
            # Extract ZIP if needed
            hdf_files = []
            if file_path.lower().endswith('.zip'):
                progress_callback(20, "Extracting ZIP file...")
                hdf_files = self.extract_zip_file(file_path, log_callback)
            else:
                hdf_files = [file_path]
            
            if not hdf_files:
                log_callback("❌ No HDF files found")
                return False
            
            if should_stop_callback():
                return False
            
            progress_callback(30, "Loading ASTER data...")
            
            # Process HDF files
            all_data = {}
            for i, hdf_file in enumerate(hdf_files):
                log_callback(f"📁 Processing HDF file {i+1}/{len(hdf_files)}: {os.path.basename(hdf_file)}")
                
                if should_stop_callback():
                    return False
                
                # Read HDF data
                file_data = self.read_hdf_with_gdal(hdf_file, log_callback)
                if file_data:
                    all_data.update(file_data)
                    
                progress_callback(30 + (i+1) * 20 // len(hdf_files), f"Processed HDF {i+1}/{len(hdf_files)}")
            
            if not all_data:
                log_callback("❌ No valid data extracted from HDF files")
                return False
            
            progress_callback(60, "Validating extracted data...")
            
            # Validate extracted data
            if not self.validate_extracted_data(all_data, log_callback):
                log_callback("❌ Data validation failed")
                return False
            
            if should_stop_callback():
                return False
            
            progress_callback(80, "Creating QGIS layers...")
            
            # Create QGIS layers
            layers_created = self.create_qgis_layers_from_data(all_data, log_callback)
            
            if layers_created > 0:
                progress_callback(100, "Processing completed!")
                log_callback(f"✅ Successfully created {layers_created} QGIS layers")
                return True
            else:
                log_callback("❌ Failed to create QGIS layers")
                return False
                
        except Exception as e:
            log_callback(f"❌ Processing error: {str(e)}")
            log_callback(f"Traceback: {traceback.format_exc()}")
            return False
        finally:
            # Cleanup temporary directories
            self.cleanup_temp_dirs()
    
    def extract_zip_file(self, zip_path, log_callback):
        """Extract ZIP file and find HDF files"""
        try:
            temp_dir = tempfile.mkdtemp(prefix='aster_processing_')
            self.temp_dirs.append(temp_dir)
            
            log_callback(f"📂 Extracting ZIP to: {temp_dir}")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find HDF files recursively
            hdf_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.lower().endswith(('.hdf', '.h5')):
                        full_path = os.path.join(root, file)
                        hdf_files.append(full_path)
                        log_callback(f"📋 Found HDF file: {file}")
            
            log_callback(f"📊 Total HDF files found: {len(hdf_files)}")
            return hdf_files
            
        except Exception as e:
            log_callback(f"❌ ZIP extraction failed: {str(e)}")
            return []
    
    def read_hdf_with_gdal(self, file_path, log_callback):
        """COMPLETE FIXED: Read HDF file using GDAL with robust band reading and correct mapping"""
        if not HAS_GDAL:
            log_callback("❌ GDAL not available for HDF reading")
            return {}
        
        gdal.UseExceptions()
        
        try:
            log_callback(f"🔧 Opening HDF file with GDAL: {os.path.basename(file_path)}")
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
                    
                    # Skip quality assurance and other non-data subdatasets
                    if any(skip_word in name for skip_word in ['QA_DATAPLANE', 'QA_', 'QUALITY', 'METADATA']):
                        log_callback(f"⏭️ Skipping QA/metadata subdataset: {name}")
                        continue
                    
                    try:
                        # Open subdataset
                        sub_ds = gdal.Open(path, gdal.GA_ReadOnly)
                        if sub_ds is None:
                            log_callback(f"❌ Could not open subdataset: {name}")
                            continue
                        
                        log_callback(f"📏 Subdataset dimensions: {sub_ds.RasterXSize}x{sub_ds.RasterYSize}, {sub_ds.RasterCount} bands")
                        
                        # Read band data using robust strategies
                        bands = []
                        for j in range(1, sub_ds.RasterCount + 1):
                            band = sub_ds.GetRasterBand(j)
                            if band is None:
                                log_callback(f"❌ Could not get band {j}")
                                continue
                            
                            band_data = self.read_band_data_robust(band, j, log_callback)
                            
                            if band_data is not None:
                                bands.append(band_data)
                                log_callback(f"✅ Successfully read band {j}: {band_data.shape}, dtype: {band_data.dtype}")
                            else:
                                log_callback(f"❌ Failed to read band {j}")
                        
                        if bands:
                            # FIXED: Store band data based on actual subdataset name
                            if 'VNIR' in name.upper():
                                if 'vnir_bands' not in data:
                                    data['vnir_bands'] = {}
                                    data['vnir_geotransform'] = sub_ds.GetGeoTransform()
                                    data['vnir_projection'] = sub_ds.GetProjection()
                                
                                # Extract actual band name from subdataset name
                                if 'BAND1' in name.upper():
                                    data['vnir_bands']['BAND1'] = bands[0]
                                    log_callback(f"✅ Stored BAND1 from VNIR subdataset")
                                elif 'BAND2' in name.upper():
                                    data['vnir_bands']['BAND2'] = bands[0]
                                    log_callback(f"✅ Stored BAND2 from VNIR subdataset")
                                elif 'BAND3N' in name.upper():
                                    data['vnir_bands']['BAND3N'] = bands[0]
                                    log_callback(f"✅ Stored BAND3N from VNIR subdataset")
                                else:
                                    # Fallback for unknown VNIR bands
                                    data['vnir_bands'][f'VNIR_UNKNOWN_{i}'] = bands[0]
                                    log_callback(f"✅ Stored unknown VNIR band from subdataset: {name}")
                            
                            elif 'SWIR' in name.upper():
                                if 'swir_bands' not in data:
                                    data['swir_bands'] = {}
                                    data['swir_geotransform'] = sub_ds.GetGeoTransform()
                                    data['swir_projection'] = sub_ds.GetProjection()
                                
                                # Extract actual band name from subdataset name
                                if 'BAND4' in name.upper():
                                    data['swir_bands']['BAND4'] = bands[0]
                                    log_callback(f"✅ Stored BAND4 from SWIR subdataset")
                                elif 'BAND5' in name.upper():
                                    data['swir_bands']['BAND5'] = bands[0]
                                    log_callback(f"✅ Stored BAND5 from SWIR subdataset")
                                elif 'BAND6' in name.upper():
                                    data['swir_bands']['BAND6'] = bands[0]
                                    log_callback(f"✅ Stored BAND6 from SWIR subdataset")
                                elif 'BAND7' in name.upper():
                                    data['swir_bands']['BAND7'] = bands[0]
                                    log_callback(f"✅ Stored BAND7 from SWIR subdataset")
                                elif 'BAND8' in name.upper():
                                    data['swir_bands']['BAND8'] = bands[0]
                                    log_callback(f"✅ Stored BAND8 from SWIR subdataset")
                                elif 'BAND9' in name.upper():
                                    data['swir_bands']['BAND9'] = bands[0]
                                    log_callback(f"✅ Stored BAND9 from SWIR subdataset")
                                else:
                                    # Fallback for unknown SWIR bands
                                    data['swir_bands'][f'SWIR_UNKNOWN_{i}'] = bands[0]
                                    log_callback(f"✅ Stored unknown SWIR band from subdataset: {name}")
                            
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
            return {}

    def read_band_data_robust(self, band, band_number, log_callback):
        """COMPLETE: Robust band data reading with multiple fallback strategies"""
        import numpy as np
        
        band_data = None
        band_type = band.DataType
        band_xsize = band.XSize
        band_ysize = band.YSize
        
        # Strategy 1: Standard ReadAsArray
        try:
            band_data = band.ReadAsArray()
            if band_data is not None and isinstance(band_data, np.ndarray) and band_data.size > 0:
                # Check for valid data
                valid_pixels = np.sum((band_data > 0) & (~np.isnan(band_data)))
                if valid_pixels > 0:
                    log_callback(f"✅ Strategy 1 (ReadAsArray) successful for band {band_number}")
                    return band_data
                else:
                    log_callback(f"⚠️ Strategy 1: All pixels are zero/invalid for band {band_number}")
                    band_data = None
            else:
                band_data = None
        except Exception as e:
            log_callback(f"⚠️ Strategy 1 failed: {str(e)}")
            band_data = None
        
        # Strategy 2: ReadAsArray with explicit parameters
        if band_data is None:
            try:
                log_callback(f"🔄 Trying Strategy 2: ReadAsArray with explicit parameters for band {band_number}...")
                band_data = band.ReadAsArray(0, 0, band_xsize, band_ysize)
                if band_data is not None and isinstance(band_data, np.ndarray) and band_data.size > 0:
                    valid_pixels = np.sum((band_data > 0) & (~np.isnan(band_data)))
                    if valid_pixels > 0:
                        log_callback(f"✅ Strategy 2 successful for band {band_number}")
                        return band_data
                    else:
                        band_data = None
                else:
                    band_data = None
            except Exception as e:
                log_callback(f"⚠️ Strategy 2 failed: {str(e)}")
                band_data = None
        
        # Strategy 3: ReadRaster with manual conversion
        if band_data is None:
            try:
                log_callback(f"🔄 Trying Strategy 3: ReadRaster for band {band_number}...")
                raw_data = band.ReadRaster(0, 0, band_xsize, band_ysize)
                
                if raw_data:
                    # Map GDAL data types to numpy
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
                            valid_pixels = np.sum((band_data > 0) & (~np.isnan(band_data)))
                            if valid_pixels > 0:
                                log_callback(f"✅ Strategy 3 successful for band {band_number}")
                                return band_data
                            else:
                                band_data = None
                    except Exception as reshape_error:
                        log_callback(f"⚠️ Strategy 3 reshape failed: {str(reshape_error)}")
                        band_data = None
                else:
                    log_callback(f"⚠️ Strategy 3: ReadRaster returned no data for band {band_number}")
                    band_data = None
            except Exception as e:
                log_callback(f"⚠️ Strategy 3 failed: {str(e)}")
                band_data = None
        
        # Strategy 4: Block-by-block reading
        if band_data is None:
            try:
                log_callback(f"🔄 Trying Strategy 4: Block-by-block reading for band {band_number}...")
                
                # Map data types
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
                        total_blocks += 1
                        
                        # Calculate actual block size
                        x_size = min(block_size, band_xsize - x)
                        y_size = min(block_size, band_ysize - y)
                        
                        try:
                            block_data = band.ReadAsArray(x, y, x_size, y_size)
                            if block_data is not None:
                                band_data[y:y+y_size, x:x+x_size] = block_data
                                successful_blocks += 1
                        except:
                            pass  # Skip failed blocks
                
                if successful_blocks > 0:
                    valid_pixels = np.sum((band_data > 0) & (~np.isnan(band_data)))
                    if valid_pixels > 0:
                        log_callback(f"✅ Strategy 4 successful for band {band_number}: {successful_blocks}/{total_blocks} blocks read")
                        return band_data
                    else:
                        band_data = None
                else:
                    band_data = None
                    
            except Exception as e:
                log_callback(f"⚠️ Strategy 4 failed: {str(e)}")
                band_data = None
        
        # Strategy 5: Try reading a small sample first
        if band_data is None:
            try:
                log_callback(f"🔄 Trying Strategy 5: Sample test for band {band_number}...")
                
                # Try reading a small sample
                sample_size = min(100, band_xsize, band_ysize)
                sample = band.ReadAsArray(0, 0, sample_size, sample_size)
                
                if sample is not None and isinstance(sample, np.ndarray):
                    # If sample works, try reading full data with different approach
                    log_callback(f"✅ Sample successful, attempting full read...")
                    
                    # Use the datatype from the sample
                    band_data = np.zeros((band_ysize, band_xsize), dtype=sample.dtype)
                    
                    # Try reading in larger chunks
                    chunk_size = min(1000, band_xsize, band_ysize)
                    for y in range(0, band_ysize, chunk_size):
                        for x in range(0, band_xsize, chunk_size):
                            x_size = min(chunk_size, band_xsize - x)
                            y_size = min(chunk_size, band_ysize - y)
                            
                            try:
                                chunk = band.ReadAsArray(x, y, x_size, y_size)
                                if chunk is not None:
                                    band_data[y:y+y_size, x:x+x_size] = chunk
                            except:
                                pass
                    
                    valid_pixels = np.sum((band_data > 0) & (~np.isnan(band_data)))
                    if valid_pixels > 0:
                        log_callback(f"✅ Strategy 5 successful for band {band_number}")
                        return band_data
                    else:
                        band_data = None
                else:
                    band_data = None
                    
            except Exception as e:
                log_callback(f"⚠️ Strategy 5 failed: {str(e)}")
                band_data = None
        
        # If all strategies failed
        log_callback(f"❌ All reading strategies failed for band {band_number}")
        return None

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

    def create_qgis_layers_from_data(self, data, log_callback):
        """FINAL FIX: Create QGIS layers from extracted ASTER data with all import issues resolved"""
        # CRITICAL FIX: Import os at the top of the method
        import os
        import datetime
        
        try:
            layers_created = 0
            
            # Determine output directory - simplified approach
            try:
                # Use Documents directory as default
                documents_dir = os.path.expanduser("~/Documents")
                output_dir = os.path.join(documents_dir, 'ASTER_Processed')
                
                # Try to find a better location if temp dirs exist
                if hasattr(self, 'temp_dirs') and self.temp_dirs:
                    try:
                        # Try to get parent directory of temp extraction
                        temp_dir = self.temp_dirs[0]
                        # Go up several levels to find a suitable directory
                        current_dir = temp_dir
                        for _ in range(4):  # Go up 4 levels
                            parent_dir = os.path.dirname(current_dir)
                            if parent_dir != current_dir:  # Not at root
                                current_dir = parent_dir
                            else:
                                break
                        
                        # Use this as base for output
                        if os.path.exists(current_dir) and os.access(current_dir, os.W_OK):
                            output_dir = os.path.join(current_dir, 'ASTER_Processed')
                    except:
                        pass  # Keep default Documents directory
                        
            except Exception as e:
                log_callback(f"⚠️ Could not determine output location: {str(e)}")
                # Final fallback - use current working directory
                output_dir = os.path.join(os.getcwd(), 'ASTER_Processed')
            
            # Create output directory
            try:
                os.makedirs(output_dir, exist_ok=True)
                log_callback(f"🗂️ Saving processed files to: {output_dir}")
            except Exception as e:
                log_callback(f"❌ Could not create output directory: {str(e)}")
                return 0
            
            # Process VNIR bands
            if 'vnir_bands' in data and data['vnir_bands']:
                try:
                    log_callback("🎨 Creating VNIR layer...")
                    vnir_file = os.path.join(output_dir, 'aster_vnir.tif')
                    
                    vnir_success = self.create_geotiff_from_bands(
                        data['vnir_bands'], 
                        data.get('vnir_geotransform'),
                        data.get('vnir_projection'),
                        vnir_file,
                        log_callback
                    )
                    
                    if vnir_success and self.add_layer_to_qgis(vnir_file, "ASTER_VNIR", log_callback):
                        layers_created += 1
                except Exception as e:
                    log_callback(f"⚠️ Could not create VNIR layer: {str(e)}")
            
            # Process SWIR bands
            if 'swir_bands' in data and data['swir_bands']:
                try:
                    log_callback("🎨 Creating SWIR layer...")
                    swir_file = os.path.join(output_dir, 'aster_swir.tif')
                    
                    swir_success = self.create_geotiff_from_bands(
                        data['swir_bands'],
                        data.get('swir_geotransform'),
                        data.get('swir_projection'),
                        swir_file,
                        log_callback
                    )
                    
                    if swir_success and self.add_layer_to_qgis(swir_file, "ASTER_SWIR", log_callback):
                        layers_created += 1
                except Exception as e:
                    log_callback(f"⚠️ Could not create SWIR layer: {str(e)}")
            
            # Create RGB composite if we have enough VNIR bands
            if 'vnir_bands' in data and len(data['vnir_bands']) >= 3:
                try:
                    log_callback("🎨 Creating RGB composite...")
                    rgb_file = os.path.join(output_dir, 'aster_rgb.tif')
                    
                    rgb_success = self.create_rgb_composite(
                        data['vnir_bands'],
                        data.get('vnir_geotransform'),
                        data.get('vnir_projection'),
                        rgb_file,
                        log_callback
                    )
                    
                    if rgb_success and self.add_layer_to_qgis(rgb_file, "ASTER_RGB", log_callback):
                        layers_created += 1
                except Exception as e:
                    log_callback(f"⚠️ Could not create RGB layer: {str(e)}")
            
            log_callback(f"🎉 Created {layers_created} QGIS layers successfully")
            
            # Refresh QGIS interface
            if layers_created > 0:
                try:
                    log_callback("🔄 Refreshing QGIS interface...")
                    from qgis.utils import iface
                    if iface:
                        iface.mapCanvas().refresh()
                        iface.layerTreeView().refreshLayerSymbology()
                        log_callback("✅ QGIS interface refreshed")
                except Exception as refresh_error:
                    log_callback(f"⚠️ Could not refresh QGIS interface: {str(refresh_error)}")
            
            return layers_created
            
        except Exception as e:
            import traceback
            log_callback(f"❌ Error creating QGIS layers: {str(e)}")
            log_callback(f"Traceback: {traceback.format_exc()}")
            return 0




    def create_combined_aster_dataset(self, vnir_bands, swir_bands, vnir_geotransform, swir_geotransform, projection, output_path, log_callback):
        """Create a combined 9-band ASTER dataset with proper resampling"""
        try:
            if not HAS_GDAL:
                log_callback("❌ GDAL not available for combined dataset creation")
                return False
            
            # Use VNIR as reference (15m resolution)
            ref_transform = vnir_geotransform
            ref_bands = vnir_bands
            
            # Get reference dimensions from first VNIR band
            first_vnir = list(ref_bands.values())[0]
            ref_height, ref_width = first_vnir.shape
            
            log_callback(f"📝 Creating combined dataset: {ref_width}x{ref_height}, 9 bands")
            
            # Create output GeoTIFF
            driver = gdal.GetDriverByName('GTiff')
            dataset = driver.Create(output_path, ref_width, ref_height, 9, gdal.GDT_Float32)
            
            if not dataset:
                raise Exception("Failed to create combined dataset")
            
            # Set geotransform and projection
            if ref_transform:
                dataset.SetGeoTransform(ref_transform)
            if projection:
                dataset.SetProjection(projection)
            
            band_index = 1
            
            # Add VNIR bands (already at 15m)
            vnir_order = ['BAND1', 'BAND2', 'BAND3N']
            for band_name in vnir_order:
                if band_name in vnir_bands:
                    band = dataset.GetRasterBand(band_index)
                    band_data = vnir_bands[band_name].astype(np.float32)
                    band.WriteArray(band_data)
                    band.SetDescription(f"ASTER_{band_name}_15m")
                    band.FlushCache()
                    log_callback(f"✅ Added {band_name} to combined dataset (band {band_index})")
                    band_index += 1
            
            # Add SWIR bands (resample from 30m to 15m)
            swir_order = ['BAND4', 'BAND5', 'BAND6', 'BAND7', 'BAND8', 'BAND9']
            for band_name in swir_order:
                if band_name in swir_bands:
                    band = dataset.GetRasterBand(band_index)
                    
                    # Resample SWIR band from 30m to 15m
                    swir_data = swir_bands[band_name]
                    resampled_data = self.resample_to_reference(swir_data, first_vnir.shape)
                    
                    band.WriteArray(resampled_data.astype(np.float32))
                    band.SetDescription(f"ASTER_{band_name}_15m_resampled")
                    band.FlushCache()
                    log_callback(f"✅ Added {band_name} to combined dataset (band {band_index}, resampled)")
                    band_index += 1
            
            dataset.FlushCache()
            dataset = None  # Close dataset
            
            log_callback(f"✅ Created combined GeoTIFF: {output_path}")
            return True
            
        except Exception as e:
            log_callback(f"❌ Combined dataset creation failed: {str(e)}")
            return False
    



    def resample_to_reference(self, data, target_shape):
        """Simple resampling using scipy"""
        try:
            from scipy import ndimage
            
            height_ratio = target_shape[0] / data.shape[0]
            width_ratio = target_shape[1] / data.shape[1]
            
            # Use zoom for resampling
            resampled = ndimage.zoom(data, (height_ratio, width_ratio), order=1)
            
            # Ensure exact target shape
            if resampled.shape != target_shape:
                # Crop or pad if needed
                if resampled.shape[0] > target_shape[0]:
                    resampled = resampled[:target_shape[0], :]
                if resampled.shape[1] > target_shape[1]:
                    resampled = resampled[:, :target_shape[1]]
                
                if resampled.shape[0] < target_shape[0] or resampled.shape[1] < target_shape[1]:
                    # Pad with zeros if too small
                    padded = np.zeros(target_shape, dtype=resampled.dtype)
                    padded[:resampled.shape[0], :resampled.shape[1]] = resampled
                    resampled = padded
            
            return resampled
            
        except ImportError:
            # Fallback to simple nearest neighbor
            from skimage.transform import resize
            return resize(data, target_shape, preserve_range=True, anti_aliasing=False)
        except:
            # Last resort: simple array repetition
            height_factor = target_shape[0] // data.shape[0]
            width_factor = target_shape[1] // data.shape[1]
            return np.repeat(np.repeat(data, height_factor, axis=0), width_factor, axis=1)



    def create_geotiff_from_bands(self, bands_dict, geotransform, projection, output_path, log_callback):
        """Create a GeoTIFF file from band data"""
        try:
            if not HAS_GDAL:
                log_callback("❌ GDAL not available for GeoTIFF creation")
                return None
            
            # Get band data as list
            band_names = sorted(bands_dict.keys())
            band_arrays = [bands_dict[name] for name in band_names]
            
            if not band_arrays:
                log_callback("❌ No band data to create GeoTIFF")
                return None
            
            # Get dimensions from first band
            height, width = band_arrays[0].shape
            num_bands = len(band_arrays)
            
            log_callback(f"📝 Creating GeoTIFF: {width}x{height}, {num_bands} bands")
            
            # Create GeoTIFF
            driver = gdal.GetDriverByName('GTiff')
            dataset = driver.Create(output_path, width, height, num_bands, gdal.GDT_Float32)
            
            if not dataset:
                raise Exception("Failed to create GeoTIFF dataset")
            
            # Set geotransform and projection
            if geotransform:
                dataset.SetGeoTransform(geotransform)
            if projection:
                dataset.SetProjection(projection)
            
            # Write band data
            for i, band_array in enumerate(band_arrays):
                band = dataset.GetRasterBand(i + 1)
                band.WriteArray(band_array)
                band.SetDescription(band_names[i])
                band.FlushCache()
            
            dataset.FlushCache()
            dataset = None  # Close dataset
            
            log_callback(f"✅ Created GeoTIFF: {output_path}")
            return output_path
            
        except Exception as e:
            log_callback(f"❌ GeoTIFF creation failed: {str(e)}")
            return None

    def create_rgb_composite(self, vnir_bands, geotransform, projection, output_path, log_callback):
        """Create RGB composite from VNIR bands"""
        try:
            # Use first 3 VNIR bands for RGB
            band_names = sorted(vnir_bands.keys())[:3]
            
            if len(band_names) < 3:
                log_callback("❌ Not enough VNIR bands for RGB composite")
                return None
            
            rgb_bands = {f'RGB_{i+1}': vnir_bands[name] for i, name in enumerate(band_names)}
            
            return self.create_geotiff_from_bands(rgb_bands, geotransform, projection, output_path, log_callback)
            
        except Exception as e:
            log_callback(f"❌ RGB composite creation failed: {str(e)}")
            return None

    def add_layer_to_qgis(self, file_path, layer_name, log_callback):
        """Add layer to QGIS project with comprehensive error handling"""
        import os
        import datetime
        
        try:
            # Verify file exists and is valid
            if not os.path.exists(file_path):
                log_callback(f"❌ File does not exist: {file_path}")
                return False
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                log_callback(f"❌ File is empty: {file_path}")
                return False
            
            log_callback(f"📁 Adding layer: {layer_name} from {os.path.basename(file_path)}")
            
            # Create layer
            layer = QgsRasterLayer(file_path, layer_name)
            
            if not layer.isValid():
                log_callback(f"❌ Invalid raster layer: {file_path}")
                
                # Try to diagnose the issue
                try:
                    from osgeo import gdal
                    ds = gdal.Open(file_path)
                    if ds:
                        log_callback(f"🔍 GDAL can read file: {ds.RasterXSize}x{ds.RasterYSize}, {ds.RasterCount} bands")
                    else:
                        log_callback(f"❌ GDAL cannot read file: {file_path}")
                except Exception as diag_error:
                    log_callback(f"⚠️ Could not diagnose file: {str(diag_error)}")
                
                return False
            
            # Set layer properties
            layer.setCustomProperty('layer_type', 'spectral')
            layer.setCustomProperty('aster_type', 'PROCESSED')
            layer.setCustomProperty('processing_date', str(datetime.datetime.now()))
            
            # Add to QGIS project
            QgsProject.instance().addMapLayer(layer)
            
            # Verify layer was added
            project_layers = QgsProject.instance().mapLayers()
            layer_found = False
            for layer_id, project_layer in project_layers.items():
                if project_layer.name() == layer_name:
                    layer_found = True
                    break
            
            if layer_found:
                log_callback(f"✅ Added layer to project: {layer.name()}")
                
                # Try to zoom to layer extent
                try:
                    extent = layer.extent()
                    if extent and not extent.isEmpty():
                        log_callback(f"📍 Layer extent: {extent.toString()}")
                        
                        # Try to zoom to the layer
                        from qgis.utils import iface
                        if iface:
                            iface.mapCanvas().setExtent(extent)
                            iface.mapCanvas().refresh()
                            log_callback("🔍 Zoomed to layer extent")
                    else:
                        log_callback("⚠️ Layer has empty extent")
                except Exception as extent_error:
                    log_callback(f"⚠️ Could not handle layer extent: {str(extent_error)}")
                
                return True
            else:
                log_callback(f"❌ Layer not found in project after adding: {layer_name}")
                return False
            
        except Exception as e:
            log_callback(f"❌ Failed to add layer to QGIS: {str(e)}")
            import traceback
            log_callback(f"Traceback: {traceback.format_exc()}")
            return False

    def cleanup_temp_dirs(self):
        """Clean up temporary directories"""
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception:
                pass  # Ignore cleanup errors
        self.temp_dirs.clear()

# Thread class for compatibility
class AsterProcessingThread(QThread):
    """Threading wrapper for ASTER processing"""
    
    progress_updated = pyqtSignal(int, str)
    log_message = pyqtSignal(str)
    processing_finished = pyqtSignal(bool, dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, file_path, processor=None):
        super().__init__()
        self.file_path = file_path
        self.processor = processor or AsterProcessor()
        self.should_stop = False
    
    def stop(self):
        """Stop processing"""
        self.should_stop = True
    
    def run(self):
        """Main processing function that runs in separate thread"""
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
                self.processing_finished.emit(True, {"status": "success"})
            elif self.should_stop:
                log_callback("⏹️ Processing cancelled by user")
                self.processing_finished.emit(False, {"status": "cancelled"})
            else:
                self.error_occurred.emit("Processing failed")
                
        except Exception as e:
            error_msg = f"Processing error: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)