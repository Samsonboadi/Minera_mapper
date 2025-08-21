import os
import numpy as np
import tempfile
import json
import zipfile
import shutil
import threading
import traceback
import datetime
from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox, QProgressDialog
from qgis.PyQt.QtCore import Qt, QThread, pyqtSignal
from qgis.core import QgsRasterLayer, QgsProject, QgsMessageLog, Qgis, QgsColorRampShader, QgsRasterShader, QgsSingleBandPseudoColorRenderer
from PyQt5.QtGui import QColor

# Safe imports with fallbacks
try:
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from osgeo import gdal, osr
    HAS_GDAL = True
except ImportError:
    HAS_GDAL = False

try:
    from algorithms.mineral_mapping import MineralMapper
    HAS_MINERAL_MAPPER = True
except ImportError:
    HAS_MINERAL_MAPPER = False

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


class AsterProcessingThread(QThread):
    """Thread for ASTER processing to prevent UI freezing"""
    
    # Signals for communication with main thread
    progress_updated = pyqtSignal(int, str)  # progress, message
    log_message = pyqtSignal(str)  # log message
    processing_finished = pyqtSignal(bool, dict)  # success, results
    error_occurred = pyqtSignal(str)  # error message
    
    def __init__(self, file_path, processor):
        super().__init__()
        self.file_path = file_path
        self.processor = processor
        self.should_stop = False
    
    def stop(self):
        """Stop the processing thread"""
        self.should_stop = True
    
    def run(self):
        """Main processing function that runs in separate thread"""
        try:
            self.log_message.emit("🚀 Starting ASTER data processing...")
            self.progress_updated.emit(5, "Initializing processing...")
            
            if self.should_stop:
                return
            
            # Validate file
            self.log_message.emit(f"📁 Validating file: {os.path.basename(self.file_path)}")
            if not self.processor.validate_aster_file(self.file_path):
                self.error_occurred.emit("File validation failed")
                return
            
            self.progress_updated.emit(10, "File validated successfully")
            
            if self.should_stop:
                return
            
            # Process the file
            result = self.processor.process_aster_file_threaded(
                self.file_path, 
                self.progress_callback,
                self.log_callback,
                self.should_stop_callback
            )
            
            if result and not self.should_stop:
                self.log_message.emit("✅ ASTER processing completed successfully!")
                self.processing_finished.emit(True, result)
            elif self.should_stop:
                self.log_message.emit("⏹️ Processing cancelled by user")
                self.processing_finished.emit(False, {})
            else:
                self.error_occurred.emit("Processing failed")
                
        except Exception as e:
            error_msg = f"Processing error: {str(e)}\n{traceback.format_exc()}"
            self.error_occurred.emit(error_msg)
    
    def progress_callback(self, value, message):
        """Callback for progress updates"""
        self.progress_updated.emit(value, message)
    
    def log_callback(self, message):
        """Callback for log messages"""
        self.log_message.emit(message)
    
    def should_stop_callback(self):
        """Callback to check if processing should stop"""
        return self.should_stop


class AsterProcessor:
    """Complete ASTER processor with threading and detailed logging"""
    
    def __init__(self, iface=None):
        self.iface = iface
        
        # Handle case where iface is not provided
        if self.iface is None:
            from qgis.utils import iface as qgis_iface
            self.iface = qgis_iface
        
        # Initialize mineral signatures
        self.mineral_signatures = {}
        
        # ASTER band specifications
        self.aster_bands = {
            'VNIR': {
                'Band1': {'wavelength': 560, 'name': 'Green', 'resolution': 15},
                'Band2': {'wavelength': 660, 'name': 'Red', 'resolution': 15},
                'Band3N': {'wavelength': 820, 'name': 'NIR', 'resolution': 15}
            },
            'SWIR': {
                'Band4': {'wavelength': 1650, 'name': 'SWIR1', 'resolution': 30},
                'Band5': {'wavelength': 2165, 'name': 'SWIR2', 'resolution': 30},
                'Band6': {'wavelength': 2205, 'name': 'SWIR3', 'resolution': 30},
                'Band7': {'wavelength': 2260, 'name': 'SWIR4', 'resolution': 30},
                'Band8': {'wavelength': 2330, 'name': 'SWIR5', 'resolution': 30},
                'Band9': {'wavelength': 2395, 'name': 'SWIR6', 'resolution': 30}
            }
        }
        
        # Mineral ratios for geological analysis
        self.mineral_ratios = {
            'Iron_Oxide': {
                'formula': 'Band3N / Band2', 
                'bands': ['Band3N', 'Band2'],
                'description': 'Iron oxide detection using NIR/Red ratio'
            },
            'Clay_Minerals': {
                'formula': 'Band5 / Band6', 
                'bands': ['Band5', 'Band6'],
                'description': 'Clay minerals using SWIR band ratio'
            },
            'Carbonate': {
                'formula': '(Band6 + Band8) / Band7', 
                'bands': ['Band6', 'Band8', 'Band7'],
                'description': 'Carbonate minerals detection'
            },
            'Silicate': {
                'formula': 'Band7 / Band6', 
                'bands': ['Band7', 'Band6'],
                'description': 'Silicate minerals detection'
            },
            'Alteration': {
                'formula': 'Band4 / Band6', 
                'bands': ['Band4', 'Band6'],
                'description': 'Hydrothermal alteration zones'
            },
            'Gossan': {
                'formula': '(Band2 / Band1) * (Band4 / Band5)', 
                'bands': ['Band2', 'Band1', 'Band4', 'Band5'],
                'description': 'Iron-rich oxidized zones (gossans)'
            }
        }
        
        # Processing state variables
        self.current_file = None
        self.temp_dirs = []
        self.processing_thread = None
        self.progress_dialog = None
        self.log_widget = None
    
    def set_log_widget(self, log_widget):
        """Set the log widget for displaying messages"""
        self.log_widget = log_widget
    
    def log_message(self, message, level="INFO"):
        """Log message - THREAD SAFE VERSION"""
        # Map levels to QGIS levels
        qgis_levels = {
            "INFO": Qgis.Info,
            "WARNING": Qgis.Warning,
            "ERROR": Qgis.Critical,
            "DEBUG": Qgis.Info,
            "SUCCESS": Qgis.Success
        }
        
        # Log to QGIS (thread safe)
        QgsMessageLog.logMessage(message, 'ASTER Processor', qgis_levels.get(level, Qgis.Info))
        
        # IMPORTANT: Do NOT update UI directly from thread!
        # The log widget will be updated via signals from the processing thread
    
    def check_dependencies(self):
        """Check if required dependencies are available"""
        missing_deps = []
        
        if not HAS_RASTERIO and not HAS_GDAL:
            missing_deps.append("rasterio or GDAL")
        if not HAS_MINERAL_MAPPER:
            missing_deps.append("MineralMapper")
        if not HAS_SKLEARN:
            missing_deps.append("sklearn")
        if not HAS_SCIPY:
            missing_deps.append("scipy")
        
        if missing_deps:
            error_msg = f"Missing required dependencies: {', '.join(missing_deps)}"
            self.log_message(error_msg, "ERROR")
            QMessageBox.critical(
                self.iface.mainWindow(),
                "Missing Dependencies",
                f"{error_msg}\n\nPlease install the required packages and restart QGIS."
            )
            return False
        
        return True
    
    def load_mineral_signatures(self, signature_file):
        """Load mineral signatures from a JSON file"""
        try:
            with open(signature_file, 'r') as f:
                data = json.load(f)
                self.mineral_signatures = data.get('minerals', {})
            self.log_message(f"Loaded {len(self.mineral_signatures)} mineral signatures", "INFO")
        except Exception as e:
            self.log_message(f"Failed to load mineral signatures: {str(e)}", "ERROR")
            self.mineral_signatures = {}  # Fallback to empty dict
    
    def select_signature_file(self):
        """Select mineral signatures JSON file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.iface.mainWindow(),
            "Select Mineral Signatures JSON File",
            "",
            "JSON files (*.json);;All files (*)"
        )
        return file_path
    
    def process_data(self, file_path=None, log_widget=None):
        """Main processing function with improved UI and threading"""
        if not self.check_dependencies():
            return False
        
        # Set log widget if provided
        if log_widget:
            self.set_log_widget(log_widget)
        
        try:
            # Load mineral signatures
            signature_file = os.path.join(os.path.dirname(__file__), 'data', 'mineral_signatures.json')
            self.load_mineral_signatures(signature_file)
            
            # Optional: Allow user to select signature file
            # signature_file = self.select_signature_file()
            # if signature_file:
            #     self.load_mineral_signatures(signature_file)
            # else:
            #     self.log_message("No mineral signature file selected, skipping advanced mineral mapping", "WARNING")
            
            # File selection if not provided
            if file_path is None:
                file_path = self.select_aster_file()
                if not file_path:
                    return False
            
            self.current_file = file_path
            self.log_message(f"Starting processing of: {os.path.basename(file_path)}", "INFO")
            
            # Create progress dialog with cancel capability
            self.progress_dialog = QProgressDialog(
                "Starting ASTER processing...", 
                "Cancel", 
                0, 100, 
                self.iface.mainWindow()
            )
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.setWindowTitle("ASTER Data Processing")
            self.progress_dialog.show()
            
            # Create and start processing thread
            self.processing_thread = AsterProcessingThread(file_path, self)
            
            # Connect thread signals
            self.processing_thread.progress_updated.connect(self.update_progress)
            self.processing_thread.log_message.connect(self.add_log_message)
            self.processing_thread.processing_finished.connect(self.processing_complete)
            self.processing_thread.error_occurred.connect(self.processing_error)
            
            # Connect cancel button
            self.progress_dialog.canceled.connect(self.cancel_processing)
            
            # Start processing
            self.processing_thread.start()
            
            return True
            
        except Exception as e:
            self.handle_processing_error(e)
            return False
    
    def process_specific_file(self, file_path, log_widget=None):
        """Process a specific ASTER file - called from main dialog"""
        self.log_message(f"Processing specific file: {file_path}", "INFO")
        return self.process_data(file_path, log_widget)
    
    def update_progress(self, value, message):
        """Update progress dialog"""
        if self.progress_dialog:
            self.progress_dialog.setValue(value)
            self.progress_dialog.setLabelText(message)
    
    def add_log_message(self, message):
        """Add message to log widget"""
        self.log_message(message, "INFO")
    
    def processing_complete(self, success, results):
        """Handle processing completion"""
        if self.progress_dialog:
            self.progress_dialog.close()
        
        if success:
            bands_count = len(results.get('bands', {}))
            indices_count = len(results.get('indices', {}))
            composites_count = len(results.get('composites', {}))
            
            success_msg = (
                f"ASTER data processed successfully!\n\n"
                f"📊 Results:\n"
                f"• {bands_count} spectral bands\n"
                f"• {indices_count} mineral indices/maps\n"
                f"• {composites_count} composite images\n\n"
                f"All layers have been added to QGIS."
            )
            
            self.log_message("🎉 Processing completed successfully!", "SUCCESS")
            
            QMessageBox.information(
                self.iface.mainWindow(),
                "✅ Processing Complete",
                success_msg
            )
        
        # Cleanup
        self.cleanup_processing()
    
    def processing_error(self, error_message):
        """Handle processing errors"""
        if self.progress_dialog:
            self.progress_dialog.close()
        
        self.log_message(f"❌ Error: {error_message}", "ERROR")
        
        # Show user-friendly error
        QMessageBox.critical(
            self.iface.mainWindow(),
            "❌ Processing Error",
            f"ASTER processing failed:\n\n{error_message}\n\n"
            f"Check the processing log for more details."
        )
        
        # Cleanup
        self.cleanup_processing()
    
    def cancel_processing(self):
        """Cancel the processing"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.log_message("⏹️ Cancelling processing...", "WARNING")
            self.processing_thread.stop()
            self.processing_thread.wait(5000)  # Wait up to 5 seconds
            
            if self.processing_thread.isRunning():
                self.processing_thread.terminate()
                self.log_message("🔄 Force terminating processing thread...", "WARNING")
        
        self.cleanup_processing()
    
    def cleanup_processing(self):
        """Clean up processing resources"""
        # Clean up thread
        if self.processing_thread:
            if self.processing_thread.isRunning():
                self.processing_thread.wait()
            self.processing_thread = None
        
        # Close progress dialog
        if self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        # Clean up temp directories
        self.cleanup_temp_dirs()
    
    def cleanup_temp_dirs(self):
        """Clean up temporary directories"""
        for temp_dir in self.temp_dirs:
            try:
                shutil.rmtree(temp_dir)
                self.log_message(f"🧹 Cleaned up temporary directory: {temp_dir}", "DEBUG")
            except Exception as e:
                self.log_message(f"⚠️ Failed to clean up temporary directory {temp_dir}: {str(e)}", "WARNING")
        self.temp_dirs = []
    
    def select_aster_file(self):
        """Select ASTER file (ZIP or HDF)"""
        file_path, _ = QFileDialog.getOpenFileName(
            self.iface.mainWindow(),
            "Select ASTER File",
            "",
            "ASTER files (*.zip *.hdf *.h5 *.hdf5);;ZIP files (*.zip);;HDF files (*.hdf *.h5 *.hdf5);;All files (*)"
        )
        return file_path
    
    def validate_aster_file(self, file_path):
        """Validate ASTER file before processing"""
        if not os.path.exists(file_path):
            self.log_message(f"File does not exist: {file_path}", "ERROR")
            return False
        
        # Check file size
        file_size = os.path.getsize(file_path)
        self.log_message(f"File size: {file_size / (1024*1024):.1f} MB", "DEBUG")
        
        if file_size < 1000:  # Very small file
            self.log_message(f"Warning: File seems very small: {file_size} bytes", "WARNING")
        
        # Check file extension and type
        file_lower = file_path.lower()
        
        if file_lower.endswith('.zip'):
            return self.validate_zip_file(file_path)
        elif any(file_lower.endswith(ext) for ext in ['.hdf', '.h5', '.hdf5']):
            return self.validate_hdf_file(file_path)
        else:
            self.log_message("Warning: File doesn't appear to be a ZIP or HDF file", "WARNING")
        
        return True
    
    def validate_zip_file(self, file_path):
        """Validate ZIP file"""
        try:
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                bad_file = zip_ref.testzip()
                if bad_file:
                    self.log_message(f"ZIP file is corrupted. First bad file: {bad_file}", "ERROR")
                    return False
                
                file_list = zip_ref.namelist()
                hdf_files = [f for f in file_list if f.lower().endswith('.hdf')]
                
                if not hdf_files:
                    self.log_message("Warning: No HDF files found in ZIP archive", "WARNING")
                else:
                    self.log_message(f"Found {len(hdf_files)} HDF files in ZIP", "INFO")
                
        except zipfile.BadZipFile:
            self.log_message("Error: Selected file is not a valid ZIP archive", "ERROR")
            return False
        except Exception as e:
            self.log_message(f"Error validating ZIP file: {str(e)}", "ERROR")
            return False
        
        return True
    
    def validate_hdf_file(self, file_path):
        """Validate HDF file"""
        try:
            # Try to open with GDAL if available
            if HAS_GDAL:
                dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
                if dataset is None:
                    raise Exception("Cannot open HDF file with GDAL")
                dataset = None
                self.log_message("HDF file validation successful", "INFO")
            
            return True
            
        except Exception as e:
            self.log_message(f"Cannot open HDF file: {str(e)}", "ERROR")
            return False
    
    def process_aster_file_threaded(self, file_path, progress_callback, log_callback, should_stop_callback):
        """Process ASTER file in thread with detailed logging"""
        log_callback("🔄 Starting ASTER file processing...")
        
        try:
            # Determine file type and process accordingly
            if file_path.lower().endswith('.zip'):
                return self.process_zip_file_threaded(file_path, progress_callback, log_callback, should_stop_callback)
            else:
                return self.process_hdf_file_threaded(file_path, progress_callback, log_callback, should_stop_callback)
                
        except Exception as e:
            log_callback(f"❌ Processing failed: {str(e)}")
            raise
    
    def process_zip_file_threaded(self, zip_path, progress_callback, log_callback, should_stop_callback):
        """Process ASTER ZIP file with threading"""
        progress_callback(15, "📦 Extracting ZIP file...")
        log_callback(f"📦 Extracting: {os.path.basename(zip_path)}")
        
        if should_stop_callback():
            return None
        
        # Extract ZIP file
        extracted_data = self.extract_aster_zip(zip_path, log_callback)
        if not extracted_data:
            return None
        
        progress_callback(25, "🔍 Locating HDF files...")
        log_callback("🔍 Searching for HDF files in extracted data...")
        
        if should_stop_callback():
            return None
        
        # Find HDF files
        hdf_files = self.find_hdf_files(extracted_data['extract_dir'], log_callback)
        if not hdf_files or not any(hdf_files.values()):
            log_callback("❌ No HDF files found in the extracted ASTER product")
            return None
        
        progress_callback(35, "📖 Reading HDF files...")
        log_callback("📖 Reading HDF file contents...")
        
        if should_stop_callback():
            return None
        
        # Read HDF files
        aster_data = self.read_multiple_hdf_files(hdf_files, log_callback, should_stop_callback)
        if not aster_data:
            return None
        
        return self.process_aster_data_threaded(aster_data, progress_callback, log_callback, should_stop_callback)
    
    def process_hdf_file_threaded(self, hdf_path, progress_callback, log_callback, should_stop_callback):
        """Process single HDF file with threading"""
        progress_callback(20, "📖 Reading HDF file...")
        log_callback(f"📖 Reading HDF file: {os.path.basename(hdf_path)}")
        
        if should_stop_callback():
            return None
        
        # Read single HDF file
        aster_data = self.read_single_hdf_file(hdf_path, log_callback)
        if not aster_data:
            return None
        
        return self.process_aster_data_threaded(aster_data, progress_callback, log_callback, should_stop_callback)
    
    def process_aster_data_threaded(self, aster_data, progress_callback, log_callback, should_stop_callback):
        """Process ASTER data with detailed progress tracking"""
        
        progress_callback(45, "🎯 Extracting spectral bands...")
        log_callback("🎯 Starting band extraction...")
        
        if should_stop_callback():
            return None
        
        # Extract bands
        extracted_bands = self.extract_bands_safe(aster_data, progress_callback, log_callback, should_stop_callback)
        if not extracted_bands:
            return None
        
        progress_callback(60, "🧮 Calculating mineral indices and maps...")
        log_callback(f"🧮 Calculating {len(self.mineral_ratios)} mineral indices and advanced maps...")
        
        if should_stop_callback():
            return None
        
        # Calculate mineral ratios and maps
        mineral_indices = self.calculate_mineral_ratios_safe(extracted_bands, log_callback, should_stop_callback)
        
        progress_callback(80, "🎨 Creating composite images...")
        log_callback("🎨 Creating false color composite images...")
        
        if should_stop_callback():
            return None
        
        # Create composites
        composites = self.create_composites_safe(extracted_bands, log_callback, should_stop_callback)
        
        progress_callback(90, "➕ Adding layers to QGIS...")
        log_callback("➕ Adding all layers to QGIS project...")
        
        if should_stop_callback():
            return None
        
        # Add to QGIS
        self.add_layers_to_qgis_safe(extracted_bands, mineral_indices, composites, log_callback)
        
        result = {
            'bands': extracted_bands,
            'indices': mineral_indices,
            'composites': composites
        }
        
        # Save metadata
        self.save_processing_metadata(result, log_callback)
        
        progress_callback(100, "✅ Processing complete!")
        log_callback("✅ ASTER processing completed successfully!")
        
        return result
    
    def extract_aster_zip(self, zip_path, log_callback):
        """Extract ASTER ZIP file with logging"""
        try:
            extract_dir = tempfile.mkdtemp(prefix='aster_extract_')
            self.temp_dirs.append(extract_dir)
            
            log_callback(f"📁 Creating extraction directory: {extract_dir}")
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                log_callback(f"📋 ZIP contains {len(file_list)} files")
                zip_ref.extractall(extract_dir)
            
            log_callback("✅ ZIP extraction completed")
            return {'extract_dir': extract_dir}
            
        except Exception as e:
            log_callback(f"❌ Failed to extract ZIP file: {str(e)}")
            return None
    
    def find_hdf_files(self, extract_dir, log_callback):
        """Find HDF files with improved detection logic"""
        hdf_files = {'VNIR': [], 'SWIR': [], 'TIR': []}
        
        try:
            log_callback("🔍 Scanning directory structure for HDF files...")
            
            total_files = 0
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    total_files += 1
                    if file.lower().endswith('.hdf'):
                        file_path = os.path.join(root, file)
                        file_upper = file.upper()
                        
                        try:
                            file_size = os.path.getsize(file_path)
                            log_callback(f"📊 Found HDF: {file} (Size: {file_size/1024:.1f} KB)")
                            
                            # Method 1: Check subdatasets to determine type
                            dataset = gdal.Open(file_path, gdal.GA_ReadOnly)
                            if dataset:
                                subdatasets = dataset.GetSubDatasets()
                                vnir_count = 0
                                swir_count = 0
                                
                                for sub in subdatasets:
                                    sub_name = sub[1].upper()
                                    if 'VNIR' in sub_name or any(band in sub_name for band in ['BAND1', 'BAND2', 'BAND3']):
                                        vnir_count += 1
                                    elif 'SWIR' in sub_name or any(band in sub_name for band in ['BAND4', 'BAND5', 'BAND6', 'BAND7', 'BAND8', 'BAND9']):
                                        swir_count += 1
                                
                                dataset = None
                                
                                # Classify based on content
                                if vnir_count > swir_count:
                                    hdf_files['VNIR'].append(file_path)
                                    log_callback(f"📊 Classified as VNIR HDF: {file} ({vnir_count} VNIR bands)")
                                elif swir_count > vnir_count:
                                    hdf_files['SWIR'].append(file_path)
                                    log_callback(f"📊 Classified as SWIR HDF: {file} ({swir_count} SWIR bands)")
                                else:
                                    # Fallback: Use file size
                                    if file_size > 100000:
                                        hdf_files['VNIR'].append(file_path)
                                        log_callback(f"📊 Classified as VNIR HDF by size: {file}")
                                    else:
                                        hdf_files['SWIR'].append(file_path)
                                        log_callback(f"📊 Classified as SWIR HDF by size: {file}")
                            else:
                                if file_size > 100000:
                                    hdf_files['VNIR'].append(file_path)
                                    log_callback(f"📊 Assumed VNIR HDF (large file): {file}")
                                else:
                                    hdf_files['SWIR'].append(file_path)
                                    log_callback(f"📊 Assumed SWIR HDF (small file): {file}")
                                    
                        except Exception as e:
                            log_callback(f"⚠️ Error analyzing {file}: {str(e)}")
                            hdf_files['VNIR'].append(file_path)
            
            total_hdf = sum(len(files) for files in hdf_files.values())
            log_callback(f"📈 Summary: {total_hdf} HDF files found in {total_files} total files")
            log_callback(f"📊 VNIR files: {len(hdf_files['VNIR'])}, SWIR files: {len(hdf_files['SWIR'])}")
            
            return hdf_files
            
        except Exception as e:
            log_callback(f"❌ Error finding HDF files: {str(e)}")
            return {}
    
    def read_multiple_hdf_files(self, hdf_files, log_callback, should_stop_callback):
        """Read multiple HDF files with progress tracking"""
        combined_data = {}
        
        try:
            # Process VNIR files
            if hdf_files['VNIR']:
                for i, vnir_file in enumerate(hdf_files['VNIR']):
                    if should_stop_callback():
                        return None
                    
                    log_callback(f"📖 Reading VNIR file {i+1}/{len(hdf_files['VNIR'])}: {os.path.basename(vnir_file)}")
                    vnir_data = self.read_single_hdf_file(vnir_file, log_callback)
                    if vnir_data:
                        combined_data.update(vnir_data)
            
            # Process SWIR files
            if hdf_files['SWIR']:
                for i, swir_file in enumerate(hdf_files['SWIR']):
                    if should_stop_callback():
                        return None
                    
                    log_callback(f"📖 Reading SWIR file {i+1}/{len(hdf_files['SWIR'])}: {os.path.basename(swir_file)}")
                    swir_data = self.read_single_hdf_file(swir_file, log_callback)
                    if swir_data:
                        combined_data.update(swir_data)
            
            if not combined_data:
                raise Exception("No data could be read from HDF files")
            
            log_callback(f"✅ Successfully read {len(combined_data)} datasets from HDF files")
            return combined_data
            
        except Exception as e:
            log_callback(f"❌ Failed to read HDF files: {str(e)}")
            return None
    
    def read_single_hdf_file(self, hdf_path, log_callback):
        """Read single HDF file with multiple fallback methods"""
        try:
            log_callback(f"🔍 Attempting to read: {os.path.basename(hdf_path)}")
            
            # Try GDAL first
            if HAS_GDAL:
                try:
                    result = self.read_hdf_with_gdal(hdf_path, log_callback)
                    if result:
                        return result
                    log_callback("⚠️ GDAL method failed, trying rasterio...")
                except Exception as gdal_error:
                    log_callback(f"⚠️ GDAL failed: {str(gdal_error)}")
            
            # Try rasterio as fallback
            if HAS_RASTERIO:
                try:
                    result = self.read_hdf_with_rasterio_subdatasets(hdf_path, log_callback)
                    if result:
                        return result
                    log_callback("⚠️ Rasterio subdataset method failed, trying direct rasterio...")
                except Exception as rasterio_error:
                    log_callback(f"⚠️ Rasterio subdatasets failed: {str(rasterio_error)}")
                
                try:
                    result = self.read_hdf_with_rasterio(hdf_path, log_callback)
                    if result:
                        return result
                except Exception as rasterio_error:
                    log_callback(f"⚠️ Direct rasterio failed: {str(rasterio_error)}")
            
            # Final fallback: Try system tools
            log_callback("🔄 All standard methods failed, trying system extraction...")
            return self.extract_hdf_with_system_tools(hdf_path, log_callback)
                
        except Exception as e:
            log_callback(f"❌ Failed to read {os.path.basename(hdf_path)}: {str(e)}")
            return None
    
    def read_hdf_with_gdal(self, file_path, log_callback):
        """Read HDF file using GDAL with improved error handling"""
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
                                band_data = band.ReadAsArray()
                                
                                if band_data is None:
                                    log_callback(f"🔄 Trying alternative read method for band {j}...")
                                    band_data = band.ReadAsArray(0, 0, band_xsize, band_ysize)
                                
                                if band_data is None:
                                    log_callback(f"🔄 Trying chunked read for band {j}...")
                                    chunk_size = min(100, band_xsize, band_ysize)
                                    test_chunk = band.ReadAsArray(0, 0, chunk_size, chunk_size)
                                    if test_chunk is not None and isinstance(test_chunk, np.ndarray):
                                        band_data = band.ReadAsArray(0, 0, band_xsize, band_ysize)
                                
                                if band_data is None:
                                    log_callback(f"🔄 Trying numpy-based read for band {j}...")
                                    dtype_map = {
                                        gdal.GDT_Byte: np.uint8,
                                        gdal.GDT_UInt16: np.uint16,
                                        gdal.GDT_Int16: np.int16,
                                        gdal.GDT_UInt32: np.uint32,
                                        gdal.GDT_Int32: np.int32,
                                        gdal.GDT_Float32: np.float32,
                                        gdal.GDT_Float64: np.float64
                                    }
                                    numpy_dtype = dtype_map.get(band_type, np.float32)
                                    band_data = np.zeros((band_ysize, band_xsize), dtype=numpy_dtype)
                                    result = band.ReadAsArray(buf_obj=band_data)
                                    if result is None:
                                        band_data = None
                                
                                if band_data is None:
                                    log_callback(f"🔄 Trying GDAL ReadRaster...")
                                    raw_data = band.ReadRaster(0, 0, band_xsize, band_ysize)
                                    if raw_data:
                                        dtype_map = {
                                            gdal.GDT_Byte: np.uint8,
                                            gdal.GDT_UInt16: np.uint16,
                                            gdal.GDT_Int16: np.int16,
                                        }
                                        numpy_dtype = dtype_map.get(band_type, np.uint16)
                                        band_data = np.frombuffer(raw_data, dtype=numpy_dtype).reshape(band_ysize, band_xsize)
                                
                                if band_data is None:
                                    log_callback(f"❌ All read methods failed for band {j}")
                                    continue
                                
                                if not isinstance(band_data, np.ndarray):
                                    band_data = np.array(band_data)
                                
                                if band_data.size == 0:
                                    log_callback(f"⚠️ Band {j} is empty")
                                    continue
                                
                                if np.all(band_data == 0) or np.all(np.isnan(band_data)):
                                    log_callback(f"⚠️ Band {j} contains no valid data")
                                    continue
                                
                                bands.append(band_data)
                                log_callback(f"✅ Successfully read band {j}, shape: {band_data.shape}, dtype: {band_data.dtype}")
                                
                            except Exception as band_error:
                                log_callback(f"❌ Error reading band {j}: {str(band_error)}")
                                continue
                        
                        if bands:
                            if 'VNIR' in name or any(band_name in name for band_name in ['BAND1', 'BAND2', 'BAND3']):
                                sensor_type = 'vnir'
                            elif 'SWIR' in name or any(band_name in name for band_name in ['BAND4', 'BAND5', 'BAND6', 'BAND7', 'BAND8', 'BAND9']):
                                sensor_type = 'swir'
                            else:
                                sensor_type = 'vnir'
                            
                            if f'{sensor_type}_bands' not in data:
                                data[f'{sensor_type}_bands'] = {}
                                data[f'{sensor_type}_geotransform'] = sub_ds.GetGeoTransform()
                                data[f'{sensor_type}_projection'] = sub_ds.GetProjection()
                            
                            for band_idx, band_data in enumerate(bands):
                                band_num = None
                                for band_name in ['BAND1', 'BAND2', 'BAND3N', 'BAND4', 'BAND5', 'BAND6', 'BAND7', 'BAND8', 'BAND9']:
                                    if band_name in name:
                                        band_num = band_name
                                        break
                                
                                if band_num:
                                    data[f'{sensor_type}_bands'][band_num] = band_data
                                    log_callback(f"✅ Stored {sensor_type.upper()} {band_num}: {band_data.shape}")
                        
                        sub_ds = None
                        
                    except Exception as e:
                        log_callback(f"⚠️ Failed to read subdataset {name}: {str(e)}")
                        continue
            else:
                log_callback("📊 No subdatasets found, trying direct read...")
                
                try:
                    bands = []
                    for i in range(1, dataset.RasterCount + 1):
                        band = dataset.GetRasterBand(i)
                        
                        try:
                            band_data = band.ReadAsArray()
                            if band_data is not None and isinstance(band_data, np.ndarray) and band_data.size > 0:
                                bands.append(band_data)
                                log_callback(f"✅ Read band {i}, shape: {band_data.shape}")
                            else:
                                log_callback(f"⚠️ Band {i} invalid or empty")
                        except Exception as e:
                            log_callback(f"⚠️ Error reading band {i}: {str(e)}")
                            continue
                    
                    if bands:
                        filename = os.path.basename(file_path).upper()
                        if 'VNIR' in filename:
                            data['vnir_bands'] = {f'Band{i+1}': bands[i] for i in range(min(len(bands), 3))}
                            data['vnir_geotransform'] = dataset.GetGeoTransform()
                            data['vnir_projection'] = dataset.GetProjection()
                            log_callback(f"✅ Stored VNIR bands: {len(data['vnir_bands'])}")
                        elif 'SWIR' in filename:
                            data['swir_bands'] = {f'Band{i+4}': bands[i] for i in range(min(len(bands), 6))}
                            data['swir_geotransform'] = dataset.GetGeoTransform()
                            data['swir_projection'] = dataset.GetProjection()
                            log_callback(f"✅ Stored SWIR bands: {len(data['swir_bands'])}")
                        else:
                            data['vnir_bands'] = {f'Band{i+1}': bands[i] for i in range(min(len(bands), 3))}
                            data['vnir_geotransform'] = dataset.GetGeoTransform()
                            data['vnir_projection'] = dataset.GetProjection()
                            log_callback(f"✅ Stored data as VNIR bands: {len(data['vnir_bands'])}")
                
                except Exception as e:
                    log_callback(f"❌ Direct read failed: {str(e)}")
            
            dataset = None
            
            total_bands = 0
            if 'vnir_bands' in data:
                total_bands += len(data['vnir_bands'])
            if 'swir_bands' in data:
                total_bands += len(data['swir_bands'])
            
            if total_bands == 0:
                raise Exception("No valid band data could be extracted from HDF file")
            
            log_callback(f"✅ Successfully extracted {total_bands} bands from HDF file")
            return data
            
        except Exception as e:
            log_callback(f"❌ GDAL read error for {file_path}: {str(e)}")
            raise
    
    def read_hdf_with_rasterio_subdatasets(self, file_path, log_callback):
        """Read HDF using rasterio with subdataset support"""
        if not HAS_RASTERIO:
            raise Exception("Rasterio not available")
        
        try:
            log_callback(f"🔧 Opening HDF file with rasterio (subdataset method)...")
            
            data = {}
            
            with rasterio.open(file_path) as main_src:
                subdatasets = main_src.subdatasets
                log_callback(f"📊 Rasterio found {len(subdatasets)} subdatasets")
                
                for i, subdataset_path in enumerate(subdatasets):
                    try:
                        log_callback(f"📋 Reading rasterio subdataset {i+1}: {subdataset_path}")
                        
                        with rasterio.open(subdataset_path) as sub_src:
                            band_data = sub_src.read(1)
                            
                            if band_data is not None and isinstance(band_data, np.ndarray) and band_data.size > 0:
                                log_callback(f"✅ Rasterio successfully read subdataset {i+1}: {band_data.shape}, dtype: {band_data.dtype}")
                                
                                path_upper = subdataset_path.upper()
                                if 'BAND1' in path_upper:
                                    if 'vnir_bands' not in data:
                                        data['vnir_bands'] = {}
                                        data['vnir_geotransform'] = sub_src.transform
                                        data['vnir_projection'] = str(sub_src.crs)
                                    data['vnir_bands']['BAND1'] = band_data
                                elif 'BAND2' in path_upper:
                                    if 'vnir_bands' not in data:
                                        data['vnir_bands'] = {}
                                        data['vnir_geotransform'] = sub_src.transform
                                        data['vnir_projection'] = str(sub_src.crs)
                                    data['vnir_bands']['BAND2'] = band_data
                                elif 'BAND3' in path_upper:
                                    if 'vnir_bands' not in data:
                                        data['vnir_bands'] = {}
                                        data['vnir_geotransform'] = sub_src.transform
                                        data['vnir_projection'] = str(sub_src.crs)
                                    data['vnir_bands']['BAND3N'] = band_data
                                elif any(band in path_upper for band in ['BAND4', 'BAND5', 'BAND6', 'BAND7', 'BAND8', 'BAND9']):
                                    if 'swir_bands' not in data:
                                        data['swir_bands'] = {}
                                        data['swir_geotransform'] = sub_src.transform
                                        data['swir_projection'] = str(sub_src.crs)
                                    
                                    for band_name in ['BAND4', 'BAND5', 'BAND6', 'BAND7', 'BAND8', 'BAND9']:
                                        if band_name in path_upper:
                                            data['swir_bands'][band_name] = band_data
                                            break
                            else:
                                log_callback(f"⚠️ Rasterio subdataset {i+1} returned invalid data")
                        
                    except Exception as sub_error:
                        log_callback(f"⚠️ Failed to read rasterio subdataset {i+1}: {str(sub_error)}")
                        continue
            
            total_bands = 0
            if 'vnir_bands' in data:
                total_bands += len(data['vnir_bands'])
            if 'swir_bands' in data:
                total_bands += len(data['swir_bands'])
            
            if total_bands > 0:
                log_callback(f"✅ Rasterio successfully extracted {total_bands} bands")
                return data
            else:
                raise Exception("No bands extracted with rasterio subdataset method")
                
        except Exception as e:
            log_callback(f"❌ Rasterio subdataset read error: {str(e)}")
            raise
    
    def read_hdf_with_rasterio(self, file_path, log_callback):
        """Read HDF file using rasterio with logging"""
        if not HAS_RASTERIO:
            raise Exception("Rasterio not available")
        
        try:
            log_callback(f"🔧 Opening HDF file with rasterio...")
            
            with rasterio.open(file_path) as src:
                data = src.read()
                profile = src.profile
                
                log_callback(f"📏 Data shape: {data.shape}")
                
                filename = os.path.basename(file_path).upper()
                result = {}
                
                if 'VNIR' in filename:
                    result['vnir_data'] = data
                    log_callback("✅ Stored as VNIR data")
                elif 'SWIR' in filename:
                    result['swir_data'] = data
                    log_callback("✅ Stored as SWIR data")
                else:
                    result['vnir_data'] = data
                    log_callback("✅ Stored as VNIR data (default)")
                
                result['profile'] = profile
                return result
                
        except Exception as e:
            log_callback(f"❌ Rasterio read error for {file_path}: {str(e)}")
            raise
    
    def extract_hdf_with_system_tools(self, file_path, log_callback):
        """Final fallback: Try to extract HDF data using system tools"""
        try:
            log_callback("🔧 Attempting system-level HDF extraction...")
            
            import subprocess
            import tempfile
            
            data = {}
            temp_dir = tempfile.mkdtemp(prefix='hdf_extract_')
            
            try:
                result = subprocess.run(['gdalinfo', file_path], 
                                      capture_output=True, text=True, timeout=30)
                
                if result.returncode == 0:
                    gdalinfo_output = result.stdout
                    log_callback("📊 Got gdalinfo output, parsing subdatasets...")
                    
                    lines = gdalinfo_output.split('\n')
                    subdatasets = []
                    
                    for line in lines:
                        if 'SUBDATASET_' in line and '_NAME=' in line:
                            parts = line.split('=', 1)
                            if len(parts) > 1:
                                subdataset_path = parts[1].strip()
                                subdatasets.append(subdataset_path)
                    
                    log_callback(f"📋 Found {len(subdatasets)} subdatasets via gdalinfo")
                    
                    for i, subdataset_path in enumerate(subdatasets):
                        if 'QA_DATAPLANE' in subdataset_path.upper():
                            continue
                        
                        try:
                            output_file = os.path.join(temp_dir, f"band_{i}.tif")
                            
                            translate_result = subprocess.run([
                                'gdal_translate', '-of', 'GTiff', 
                                subdataset_path, output_file
                            ], capture_output=True, text=True, timeout=60)
                            
                            if translate_result.returncode == 0 and os.path.exists(output_file):
                                log_callback(f"✅ Successfully extracted subdataset {i} to {output_file}")
                                
                                with rasterio.open(output_file) as src:
                                    band_data = src.read(1)
                                    
                                    if band_data is not None and band_data.size > 0:
                                        if any(band in subdataset_path.upper() for band in ['BAND1', 'BAND2', 'BAND3']):
                                            if 'vnir_bands' not in data:
                                                data['vnir_bands'] = {}
                                                data['vnir_geotransform'] = src.transform
                                                data['vnir_projection'] = str(src.crs)
                                            
                                            for band_name in ['BAND1', 'BAND2', 'BAND3N']:
                                                if band_name in subdataset_path.upper():
                                                    data['vnir_bands'][band_name] = band_data
                                                    log_callback(f"✅ Stored VNIR {band_name} via system extraction")
                                                    break
                                        
                                        elif any(band in subdataset_path.upper() for band in ['BAND4', 'BAND5', 'BAND6', 'BAND7', 'BAND8', 'BAND9']):
                                            if 'swir_bands' not in data:
                                                data['swir_bands'] = {}
                                                data['swir_geotransform'] = src.transform
                                                data['swir_projection'] = str(src.crs)
                                            
                                            for band_name in ['BAND4', 'BAND5', 'BAND6', 'BAND7', 'BAND8', 'BAND9']:
                                                if band_name in subdataset_path.upper():
                                                    data['swir_bands'][band_name] = band_data
                                                    log_callback(f"✅ Stored SWIR {band_name} via system extraction")
                                                    break
                            else:
                                log_callback(f"⚠️ gdal_translate failed for subdataset {i}")
                        
                        except Exception as extract_error:
                            log_callback(f"⚠️ Failed to extract subdataset {i}: {str(extract_error)}")
                            continue
                
                else:
                    log_callback("⚠️ gdalinfo command failed")
            
            except subprocess.TimeoutExpired:
                log_callback("⚠️ gdalinfo command timed out")
            except FileNotFoundError:
                log_callback("⚠️ gdalinfo command not found in system PATH")
            
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
            
            total_bands = 0
            if 'vnir_bands' in data:
                total_bands += len(data['vnir_bands'])
            if 'swir_bands' in data:
                total_bands += len(data['swir_bands'])
            
            if total_bands > 0:
                log_callback(f"✅ System extraction succeeded with {total_bands} bands")
                return data
            else:
                log_callback("❌ System extraction found no valid bands")
                return None
                
        except Exception as e:
            log_callback(f"❌ System extraction failed: {str(e)}")
            return None
    
    def extract_bands_safe(self, aster_data, progress_callback, log_callback, should_stop_callback):
        """Extract bands with error handling and detailed logging"""
        extracted_bands = {}
        
        try:
            temp_dir = tempfile.mkdtemp(prefix='aster_bands_')
            self.temp_dirs.append(temp_dir)
            
            log_callback(f"📁 Created temp directory for bands: {temp_dir}")
            
            total_bands = 0
            if 'vnir_bands' in aster_data:
                total_bands += len(aster_data['vnir_bands'])
            if 'swir_bands' in aster_data:
                total_bands += len(aster_data['swir_bands'])
            
            if total_bands == 0:
                raise Exception("No band data found in ASTER file")
            
            log_callback(f"📊 Processing {total_bands} bands total")
            current_band = 0
            
            if 'vnir_bands' in aster_data:
                vnir_bands = aster_data['vnir_bands']
                log_callback(f"📊 Processing {len(vnir_bands)} VNIR bands")
                
                for band_name, band_data in vnir_bands.items():
                    if should_stop_callback():
                        return None
                    
                    if band_name.upper() in ['BAND1']:
                        standard_name = 'Band1'
                    elif band_name.upper() in ['BAND2']:
                        standard_name = 'Band2'
                    elif band_name.upper() in ['BAND3N', 'BAND3']:
                        standard_name = 'Band3N'
                    else:
                        standard_name = band_name
                    
                    if standard_name in self.aster_bands['VNIR']:
                        try:
                            log_callback(f"🎯 Extracting VNIR {standard_name} ({self.aster_bands['VNIR'][standard_name]['name']})...")
                            
                            band_file = self.save_band_as_geotiff_safe(
                                band_data, standard_name, self.aster_bands['VNIR'][standard_name], 
                                temp_dir, aster_data, log_callback
                            )
                            if band_file:
                                extracted_bands[standard_name] = band_file
                                log_callback(f"✅ Successfully extracted VNIR band: {standard_name}")
                            
                            current_band += 1
                            progress_val = 45 + int(20 * current_band / total_bands)
                            progress_callback(progress_val, f"Extracted {standard_name}...")
                                
                        except Exception as e:
                            log_callback(f"⚠️ Failed to extract VNIR band {standard_name}: {str(e)}")
                            continue
            
            if 'swir_bands' in aster_data:
                swir_bands = aster_data['swir_bands']
                log_callback(f"📊 Processing {len(swir_bands)} SWIR bands")
                
                for band_name, band_data in swir_bands.items():
                    if should_stop_callback():
                        return None
                    
                    band_mapping = {
                        'BAND4': 'Band4', 'BAND5': 'Band5', 'BAND6': 'Band6',
                        'BAND7': 'Band7', 'BAND8': 'Band8', 'BAND9': 'Band9'
                    }
                    
                    standard_name = band_mapping.get(band_name.upper(), band_name)
                    
                    if standard_name in self.aster_bands['SWIR']:
                        try:
                            log_callback(f"🎯 Extracting SWIR {standard_name} ({self.aster_bands['SWIR'][standard_name]['name']})...")
                            
                            band_file = self.save_band_as_geotiff_safe(
                                band_data, standard_name, self.aster_bands['SWIR'][standard_name], 
                                temp_dir, aster_data, log_callback
                            )
                            if band_file:
                                extracted_bands[standard_name] = band_file
                                log_callback(f"✅ Successfully extracted SWIR band: {standard_name}")
                            
                            current_band += 1
                            progress_val = 45 + int(20 * current_band / total_bands)
                            progress_callback(progress_val, f"Extracted {standard_name}...")
                                
                        except Exception as e:
                            log_callback(f"⚠️ Failed to extract SWIR band {standard_name}: {str(e)}")
                            continue
            
            if not extracted_bands:
                raise Exception("No bands could be extracted from ASTER data")
            
            log_callback(f"🎉 Successfully extracted {len(extracted_bands)} bands total")
            return extracted_bands
            
        except Exception as e:
            log_callback(f"❌ Band extraction failed: {str(e)}")
            return None
    
    def save_band_as_geotiff_safe(self, band_data, band_name, band_info, output_dir, aster_data, log_callback):
        """Save band as GeoTIFF with validation and logging"""
        try:
            output_path = os.path.join(output_dir, f"ASTER_{band_name}.tif")
            
            if band_data is None or band_data.size == 0:
                raise Exception(f"Band {band_name} contains no data")
            
            log_callback(f"💾 Saving {band_name} to: {os.path.basename(output_path)}")
            
            original_dtype = band_data.dtype
            log_callback(f"📊 Original data type: {original_dtype}, range: {np.min(band_data):.2f} - {np.max(band_data):.2f}")
            
            if band_data.dtype == np.uint8:
                processed_data = band_data
            elif band_data.dtype in [np.int16, np.uint16]:
                if np.max(band_data) > 10000:
                    processed_data = (band_data / np.max(band_data) * 10000).astype(np.uint16)
                    log_callback("🔧 Normalized high values to 0-10000 range")
                else:
                    processed_data = band_data.astype(np.uint16)
            else:
                if np.max(band_data) <= 1.0:
                    processed_data = (band_data * 10000).astype(np.uint16)
                    log_callback("🔧 Scaled 0-1 data to 0-10000 range")
                else:
                    processed_data = np.clip(band_data, 0, 10000).astype(np.uint16)
                    log_callback("🔧 Clipped data to 0-10000 range")
            
            geotransform, projection = self.get_geospatial_info(aster_data)
            
            if HAS_RASTERIO:
                self.save_with_rasterio(output_path, processed_data, geotransform, projection, band_name, band_info, log_callback)
            elif HAS_GDAL:
                self.save_with_gdal(output_path, processed_data, geotransform, projection, band_name, band_info, log_callback)
            else:
                raise Exception("No suitable library for saving GeoTIFF")
            
            if not os.path.exists(output_path):
                raise Exception(f"Failed to create output file: {output_path}")
            
            file_size = os.path.getsize(output_path)
            log_callback(f"✅ Saved {band_name}: {file_size} bytes")
            
            return output_path
            
        except Exception as e:
            log_callback(f"❌ Failed to save band {band_name}: {str(e)}")
            return None
    
    def save_with_rasterio(self, output_path, data, geotransform, projection, band_name, band_info, log_callback):
        """Save using rasterio with logging"""
        height, width = data.shape
        
        if hasattr(geotransform, '__len__') and len(geotransform) == 6:
            transform = rasterio.transform.Affine(*geotransform)
        else:
            transform = geotransform
        
        try:
            if projection:
                crs = rasterio.crs.CRS.from_string(projection)
            else:
                crs = None
        except:
            crs = None
        
        profile = {
            'driver': 'GTiff',
            'dtype': data.dtype,
            'nodata': 0,
            'width': width,
            'height': height,
            'count': 1,
            'crs': crs,
            'transform': transform,
            'compress': 'lzw',
            'tiled': True
        }
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(data, 1)
            dst.set_band_description(1, f"{band_name} - {band_info['name']}")
        
        log_callback(f"💾 Saved with rasterio: {width}x{height}")
    
    def save_with_gdal(self, output_path, data, geotransform, projection, band_name, band_info, log_callback):
        """Save using GDAL with logging"""
        height, width = data.shape
        
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(
            output_path, width, height, 1, gdal.GDT_UInt16,
            options=['COMPRESS=LZW', 'TILED=YES']
        )
        
        if geotransform:
            dataset.SetGeoTransform(geotransform)
        if projection:
            dataset.SetProjection(projection)
        
        band = dataset.GetRasterBand(1)
        band.WriteArray(data)
        band.SetDescription(f"{band_name} - {band_info['name']}")
        band.SetNoDataValue(0)
        
        band.SetMetadataItem('Wavelength', str(band_info['wavelength']))
        band.SetMetadataItem('Resolution', str(band_info['resolution']))
        
        dataset = None
        log_callback(f"💾 Saved with GDAL: {width}x{height}")
    
    def get_geospatial_info(self, aster_data):
        """Get geospatial information with fallbacks"""
        if 'vnir_geotransform' in aster_data:
            geotransform = aster_data['vnir_geotransform']
            projection = aster_data.get('vnir_projection', '')
        elif 'swir_geotransform' in aster_data:
            geotransform = aster_data['swir_geotransform']
            projection = aster_data.get('swir_projection', '')
        elif 'profile' in aster_data:
            profile = aster_data['profile']
            geotransform = profile.get('transform', (0, 1, 0, 0, 0, -1))
            projection = str(profile.get('crs', ''))
        else:
            geotransform = (0, 1, 0, 0, 0, -1)
            projection = ''
        
        return geotransform, projection
    
    def calculate_mineral_ratios_safe(self, extracted_bands, log_callback, should_stop_callback):
        """Calculate mineral indices and advanced mineral maps"""
        mineral_indices = {}
        
        try:
            temp_dir = tempfile.mkdtemp(prefix='aster_indices_')
            self.temp_dirs.append(temp_dir)
            log_callback(f"📁 Created temp directory for indices: {temp_dir}")
            
            # Calculate simple band ratios
            for ratio_name, ratio_info in self.mineral_ratios.items():
                if should_stop_callback():
                    return None
                
                try:
                    log_callback(f"🧮 Calculating {ratio_name} index...")
                    
                    bands_needed = ratio_info['bands']
                    if all(band in extracted_bands for band in bands_needed):
                        band_data = {}
                        for band in bands_needed:
                            with rasterio.open(extracted_bands[band]) as src:
                                band_data[band] = src.read(1)
                        
                        shapes = [data.shape for data in band_data.values()]
                        if len(set(shapes)) != 1:
                            log_callback(f"⚠️ Skipping {ratio_name}: Bands have different shapes")
                            continue
                        
                        formula = ratio_info['formula']
                        result = self._evaluate_ratio_formula(band_data, formula, log_callback)
                        
                        if result is not None:
                            output_path = os.path.join(temp_dir, f"{ratio_name}_index.tif")
                            self._save_index_as_geotiff(result, output_path, extracted_bands[bands_needed[0]], ratio_name, log_callback)
                            mineral_indices[ratio_name] = output_path
                            log_callback(f"✅ Saved {ratio_name} index")
                        else:
                            log_callback(f"⚠️ Failed to calculate {ratio_name} index")
                    else:
                        log_callback(f"⚠️ Skipping {ratio_name}: Missing required bands")
                
                except Exception as e:
                    log_callback(f"⚠️ Error calculating {ratio_name}: {str(e)}")
                    continue
            
            # Advanced mineral mapping with MineralMapper
            if HAS_MINERAL_MAPPER and HAS_RASTERIO and HAS_SKLEARN and HAS_SCIPY:
                if not self.mineral_signatures:
                    log_callback("⚠️ No mineral signatures loaded, skipping advanced mineral mapping")
                else:
                    try:
                        log_callback("🌟 Starting advanced mineral mapping...")
                        
                        band_files = list(extracted_bands.values())
                        if band_files:
                            stack_path = os.path.join(temp_dir, "stacked_bands.tif")
                            with rasterio.open(band_files[0]) as first_band:
                                profile = first_band.profile.copy()
                                profile.update(count=len(band_files))
                            
                            with rasterio.open(stack_path, 'w', **profile) as dst:
                                for i, band_file in enumerate(band_files, 1):
                                    with rasterio.open(band_file) as src:
                                        dst.write(src.read(1), i)
                            
                            mineral_mapper = MineralMapper()
                            if mineral_mapper.load_data(stack_path):
                                mineral_mapper.set_mineral_signatures(self.mineral_signatures)
                                
                                minerals_to_map = list(self.mineral_signatures.keys())
                                mineral_maps = mineral_mapper.create_mineral_maps(minerals_to_map, method='spectral_unmixing')
                                
                                mineral_map_dir = os.path.join(temp_dir, "mineral_maps")
                                saved_files = mineral_mapper.save_mineral_maps(mineral_maps, mineral_map_dir)
                                
                                for mineral_name, file_path in zip(mineral_maps.keys(), saved_files):
                                    mineral_indices[f"Mineral_{mineral_name}"] = file_path
                                    log_callback(f"✅ Saved mineral map for {mineral_name}")
                                
                                validation_results = mineral_mapper.validate_mineral_maps(mineral_maps)
                                for mineral_name, stats in validation_results.items():
                                    log_callback(f"📊 Validation for {mineral_name}: {stats}")
                            else:
                                log_callback("⚠️ Failed to load data for mineral mapping")
                        else:
                            log_callback("⚠️ No bands available for advanced mineral mapping")
                    except Exception as e:
                        log_callback(f"❌ Advanced mineral mapping failed: {str(e)}")
            else:
                log_callback("⚠️ Skipping advanced mineral mapping: Required dependencies missing")
            
            if not mineral_indices:
                raise Exception("No mineral indices or maps could be generated")
            
            log_callback(f"🎉 Generated {len(mineral_indices)} mineral indices/maps")
            return mineral_indices
            
        except Exception as e:
            log_callback(f"❌ Mineral index calculation failed: {str(e)}")
            return None
    
    def _evaluate_ratio_formula(self, band_data, formula, log_callback):
        """Evaluate band ratio formula safely"""
        try:
            local_vars = band_data.copy()
            local_vars['np'] = np
            
            for band_name in band_data:
                band_data[band_name] = np.where(band_data[band_name] == 0, np.finfo(float).eps, band_data[band_name])
            
            result = eval(formula, {"__builtins__": {}}, local_vars)
            
            result = np.array(result, dtype=np.float32)
            result[~np.isfinite(result)] = np.nan
            
            return result
        except Exception as e:
            log_callback(f"⚠️ Error evaluating formula {formula}: {str(e)}")
            return None
    
    def _save_index_as_geotiff(self, data, output_path, reference_file, index_name, log_callback):
        """Save mineral index as GeoTIFF"""
        try:
            with rasterio.open(reference_file) as ref:
                profile = ref.profile.copy()
                profile.update(dtype='float32', nodata=np.nan, count=1)
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(data, 1)
                dst.set_band_description(1, index_name)
            
            log_callback(f"💾 Saved index {index_name} to {output_path}")
        except Exception as e:
            log_callback(f"❌ Failed to save index {index_name}: {str(e)}")
    
    def create_composites_safe(self, extracted_bands, log_callback, should_stop_callback):
        """Create false color composite images"""
        composites = {}
        
        try:
            temp_dir = tempfile.mkdtemp(prefix='aster_composites_')
            self.temp_dirs.append(temp_dir)
            log_callback(f"📁 Created temp directory for composites: {temp_dir}")
            
            composite_configs = [
                {
                    'name': 'RGB_321',
                    'bands': ['Band3N', 'Band2', 'Band1'],
                    'description': 'False color composite (NIR, Red, Green)'
                },
                {
                    'name': 'RGB_742',
                    'bands': ['Band7', 'Band4', 'Band2'],
                    'description': 'False color composite for mineral mapping'
                }
            ]
            
            for config in composite_configs:
                if should_stop_callback():
                    return None
                
                try:
                    log_callback(f"🎨 Creating composite {config['name']}...")
                    
                    if all(band in extracted_bands for band in config['bands']):
                        band_data = []
                        for band in config['bands']:
                            with rasterio.open(extracted_bands[band]) as src:
                                band_data.append(src.read(1))
                        
                        shapes = [data.shape for data in band_data]
                        if len(set(shapes)) != 1:
                            log_callback(f"⚠️ Skipping {config['name']}: Bands have different shapes")
                            continue
                        
                        # Stack bands into RGB
                        rgb_data = np.stack(band_data, axis=0)
                        
                        # Normalize for visualization
                        for i in range(3):
                            band = rgb_data[i]
                            valid_data = band[~np.isnan(band)]
                            if valid_data.size > 0:
                                min_val, max_val = np.percentile(valid_data, [2, 98])
                                rgb_data[i] = np.clip((band - min_val) / (max_val - min_val) * 255, 0, 255).astype(np.uint8)
                        
                        output_path = os.path.join(temp_dir, f"{config['name']}_composite.tif")
                        with rasterio.open(extracted_bands[config['bands'][0]]) as ref:
                            profile = ref.profile.copy()
                            profile.update(dtype='uint8', count=3, nodata=0)
                        
                        with rasterio.open(output_path, 'w', **profile) as dst:
                            for i in range(3):
                                dst.write(rgb_data[i], i + 1)
                            dst.set_band_description(1, f"{config['name']} - R")
                            dst.set_band_description(2, f"{config['name']} - G")
                            dst.set_band_description(3, f"{config['name']} - B")
                        
                        composites[config['name']] = output_path
                        log_callback(f"✅ Saved composite {config['name']}")
                    else:
                        log_callback(f"⚠️ Skipping {config['name']}: Missing required bands")
                
                except Exception as e:
                    log_callback(f"⚠️ Error creating composite {config['name']}: {str(e)}")
                    continue
            
            log_callback(f"🎉 Generated {len(composites)} composite images")
            return composites
            
        except Exception as e:
            log_callback(f"❌ Composite creation failed: {str(e)}")
            return {}
    
    def add_layers_to_qgis_safe(self, extracted_bands, mineral_indices, composites, log_callback):
        """Add all layers to QGIS with appropriate styling"""
        try:
            for band_name, band_file in extracted_bands.items():
                if os.path.exists(band_file):
                    layer_name = f"ASTER_{band_name}"
                    layer = QgsRasterLayer(band_file, layer_name)
                    if layer.isValid():
                        QgsProject.instance().addMapLayer(layer)
                        log_callback(f"➕ Added band layer: {layer_name}")
                    else:
                        log_callback(f"⚠️ Invalid band layer: {layer_name}")
            
            for index_name, index_file in mineral_indices.items():
                if os.path.exists(index_file):
                    layer_name = f"ASTER_{index_name}"
                    layer = QgsRasterLayer(index_file, layer_name)
                    if layer.isValid():
                        shader = QgsRasterShader()
                        color_ramp = QgsColorRampShader()
                        color_ramp.setColorRampType(QgsColorRampShader.Interpolated)
                        color_ramp.setColorRampItemList([
                            QgsColorRampShader.ColorRampItem(0, QColor(0, 0, 255)),
                            QgsColorRampShader.ColorRampItem(0.5, QColor(255, 255, 0)),
                            QgsColorRampShader.ColorRampItem(1, QColor(255, 0, 0))
                        ])
                        shader.setRasterShaderFunction(color_ramp)
                        renderer = QgsSingleBandPseudoColorRenderer(layer.dataProvider(), 1, shader)
                        layer.setRenderer(renderer)
                        QgsProject.instance().addMapLayer(layer)
                        log_callback(f"➕ Added index/map layer: {layer_name}")
                    else:
                        log_callback(f"⚠️ Invalid index/map layer: {layer_name}")
            
            for composite_name, composite_file in composites.items():
                if os.path.exists(composite_file):
                    layer_name = f"ASTER_{composite_name}"
                    layer = QgsRasterLayer(composite_file, layer_name)
                    if layer.isValid():
                        QgsProject.instance().addMapLayer(layer)
                        log_callback(f"➕ Added composite layer: {layer_name}")
                    else:
                        log_callback(f"⚠️ Invalid composite layer: {layer_name}")
            
            log_callback("✅ All valid layers added to QGIS")
        except Exception as e:
            log_callback(f"❌ Failed to add layers to QGIS: {str(e)}")
    
    def save_processing_metadata(self, result, log_callback):
        """Save processing metadata to JSON"""
        try:
            temp_dir = tempfile.mkdtemp(prefix='aster_metadata_')
            self.temp_dirs.append(temp_dir)
            
            metadata = {
                'input_file': os.path.basename(self.current_file) if self.current_file else 'Unknown',
                'bands': list(result.get('bands', {}).keys()),
                'indices': list(result.get('indices', {}).keys()),
                'composites': list(result.get('composites', {}).keys()),
                'timestamp': datetime.datetime.now().isoformat(),
                'temp_dirs': self.temp_dirs
            }
            
            output_path = os.path.join(temp_dir, 'processing_metadata.json')
            with open(output_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            log_callback(f"💾 Saved processing metadata to {output_path}")
            
        except Exception as e:
            log_callback(f"❌ Failed to save processing metadata: {str(e)}")
    
    def handle_processing_error(self, error):
        """Handle processing errors"""
        self.log_message(f"❌ Processing error: {str(error)}", "ERROR")
        QMessageBox.critical(
            self.iface.mainWindow(),
            "Processing Error",
            f"An error occurred during processing:\n\n{str(error)}\n\nCheck the log for details."
        )