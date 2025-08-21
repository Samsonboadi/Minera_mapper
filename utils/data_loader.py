"""
Data loading utilities for various geospatial formats
"""

import os
import numpy as np
import rasterio
import geopandas as gpd
from osgeo import gdal, ogr
from qgis.core import (QgsRasterLayer, QgsVectorLayer, QgsProject, QgsMessageLog, 
                       Qgis, QgsCoordinateReferenceSystem, QgsRectangle)
from qgis.PyQt.QtWidgets import QMessageBox
# Removed h5py dependency - using GDAL for HDF5 support instead
import netCDF4 as nc
import json
import tempfile
import zipfile
import tarfile

class DataLoader:
    """Universal data loader for geospatial formats"""
    
    def __init__(self):
        self.supported_raster_formats = [
            '.tif', '.tiff', '.img', '.hdf', '.h5', '.hdf5', '.nc', '.jp2', 
            '.bsq', '.bil', '.bip', '.rst', '.grd', '.asc', '.dem', '.dt0',
            '.dt1', '.dt2', '.ers', '.gen', '.vrt'
        ]
        
        self.supported_vector_formats = [
            '.shp', '.gpkg', '.geojson', '.json', '.kml', '.kmz', '.gml', 
            '.tab', '.mif', '.mid', '.dgn', '.dxf', '.csv'
        ]
        
        self.supported_archives = ['.zip', '.tar', '.gz', '.7z']
        
        self.loaded_layers = {}
        self.metadata_cache = {}
    
    def load_raster(self, file_path, layer_name=None):
        """Load raster data with automatic format detection and ZIP processing"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            # Handle ZIP files with ASTER datasets
            if file_ext == '.zip':
                return self.process_aster_zip(file_path)
            elif file_ext in ['.hdf', '.h5', '.hdf5']:
                return self.load_hdf_raster(file_path, layer_name)
            elif file_ext == '.nc':
                return self.load_netcdf_raster(file_path, layer_name)
            elif file_ext == '.jp2':
                return self.load_jpeg2000_raster(file_path, layer_name)
            else:
                return self.load_standard_raster(file_path, layer_name)
                
        except Exception as e:
            QgsMessageLog.logMessage(f"Error loading raster {file_path}: {str(e)}", 
                                    'Mineral Prospectivity', Qgis.Critical)
            raise
    
    def load_vector(self, file_path, layer_name=None):
        """Load vector data with automatic format detection"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            if layer_name is None:
                layer_name = os.path.splitext(os.path.basename(file_path))[0]
            
            # Handle CSV files specially
            if file_path.lower().endswith('.csv'):
                return self.load_csv_as_vector(file_path, layer_name)
            
            # Standard vector loading
            layer = QgsVectorLayer(file_path, layer_name, "ogr")
            
            if not layer.isValid():
                raise Exception(f"Invalid vector layer: {file_path}")
            
            self.loaded_layers[layer_name] = layer
            return layer
            
        except Exception as e:
            QgsMessageLog.logMessage(f"Error loading vector {file_path}: {str(e)}", 
                                    'Mineral Prospectivity', Qgis.Critical)
            raise
    
    def load_standard_raster(self, file_path, layer_name=None):
        """Load standard raster formats using GDAL"""
        if layer_name is None:
            layer_name = os.path.splitext(os.path.basename(file_path))[0]
        
        layer = QgsRasterLayer(file_path, layer_name)
        
        if not layer.isValid():
            raise Exception(f"Invalid raster layer: {file_path}")
        
        # Cache metadata
        self.cache_raster_metadata(layer, file_path)
        
        # Cache metadata and determine layer type
        self.cache_raster_metadata(layer, file_path)
        
        # Determine layer type based on filename and metadata
        layer_type = self.determine_layer_type(file_path, layer_name)
        layer.setCustomProperty('layer_type', layer_type)
        
        self.loaded_layers[layer_name] = layer
        return layer
    
    def load_hdf_raster(self, file_path, layer_name=None):
        """Load HDF4/HDF5 raster data with proper ASTER VNIR/SWIR multi-band layer creation"""
        if layer_name is None:
            layer_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Use GDAL for both HDF4 and HDF5 files
        dataset = gdal.Open(file_path)
        if dataset is None:
            raise Exception(f"Cannot open HDF file: {file_path}")
        
        # Check for subdatasets (common in HDF files)
        subdatasets = dataset.GetSubDatasets()
        if subdatasets:
            # Log available datasets
            dataset_names = [sub[1] for sub in subdatasets]
            QgsMessageLog.logMessage(
                f"Found {len(subdatasets)} subdatasets in {file_path}", 
                'Mineral Prospectivity', Qgis.Info
            )
            
            # For HDF files loaded individually, use first subdataset
            # (Combined ASTER processing handled at ZIP level)
            layer = QgsRasterLayer(subdatasets[0][0], f"{layer_name}_{subdatasets[0][1]}")
        else:
            # Single dataset - load directly
            layer = QgsRasterLayer(file_path, layer_name)
        
        if not layer.isValid():
            raise Exception(f"Invalid HDF raster layer: {file_path}")
        
        # Cache metadata and determine layer type
        self.cache_raster_metadata(layer, file_path)
        
        # Determine layer type based on filename and metadata
        layer_type = self.determine_layer_type(file_path, layer_name)
        layer.setCustomProperty('layer_type', layer_type)
        
        self.loaded_layers[layer_name] = layer
        return layer
    
    def create_combined_aster_dataset(self, hdf_files, group_name, extract_dir, zip_path):
        """Create a single combined ASTER dataset with VNIR+SWIR bands properly resampled"""
        from osgeo import gdal, gdalconst
        
        try:
            # Create directory for processed files
            vrt_dir = os.path.join(extract_dir, 'combined_aster')
            if not os.path.exists(vrt_dir):
                os.makedirs(vrt_dir)
            
            # Collect all VNIR and SWIR bands from all HDF files
            all_bands = []
            
            for hdf_file in hdf_files:
                dataset = gdal.Open(hdf_file)
                if dataset:
                    subdatasets = dataset.GetSubDatasets()
                    
                    # Collect VNIR bands (15m resolution)
                    vnir_bands = [(sub[0], sub[1]) for sub in subdatasets 
                                 if 'VNIR' in sub[1] and 'Band' in sub[1]]
                    
                    # Collect SWIR bands (30m resolution)  
                    swir_bands = [(sub[0], sub[1]) for sub in subdatasets 
                                 if 'SWIR' in sub[1] and 'Band' in sub[1]]
                    
                    all_bands.extend(vnir_bands)
                    all_bands.extend(swir_bands)
            
            if not all_bands:
                raise Exception("No ASTER bands found in HDF files")
            
            # Sort bands by band number for proper spectral order
            def extract_band_number(desc):
                import re
                match = re.search(r'Band(\d+)', desc)
                return int(match.group(1)) if match else 999
            
            all_bands.sort(key=lambda x: extract_band_number(x[1]))
            
            QgsMessageLog.logMessage(
                f"Found {len(all_bands)} ASTER bands to combine: " + 
                ", ".join([f"Band{extract_band_number(b[1])}" for b in all_bands]),
                'Mineral Prospectivity', Qgis.Info
            )
            
            # Create resampled and combined VRT
            combined_vrt_path = os.path.join(vrt_dir, f"{group_name}_COMBINED_ASTER.vrt")
            
            # Use GDAL to create a properly resampled combined VRT
            # Target resolution: 15m (VNIR native resolution) for best quality
            source_paths = [band[0] for band in all_bands]
            
            vrt_options = gdal.BuildVRTOptions(
                separate=True,           # Each input as separate band
                resolution='highest',    # Use 15m resolution (VNIR)
                resampleAlg='bilinear',  # Good for continuous spectral data
                outputSRS='EPSG:4326',   # Ensure consistent projection
                addAlpha=False,          # No alpha channel
                srcNodata=0,             # Set nodata value
                VRTNodata=0              # VRT nodata value
            )
            
            vrt_ds = gdal.BuildVRT(combined_vrt_path, source_paths, options=vrt_options)
            
            if vrt_ds is None:
                raise Exception("Failed to create combined VRT")
            
            # Set ASTER-specific metadata
            self.set_aster_band_metadata(vrt_ds, all_bands)
            
            # Close to flush
            vrt_ds = None
            
            # Create QGIS layer
            layer_name = f"{group_name}_COMBINED"
            layer = QgsRasterLayer(combined_vrt_path, layer_name)
            
            if not layer.isValid():
                raise Exception(f"Invalid combined layer: {combined_vrt_path}")
            
            # Set layer properties
            layer.setCustomProperty('layer_type', 'spectral')
            layer.setCustomProperty('aster_type', 'COMBINED_VNIR_SWIR')
            layer.setCustomProperty('source_zip', zip_path)
            layer.setCustomProperty('extract_dir', extract_dir)
            layer.setCustomProperty('resolution', '15m_resampled')
            
            # Add to QGIS project
            QgsProject.instance().addMapLayer(layer)
            
            # Cache the layer
            self.loaded_layers[layer_name] = layer
            
            QgsMessageLog.logMessage(
                f"Created combined ASTER dataset: {layer.name()} with {layer.bandCount()} bands at 15m resolution",
                'Mineral Prospectivity', Qgis.Info
            )
            
            return layer
            
        except Exception as e:
            QgsMessageLog.logMessage(f"Error creating combined ASTER dataset: {str(e)}", 'Mineral Prospectivity', Qgis.Critical)
            import traceback
            QgsMessageLog.logMessage(f"Traceback: {traceback.format_exc()}", 'Mineral Prospectivity', Qgis.Critical)
            return None
    
    def create_permanent_multiband_vrt(self, datasets, layer_name, vrt_directory):
        """Create a permanent VRT file combining multiple subdatasets into a multi-band layer"""
        from osgeo import gdal
        
        try:
            # Create permanent VRT file in the extraction directory
            vrt_path = os.path.join(vrt_directory, f"{layer_name}.vrt")
            
            # Sort datasets by band name to ensure proper band order
            # Extract band numbers for proper sorting
            def extract_band_number(desc):
                import re
                match = re.search(r'Band(\d+)', desc)
                return int(match.group(1)) if match else 999
            
            sorted_datasets = sorted(datasets, key=lambda x: extract_band_number(x[1]))
            
            # Get source paths for VRT creation
            source_paths = [ds[0] for ds in sorted_datasets]
            
            QgsMessageLog.logMessage(
                f"Creating VRT with {len(source_paths)} bands in order: " + 
                ", ".join([f"Band{extract_band_number(ds[1])}" for ds in sorted_datasets]),
                'Mineral Prospectivity', Qgis.Info
            )
            
            # Create VRT using GDAL buildvrt
            vrt_options = gdal.BuildVRTOptions(
                separate=True,  # Each input as separate band
                resolution='highest',  # Use highest resolution
                resampleAlg='bilinear',  # Better resampling for spectral data
                addAlpha=False,  # No alpha channel needed
                srcNodata=0  # Set nodata value
            )
            
            vrt_ds = gdal.BuildVRT(vrt_path, source_paths, options=vrt_options)
            
            if vrt_ds is None:
                QgsMessageLog.logMessage(f"Failed to create VRT for {layer_name}", 'Mineral Prospectivity', Qgis.Warning)
                return None
            
            # Set proper band names and wavelengths for ASTER
            self.set_aster_band_metadata(vrt_ds, sorted_datasets)
            
            # Close the dataset to flush to disk
            vrt_ds = None
            
            # Verify VRT file exists and is readable
            if not os.path.exists(vrt_path):
                QgsMessageLog.logMessage(f"VRT file not created: {vrt_path}", 'Mineral Prospectivity', Qgis.Warning)
                return None
            
            # Create QGIS layer from permanent VRT
            layer = QgsRasterLayer(vrt_path, layer_name)
            
            if layer.isValid():
                QgsMessageLog.logMessage(
                    f"Created permanent VRT: {layer_name} with {layer.bandCount()} bands at {vrt_path}", 
                    'Mineral Prospectivity', Qgis.Info
                )
                return layer
            else:
                QgsMessageLog.logMessage(f"Invalid VRT layer created: {layer_name}", 'Mineral Prospectivity', Qgis.Warning)
                return None
                
        except Exception as e:
            QgsMessageLog.logMessage(f"Error creating permanent VRT for {layer_name}: {str(e)}", 'Mineral Prospectivity', Qgis.Warning)
            import traceback
            QgsMessageLog.logMessage(f"VRT creation traceback: {traceback.format_exc()}", 'Mineral Prospectivity', Qgis.Warning)
            return None
    
    def set_aster_band_metadata(self, vrt_dataset, sorted_datasets):
        """Set proper band names and wavelengths for ASTER data"""
        try:
            # ASTER wavelengths (in micrometers)
            aster_wavelengths = {
                1: 0.56,   # VNIR Band 1
                2: 0.66,   # VNIR Band 2  
                3: 0.82,   # VNIR Band 3N
                4: 1.65,   # SWIR Band 4
                5: 2.165,  # SWIR Band 5
                6: 2.205,  # SWIR Band 6
                7: 2.260,  # SWIR Band 7
                8: 2.330,  # SWIR Band 8
                9: 2.395   # SWIR Band 9
            }
            
            for i, (sub_name, sub_desc) in enumerate(sorted_datasets, 1):
                band = vrt_dataset.GetRasterBand(i)
                if band:
                    # Extract band number from description
                    import re
                    match = re.search(r'Band(\d+)', sub_desc)
                    if match:
                        band_num = int(match.group(1))
                        
                        # Set band name
                        band.SetDescription(f"ASTER Band {band_num}")
                        
                        # Set wavelength metadata
                        if band_num in aster_wavelengths:
                            band.SetMetadataItem('wavelength', str(aster_wavelengths[band_num]))
                            band.SetMetadataItem('wavelength_units', 'micrometers')
            
        except Exception as e:
            QgsMessageLog.logMessage(f"Warning: Could not set ASTER band metadata: {str(e)}", 'Mineral Prospectivity', Qgis.Info)
    
    def load_netcdf_raster(self, file_path, layer_name=None, variable=None):
        """Load NetCDF raster data"""
        if layer_name is None:
            layer_name = os.path.splitext(os.path.basename(file_path))[0]
        
        try:
            with nc.Dataset(file_path, 'r') as ncfile:
                variables = [var for var in ncfile.variables 
                           if len(ncfile.variables[var].dimensions) >= 2]
                
                if not variables:
                    raise Exception("No suitable variables found in NetCDF file")
                
                if variable is None:
                    variable = variables[0]  # Use first suitable variable
                
                # Create GDAL NetCDF layer reference
                dataset_path = f"NETCDF:\"{file_path}\":{variable}"
                layer = QgsRasterLayer(dataset_path, f"{layer_name}_{variable}")
        
        except Exception as e:
            raise Exception(f"Error loading NetCDF file: {str(e)}")
        
        if not layer.isValid():
            raise Exception(f"Invalid NetCDF raster layer: {file_path}")
        
        # Cache metadata and determine layer type
        self.cache_raster_metadata(layer, file_path)
        
        # Determine layer type based on filename and metadata
        layer_type = self.determine_layer_type(file_path, layer_name)
        layer.setCustomProperty('layer_type', layer_type)
        
        self.loaded_layers[layer_name] = layer
        return layer
    
    def load_jpeg2000_raster(self, file_path, layer_name=None):
        """Load JPEG 2000 raster data"""
        if layer_name is None:
            layer_name = os.path.splitext(os.path.basename(file_path))[0]
        
        layer = QgsRasterLayer(file_path, layer_name)
        
        if not layer.isValid():
            # Try with GDAL JP2 driver explicitly
            dataset_path = f"JP2:\"{file_path}\""
            layer = QgsRasterLayer(dataset_path, layer_name)
        
        if not layer.isValid():
            raise Exception(f"Invalid JPEG 2000 raster layer: {file_path}")
        
        # Cache metadata and determine layer type
        self.cache_raster_metadata(layer, file_path)
        
        # Determine layer type based on filename and metadata
        layer_type = self.determine_layer_type(file_path, layer_name)
        layer.setCustomProperty('layer_type', layer_type)
        
        self.loaded_layers[layer_name] = layer
        return layer
    
    def load_csv_as_vector(self, file_path, layer_name):
        """Load CSV file as vector layer (requires x,y coordinates)"""
        try:
            # Read CSV to determine coordinate columns
            import pandas as pd
            df = pd.read_csv(file_path, nrows=5)  # Read first 5 rows to check structure
            
            # Look for coordinate columns
            x_columns = [col for col in df.columns if any(keyword in col.lower() 
                        for keyword in ['x', 'lon', 'longitude', 'easting'])]
            y_columns = [col for col in df.columns if any(keyword in col.lower() 
                        for keyword in ['y', 'lat', 'latitude', 'northing'])]
            
            if not x_columns or not y_columns:
                raise Exception("No coordinate columns found in CSV")
            
            x_col = x_columns[0]
            y_col = y_columns[0]
            
            # Create layer URI
            uri = (f"file:///{file_path}?delimiter=,&xField={x_col}&yField={y_col}"
                   f"&crs=EPSG:4326&spatialIndex=yes&subsetIndex=no&watchFile=no")
            
            layer = QgsVectorLayer(uri, layer_name, "delimitedtext")
            
            if not layer.isValid():
                raise Exception(f"Invalid CSV vector layer: {file_path}")
            
            self.loaded_layers[layer_name] = layer
            return layer
            
        except Exception as e:
            raise Exception(f"Error loading CSV as vector: {str(e)}")
    
    def process_aster_zip(self, zip_path):
        """Process ZIP files containing ASTER VNIR/SWIR datasets with permanent extraction"""
        import shutil
        
        # Create permanent extraction directory next to the ZIP file
        project_dir = os.path.dirname(zip_path)
        zip_name = os.path.splitext(os.path.basename(zip_path))[0]
        extract_dir = os.path.join(project_dir, f'{zip_name}_extracted')
        
        # Create permanent directory if it doesn't exist
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)
            
            # Extract ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            
            QgsMessageLog.logMessage(f"Extracted ZIP to permanent directory: {extract_dir}", 'Mineral Prospectivity', Qgis.Info)
        else:
            QgsMessageLog.logMessage(f"Using existing extraction directory: {extract_dir}", 'Mineral Prospectivity', Qgis.Info)
        
        try:
            # Find all HDF files in extracted directory
            hdf_files = []
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    if file.lower().endswith(('.hdf', '.hdf5', '.h5')):
                        hdf_files.append(os.path.join(root, file))
            
            if not hdf_files:
                raise Exception("No HDF files found in ZIP archive")
            
            QgsMessageLog.logMessage(f"Found {len(hdf_files)} HDF files", 'Mineral Prospectivity', Qgis.Info)
            
            # Group and process ASTER files to create single combined dataset
            dataset_groups = self.group_aster_datasets(hdf_files)
            
            loaded_layers = []
            for group_name, files in dataset_groups.items():
                QgsMessageLog.logMessage(f"Creating combined ASTER dataset for: {group_name}", 'Mineral Prospectivity', Qgis.Info)
                
                try:
                    # Create single combined ASTER dataset from all HDF files in the group
                    combined_layer = self.create_combined_aster_dataset(files, group_name, extract_dir, zip_path)
                    
                    if combined_layer and combined_layer.isValid():
                        loaded_layers.append(combined_layer)
                        QgsMessageLog.logMessage(f"Created combined ASTER dataset: {combined_layer.name()}", 'Mineral Prospectivity', Qgis.Info)
                    else:
                        QgsMessageLog.logMessage(f"Failed to create combined ASTER dataset for {group_name}", 'Mineral Prospectivity', Qgis.Warning)
                        
                except Exception as e:
                    QgsMessageLog.logMessage(f"Error creating combined dataset for {group_name}: {str(e)}", 'Mineral Prospectivity', Qgis.Warning)
            
            QgsMessageLog.logMessage(f"Successfully loaded {len(loaded_layers)} ASTER datasets", 
                                   'Mineral Prospectivity', Qgis.Info)
            
            return loaded_layers[0] if loaded_layers else None
            
        except Exception as e:
            QgsMessageLog.logMessage(f"Error processing ASTER ZIP: {str(e)}", 
                                   'Mineral Prospectivity', Qgis.Critical)
            raise
    
    def group_aster_datasets(self, hdf_files):
        """Group ASTER VNIR and SWIR datasets by acquisition"""
        groups = {}
        
        for hdf_file in hdf_files:
            filename = os.path.basename(hdf_file)
            
            # Extract base name (remove VNIR/SWIR suffixes)
            base_name = filename
            
            # Common ASTER naming patterns
            for suffix in ['_VNIR', '_SWIR', '_TIR', '.hdf', '.h5']:
                base_name = base_name.replace(suffix, '')
            
            # Remove timestamp parts to group by location/date
            import re
            # Match pattern like AST_07XT_00312062007104513_20250818190553_18831
            match = re.match(r'(AST_[^_]+_[^_]+_[^_]+)_', base_name)
            if match:
                group_key = match.group(1)
            else:
                group_key = base_name
            
            if group_key not in groups:
                groups[group_key] = []
            
            groups[group_key].append(hdf_file)
        
        return groups
    
    def cache_raster_metadata(self, layer, file_path):
        """Cache raster metadata for later use"""
        metadata = {
            'file_path': file_path,
            'layer_name': layer.name(),
            'extent': layer.extent(),
            'width': layer.width(),
            'height': layer.height(),
            'band_count': layer.bandCount(),
            'crs': layer.crs().authid(),
            'pixel_size_x': layer.rasterUnitsPerPixelX(),
            'pixel_size_y': layer.rasterUnitsPerPixelY()
        }
        
        # Get band information
        metadata['bands'] = []
        for i in range(1, layer.bandCount() + 1):
            band_metadata = {
                'band_number': i,
                'name': layer.bandName(i),
                'data_type': layer.dataProvider().dataType(i),
                'no_data_value': layer.dataProvider().sourceNoDataValue(i)
            }
            metadata['bands'].append(band_metadata)
        
        self.metadata_cache[layer.name()] = metadata
    
    def extract_archive(self, archive_path, extract_to=None):
        """Extract archive files"""
        if extract_to is None:
            extract_to = tempfile.mkdtemp(prefix='extracted_')
        
        file_ext = os.path.splitext(archive_path)[1].lower()
        
        try:
            if file_ext == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as archive:
                    archive.extractall(extract_to)
            
            elif file_ext in ['.tar', '.gz']:
                with tarfile.open(archive_path, 'r:*') as archive:
                    archive.extractall(extract_to)
            
            else:
                raise Exception(f"Unsupported archive format: {file_ext}")
            
            return extract_to
            
        except Exception as e:
            raise Exception(f"Error extracting archive: {str(e)}")
    
    def auto_load_directory(self, directory_path):
        """Automatically load all supported files in a directory"""
        loaded_files = []
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()
                
                try:
                    if file_ext in self.supported_raster_formats:
                        layer = self.load_raster(file_path)
                        loaded_files.append(('raster', file_path, layer))
                    
                    elif file_ext in self.supported_vector_formats:
                        layer = self.load_vector(file_path)
                        loaded_files.append(('vector', file_path, layer))
                    
                except Exception as e:
                    QgsMessageLog.logMessage(
                        f"Skipped {file_path}: {str(e)}", 
                        'Mineral Prospectivity', Qgis.Warning
                    )
                    continue
        
        return loaded_files
    
    def load_from_url(self, url, layer_name=None):
        """Load data from URL (for WMS, WFS, etc.)"""
        try:
            if 'wms' in url.lower():
                return self.load_wms_layer(url, layer_name)
            elif 'wfs' in url.lower():
                return self.load_wfs_layer(url, layer_name)
            else:
                # Try to download and load as file
                return self.download_and_load(url, layer_name)
        
        except Exception as e:
            raise Exception(f"Error loading from URL: {str(e)}")
    
    def load_wms_layer(self, wms_url, layer_name):
        """Load WMS layer"""
        if layer_name is None:
            layer_name = "WMS_Layer"
        
        layer = QgsRasterLayer(wms_url, layer_name, "wms")
        
        if not layer.isValid():
            raise Exception(f"Invalid WMS layer: {wms_url}")
        
        # Cache metadata and determine layer type
        self.cache_raster_metadata(layer, file_path)
        
        # Determine layer type based on filename and metadata
        layer_type = self.determine_layer_type(file_path, layer_name)
        layer.setCustomProperty('layer_type', layer_type)
        
        self.loaded_layers[layer_name] = layer
        return layer
    
    def load_wfs_layer(self, wfs_url, layer_name):
        """Load WFS layer"""
        if layer_name is None:
            layer_name = "WFS_Layer"
        
        layer = QgsVectorLayer(wfs_url, layer_name, "WFS")
        
        if not layer.isValid():
            raise Exception(f"Invalid WFS layer: {wfs_url}")
        
        # Cache metadata and determine layer type
        self.cache_raster_metadata(layer, file_path)
        
        # Determine layer type based on filename and metadata
        layer_type = self.determine_layer_type(file_path, layer_name)
        layer.setCustomProperty('layer_type', layer_type)
        
        self.loaded_layers[layer_name] = layer
        return layer
    
    def download_and_load(self, url, layer_name):
        """Download file from URL and load"""
        import urllib.request
        
        # Create temporary file
        temp_dir = tempfile.mkdtemp(prefix='download_')
        filename = os.path.basename(url.split('?')[0])  # Remove query parameters
        if not filename:
            filename = "downloaded_file"
        
        temp_file = os.path.join(temp_dir, filename)
        
        try:
            urllib.request.urlretrieve(url, temp_file)
            
            # Determine file type and load
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext in self.supported_raster_formats:
                return self.load_raster(temp_file, layer_name)
            elif file_ext in self.supported_vector_formats:
                return self.load_vector(temp_file, layer_name)
            else:
                raise Exception(f"Unsupported file format: {file_ext}")
        
        except Exception as e:
            raise Exception(f"Error downloading and loading file: {str(e)}")
    
    def get_layer_info(self, layer_name):
        """Get information about loaded layer"""
        if layer_name not in self.loaded_layers:
            return None
        
        layer = self.loaded_layers[layer_name]
        
        if isinstance(layer, QgsRasterLayer):
            return self.get_raster_info(layer)
        else:
            return self.get_vector_info(layer)
    
    def get_raster_info(self, layer):
        """Get raster layer information"""
        return {
            'type': 'raster',
            'name': layer.name(),
            'extent': {
                'xmin': layer.extent().xMinimum(),
                'xmax': layer.extent().xMaximum(),
                'ymin': layer.extent().yMinimum(),
                'ymax': layer.extent().yMaximum()
            },
            'dimensions': {
                'width': layer.width(),
                'height': layer.height(),
                'bands': layer.bandCount()
            },
            'crs': layer.crs().authid(),
            'pixel_size': {
                'x': layer.rasterUnitsPerPixelX(),
                'y': layer.rasterUnitsPerPixelY()
            }
        }
    
    def get_vector_info(self, layer):
        """Get vector layer information"""
        return {
            'type': 'vector',
            'name': layer.name(),
            'geometry_type': QgsWkbTypes.displayString(layer.wkbType()),
            'feature_count': layer.featureCount(),
            'extent': {
                'xmin': layer.extent().xMinimum(),
                'xmax': layer.extent().xMaximum(),
                'ymin': layer.extent().yMinimum(),
                'ymax': layer.extent().yMaximum()
            },
            'crs': layer.crs().authid(),
            'fields': [field.name() for field in layer.fields()]
        }
    
    def validate_data_compatibility(self, layer1_name, layer2_name):
        """Check if two layers are spatially compatible"""
        if layer1_name not in self.loaded_layers or layer2_name not in self.loaded_layers:
            return False, "One or both layers not found"
        
        layer1 = self.loaded_layers[layer1_name]
        layer2 = self.loaded_layers[layer2_name]
        
        # Check CRS compatibility
        if layer1.crs() != layer2.crs():
            return False, f"CRS mismatch: {layer1.crs().authid()} vs {layer2.crs().authid()}"
        
        # Check extent overlap
        extent1 = layer1.extent()
        extent2 = layer2.extent()
        
        if not extent1.intersects(extent2):
            return False, "No spatial overlap between layers"
        
        return True, "Layers are compatible"
    
    def get_common_extent(self, layer_names):
        """Get common extent of multiple layers"""
        if not layer_names:
            return None
        
        extents = []
        for layer_name in layer_names:
            if layer_name in self.loaded_layers:
                extents.append(self.loaded_layers[layer_name].extent())
        
        if not extents:
            return None
        
        # Find intersection of all extents
        common_extent = extents[0]
        for extent in extents[1:]:
            common_extent = common_extent.intersect(extent)
        
        return common_extent
    
    def export_layer_list(self, output_file):
        """Export list of loaded layers to JSON"""
        layer_info = {}
        
        for name, layer in self.loaded_layers.items():
            layer_info[name] = self.get_layer_info(name)
        
        with open(output_file, 'w') as f:
            json.dump(layer_info, f, indent=2, default=str)
        
        return output_file
    
    def determine_layer_type(self, file_path, layer_name):
        """Determine the type of geological data based on filename and metadata"""
        filename = os.path.basename(file_path).lower()
        layer_name_lower = layer_name.lower() if layer_name else ''
        
        # ASTER data type detection
        if 'aster' in filename or 'ast_' in filename:
            if 'vnir' in filename or 'swir' in filename:
                return 'spectral'  # Primary spectral data
        
        # Magnetic data indicators
        if any(keyword in filename or keyword in layer_name_lower for keyword in 
               ['mag', 'magnetic', 'anomaly', 'tmi', 'rtp']):
            return 'magnetic'
        
        # Radiometric data indicators
        if any(keyword in filename or keyword in layer_name_lower for keyword in 
               ['radio', 'gamma', 'potassium', 'uranium', 'thorium', 'tc', 'k', 'u', 'th']):
            return 'radiometric'
        
        # Geological map indicators
        if any(keyword in filename or keyword in layer_name_lower for keyword in 
               ['geology', 'geological', 'lithology', 'formation', 'structure']):
            return 'geological'
        
        # Default to spectral for primary raster data
        return 'spectral'
    
    def group_aster_datasets(self, hdf_files):
        """Group ASTER VNIR and SWIR datasets by acquisition"""
        groups = {}
        
        for hdf_file in hdf_files:
            filename = os.path.basename(hdf_file)
            
            # Extract base name (remove VNIR/SWIR suffixes)
            base_name = filename
            
            # Common ASTER naming patterns
            for suffix in ['_VNIR', '_SWIR', '_TIR', '.hdf', '.h5']:
                base_name = base_name.replace(suffix, '')
            
            # Remove timestamp parts to group by location/date
            import re
            # Match pattern like AST_07XT_00312062007104513_20250818190553_18831
            match = re.match(r'(AST_[^_]+_[^_]+_[^_]+)_', base_name)
            if match:
                group_key = match.group(1)
            else:
                group_key = base_name
            
            if group_key not in groups:
                groups[group_key] = []
            
            groups[group_key].append(hdf_file)
        
        return groups
