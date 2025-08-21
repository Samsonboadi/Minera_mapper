"""
Sentinel-2 MSI data processor for mineral exploration
"""

import os
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import reproject, Resampling
import xml.etree.ElementTree as ET
from qgis.PyQt.QtWidgets import QFileDialog, QMessageBox, QProgressDialog
from qgis.PyQt.QtCore import Qt
from qgis.core import QgsRasterLayer, QgsProject, QgsMessageLog, Qgis
import zipfile
import tempfile
import json
import glob

class Sentinel2Processor:
    """Process Sentinel-2 MSI data for mineral exploration"""
    
    def __init__(self, iface):
        self.iface = iface
        
        # Sentinel-2 band specifications
        self.s2_bands = {
            'B01': {'name': 'Coastal aerosol', 'wavelength': 443, 'resolution': 60, 'width': 21},
            'B02': {'name': 'Blue', 'wavelength': 490, 'resolution': 10, 'width': 66},
            'B03': {'name': 'Green', 'wavelength': 560, 'resolution': 10, 'width': 36},
            'B04': {'name': 'Red', 'wavelength': 665, 'resolution': 10, 'width': 31},
            'B05': {'name': 'Vegetation Red Edge', 'wavelength': 705, 'resolution': 20, 'width': 15},
            'B06': {'name': 'Vegetation Red Edge', 'wavelength': 740, 'resolution': 20, 'width': 15},
            'B07': {'name': 'Vegetation Red Edge', 'wavelength': 783, 'resolution': 20, 'width': 20},
            'B08': {'name': 'NIR', 'wavelength': 842, 'resolution': 10, 'width': 106},
            'B8A': {'name': 'Vegetation Red Edge', 'wavelength': 865, 'resolution': 20, 'width': 21},
            'B09': {'name': 'Water vapour', 'wavelength': 945, 'resolution': 60, 'width': 20},
            'B10': {'name': 'SWIR – Cirrus', 'wavelength': 1375, 'resolution': 60, 'width': 31},
            'B11': {'name': 'SWIR', 'wavelength': 1610, 'resolution': 20, 'width': 91},
            'B12': {'name': 'SWIR', 'wavelength': 2190, 'resolution': 20, 'width': 175}
        }
        
        # Mineral exploration indices for Sentinel-2
        self.mineral_indices = {
            'NDVI': {
                'formula': '(B08 - B04) / (B08 + B04)',
                'bands': ['B08', 'B04'],
                'description': 'Normalized Difference Vegetation Index'
            },
            'Clay_Index': {
                'formula': 'B11 / B12',
                'bands': ['B11', 'B12'],
                'description': 'Clay Mineral Index'
            },
            'Iron_Oxide': {
                'formula': 'B04 / B02',
                'bands': ['B04', 'B02'],
                'description': 'Iron Oxide Index'
            },
            'Ferrous_Iron': {
                'formula': 'B08 / B04',
                'bands': ['B08', 'B04'],
                'description': 'Ferrous Iron Index'
            },
            'Alteration_Index': {
                'formula': '(B06 + B07) / B08',
                'bands': ['B06', 'B07', 'B08'],
                'description': 'Alteration Zone Index'
            },
            'Carbonate_Index': {
                'formula': 'B12 / B11',
                'bands': ['B12', 'B11'],
                'description': 'Carbonate Index'
            },
            'Silica_Index': {
                'formula': 'B11 / (B11 + B12)',
                'bands': ['B11', 'B12'],
                'description': 'Silica Index'
            }
        }
        
        self.current_product = None
        self.extracted_bands = {}
        
    def process_data(self):
        """Main processing function for Sentinel-2 data"""
        # File/folder selection
        input_path = QFileDialog.getExistingDirectory(
            self.iface.mainWindow(),
            "Select Sentinel-2 Product Directory (or select .zip file)",
            ""
        )
        
        if not input_path:
            # Try zip file selection
            zip_path, _ = QFileDialog.getOpenFileName(
                self.iface.mainWindow(),
                "Select Sentinel-2 ZIP file",
                "",
                "ZIP files (*.zip);;All files (*)"
            )
            if not zip_path:
                return
            input_path = zip_path
        
        try:
            # Progress dialog
            progress = QProgressDialog("Processing Sentinel-2 data...", "Cancel", 0, 100)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            
            progress.setValue(5)
            progress.setLabelText("Extracting Sentinel-2 product...")
            
            # Extract or locate product
            product_path = self.extract_or_locate_product(input_path)
            self.current_product = product_path
            
            progress.setValue(15)
            progress.setLabelText("Reading metadata...")
            
            # Read metadata
            metadata = self.read_sentinel2_metadata(product_path)
            
            progress.setValue(25)
            progress.setLabelText("Locating and resampling bands...")
            
            # Process bands (resample to common resolution)
            processed_bands = self.process_bands(product_path, target_resolution=10, progress=progress)
            
            progress.setValue(50)
            progress.setLabelText("Calculating mineral indices...")
            
            # Calculate mineral exploration indices
            mineral_maps = self.calculate_mineral_indices(processed_bands)
            
            progress.setValue(70)
            progress.setLabelText("Creating composite images...")
            
            # Create composites
            composites = self.create_composites(processed_bands)
            
            progress.setValue(85)
            progress.setLabelText("Adding layers to QGIS...")
            
            # Add to QGIS
            self.add_layers_to_qgis(processed_bands, mineral_maps, composites, metadata)
            
            progress.setValue(100)
            progress.close()
            
            QMessageBox.information(
                self.iface.mainWindow(),
                "Success",
                f"Sentinel-2 data processed successfully!\n"
                f"Product: {metadata.get('PRODUCT_ID', 'Unknown')}\n"
                f"Added {len(processed_bands)} bands, {len(mineral_maps)} mineral indices, "
                f"and {len(composites)} composite images."
            )
            
        except Exception as e:
            if 'progress' in locals():
                progress.close()
            QMessageBox.critical(
                self.iface.mainWindow(),
                "Error",
                f"Failed to process Sentinel-2 data: {str(e)}"
            )
            QgsMessageLog.logMessage(f"Sentinel-2 processing error: {str(e)}", 'Mineral Prospectivity', Qgis.Critical)
    
    def extract_or_locate_product(self, input_path):
        """Extract ZIP or locate SAFE directory"""
        if input_path.endswith('.zip'):
            # Extract ZIP file
            temp_dir = tempfile.mkdtemp(prefix='sentinel2_')
            with zipfile.ZipFile(input_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find SAFE directory
            safe_dirs = [d for d in os.listdir(temp_dir) if d.endswith('.SAFE')]
            if not safe_dirs:
                raise Exception("No SAFE directory found in ZIP file")
            
            return os.path.join(temp_dir, safe_dirs[0])
        
        elif input_path.endswith('.SAFE') or os.path.isdir(input_path):
            return input_path
        
        else:
            raise Exception("Invalid Sentinel-2 product format")
    
    def read_sentinel2_metadata(self, product_path):
        """Read Sentinel-2 metadata from MTD_MSIL*xml file"""
        metadata = {}
        
        # Find metadata file
        metadata_files = glob.glob(os.path.join(product_path, 'MTD_MSIL*.xml'))
        if not metadata_files:
            QgsMessageLog.logMessage("No metadata file found, using defaults", 'Mineral Prospectivity', Qgis.Warning)
            return {'PRODUCT_ID': os.path.basename(product_path)}
        
        metadata_file = metadata_files[0]
        
        try:
            tree = ET.parse(metadata_file)
            root = tree.getroot()
            
            # Extract key metadata
            general_info = root.find('.//General_Info')
            if general_info is not None:
                product_info = general_info.find('Product_Info')
                if product_info is not None:
                    metadata['PRODUCT_ID'] = self._get_text(product_info, 'PRODUCT_URI', 'Unknown')
                    metadata['PROCESSING_LEVEL'] = self._get_text(product_info, 'PROCESSING_LEVEL', 'Unknown')
                    metadata['PRODUCT_TYPE'] = self._get_text(product_info, 'PRODUCT_TYPE', 'Unknown')
            
            # Get acquisition info
            product_info = root.find('.//Product_Info')
            if product_info is not None:
                datatake_info = product_info.find('Datatake')
                if datatake_info is not None:
                    metadata['DATATAKE_ID'] = self._get_text(datatake_info, 'DATATAKE_IDENTIFIER', 'Unknown')
                    metadata['SENSING_TIME'] = self._get_text(datatake_info, 'DATATAKE_SENSING_START', 'Unknown')
            
            # Get geometric info
            geometric_info = root.find('.//Geometric_Info')
            if geometric_info is not None:
                product_footprint = geometric_info.find('.//Product_Footprint')
                if product_footprint is not None:
                    global_footprint = product_footprint.find('Global_Footprint')
                    if global_footprint is not None:
                        metadata['FOOTPRINT'] = global_footprint.text
                        
        except Exception as e:
            QgsMessageLog.logMessage(f"Error reading metadata: {str(e)}", 'Mineral Prospectivity', Qgis.Warning)
            metadata['PRODUCT_ID'] = os.path.basename(product_path)
        
        return metadata
    
    def _get_text(self, parent, tag, default=''):
        """Safely get text from XML element"""
        element = parent.find(tag)
        return element.text if element is not None else default
    
    def process_bands(self, product_path, target_resolution=10, progress=None):
        """Process and resample Sentinel-2 bands to common resolution"""
        processed_bands = {}
        temp_dir = tempfile.mkdtemp(prefix='s2_bands_')
        
        # Find GRANULE directory
        granule_dir = os.path.join(product_path, 'GRANULE')
        if not os.path.exists(granule_dir):
            raise Exception("GRANULE directory not found")
        
        # Get first (usually only) granule
        granules = [d for d in os.listdir(granule_dir) if os.path.isdir(os.path.join(granule_dir, d))]
        if not granules:
            raise Exception("No granule directories found")
        
        granule_path = os.path.join(granule_dir, granules[0])
        img_data_path = os.path.join(granule_path, 'IMG_DATA')
        
        # For L2A products, look in subdirectories
        if not glob.glob(os.path.join(img_data_path, '*B*.jp2')):
            # Check for resolution subdirectories (L2A format)
            for res_dir in ['R10m', 'R20m', 'R60m']:
                res_path = os.path.join(img_data_path, res_dir)
                if os.path.exists(res_path):
                    img_data_path = res_path
                    break
        
        # Find band files
        band_files = {}
        for band_id in self.s2_bands.keys():
            # Look for band files
            pattern = os.path.join(img_data_path, f'*{band_id}*.jp2')
            if not pattern:
                pattern = os.path.join(img_data_path, '..', 'R*m', f'*{band_id}*.jp2')
            
            files = glob.glob(pattern)
            if not files:
                # Try different resolution directories
                for res_dir in ['R10m', 'R20m', 'R60m']:
                    res_pattern = os.path.join(img_data_path, '..', res_dir, f'*{band_id}*.jp2')
                    files = glob.glob(res_pattern)
                    if files:
                        break
            
            if files:
                band_files[band_id] = files[0]
        
        if not band_files:
            raise Exception("No band files found")
        
        # Get reference for resampling (use 10m band as reference)
        reference_band = None
        for band_id in ['B02', 'B03', 'B04', 'B08']:  # 10m bands
            if band_id in band_files:
                reference_band = band_files[band_id]
                break
        
        if not reference_band:
            raise Exception("No reference band found for resampling")
        
        # Get reference profile
        with rasterio.open(reference_band) as ref_src:
            ref_profile = ref_src.profile
            ref_transform = ref_src.transform
            ref_crs = ref_src.crs
            ref_width = ref_src.width
            ref_height = ref_src.height
        
        # Process each band
        total_bands = len(band_files)
        for i, (band_id, band_path) in enumerate(band_files.items()):
            try:
                if progress:
                    progress.setValue(25 + int(25 * i / total_bands))
                    progress.setLabelText(f"Processing band {band_id}...")
                
                output_path = os.path.join(temp_dir, f"S2_{band_id}_resampled.tif")
                
                with rasterio.open(band_path) as src:
                    band_data = src.read(1)
                    
                    # Check if resampling is needed
                    if (src.width != ref_width or src.height != ref_height or 
                        src.transform != ref_transform):
                        
                        # Resample to reference grid
                        resampled_data = np.empty((ref_height, ref_width), dtype=src.dtypes[0])
                        
                        reproject(
                            source=band_data,
                            destination=resampled_data,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=ref_transform,
                            dst_crs=ref_crs,
                            resampling=Resampling.bilinear
                        )
                        
                        band_data = resampled_data
                
                # Save resampled band
                output_profile = ref_profile.copy()
                output_profile.update({
                    'dtype': 'uint16',
                    'compress': 'lzw',
                    'nodata': 0
                })
                
                with rasterio.open(output_path, 'w', **output_profile) as dst:
                    dst.write(band_data.astype(np.uint16), 1)
                    dst.set_band_description(1, f"Sentinel-2 {band_id} - {self.s2_bands[band_id]['name']}")
                
                processed_bands[band_id] = output_path
                
            except Exception as e:
                QgsMessageLog.logMessage(
                    f"Error processing band {band_id}: {str(e)}", 
                    'Mineral Prospectivity', Qgis.Warning
                )
                continue
        
        return processed_bands
    
    def calculate_mineral_indices(self, processed_bands):
        """Calculate mineral exploration indices"""
        mineral_maps = {}
        temp_dir = tempfile.mkdtemp(prefix='s2_indices_')
        
        for index_name, index_info in self.mineral_indices.items():
            try:
                # Check if required bands are available
                required_bands = index_info['bands']
                available_bands = {band: path for band, path in processed_bands.items() 
                                 if band in required_bands}
                
                if len(available_bands) != len(required_bands):
                    QgsMessageLog.logMessage(
                        f"Skipping {index_name}: missing required bands", 
                        'Mineral Prospectivity', Qgis.Warning
                    )
                    continue
                
                # Load band data
                band_arrays = {}
                reference_profile = None
                
                for band_id, band_path in available_bands.items():
                    with rasterio.open(band_path) as src:
                        band_data = src.read(1).astype(np.float32)
                        # Convert DN to reflectance (assuming L2A data is already in reflectance * 10000)
                        band_data = band_data / 10000.0
                        band_arrays[band_id] = band_data
                        
                        if reference_profile is None:
                            reference_profile = src.profile
                
                # Calculate index
                index_result = self._calculate_index_formula(
                    index_info['formula'], band_arrays
                )
                
                # Save index as GeoTIFF
                output_path = os.path.join(temp_dir, f"S2_{index_name}.tif")
                
                output_profile = reference_profile.copy()
                output_profile.update({
                    'dtype': 'float32',
                    'compress': 'lzw',
                    'nodata': -9999
                })
                
                with rasterio.open(output_path, 'w', **output_profile) as dst:
                    dst.write(index_result, 1)
                    dst.set_band_description(1, f"{index_name} - {index_info['description']}")
                
                mineral_maps[index_name] = output_path
                
            except Exception as e:
                QgsMessageLog.logMessage(
                    f"Error calculating {index_name}: {str(e)}", 
                    'Mineral Prospectivity', Qgis.Warning
                )
                continue
        
        return mineral_maps
    
    def _calculate_index_formula(self, formula, band_arrays):
        """Calculate spectral index formula"""
        expression = formula
        
        # Handle division by zero and create safe arrays
        safe_arrays = {}
        for band_id, array in band_arrays.items():
            # Set very small values to avoid division issues
            safe_array = np.where(np.abs(array) < 1e-10, 1e-10, array)
            safe_arrays[band_id] = safe_array
            
            # Replace band ID in formula
            expression = expression.replace(band_id, f"safe_arrays['{band_id}']")
        
        try:
            # Evaluate the expression
            result = eval(expression, {"__builtins__": {}}, 
                         {"safe_arrays": safe_arrays, "np": np})
            
            # Handle invalid results
            result = np.where(np.isfinite(result), result, -9999)
            
            return result.astype(np.float32)
            
        except Exception as e:
            raise Exception(f"Error evaluating formula '{formula}': {str(e)}")
    
    def create_composites(self, processed_bands):
        """Create false color composite images"""
        composites = {}
        temp_dir = tempfile.mkdtemp(prefix='s2_composites_')
        
        # Define composite combinations
        composite_definitions = {
            'True_Color': ['B04', 'B03', 'B02'],  # Red, Green, Blue
            'False_Color_IR': ['B08', 'B04', 'B03'],  # NIR, Red, Green
            'Agriculture': ['B11', 'B08', 'B02'],  # SWIR, NIR, Blue
            'Geology': ['B12', 'B11', 'B02'],  # SWIR, SWIR, Blue
            'Vegetation': ['B08', 'B11', 'B04'],  # NIR, SWIR, Red
            'Urban': ['B12', 'B11', 'B04']  # SWIR, SWIR, Red
        }
        
        for composite_name, band_combo in composite_definitions.items():
            try:
                # Check band availability
                available_bands = [band for band in band_combo if band in processed_bands]
                if len(available_bands) < 3:
                    continue
                
                # Load the three bands
                composite_arrays = []
                reference_profile = None
                
                for band_id in available_bands[:3]:
                    with rasterio.open(processed_bands[band_id]) as src:
                        band_data = src.read(1).astype(np.float32)
                        
                        # Normalize band data (stretch to 0-255)
                        valid_data = band_data[band_data > 0]
                        if len(valid_data) > 0:
                            p2, p98 = np.percentile(valid_data, [2, 98])
                            band_data = np.clip((band_data - p2) / (p98 - p2) * 255, 0, 255)
                        
                        composite_arrays.append(band_data)
                        
                        if reference_profile is None:
                            reference_profile = src.profile
                
                if len(composite_arrays) == 3:
                    # Stack bands as RGB composite
                    rgb_composite = np.stack(composite_arrays, axis=0)
                    
                    # Save composite
                    output_path = os.path.join(temp_dir, f"S2_{composite_name}.tif")
                    
                    output_profile = reference_profile.copy()
                    output_profile.update({
                        'count': 3,
                        'dtype': 'uint8',
                        'compress': 'lzw',
                        'photometric': 'rgb'
                    })
                    
                    with rasterio.open(output_path, 'w', **output_profile) as dst:
                        dst.write(rgb_composite.astype(np.uint8))
                        dst.colorinterp = [rasterio.enums.ColorInterp.red,
                                         rasterio.enums.ColorInterp.green,
                                         rasterio.enums.ColorInterp.blue]
                    
                    composites[composite_name] = output_path
                
            except Exception as e:
                QgsMessageLog.logMessage(
                    f"Error creating {composite_name} composite: {str(e)}", 
                    'Mineral Prospectivity', Qgis.Warning
                )
                continue
        
        return composites
    
    def add_layers_to_qgis(self, processed_bands, mineral_maps, composites, metadata):
        """Add processed layers to QGIS"""
        project = QgsProject.instance()
        
        # Create main group for this product
        product_id = metadata.get('PRODUCT_ID', 'Sentinel2_Product')
        main_group = project.layerTreeRoot().addGroup(f"Sentinel-2: {product_id}")
        
        # Add individual bands
        band_group = main_group.addGroup("Bands")
        for band_id, band_path in processed_bands.items():
            band_info = self.s2_bands.get(band_id, {})
            layer_name = f"S2 {band_id} - {band_info.get('name', 'Unknown')}"
            layer = QgsRasterLayer(band_path, layer_name)
            if layer.isValid():
                project.addMapLayer(layer, False)
                band_group.addLayer(layer)
        
        # Add mineral indices
        if mineral_maps:
            index_group = main_group.addGroup("Mineral Indices")
            for index_name, index_path in mineral_maps.items():
                layer = QgsRasterLayer(index_path, f"S2 {index_name}")
                if layer.isValid():
                    project.addMapLayer(layer, False)
                    index_group.addLayer(layer)
        
        # Add composites
        if composites:
            composite_group = main_group.addGroup("Composites")
            for composite_name, composite_path in composites.items():
                layer = QgsRasterLayer(composite_path, f"S2 {composite_name}")
                if layer.isValid():
                    project.addMapLayer(layer, False)
                    composite_group.addLayer(layer)
        
        # Refresh canvas
        self.iface.mapCanvas().refresh()
    
    def atmospheric_correction_sen2cor(self, product_path):
        """Apply Sen2Cor atmospheric correction (if available)"""
        # This would require Sen2Cor to be installed
        # For now, just log that this feature is available
        QgsMessageLog.logMessage(
            "For atmospheric correction, use Sen2Cor processor on L1C products", 
            'Mineral Prospectivity', Qgis.Info
        )
        return False
    
    def cloud_masking(self, product_path, processed_bands):
        """Apply cloud masking using SCL (Scene Classification Layer)"""
        # Look for SCL band
        granule_dir = os.path.join(product_path, 'GRANULE')
        granules = [d for d in os.listdir(granule_dir) if os.path.isdir(os.path.join(granule_dir, d))]
        
        if not granules:
            return None
        
        granule_path = os.path.join(granule_dir, granules[0])
        img_data_path = os.path.join(granule_path, 'IMG_DATA')
        
        # Look for SCL file
        scl_files = glob.glob(os.path.join(img_data_path, '**', '*SCL*.jp2'), recursive=True)
        
        if scl_files:
            scl_path = scl_files[0]
            
            # Create cloud mask
            temp_dir = tempfile.mkdtemp(prefix='s2_mask_')
            mask_path = os.path.join(temp_dir, 'cloud_mask.tif')
            
            try:
                with rasterio.open(scl_path) as src:
                    scl_data = src.read(1)
                    
                    # Create cloud mask (SCL values 8, 9, 10 are clouds/cirrus)
                    cloud_mask = np.isin(scl_data, [8, 9, 10])
                    
                    # Save mask
                    mask_profile = src.profile.copy()
                    mask_profile.update({'dtype': 'uint8', 'nodata': 255})
                    
                    with rasterio.open(mask_path, 'w', **mask_profile) as dst:
                        dst.write(cloud_mask.astype(np.uint8), 1)
                
                return mask_path
                
            except Exception as e:
                QgsMessageLog.logMessage(
                    f"Error creating cloud mask: {str(e)}", 
                    'Mineral Prospectivity', Qgis.Warning
                )
        
        return None
    
    def save_processing_metadata(self, output_dir, metadata):
        """Save processing metadata"""
        processing_metadata = {
            'source_product': self.current_product,
            'product_metadata': metadata,
            'processing_date': str(np.datetime64('now')),
            'band_information': self.s2_bands,
            'mineral_indices': self.mineral_indices,
            'processing_steps': [
                'Band resampling to 10m',
                'Reflectance conversion',
                'Mineral index calculation',
                'Composite generation'
            ]
        }
        
        metadata_path = os.path.join(output_dir, 'sentinel2_processing_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(processing_metadata, f, indent=2, default=str)
        
        return metadata_path
