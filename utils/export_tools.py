"""
Export tools for analysis results and reports
"""

import os
import json
import csv
import numpy as np
import pandas as pd
import rasterio
from datetime import datetime
from qgis.core import (QgsProject, QgsRasterLayer, QgsVectorLayer, QgsMessageLog, 
                       Qgis, QgsLayoutManager, QgsLayout, QgsLayoutItemMap,
                       QgsLayoutItemLabel, QgsLayoutItemLegend, QgsLayoutSize,
                       QgsUnitTypes, QgsLayoutExporter, QgsLayoutPoint)
from qgis.PyQt.QtCore import QSizeF, QPointF
from qgis.PyQt.QtGui import QFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

class ExportTools:
    """Comprehensive export tools for mineral prospectivity analysis"""
    
    def __init__(self):
        self.project = QgsProject.instance()
        self.export_formats = {
            'raster': ['.tif', '.img', '.rst'],
            'vector': ['.shp', '.gpkg', '.geojson'],
            'table': ['.csv', '.xlsx', '.json'],
            'report': ['.pdf', '.html', '.docx']
        }
        
    def export_all_results(self, output_directory):
        """Export all analysis results to specified directory"""
        os.makedirs(output_directory, exist_ok=True)
        
        export_summary = {
            'export_date': datetime.now().isoformat(),
            'output_directory': output_directory,
            'exported_items': []
        }
        
        try:
            # Export raster layers
            raster_dir = os.path.join(output_directory, 'rasters')
            os.makedirs(raster_dir, exist_ok=True)
            raster_exports = self.export_raster_layers(raster_dir)
            export_summary['exported_items'].extend(raster_exports)
            
            # Export vector layers
            vector_dir = os.path.join(output_directory, 'vectors')
            os.makedirs(vector_dir, exist_ok=True)
            vector_exports = self.export_vector_layers(vector_dir)
            export_summary['exported_items'].extend(vector_exports)
            
            # Export statistical reports
            stats_dir = os.path.join(output_directory, 'statistics')
            os.makedirs(stats_dir, exist_ok=True)
            stats_exports = self.export_statistical_reports(stats_dir)
            export_summary['exported_items'].extend(stats_exports)
            
            # Create comprehensive report
            report_path = os.path.join(output_directory, 'analysis_report.pdf')
            self.create_comprehensive_report(report_path)
            export_summary['exported_items'].append({
                'type': 'report',
                'name': 'Comprehensive Analysis Report',
                'path': report_path
            })
            
            # Export project metadata
            metadata_path = os.path.join(output_directory, 'project_metadata.json')
            self.export_project_metadata(metadata_path)
            export_summary['exported_items'].append({
                'type': 'metadata',
                'name': 'Project Metadata',
                'path': metadata_path
            })
            
            # Save export summary
            summary_path = os.path.join(output_directory, 'export_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(export_summary, f, indent=2)
            
            QgsMessageLog.logMessage(
                f"Export completed successfully to {output_directory}",
                'Mineral Prospectivity', Qgis.Info
            )
            
            return export_summary
            
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Export failed: {str(e)}",
                'Mineral Prospectivity', Qgis.Critical
            )
            raise
    
    def export_raster_layers(self, output_directory):
        """Export all raster layers to GeoTIFF format"""
        exported_items = []
        
        for layer_id, layer in self.project.mapLayers().items():
            if isinstance(layer, QgsRasterLayer) and layer.isValid():
                try:
                    output_path = os.path.join(
                        output_directory, 
                        f"{self._sanitize_filename(layer.name())}.tif"
                    )
                    
                    # Copy raster data
                    source_path = layer.source()
                    if os.path.exists(source_path):
                        with rasterio.open(source_path) as src:
                            profile = src.profile.copy()
                            profile.update({
                                'driver': 'GTiff',
                                'compress': 'lzw',
                                'tiled': True
                            })
                            
                            with rasterio.open(output_path, 'w', **profile) as dst:
                                for i in range(1, src.count + 1):
                                    dst.write(src.read(i), i)
                    
                    exported_items.append({
                        'type': 'raster',
                        'name': layer.name(),
                        'path': output_path,
                        'bands': layer.bandCount(),
                        'extent': {
                            'xmin': layer.extent().xMinimum(),
                            'xmax': layer.extent().xMaximum(),
                            'ymin': layer.extent().yMinimum(),
                            'ymax': layer.extent().yMaximum()
                        }
                    })
                    
                except Exception as e:
                    QgsMessageLog.logMessage(
                        f"Failed to export raster layer {layer.name()}: {str(e)}",
                        'Mineral Prospectivity', Qgis.Warning
                    )
        
        return exported_items
    
    def export_vector_layers(self, output_directory):
        """Export all vector layers to multiple formats"""
        exported_items = []
        
        for layer_id, layer in self.project.mapLayers().items():
            if isinstance(layer, QgsVectorLayer) and layer.isValid():
                try:
                    base_name = self._sanitize_filename(layer.name())
                    
                    # Export to Shapefile
                    shp_path = os.path.join(output_directory, f"{base_name}.shp")
                    self._export_vector_to_format(layer, shp_path, "ESRI Shapefile")
                    
                    # Export to GeoPackage
                    gpkg_path = os.path.join(output_directory, f"{base_name}.gpkg")
                    self._export_vector_to_format(layer, gpkg_path, "GPKG")
                    
                    # Export to GeoJSON
                    geojson_path = os.path.join(output_directory, f"{base_name}.geojson")
                    self._export_vector_to_format(layer, geojson_path, "GeoJSON")
                    
                    exported_items.append({
                        'type': 'vector',
                        'name': layer.name(),
                        'geometry_type': layer.geometryType().name,
                        'feature_count': layer.featureCount(),
                        'formats': {
                            'shapefile': shp_path,
                            'geopackage': gpkg_path,
                            'geojson': geojson_path
                        }
                    })
                    
                except Exception as e:
                    QgsMessageLog.logMessage(
                        f"Failed to export vector layer {layer.name()}: {str(e)}",
                        'Mineral Prospectivity', Qgis.Warning
                    )
        
        return exported_items
    
    def _export_vector_to_format(self, layer, output_path, driver_name):
        """Export vector layer to specific format"""
        from qgis.core import QgsVectorFileWriter
        
        options = QgsVectorFileWriter.SaveVectorOptions()
        options.driverName = driver_name
        options.fileEncoding = "UTF-8"
        
        result = QgsVectorFileWriter.writeAsVectorFormatV3(
            layer, output_path, layer.transformContext(), options
        )
        
        if result[0] != QgsVectorFileWriter.NoError:
            raise Exception(f"Export failed: {result[1]}")
    
    def export_statistical_reports(self, output_directory):
        """Export statistical analysis reports"""
        exported_items = []
        
        try:
            # Collect layer statistics
            layer_stats = self._collect_layer_statistics()
            
            # Export to CSV
            csv_path = os.path.join(output_directory, 'layer_statistics.csv')
            self._export_statistics_to_csv(layer_stats, csv_path)
            
            # Export to JSON
            json_path = os.path.join(output_directory, 'layer_statistics.json')
            with open(json_path, 'w') as f:
                json.dump(layer_stats, f, indent=2, default=str)
            
            # Create statistical plots
            plots_dir = os.path.join(output_directory, 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            plot_files = self._create_statistical_plots(layer_stats, plots_dir)
            
            exported_items.extend([
                {
                    'type': 'statistics',
                    'name': 'Layer Statistics CSV',
                    'path': csv_path
                },
                {
                    'type': 'statistics',
                    'name': 'Layer Statistics JSON',
                    'path': json_path
                }
            ])
            
            exported_items.extend(plot_files)
            
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Failed to export statistical reports: {str(e)}",
                'Mineral Prospectivity', Qgis.Warning
            )
        
        return exported_items
    
    def _collect_layer_statistics(self):
        """Collect statistics from all raster layers"""
        layer_stats = {}
        
        for layer_id, layer in self.project.mapLayers().items():
            if isinstance(layer, QgsRasterLayer) and layer.isValid():
                stats = {
                    'name': layer.name(),
                    'bands': layer.bandCount(),
                    'extent': {
                        'xmin': layer.extent().xMinimum(),
                        'xmax': layer.extent().xMaximum(),
                        'ymin': layer.extent().yMinimum(),
                        'ymax': layer.extent().yMaximum()
                    },
                    'band_statistics': []
                }
                
                # Get statistics for each band
                for i in range(1, layer.bandCount() + 1):
                    try:
                        provider = layer.dataProvider()
                        band_stats = provider.bandStatistics(i)
                        
                        stats['band_statistics'].append({
                            'band': i,
                            'min': float(band_stats.minimumValue),
                            'max': float(band_stats.maximumValue),
                            'mean': float(band_stats.mean),
                            'stddev': float(band_stats.stdDev)
                        })
                    except:
                        stats['band_statistics'].append({
                            'band': i,
                            'min': None,
                            'max': None,
                            'mean': None,
                            'stddev': None
                        })
                
                layer_stats[layer.name()] = stats
        
        return layer_stats
    
    def _export_statistics_to_csv(self, layer_stats, output_path):
        """Export layer statistics to CSV"""
        rows = []
        
        for layer_name, stats in layer_stats.items():
            for band_stat in stats['band_statistics']:
                rows.append({
                    'layer_name': layer_name,
                    'band': band_stat['band'],
                    'min_value': band_stat['min'],
                    'max_value': band_stat['max'],
                    'mean_value': band_stat['mean'],
                    'std_deviation': band_stat['stddev'],
                    'extent_xmin': stats['extent']['xmin'],
                    'extent_xmax': stats['extent']['xmax'],
                    'extent_ymin': stats['extent']['ymin'],
                    'extent_ymax': stats['extent']['ymax']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
    
    def _create_statistical_plots(self, layer_stats, plots_directory):
        """Create statistical visualization plots"""
        plot_files = []
        
        try:
            # Create summary statistics plot
            summary_path = os.path.join(plots_directory, 'summary_statistics.png')
            self._plot_summary_statistics(layer_stats, summary_path)
            plot_files.append({
                'type': 'plot',
                'name': 'Summary Statistics Plot',
                'path': summary_path
            })
            
            # Create individual layer histograms
            for layer_name, stats in layer_stats.items():
                if stats['band_statistics']:
                    hist_path = os.path.join(
                        plots_directory, 
                        f"{self._sanitize_filename(layer_name)}_histogram.png"
                    )
                    self._plot_layer_histogram(layer_name, stats, hist_path)
                    plot_files.append({
                        'type': 'plot',
                        'name': f'{layer_name} Histogram',
                        'path': hist_path
                    })
            
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Failed to create plots: {str(e)}",
                'Mineral Prospectivity', Qgis.Warning
            )
        
        return plot_files
    
    def _plot_summary_statistics(self, layer_stats, output_path):
        """Create summary statistics plot"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Layer Statistics Summary', fontsize=16)
        
        # Extract data for plotting
        layer_names = []
        means = []
        stds = []
        ranges = []
        
        for layer_name, stats in layer_stats.items():
            if stats['band_statistics']:
                layer_names.append(layer_name[:20])  # Truncate long names
                band_means = [bs['mean'] for bs in stats['band_statistics'] if bs['mean'] is not None]
                band_stds = [bs['stddev'] for bs in stats['band_statistics'] if bs['stddev'] is not None]
                band_ranges = [(bs['max'] - bs['min']) for bs in stats['band_statistics'] 
                              if bs['max'] is not None and bs['min'] is not None]
                
                means.append(np.mean(band_means) if band_means else 0)
                stds.append(np.mean(band_stds) if band_stds else 0)
                ranges.append(np.mean(band_ranges) if band_ranges else 0)
        
        if layer_names:
            # Mean values bar chart
            axes[0, 0].bar(range(len(layer_names)), means)
            axes[0, 0].set_title('Mean Values by Layer')
            axes[0, 0].set_xticks(range(len(layer_names)))
            axes[0, 0].set_xticklabels(layer_names, rotation=45, ha='right')
            
            # Standard deviation bar chart
            axes[0, 1].bar(range(len(layer_names)), stds)
            axes[0, 1].set_title('Standard Deviation by Layer')
            axes[0, 1].set_xticks(range(len(layer_names)))
            axes[0, 1].set_xticklabels(layer_names, rotation=45, ha='right')
            
            # Value ranges bar chart
            axes[1, 0].bar(range(len(layer_names)), ranges)
            axes[1, 0].set_title('Value Ranges by Layer')
            axes[1, 0].set_xticks(range(len(layer_names)))
            axes[1, 0].set_xticklabels(layer_names, rotation=45, ha='right')
            
            # Band count pie chart
            band_counts = [len(stats['band_statistics']) for stats in layer_stats.values()]
            if band_counts:
                axes[1, 1].pie(band_counts, labels=layer_names, autopct='%1.0f%%')
                axes[1, 1].set_title('Band Distribution')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_layer_histogram(self, layer_name, stats, output_path):
        """Create histogram for individual layer"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create histogram data
        x_labels = [f"Band {bs['band']}" for bs in stats['band_statistics']]
        means = [bs['mean'] for bs in stats['band_statistics'] if bs['mean'] is not None]
        
        if means:
            ax.bar(range(len(means)), means)
            ax.set_title(f'Band Statistics - {layer_name}')
            ax.set_xlabel('Bands')
            ax.set_ylabel('Mean Value')
            ax.set_xticks(range(len(means)))
            ax.set_xticklabels(x_labels[:len(means)])
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comprehensive_report(self, output_path):
        """Create comprehensive PDF report"""
        try:
            with PdfPages(output_path) as pdf:
                # Title page
                fig = plt.figure(figsize=(8.5, 11))
                fig.text(0.5, 0.7, 'Mineral Prospectivity Analysis Report', 
                        ha='center', va='center', fontsize=24, weight='bold')
                fig.text(0.5, 0.6, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                        ha='center', va='center', fontsize=12)
                fig.text(0.5, 0.5, f'Project: {self.project.baseName()}', 
                        ha='center', va='center', fontsize=14)
                
                # Add project summary
                self._add_project_summary_to_figure(fig)
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
                # Layer summary page
                self._create_layer_summary_page(pdf)
                
                # Statistical analysis pages
                self._create_statistical_analysis_pages(pdf)
                
                # Processing methodology page
                self._create_methodology_page(pdf)
            
            QgsMessageLog.logMessage(
                f"Comprehensive report created: {output_path}",
                'Mineral Prospectivity', Qgis.Info
            )
            
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Failed to create comprehensive report: {str(e)}",
                'Mineral Prospectivity', Qgis.Warning
            )
    
    def _add_project_summary_to_figure(self, fig):
        """Add project summary information to figure"""
        layer_count = len(self.project.mapLayers())
        raster_count = sum(1 for layer in self.project.mapLayers().values() 
                          if isinstance(layer, QgsRasterLayer))
        vector_count = sum(1 for layer in self.project.mapLayers().values() 
                          if isinstance(layer, QgsVectorLayer))
        
        summary_text = f"""
        Project Summary:
        • Total Layers: {layer_count}
        • Raster Layers: {raster_count}
        • Vector Layers: {vector_count}
        • CRS: {self.project.crs().authid()}
        """
        
        fig.text(0.5, 0.3, summary_text, ha='center', va='center', fontsize=12)
    
    def _create_layer_summary_page(self, pdf):
        """Create layer summary page"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        # Create layer summary table
        layer_data = []
        for layer in self.project.mapLayers().values():
            layer_data.append([
                layer.name(),
                'Raster' if isinstance(layer, QgsRasterLayer) else 'Vector',
                layer.crs().authid(),
                'Valid' if layer.isValid() else 'Invalid'
            ])
        
        if layer_data:
            table = ax.table(cellText=layer_data,
                           colLabels=['Layer Name', 'Type', 'CRS', 'Status'],
                           cellLoc='center',
                           loc='center',
                           colWidths=[0.4, 0.15, 0.25, 0.2])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
        
        ax.set_title('Layer Summary', fontsize=16, weight='bold', pad=20)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def _create_statistical_analysis_pages(self, pdf):
        """Create statistical analysis pages"""
        layer_stats = self._collect_layer_statistics()
        
        for layer_name, stats in layer_stats.items():
            if stats['band_statistics']:
                fig, axes = plt.subplots(2, 1, figsize=(8.5, 11))
                
                # Band statistics table
                band_data = []
                for bs in stats['band_statistics']:
                    band_data.append([
                        bs['band'],
                        f"{bs['min']:.3f}" if bs['min'] is not None else 'N/A',
                        f"{bs['max']:.3f}" if bs['max'] is not None else 'N/A',
                        f"{bs['mean']:.3f}" if bs['mean'] is not None else 'N/A',
                        f"{bs['stddev']:.3f}" if bs['stddev'] is not None else 'N/A'
                    ])
                
                axes[0].axis('off')
                table = axes[0].table(cellText=band_data,
                                    colLabels=['Band', 'Min', 'Max', 'Mean', 'Std Dev'],
                                    cellLoc='center',
                                    loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                table.scale(1, 1.5)
                axes[0].set_title(f'{layer_name} - Band Statistics', fontsize=14, weight='bold')
                
                # Statistical plot
                means = [bs['mean'] for bs in stats['band_statistics'] if bs['mean'] is not None]
                bands = [f"Band {bs['band']}" for bs in stats['band_statistics'] if bs['mean'] is not None]
                
                if means:
                    axes[1].bar(range(len(means)), means)
                    axes[1].set_title('Mean Values by Band')
                    axes[1].set_xlabel('Bands')
                    axes[1].set_ylabel('Mean Value')
                    axes[1].set_xticks(range(len(means)))
                    axes[1].set_xticklabels(bands, rotation=45)
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
    
    def _create_methodology_page(self, pdf):
        """Create methodology documentation page"""
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis('off')
        
        methodology_text = """
        MINERAL PROSPECTIVITY MAPPING METHODOLOGY
        
        1. DATA SOURCES
        • ASTER L2 VNIR SWIR surface reflectance data
        • Sentinel-2 MSI multispectral imagery
        • Geological maps and structural data
        • Magnetic anomaly maps
        • Radiometric data (gamma-ray spectrometry)
        • Topographic and morphological data
        
        2. PROCESSING ALGORITHMS
        • Spectral Angle Mapper (SAM)
        • Linear Spectral Unmixing
        • Minimum Noise Fraction (MNF)
        • Principal Component Analysis (PCA)
        • Matched Filtering
        
        3. MINERAL MAPPING
        • Multi-criteria decision analysis (MCDA)
        • Fuzzy logic integration
        • Weighted overlay analysis
        • Statistical correlation assessment
        
        4. PROSPECTIVITY MAPPING
        • Evidence-based modeling
        • Neural network analysis
        • Analytic Hierarchy Process (AHP)
        • Uncertainty quantification
        
        5. VALIDATION
        • Cross-validation techniques
        • Statistical significance testing
        • Quality assessment metrics
        • Confidence mapping
        """
        
        ax.text(0.05, 0.95, methodology_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    def export_project_metadata(self, output_path):
        """Export comprehensive project metadata"""
        metadata = {
            'project_info': {
                'name': self.project.baseName(),
                'title': self.project.title(),
                'crs': self.project.crs().authid(),
                'creation_date': datetime.now().isoformat(),
                'qgis_version': '3.22+'  # Assuming minimum version
            },
            'layers': {},
            'processing_history': {
                'spectral_analysis': 'Applied to ASTER and Sentinel-2 data',
                'mineral_mapping': 'Created using multiple algorithms',
                'data_fusion': 'Multi-source integration performed',
                'prospectivity_mapping': 'Generated using weighted overlay analysis'
            },
            'analysis_parameters': {
                'target_minerals': [
                    'Gold', 'Iron Oxide', 'Iron Hydroxide', 'Clay Minerals',
                    'Carbonate', 'Silica', 'Lithium', 'Diamond Indicator Minerals',
                    'Alteration Minerals', 'Gossans'
                ],
                'spectral_methods': [
                    'Spectral Angle Mapper', 'Spectral Unmixing',
                    'Minimum Noise Fraction', 'Principal Component Analysis',
                    'Matched Filtering'
                ]
            }
        }
        
        # Add layer information
        for layer_id, layer in self.project.mapLayers().items():
            metadata['layers'][layer.name()] = {
                'id': layer_id,
                'type': 'raster' if isinstance(layer, QgsRasterLayer) else 'vector',
                'source': layer.source(),
                'crs': layer.crs().authid(),
                'extent': [
                    layer.extent().xMinimum(),
                    layer.extent().yMinimum(),
                    layer.extent().xMaximum(),
                    layer.extent().yMaximum()
                ],
                'valid': layer.isValid()
            }
            
            if isinstance(layer, QgsRasterLayer):
                metadata['layers'][layer.name()].update({
                    'bands': layer.bandCount(),
                    'width': layer.width(),
                    'height': layer.height(),
                    'pixel_size_x': layer.rasterUnitsPerPixelX(),
                    'pixel_size_y': layer.rasterUnitsPerPixelY()
                })
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _sanitize_filename(self, filename):
        """Sanitize filename for safe file system use"""
        import re
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        filename = re.sub(r'\s+', '_', filename)
        return filename[:50]  # Limit length
    
    def export_layer_to_format(self, layer_name, output_path, format_type):
        """Export specific layer to specified format"""
        layer = None
        for l in self.project.mapLayers().values():
            if l.name() == layer_name:
                layer = l
                break
        
        if not layer:
            raise ValueError(f"Layer '{layer_name}' not found")
        
        if isinstance(layer, QgsRasterLayer):
            return self._export_raster_to_format(layer, output_path, format_type)
        else:
            return self._export_vector_to_format(layer, output_path, format_type)
    
    def _export_raster_to_format(self, layer, output_path, format_type):
        """Export raster layer to specific format"""
        source_path = layer.source()
        
        with rasterio.open(source_path) as src:
            profile = src.profile.copy()
            
            if format_type.upper() == 'GEOTIFF':
                profile['driver'] = 'GTiff'
            elif format_type.upper() == 'IMAGINE':
                profile['driver'] = 'HFA'
            elif format_type.upper() == 'ENVI':
                profile['driver'] = 'ENVI'
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                for i in range(1, src.count + 1):
                    dst.write(src.read(i), i)
        
        return output_path
    
    def create_layout_export(self, output_path, layout_name="Analysis Layout"):
        """Create and export a map layout"""
        try:
            manager = self.project.layoutManager()
            
            # Create new layout
            layout = QgsLayout(self.project)
            layout.initializeDefaults()
            layout.setName(layout_name)
            
            # Set page size
            page = layout.pageCollection().page(0)
            page.setPageSize(QgsLayoutSize(297, 210, QgsUnitTypes.LayoutMillimeters))  # A4 landscape
            
            # Add map item
            map_item = QgsLayoutItemMap(layout)
            map_item.attemptSetSceneRect(QgsLayoutPoint(10, 10), QgsLayoutSize(200, 150))
            map_item.setExtent(self.project.mapCanvas().extent())
            layout.addLayoutItem(map_item)
            
            # Add title
            title = QgsLayoutItemLabel(layout)
            title.setText("Mineral Prospectivity Analysis")
            title.setFont(QFont("Arial", 16, QFont.Bold))
            title.attemptSetSceneRect(QgsLayoutPoint(10, 170), QgsLayoutSize(200, 20))
            layout.addLayoutItem(title)
            
            # Add legend
            legend = QgsLayoutItemLegend(layout)
            legend.setLinkedMap(map_item)
            legend.attemptSetSceneRect(QgsLayoutPoint(220, 10), QgsLayoutSize(60, 150))
            layout.addLayoutItem(legend)
            
            # Export layout
            exporter = QgsLayoutExporter(layout)
            
            if output_path.lower().endswith('.pdf'):
                result = exporter.exportToPdf(output_path, QgsLayoutExporter.PdfExportSettings())
            elif output_path.lower().endswith('.png'):
                result = exporter.exportToImage(output_path, QgsLayoutExporter.ImageExportSettings())
            else:
                raise ValueError("Unsupported export format")
            
            if result == QgsLayoutExporter.Success:
                QgsMessageLog.logMessage(
                    f"Layout exported successfully: {output_path}",
                    'Mineral Prospectivity', Qgis.Info
                )
            else:
                raise Exception(f"Layout export failed with code: {result}")
            
            return output_path
            
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Failed to create layout export: {str(e)}",
                'Mineral Prospectivity', Qgis.Critical
            )
            raise
