"""
Modern data fusion dialog for multi-source geological data integration
"""

import os
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
                                QLabel, QPushButton, QProgressBar, QTextEdit,
                                QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox,
                                QCheckBox, QListWidget, QListWidgetItem, QSlider,
                                QFrame, QMessageBox, QFileDialog)
from qgis.PyQt.QtGui import QFont, QPixmap, QIcon
from qgis.core import QgsProject, QgsRasterLayer, QgsMessageLog, Qgis


class DataFusionDialog(QDialog):
    """Modern dialog for multi-source data fusion"""
    
    def __init__(self, iface):
        super().__init__()
        self.iface = iface
        self.setWindowTitle("Multi-Source Data Fusion")
        self.setMinimumSize(800, 600)
        
        self.selected_layers = {}
        self.layer_weights = {}
        
        self.setup_ui()
        self.apply_modern_style()
        self.refresh_layers()
    
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        self.create_header(layout)
        
        # Main content
        self.create_main_content(layout)
        
        # Footer
        self.create_footer(layout)
    
    def create_header(self, layout):
        """Create header section"""
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #FF6B35, stop: 1 #F7931E);
                border-radius: 10px;
                padding: 20px;
            }
        """)
        
        header_layout = QVBoxLayout(header_frame)
        
        title = QLabel("Multi-Source Data Fusion")
        title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 20px;
                font-weight: bold;
                margin: 0;
            }
        """)
        
        subtitle = QLabel("Integrate multiple geological datasets for enhanced analysis")
        subtitle.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 0.9);
                font-size: 12px;
                margin: 5px 0 0 0;
            }
        """)
        
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        layout.addWidget(header_frame)
    
    def create_main_content(self, layout):
        """Create main content area"""
        content_layout = QHBoxLayout()
        
        # Left panel - Layer selection
        left_panel = self.create_layer_panel()
        content_layout.addWidget(left_panel, 1)
        
        # Right panel - Fusion settings
        right_panel = self.create_settings_panel()
        content_layout.addWidget(right_panel, 1)
        
        layout.addLayout(content_layout)
    
    def create_layer_panel(self):
        """Create layer selection panel"""
        group = QGroupBox("Input Layers")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #333333;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
                background-color: white;
            }
        """)
        
        layout = QVBoxLayout(group)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh Layers")
        refresh_btn.clicked.connect(self.refresh_layers)
        refresh_btn.setStyleSheet(self.get_button_style())
        layout.addWidget(refresh_btn)
        
        # Layer list
        self.layer_list = QListWidget()
        self.layer_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                background-color: #fafafa;
                selection-background-color: #2196F3;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #f0f0f0;
            }
            QListWidget::item:hover {
                background-color: #e3f2fd;
            }
            QListWidget::item:selected {
                background-color: #2196F3;
                color: white;
            }
        """)
        layout.addWidget(self.layer_list)
        
        # Weight adjustment
        weight_layout = QVBoxLayout()
        weight_layout.addWidget(QLabel("Layer Weight:"))
        
        self.weight_slider = QSlider(Qt.Horizontal)
        self.weight_slider.setRange(1, 100)
        self.weight_slider.setValue(50)
        self.weight_slider.valueChanged.connect(self.update_weight_label)
        
        self.weight_label = QLabel("0.5")
        self.weight_label.setAlignment(Qt.AlignCenter)
        self.weight_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
                border-radius: 4px;
                padding: 4px;
                min-width: 40px;
            }
        """)
        
        weight_layout.addWidget(self.weight_slider)
        weight_layout.addWidget(self.weight_label)
        layout.addLayout(weight_layout)
        
        return group
    
    def create_settings_panel(self):
        """Create fusion settings panel"""
        group = QGroupBox("Fusion Settings")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 14px;
                color: #333333;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
                background-color: white;
            }
        """)
        
        layout = QVBoxLayout(group)
        
        # Fusion method
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Fusion Method:"))
        
        self.fusion_method = QComboBox()
        self.fusion_method.addItems([
            "Weighted Average",
            "Principal Component Analysis",
            "Fuzzy Logic",
            "Neural Network",
            "Bayesian Fusion"
        ])
        self.fusion_method.setStyleSheet(self.get_combobox_style())
        method_layout.addWidget(self.fusion_method)
        layout.addLayout(method_layout)
        
        # Normalization method
        norm_layout = QHBoxLayout()
        norm_layout.addWidget(QLabel("Normalization:"))
        
        self.norm_method = QComboBox()
        self.norm_method.addItems([
            "Min-Max Scaling",
            "Z-Score Normalization", 
            "Percentile Scaling",
            "Robust Scaling"
        ])
        self.norm_method.setStyleSheet(self.get_combobox_style())
        norm_layout.addWidget(self.norm_method)
        layout.addLayout(norm_layout)
        
        # Advanced options
        self.create_advanced_options(layout)
        
        # Preview button
        preview_btn = QPushButton("Generate Preview")
        preview_btn.clicked.connect(self.generate_preview)
        preview_btn.setStyleSheet(self.get_primary_button_style())
        layout.addWidget(preview_btn)
        
        # Results area
        results_label = QLabel("Fusion Results:")
        results_label.setStyleSheet("font-weight: bold; margin-top: 15px;")
        layout.addWidget(results_label)
        
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(150)
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                padding: 8px;
            }
        """)
        self.results_text.setPlainText("Preview results will appear here...")
        layout.addWidget(self.results_text)
        
        return group
    
    def create_advanced_options(self, layout):
        """Create advanced fusion options"""
        advanced_group = QGroupBox("Advanced Options")
        advanced_group.setCheckable(True)
        advanced_group.setChecked(False)
        advanced_group.setStyleSheet("""
            QGroupBox {
                font-size: 12px;
                color: #555555;
                border: 1px solid #d0d0d0;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 8px 0 8px;
                background-color: white;
            }
        """)
        
        advanced_layout = QVBoxLayout(advanced_group)
        
        # Quality assessment
        self.quality_assessment = QCheckBox("Enable quality assessment")
        self.quality_assessment.setChecked(True)
        advanced_layout.addWidget(self.quality_assessment)
        
        # Uncertainty analysis
        self.uncertainty_analysis = QCheckBox("Perform uncertainty analysis")
        advanced_layout.addWidget(self.uncertainty_analysis)
        
        # Spatial filtering
        filter_layout = QHBoxLayout()
        self.spatial_filter = QCheckBox("Apply spatial filtering")
        filter_layout.addWidget(self.spatial_filter)
        
        self.filter_size = QSpinBox()
        self.filter_size.setRange(1, 10)
        self.filter_size.setValue(3)
        self.filter_size.setSuffix(" pixels")
        filter_layout.addWidget(self.filter_size)
        advanced_layout.addLayout(filter_layout)
        
        layout.addWidget(advanced_group)
    
    def create_footer(self, layout):
        """Create footer with action buttons"""
        footer_layout = QHBoxLayout()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                text-align: center;
                background-color: #f5f5f5;
            }
            QProgressBar::chunk {
                background-color: #FF6B35;
                border-radius: 3px;
            }
        """)
        footer_layout.addWidget(self.progress_bar)
        
        # Buttons
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self.export_results)
        export_btn.setStyleSheet(self.get_button_style())
        
        run_btn = QPushButton("Run Fusion")
        run_btn.clicked.connect(self.run_fusion)
        run_btn.setStyleSheet(self.get_primary_button_style())
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet(self.get_button_style())
        
        footer_layout.addWidget(export_btn)
        footer_layout.addWidget(run_btn)
        footer_layout.addWidget(close_btn)
        
        layout.addLayout(footer_layout)
    
    def apply_modern_style(self):
        """Apply modern styling"""
        self.setStyleSheet("""
            QDialog {
                background-color: #ffffff;
                color: #333333;
            }
        """)
    
    def get_button_style(self):
        """Standard button style"""
        return """
            QPushButton {
                background-color: #f8f9fa;
                color: #333333;
                border: 1px solid #dee2e6;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
                border-color: #adb5bd;
            }
            QPushButton:pressed {
                background-color: #dee2e6;
            }
        """
    
    def get_primary_button_style(self):
        """Primary button style"""
        return """
            QPushButton {
                background-color: #FF6B35;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #E55A2B;
            }
            QPushButton:pressed {
                background-color: #CC4E21;
            }
        """
    
    def get_combobox_style(self):
        """Combobox style"""
        return """
            QComboBox {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 6px 10px;
                background-color: white;
                min-width: 120px;
            }
            QComboBox:hover {
                border-color: #FF6B35;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-style: solid;
                border-width: 4px 4px 0 4px;
                border-color: #666666 transparent transparent transparent;
            }
        """
    
    def refresh_layers(self):
        """Refresh available raster layers"""
        self.layer_list.clear()
        self.selected_layers.clear()
        self.layer_weights.clear()
        
        project = QgsProject.instance()
        for layer in project.mapLayers().values():
            if isinstance(layer, QgsRasterLayer) and layer.isValid():
                item = QListWidgetItem(layer.name())
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                item.setData(Qt.UserRole, layer.id())
                self.layer_list.addItem(item)
        
        if self.layer_list.count() == 0:
            item = QListWidgetItem("No raster layers available")
            item.setFlags(Qt.NoItemFlags)
            self.layer_list.addItem(item)
    
    def update_weight_label(self, value):
        """Update weight label"""
        weight = value / 100.0
        self.weight_label.setText(f"{weight:.2f}")
        
        # Update weight for selected layers
        for i in range(self.layer_list.count()):
            item = self.layer_list.item(i)
            if item.checkState() == Qt.Checked:
                layer_id = item.data(Qt.UserRole)
                if layer_id:
                    self.layer_weights[layer_id] = weight
    
    def get_selected_layers(self):
        """Get selected layers and their weights"""
        selected = {}
        weights = {}
        
        for i in range(self.layer_list.count()):
            item = self.layer_list.item(i)
            if item.checkState() == Qt.Checked:
                layer_id = item.data(Qt.UserRole)
                if layer_id:
                    selected[layer_id] = item.text()
                    weights[layer_id] = self.layer_weights.get(layer_id, 0.5)
        
        return selected, weights
    
    def generate_preview(self):
        """Generate fusion preview"""
        selected_layers, weights = self.get_selected_layers()
        
        if len(selected_layers) < 2:
            QMessageBox.warning(self, "Warning", "Please select at least 2 layers for fusion.")
            return
        
        try:
            # Show progress
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Import fusion processor
            from ..algorithms.data_fusion import DataFusionProcessor
            
            processor = DataFusionProcessor()
            
            self.progress_bar.setValue(25)
            
            # Load layers
            layer_ids = list(selected_layers.keys())
            weight_list = [weights[lid] for lid in layer_ids]
            
            if processor.load_layers(layer_ids, weight_list):
                self.progress_bar.setValue(50)
                
                # Normalize layers
                norm_method = self.norm_method.currentText().lower().replace(' ', '_').replace('-', '_')
                processor.normalize_layers(norm_method)
                
                self.progress_bar.setValue(75)
                
                # Create preview
                fusion_method = self.fusion_method.currentText().lower().replace(' ', '_')
                result = processor.create_preview(fusion_method)
                
                self.progress_bar.setValue(100)
                
                # Display results
                if result is not None:
                    quality_metrics = processor.calculate_fusion_quality_metrics(result)
                    
                    results_text = f"Fusion Preview Results:\n"
                    results_text += f"Method: {self.fusion_method.currentText()}\n"
                    results_text += f"Normalization: {self.norm_method.currentText()}\n"
                    results_text += f"Layers: {len(selected_layers)}\n"
                    results_text += f"Coverage: {quality_metrics.get('coverage_percent', 0):.1f}%\n"
                    results_text += f"Value Range: {quality_metrics.get('min', 0):.3f} - {quality_metrics.get('max', 1):.3f}\n"
                    
                    self.results_text.setPlainText(results_text)
                    
                    QMessageBox.information(self, "Success", "Fusion preview generated successfully!")
                else:
                    QMessageBox.critical(self, "Error", "Failed to generate fusion preview.")
            
            else:
                QMessageBox.critical(self, "Error", "Failed to load selected layers.")
            
            self.progress_bar.setVisible(False)
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Fusion preview failed: {str(e)}")
    
    def run_fusion(self):
        """Run complete data fusion"""
        selected_layers, weights = self.get_selected_layers()
        
        if len(selected_layers) < 2:
            QMessageBox.warning(self, "Warning", "Please select at least 2 layers for fusion.")
            return
        
        try:
            # Get output file
            output_file, _ = QFileDialog.getSaveFileName(
                self, "Save Fusion Result", "",
                "GeoTIFF files (*.tif);;All files (*)"
            )
            
            if not output_file:
                return
            
            # Show progress
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            from ..algorithms.data_fusion import DataFusionProcessor
            
            processor = DataFusionProcessor()
            
            # Load and process
            layer_ids = list(selected_layers.keys())
            weight_list = [weights[lid] for lid in layer_ids]
            
            if processor.load_layers(layer_ids, weight_list):
                self.progress_bar.setValue(20)
                
                norm_method = self.norm_method.currentText().lower().replace(' ', '_').replace('-', '_')
                processor.normalize_layers(norm_method)
                
                self.progress_bar.setValue(40)
                
                fusion_method = self.fusion_method.currentText().lower().replace(' ', '_')
                
                if fusion_method == 'weighted_average':
                    result = processor.weighted_average_fusion()
                elif fusion_method == 'principal_component_analysis':
                    result = processor.principal_component_fusion()
                elif fusion_method == 'fuzzy_logic':
                    result = processor.fuzzy_logic_fusion()
                elif fusion_method == 'neural_network':
                    result = processor.neural_network_fusion()
                elif fusion_method == 'bayesian_fusion':
                    result = processor.bayesian_fusion()
                else:
                    result = processor.weighted_average_fusion()
                
                self.progress_bar.setValue(80)
                
                # Apply spatial filtering if enabled
                if self.spatial_filter.isChecked():
                    from scipy import ndimage
                    sigma = self.filter_size.value() / 3.0
                    result = ndimage.gaussian_filter(result, sigma=sigma)
                
                # Save result
                saved_path = processor.save_result(result, output_file)
                
                self.progress_bar.setValue(100)
                
                # Add to QGIS
                layer = QgsRasterLayer(saved_path, "Data Fusion Result")
                if layer.isValid():
                    QgsProject.instance().addMapLayer(layer)
                    
                    # Update results
                    quality_metrics = processor.calculate_fusion_quality_metrics(result)
                    results_text = f"Data Fusion Completed!\n"
                    results_text += f"Output: {os.path.basename(saved_path)}\n"
                    results_text += f"Method: {self.fusion_method.currentText()}\n"
                    results_text += f"Layers Fused: {len(selected_layers)}\n"
                    results_text += f"Coverage: {quality_metrics.get('coverage_percent', 0):.1f}%\n"
                    
                    self.results_text.setPlainText(results_text)
                    
                    QMessageBox.information(self, "Success", 
                                          f"Data fusion completed successfully!\n"
                                          f"Result saved to: {saved_path}")
                else:
                    QMessageBox.warning(self, "Warning", 
                                      "Fusion completed but layer could not be added to QGIS.")
            
            else:
                QMessageBox.critical(self, "Error", "Failed to load selected layers.")
            
            self.progress_bar.setVisible(False)
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(self, "Error", f"Data fusion failed: {str(e)}")
    
    def export_results(self):
        """Export fusion results"""
        if self.results_text.toPlainText().strip() == "Preview results will appear here...":
            QMessageBox.information(self, "Info", "No results to export. Please run fusion first.")
            return
        
        output_file, _ = QFileDialog.getSaveFileName(
            self, "Save Results Report", "",
            "Text files (*.txt);;All files (*)"
        )
        
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    f.write("Data Fusion Results Report\n")
                    f.write("=" * 30 + "\n\n")
                    f.write(self.results_text.toPlainText())
                    f.write("\n\nGenerated by Mineral Prospectivity Mapping Plugin")
                
                QMessageBox.information(self, "Success", f"Results exported to: {output_file}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export results: {str(e)}")