"""
Modern correlation analysis dialog for geological datasets
"""

import os
import numpy as np
from qgis.PyQt.QtCore import Qt, pyqtSignal, QThread
from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
                                QLabel, QPushButton, QProgressBar, QTextEdit,
                                QGroupBox, QComboBox, QSpinBox, QDoubleSpinBox,
                                QCheckBox, QListWidget, QListWidgetItem, QTableWidget,
                                QTableWidgetItem, QFrame, QMessageBox, QFileDialog,
                                QTabWidget, QWidget, QScrollArea)
from qgis.PyQt.QtGui import QFont, QPixmap, QIcon, QColor
from qgis.core import QgsProject, QgsRasterLayer, QgsMessageLog, Qgis


class CorrelationWorker(QThread):
    """Worker thread for correlation analysis"""
    
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, dict)
    
    def __init__(self, layer_ids, method):
        super().__init__()
        self.layer_ids = layer_ids
        self.method = method
    
    def run(self):
        try:
            from ..algorithms.correlation_analysis import CorrelationAnalyzer
            
            self.status.emit("Initializing correlation analyzer...")
            self.progress.emit(10)
            
            analyzer = CorrelationAnalyzer()
            
            self.status.emit("Loading layer data...")
            self.progress.emit(20)
            
            data_arrays = analyzer.load_layers_data(self.layer_ids)
            
            self.status.emit("Calculating correlation matrix...")
            self.progress.emit(40)
            
            correlation_matrix = analyzer.calculate_correlation_matrix(data_arrays, self.method)
            
            self.status.emit("Calculating p-values...")
            self.progress.emit(60)
            
            p_values = analyzer.calculate_p_values(data_arrays, self.method)
            
            self.status.emit("Computing advanced statistics...")
            self.progress.emit(80)
            
            advanced_stats = analyzer.compute_advanced_statistics(data_arrays)
            
            self.progress.emit(100)
            
            results = {
                'correlation_matrix': correlation_matrix,
                'p_values': p_values,
                'advanced_stats': advanced_stats,
                'layer_names': {lid: analyzer.get_layer_name(lid) for lid in self.layer_ids}
            }
            
            self.finished.emit(True, results)
            
        except Exception as e:
            self.finished.emit(False, {'error': str(e)})


class CorrelationDialog(QDialog):
    """Modern dialog for correlation analysis"""
    
    def __init__(self, iface):
        super().__init__()
        self.iface = iface
        self.setWindowTitle("Correlation Analysis")
        self.setMinimumSize(900, 700)
        
        self.correlation_results = None
        self.worker = None
        
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
        
        # Main content with tabs
        self.create_main_content(layout)
        
        # Footer
        self.create_footer(layout)
    
    def create_header(self, layout):
        """Create header section"""
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #6A1B9A, stop: 1 #9C27B0);
                border-radius: 10px;
                padding: 20px;
            }
        """)
        
        header_layout = QVBoxLayout(header_frame)
        
        title = QLabel("Correlation Analysis")
        title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 20px;
                font-weight: bold;
                margin: 0;
            }
        """)
        
        subtitle = QLabel("Analyze statistical relationships between geological datasets")
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
        """Create main tabbed content"""
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background-color: #fafafa;
                padding: 10px;
            }
            QTabBar::tab {
                background-color: #f5f5f5;
                padding: 10px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                color: #666666;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #6A1B9A;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background-color: #e1bee7;
                color: #6A1B9A;
            }
        """)
        
        # Setup tab
        self.create_setup_tab()
        
        # Results tab
        self.create_results_tab()
        
        # Advanced tab
        self.create_advanced_tab()
        
        layout.addWidget(self.tab_widget)
    
    def create_setup_tab(self):
        """Create analysis setup tab"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Left panel - Layer selection
        left_panel = self.create_layer_selection_panel()
        layout.addWidget(left_panel, 1)
        
        # Right panel - Analysis settings
        right_panel = self.create_analysis_settings_panel()
        layout.addWidget(right_panel, 1)
        
        self.tab_widget.addTab(tab, "Setup")
    
    def create_layer_selection_panel(self):
        """Create layer selection panel"""
        group = QGroupBox("Select Layers for Analysis")
        group.setStyleSheet(self.get_groupbox_style())
        
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
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #f0f0f0;
            }
            QListWidget::item:hover {
                background-color: #f3e5f5;
            }
            QListWidget::item:selected {
                background-color: #6A1B9A;
                color: white;
            }
        """)
        layout.addWidget(self.layer_list)
        
        # Selection info
        self.selection_label = QLabel("0 layers selected")
        self.selection_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                padding: 5px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.selection_label)
        
        return group
    
    def create_analysis_settings_panel(self):
        """Create analysis settings panel"""
        group = QGroupBox("Analysis Settings")
        group.setStyleSheet(self.get_groupbox_style())
        
        layout = QVBoxLayout(group)
        
        # Correlation method
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Correlation Method:"))
        
        self.correlation_method = QComboBox()
        self.correlation_method.addItems([
            "Pearson",
            "Spearman", 
            "Kendall"
        ])
        self.correlation_method.setStyleSheet(self.get_combobox_style())
        method_layout.addWidget(self.correlation_method)
        layout.addLayout(method_layout)
        
        # Significance level
        sig_layout = QHBoxLayout()
        sig_layout.addWidget(QLabel("Significance Level:"))
        
        self.significance_level = QDoubleSpinBox()
        self.significance_level.setRange(0.001, 0.1)
        self.significance_level.setSingleStep(0.001)
        self.significance_level.setValue(0.05)
        self.significance_level.setDecimals(3)
        sig_layout.addWidget(self.significance_level)
        layout.addLayout(sig_layout)
        
        # Advanced options
        self.partial_correlation = QCheckBox("Calculate partial correlations")
        self.mutual_information = QCheckBox("Calculate mutual information")
        self.distance_correlation = QCheckBox("Calculate distance correlation")
        
        layout.addWidget(self.partial_correlation)
        layout.addWidget(self.mutual_information) 
        layout.addWidget(self.distance_correlation)
        
        # Run button
        run_btn = QPushButton("Run Correlation Analysis")
        run_btn.clicked.connect(self.run_analysis)
        run_btn.setStyleSheet(self.get_primary_button_style())
        layout.addWidget(run_btn)
        
        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: #e8f5e8;
                color: #2e7d32;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.status_label)
        
        return group
    
    def create_results_tab(self):
        """Create results display tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Results summary
        summary_group = QGroupBox("Analysis Summary")
        summary_group.setStyleSheet(self.get_groupbox_style())
        summary_layout = QVBoxLayout(summary_group)
        
        self.summary_text = QTextEdit()
        self.summary_text.setMaximumHeight(120)
        self.summary_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                padding: 8px;
            }
        """)
        self.summary_text.setPlainText("Analysis results will appear here...")
        summary_layout.addWidget(self.summary_text)
        layout.addWidget(summary_group)
        
        # Correlation matrix
        matrix_group = QGroupBox("Correlation Matrix")
        matrix_group.setStyleSheet(self.get_groupbox_style())
        matrix_layout = QVBoxLayout(matrix_group)
        
        self.correlation_table = QTableWidget()
        self.correlation_table.setStyleSheet("""
            QTableWidget {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                background-color: white;
                gridline-color: #f0f0f0;
            }
            QTableWidget::item {
                padding: 4px;
                border: none;
            }
            QHeaderView::section {
                background-color: #f5f5f5;
                border: 1px solid #e0e0e0;
                padding: 4px;
                font-weight: bold;
            }
        """)
        matrix_layout.addWidget(self.correlation_table)
        layout.addWidget(matrix_group)
        
        self.tab_widget.addTab(tab, "Results")
    
    def create_advanced_tab(self):
        """Create advanced statistics tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # PCA results
        pca_group = QGroupBox("Principal Component Analysis")
        pca_group.setStyleSheet(self.get_groupbox_style())
        pca_layout = QVBoxLayout(pca_group)
        
        self.pca_text = QTextEdit()
        self.pca_text.setMaximumHeight(150)
        self.pca_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                padding: 8px;
            }
        """)
        pca_layout.addWidget(self.pca_text)
        layout.addWidget(pca_group)
        
        # Cluster analysis
        cluster_group = QGroupBox("Cluster Analysis")
        cluster_group.setStyleSheet(self.get_groupbox_style())
        cluster_layout = QVBoxLayout(cluster_group)
        
        self.cluster_text = QTextEdit()
        self.cluster_text.setMaximumHeight(150)
        self.cluster_text.setStyleSheet("""
            QTextEdit {
                background-color: #f8f9fa;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                padding: 8px;
            }
        """)
        cluster_layout.addWidget(self.cluster_text)
        layout.addWidget(cluster_group)
        
        self.tab_widget.addTab(tab, "Advanced")
    
    def create_footer(self, layout):
        """Create footer with progress and controls"""
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
                background-color: #6A1B9A;
                border-radius: 3px;
            }
        """)
        footer_layout.addWidget(self.progress_bar)
        
        # Export button
        export_btn = QPushButton("Export Results")
        export_btn.clicked.connect(self.export_results)
        export_btn.setStyleSheet(self.get_button_style())
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet(self.get_button_style())
        
        footer_layout.addWidget(export_btn)
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
    
    def get_groupbox_style(self):
        """GroupBox style"""
        return """
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
        """
    
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
                background-color: #6A1B9A;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #4A148C;
            }
            QPushButton:pressed {
                background-color: #38006B;
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
                min-width: 100px;
            }
            QComboBox:hover {
                border-color: #6A1B9A;
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
        
        self.update_selection_count()
    
    def update_selection_count(self):
        """Update selection count label"""
        count = 0
        for i in range(self.layer_list.count()):
            item = self.layer_list.item(i)
            if item.checkState() == Qt.Checked:
                count += 1
        
        self.selection_label.setText(f"{count} layers selected")
        
        # Update style based on minimum requirement
        if count >= 2:
            self.selection_label.setStyleSheet("""
                QLabel {
                    background-color: #e8f5e8;
                    color: #2e7d32;
                    padding: 5px;
                    border-radius: 4px;
                    font-weight: bold;
                }
            """)
        else:
            self.selection_label.setStyleSheet("""
                QLabel {
                    background-color: #fff3e0;
                    color: #ef6c00;
                    padding: 5px;
                    border-radius: 4px;
                    font-weight: bold;
                }
            """)
    
    def get_selected_layers(self):
        """Get selected layer IDs"""
        selected = []
        for i in range(self.layer_list.count()):
            item = self.layer_list.item(i)
            if item.checkState() == Qt.Checked:
                layer_id = item.data(Qt.UserRole)
                if layer_id:
                    selected.append(layer_id)
        return selected
    
    def run_analysis(self):
        """Run correlation analysis"""
        selected_layers = self.get_selected_layers()
        
        if len(selected_layers) < 2:
            QMessageBox.warning(self, "Warning", 
                              "Please select at least 2 layers for correlation analysis.")
            return
        
        # Update UI
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting analysis...")
        
        # Start worker thread
        method = self.correlation_method.currentText().lower()
        self.worker = CorrelationWorker(selected_layers, method)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.status_label.setText)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.start()
    
    def on_analysis_finished(self, success, results):
        """Handle analysis completion"""
        self.progress_bar.setVisible(False)
        
        if success and 'error' not in results:
            self.correlation_results = results
            self.display_results(results)
            self.status_label.setText("Analysis completed successfully")
            
            # Switch to results tab
            self.tab_widget.setCurrentIndex(1)
            
            QMessageBox.information(self, "Success", "Correlation analysis completed successfully!")
        else:
            error_msg = results.get('error', 'Unknown error occurred')
            self.status_label.setText("Analysis failed")
            QMessageBox.critical(self, "Error", f"Correlation analysis failed: {error_msg}")
        
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
    
    def display_results(self, results):
        """Display analysis results"""
        # Summary
        correlation_matrix = results['correlation_matrix']
        layer_names = results['layer_names']
        
        summary = f"Correlation Analysis Results\n"
        summary += f"Method: {self.correlation_method.currentText()}\n"
        summary += f"Layers: {len(layer_names)}\n"
        summary += f"Matrix size: {correlation_matrix.shape[0]}x{correlation_matrix.shape[1]}\n"
        
        # Find strongest correlations
        n = correlation_matrix.shape[0]
        max_corr = 0
        max_pair = None
        
        for i in range(n):
            for j in range(i+1, n):
                corr_val = abs(correlation_matrix[i, j])
                if corr_val > max_corr and np.isfinite(corr_val):
                    max_corr = corr_val
                    max_pair = (i, j)
        
        if max_pair:
            layer_ids = list(layer_names.keys())
            name1 = layer_names[layer_ids[max_pair[0]]]
            name2 = layer_names[layer_ids[max_pair[1]]]
            summary += f"Strongest correlation: {name1} - {name2} ({max_corr:.3f})"
        
        self.summary_text.setPlainText(summary)
        
        # Correlation matrix table
        self.display_correlation_matrix(correlation_matrix, layer_names)
        
        # Advanced statistics
        if 'advanced_stats' in results:
            self.display_advanced_stats(results['advanced_stats'])
    
    def display_correlation_matrix(self, matrix, layer_names):
        """Display correlation matrix in table"""
        layer_ids = list(layer_names.keys())
        n = len(layer_ids)
        
        self.correlation_table.setRowCount(n)
        self.correlation_table.setColumnCount(n)
        
        # Set headers
        headers = [layer_names[lid][:15] + "..." if len(layer_names[lid]) > 15 
                  else layer_names[lid] for lid in layer_ids]
        self.correlation_table.setHorizontalHeaderLabels(headers)
        self.correlation_table.setVerticalHeaderLabels(headers)
        
        # Fill matrix
        for i in range(n):
            for j in range(n):
                value = matrix[i, j]
                if np.isfinite(value):
                    item = QTableWidgetItem(f"{value:.3f}")
                    
                    # Color code based on correlation strength
                    if i == j:
                        item.setBackground(QColor(220, 220, 220))  # Diagonal
                    elif abs(value) > 0.7:
                        item.setBackground(QColor(255, 182, 193))  # Strong
                    elif abs(value) > 0.5:
                        item.setBackground(QColor(255, 218, 185))  # Moderate
                    elif abs(value) > 0.3:
                        item.setBackground(QColor(255, 255, 186))  # Weak
                    
                    item.setTextAlignment(Qt.AlignCenter)
                else:
                    item = QTableWidgetItem("N/A")
                    item.setTextAlignment(Qt.AlignCenter)
                
                self.correlation_table.setItem(i, j, item)
        
        # Resize columns
        self.correlation_table.resizeColumnsToContents()
    
    def display_advanced_stats(self, advanced_stats):
        """Display advanced statistics"""
        # PCA results
        if 'pca' in advanced_stats:
            pca_data = advanced_stats['pca']
            pca_text = "Principal Component Analysis:\n"
            pca_text += f"Explained Variance Ratios:\n"
            
            for i, ratio in enumerate(pca_data['explained_variance_ratio'][:5]):
                pca_text += f"  PC{i+1}: {ratio:.3f} ({ratio*100:.1f}%)\n"
            
            cumulative = pca_data['cumulative_variance']
            pca_text += f"\nCumulative variance (first 3 PCs): {cumulative[2]:.3f}"
            
            self.pca_text.setPlainText(pca_text)
        
        # Clustering results
        if 'clustering' in advanced_stats:
            cluster_data = advanced_stats['clustering']
            cluster_text = "Cluster Analysis:\n"
            cluster_text += f"Optimal clusters: {cluster_data['optimal_clusters']}\n"
            cluster_text += f"Silhouette scores:\n"
            
            for i, score in enumerate(cluster_data['silhouette_scores'][:5]):
                cluster_text += f"  {i+2} clusters: {score:.3f}\n"
            
            self.cluster_text.setPlainText(cluster_text)
    
    def export_results(self):
        """Export correlation results"""
        if self.correlation_results is None:
            QMessageBox.information(self, "Info", 
                                  "No results to export. Please run analysis first.")
            return
        
        output_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory"
        )
        
        if output_dir:
            try:
                from ..algorithms.correlation_analysis import CorrelationAnalyzer
                
                analyzer = CorrelationAnalyzer()
                analyzer.correlation_matrix = self.correlation_results['correlation_matrix']
                analyzer.p_values_matrix = self.correlation_results['p_values']
                analyzer.advanced_stats = self.correlation_results.get('advanced_stats', {})
                analyzer.data_arrays = {lid: None for lid in self.correlation_results['layer_names'].keys()}
                
                # Mock the get_layer_name method
                def mock_get_layer_name(layer_id):
                    return self.correlation_results['layer_names'].get(layer_id, f"Layer_{layer_id}")
                
                analyzer.get_layer_name = mock_get_layer_name
                
                # Export results
                exported_path = analyzer.export_correlation_results(output_dir)
                
                QMessageBox.information(self, "Success", 
                                      f"Results exported successfully to: {exported_path}")
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export results: {str(e)}")