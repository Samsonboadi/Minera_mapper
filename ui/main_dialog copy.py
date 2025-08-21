"""
Modern main dialog for Mineral Prospectivity Mapping plugin
"""

import os
from qgis.PyQt.QtCore import Qt, pyqtSignal, QThread, QTimer
from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
                                QLabel, QPushButton, QProgressBar, QTextEdit,
                                QTabWidget, QWidget, QComboBox, QSpinBox,
                                QDoubleSpinBox, QCheckBox, QGroupBox, QFrame,
                                QScrollArea, QSplitter, QFileDialog, QMessageBox,
                                QListWidget, QListWidgetItem, QSlider)
from qgis.PyQt.QtGui import QFont, QPixmap, QIcon, QPalette, QColor
from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer, QgsMessageLog, Qgis


class ModernCard(QFrame):
    """Modern card-style widget"""
    
    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 10px;
                margin: 5px;
            }
            QFrame:hover {
                border: 1px solid #2196F3;
                box-shadow: 0 2px 8px rgba(33, 150, 243, 0.3);
            }
        """)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(15, 15, 15, 15)
        
        if title:
            title_label = QLabel(title)
            title_label.setFont(QFont("Arial", 12, QFont.Bold))
            title_label.setStyleSheet("color: #333333; margin-bottom: 10px;")
            self.layout.addWidget(title_label)


class ProcessingWorker(QThread):
    """Worker thread for data processing"""
    
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, processor, method, *args, **kwargs):
        super().__init__()
        self.processor = processor
        self.method = method
        self.args = args
        self.kwargs = kwargs
    
    def run(self):
        try:
            self.status.emit("Starting processing...")
            self.progress.emit(10)
            
            # Execute the processing method
            result = getattr(self.processor, self.method)(*self.args, **self.kwargs)
            
            self.progress.emit(100)
            self.finished.emit(True, "Processing completed successfully!")
            
        except Exception as e:
            self.finished.emit(False, f"Processing failed: {str(e)}")


class MainDialog(QDialog):
    """Modern main dialog for mineral prospectivity mapping"""
    
    def __init__(self, iface):
        super().__init__()
        self.iface = iface
        self.setWindowTitle("Mineral Prospectivity Mapping")
        self.setMinimumSize(1000, 700)
        self.worker = None
        
        self.setup_ui()
        self.apply_modern_style()
    
    def setup_ui(self):
        """Setup the user interface"""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        self.create_header(main_layout)
        
        # Main content with tabs
        self.create_main_content(main_layout)
        
        # Footer with buttons
        self.create_footer(main_layout)
    
    def create_header(self, layout):
        """Create modern header"""
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #2196F3, stop: 1 #21CBF3);
                border-radius: 10px;
                padding: 20px;
            }
        """)
        
        header_layout = QHBoxLayout(header_frame)
        
        # Title and subtitle
        title_layout = QVBoxLayout()
        
        title = QLabel("Mineral Prospectivity Mapping")
        title.setStyleSheet("""
            QLabel {
                color: white;
                font-size: 24px;
                font-weight: bold;
                margin: 0;
            }
        """)
        
        subtitle = QLabel("Advanced geological data processing and analysis")
        subtitle.setStyleSheet("""
            QLabel {
                color: rgba(255, 255, 255, 0.8);
                font-size: 14px;
                margin: 0;
            }
        """)
        
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        header_layout.addLayout(title_layout)
        
        # Status indicator
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 0.2);
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: bold;
            }
        """)
        header_layout.addWidget(self.status_label)
        
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
                padding: 12px 20px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                color: #666666;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #2196F3;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background-color: #e3f2fd;
                color: #2196F3;
            }
        """)
        
        # Data Processing Tab
        self.create_processing_tab()
        
        # Analysis Tab
        self.create_analysis_tab()
        
        # Results Tab
        self.create_results_tab()
        
        layout.addWidget(self.tab_widget)
    
    def create_processing_tab(self):
        """Create data processing tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Scroll area for cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(15)
        
        # ASTER Processing Card
        aster_card = self.create_aster_card()
        scroll_layout.addWidget(aster_card)
        
        # Sentinel-2 Processing Card
        sentinel_card = self.create_sentinel_card()
        scroll_layout.addWidget(sentinel_card)
        
        # Geological Data Card
        geo_card = self.create_geological_card()
        scroll_layout.addWidget(geo_card)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        self.tab_widget.addTab(tab, "Data Processing")
    
    def create_aster_card(self):
        """Create ASTER processing card"""
        card = ModernCard("ASTER L2 Surface Reflectance Processing")
        
        # Description
        desc = QLabel("Process ASTER L2 VNIR/SWIR data for mineral mapping")
        desc.setStyleSheet("color: #666666; margin-bottom: 15px;")
        card.layout.addWidget(desc)
        
        # Options
        options_layout = QGridLayout()
        
        # File selection
        self.aster_file_label = QLabel("No file selected")
        self.aster_file_label.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                padding: 8px;
                border-radius: 4px;
                border: 1px solid #e0e0e0;
            }
        """)
        
        browse_btn = QPushButton("Browse ASTER File")
        browse_btn.clicked.connect(self.browse_aster_file)
        browse_btn.setStyleSheet(self.get_button_style())
        
        options_layout.addWidget(QLabel("ASTER File:"), 0, 0)
        options_layout.addWidget(self.aster_file_label, 0, 1)
        options_layout.addWidget(browse_btn, 0, 2)
        
        # Processing options
        self.aster_atmospheric = QCheckBox("Apply atmospheric correction")
        self.aster_ratios = QCheckBox("Calculate mineral ratios")
        self.aster_ratios.setChecked(True)
        self.aster_composites = QCheckBox("Create false color composites")
        self.aster_composites.setChecked(True)
        
        options_layout.addWidget(self.aster_atmospheric, 1, 0, 1, 3)
        options_layout.addWidget(self.aster_ratios, 2, 0, 1, 3)
        options_layout.addWidget(self.aster_composites, 3, 0, 1, 3)
        
        card.layout.addLayout(options_layout)
        
        # Process button
        process_btn = QPushButton("Process ASTER Data")
        process_btn.clicked.connect(self.process_aster_data)
        process_btn.setStyleSheet(self.get_primary_button_style())
        card.layout.addWidget(process_btn)
        
        return card
    
    def create_sentinel_card(self):
        """Create Sentinel-2 processing card"""
        card = ModernCard("Sentinel-2 MSI Processing")
        
        desc = QLabel("Process Sentinel-2 MSI data for mineral exploration")
        desc.setStyleSheet("color: #666666; margin-bottom: 15px;")
        card.layout.addWidget(desc)
        
        options_layout = QGridLayout()
        
        # File selection
        self.s2_file_label = QLabel("No directory selected")
        self.s2_file_label.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                padding: 8px;
                border-radius: 4px;
                border: 1px solid #e0e0e0;
            }
        """)
        
        browse_btn = QPushButton("Browse S2 Directory")
        browse_btn.clicked.connect(self.browse_s2_directory)
        browse_btn.setStyleSheet(self.get_button_style())
        
        options_layout.addWidget(QLabel("S2 Product:"), 0, 0)
        options_layout.addWidget(self.s2_file_label, 0, 1)
        options_layout.addWidget(browse_btn, 0, 2)
        
        # Processing options
        self.s2_resample = QCheckBox("Resample to 10m resolution")
        self.s2_resample.setChecked(True)
        self.s2_indices = QCheckBox("Calculate mineral indices")
        self.s2_indices.setChecked(True)
        self.s2_composites = QCheckBox("Create composite images")
        self.s2_composites.setChecked(True)
        
        options_layout.addWidget(self.s2_resample, 1, 0, 1, 3)
        options_layout.addWidget(self.s2_indices, 2, 0, 1, 3)
        options_layout.addWidget(self.s2_composites, 3, 0, 1, 3)
        
        card.layout.addLayout(options_layout)
        
        # Process button
        process_btn = QPushButton("Process Sentinel-2 Data")
        process_btn.clicked.connect(self.process_s2_data)
        process_btn.setStyleSheet(self.get_primary_button_style())
        card.layout.addWidget(process_btn)
        
        return card
    
    def create_geological_card(self):
        """Create geological data processing card"""
        card = ModernCard("Geological Data Processing")
        
        desc = QLabel("Process geological maps and structural data")
        desc.setStyleSheet("color: #666666; margin-bottom: 15px;")
        card.layout.addWidget(desc)
        
        # File selection
        file_layout = QHBoxLayout()
        self.geo_file_label = QLabel("No file selected")
        self.geo_file_label.setStyleSheet("""
            QLabel {
                background-color: #f5f5f5;
                padding: 8px;
                border-radius: 4px;
                border: 1px solid #e0e0e0;
            }
        """)
        
        browse_btn = QPushButton("Browse File")
        browse_btn.clicked.connect(self.browse_geological_file)
        browse_btn.setStyleSheet(self.get_button_style())
        
        file_layout.addWidget(self.geo_file_label)
        file_layout.addWidget(browse_btn)
        card.layout.addLayout(file_layout)
        
        # Options
        self.geo_favorability = QCheckBox("Calculate geological favorability")
        self.geo_favorability.setChecked(True)
        self.geo_structural = QCheckBox("Analyze structural patterns")
        self.geo_structural.setChecked(True)
        
        card.layout.addWidget(self.geo_favorability)
        card.layout.addWidget(self.geo_structural)
        
        # Process button
        process_btn = QPushButton("Process Geological Data")
        process_btn.clicked.connect(self.process_geological_data)
        process_btn.setStyleSheet(self.get_primary_button_style())
        card.layout.addWidget(process_btn)
        
        return card
    
    def create_analysis_tab(self):
        """Create analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Analysis cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        # Spectral Analysis Card
        spectral_card = self.create_spectral_analysis_card()
        scroll_layout.addWidget(spectral_card)
        
        # Prospectivity Mapping Card
        prospect_card = self.create_prospectivity_card()
        scroll_layout.addWidget(prospect_card)
        
        # Data Fusion Card
        fusion_card = self.create_fusion_card()
        scroll_layout.addWidget(fusion_card)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        self.tab_widget.addTab(tab, "Analysis")
    
    def create_spectral_analysis_card(self):
        """Create spectral analysis card"""
        card = ModernCard("Spectral Analysis")
        
        desc = QLabel("Perform spectral unmixing and mineral identification")
        desc.setStyleSheet("color: #666666; margin-bottom: 15px;")
        card.layout.addWidget(desc)
        
        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        
        self.spectral_method = QComboBox()
        self.spectral_method.addItems([
            "Spectral Angle Mapper (SAM)",
            "Spectral Unmixing",
            "Minimum Noise Fraction (MNF)",
            "Principal Component Analysis (PCA)",
            "Matched Filtering"
        ])
        self.spectral_method.setStyleSheet(self.get_combobox_style())
        method_layout.addWidget(self.spectral_method)
        
        card.layout.addLayout(method_layout)
        
        # Target minerals
        minerals_label = QLabel("Target Minerals:")
        minerals_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        card.layout.addWidget(minerals_label)
        
        self.mineral_list = QListWidget()
        self.mineral_list.setMaximumHeight(150)
        self.mineral_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                background-color: #fafafa;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #f0f0f0;
            }
            QListWidget::item:selected {
                background-color: #2196F3;
                color: white;
            }
        """)
        
        # Add some default minerals
        minerals = ["Gold", "Iron Oxide", "Clay Minerals", "Carbonate", "Silica", 
                   "Alteration Minerals", "Gossans"]
        for mineral in minerals:
            item = QListWidgetItem(mineral)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.mineral_list.addItem(item)
        
        card.layout.addWidget(self.mineral_list)
        
        # Analyze button
        analyze_btn = QPushButton("Run Spectral Analysis")
        analyze_btn.clicked.connect(self.run_spectral_analysis)
        analyze_btn.setStyleSheet(self.get_primary_button_style())
        card.layout.addWidget(analyze_btn)
        
        return card
    
    def create_prospectivity_card(self):
        """Create prospectivity mapping card"""
        card = ModernCard("Prospectivity Mapping")
        
        desc = QLabel("Generate mineral prospectivity maps from multiple data layers")
        desc.setStyleSheet("color: #666666; margin-bottom: 15px;")
        card.layout.addWidget(desc)
        
        # Method selection
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        
        self.prospect_method = QComboBox()
        self.prospect_method.addItems([
            "Weighted Overlay",
            "Fuzzy Logic",
            "Neural Network",
            "Evidence Weights",
            "Analytic Hierarchy Process"
        ])
        self.prospect_method.setStyleSheet(self.get_combobox_style())
        method_layout.addWidget(self.prospect_method)
        
        card.layout.addLayout(method_layout)
        
        # Layer selection
        layers_label = QLabel("Input Layers:")
        layers_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        card.layout.addWidget(layers_label)
        
        self.layer_list = QListWidget()
        self.layer_list.setMaximumHeight(150)
        self.layer_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                background-color: #fafafa;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #f0f0f0;
            }
            QListWidget::item:selected {
                background-color: #2196F3;
                color: white;
            }
        """)
        
        # Populate with available raster layers
        self.refresh_layer_list()
        
        card.layout.addWidget(self.layer_list)
        
        # Refresh and map buttons
        button_layout = QHBoxLayout()
        refresh_btn = QPushButton("Refresh Layers")
        refresh_btn.clicked.connect(self.refresh_layer_list)
        refresh_btn.setStyleSheet(self.get_button_style())
        
        map_btn = QPushButton("Generate Prospectivity Map")
        map_btn.clicked.connect(self.generate_prospectivity_map)
        map_btn.setStyleSheet(self.get_primary_button_style())
        
        button_layout.addWidget(refresh_btn)
        button_layout.addWidget(map_btn)
        card.layout.addLayout(button_layout)
        
        return card
    
    def create_fusion_card(self):
        """Create data fusion card"""
        card = ModernCard("Multi-Source Data Fusion")
        
        desc = QLabel("Fuse multiple data sources for enhanced analysis")
        desc.setStyleSheet("color: #666666; margin-bottom: 15px;")
        card.layout.addWidget(desc)
        
        # Fusion method
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        
        self.fusion_method = QComboBox()
        self.fusion_method.addItems([
            "Weighted Average",
            "Principal Component",
            "Fuzzy Logic",
            "Neural Network",
            "Bayesian"
        ])
        self.fusion_method.setStyleSheet(self.get_combobox_style())
        method_layout.addWidget(self.fusion_method)
        
        card.layout.addLayout(method_layout)
        
        # Normalization
        norm_layout = QHBoxLayout()
        norm_layout.addWidget(QLabel("Normalization:"))
        
        self.norm_method = QComboBox()
        self.norm_method.addItems(["Min-Max", "Z-Score", "Percentile", "Robust"])
        self.norm_method.setStyleSheet(self.get_combobox_style())
        norm_layout.addWidget(self.norm_method)
        
        card.layout.addLayout(norm_layout)
        
        # Fuse button
        fuse_btn = QPushButton("Run Data Fusion")
        fuse_btn.clicked.connect(self.run_data_fusion)
        fuse_btn.setStyleSheet(self.get_primary_button_style())
        card.layout.addWidget(fuse_btn)
        
        return card
    
    def create_results_tab(self):
        """Create results tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Results display
        results_card = ModernCard("Processing Results")
        
        self.results_text = QTextEdit()
        self.results_text.setMinimumHeight(300)
        self.results_text.setStyleSheet("""
            QTextEdit {
                background-color: #fafafa;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                font-family: 'Courier New', monospace;
                font-size: 11px;
                padding: 10px;
            }
        """)
        self.results_text.setPlainText("Processing results will appear here...")
        
        results_card.layout.addWidget(self.results_text)
        layout.addWidget(results_card)
        
        # Export options
        export_card = ModernCard("Export Results")
        
        export_layout = QHBoxLayout()
        
        export_report_btn = QPushButton("Export Report")
        export_report_btn.setStyleSheet(self.get_button_style())
        
        export_data_btn = QPushButton("Export Data")
        export_data_btn.setStyleSheet(self.get_button_style())
        
        clear_btn = QPushButton("Clear Results")
        clear_btn.clicked.connect(self.clear_results)
        clear_btn.setStyleSheet(self.get_button_style())
        
        export_layout.addWidget(export_report_btn)
        export_layout.addWidget(export_data_btn)
        export_layout.addWidget(clear_btn)
        export_layout.addStretch()
        
        export_card.layout.addLayout(export_layout)
        layout.addWidget(export_card)
        
        self.tab_widget.addTab(tab, "Results")
    
    def create_footer(self, layout):
        """Create footer with progress and controls"""
        footer_frame = QFrame()
        footer_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa;
                border-top: 1px solid #e0e0e0;
                padding: 15px;
            }
        """)
        
        footer_layout = QVBoxLayout(footer_frame)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                text-align: center;
                background-color: #f5f5f5;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                border-radius: 7px;
            }
        """)
        footer_layout.addWidget(self.progress_bar)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        help_btn = QPushButton("Help")
        help_btn.clicked.connect(self.show_help)
        help_btn.setStyleSheet(self.get_button_style())
        
        about_btn = QPushButton("About")
        about_btn.clicked.connect(self.show_about)
        about_btn.setStyleSheet(self.get_button_style())
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        close_btn.setStyleSheet(self.get_button_style())
        
        button_layout.addWidget(help_btn)
        button_layout.addWidget(about_btn)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)
        
        footer_layout.addLayout(button_layout)
        layout.addWidget(footer_frame)
    
    def apply_modern_style(self):
        """Apply modern styling to the dialog"""
        self.setStyleSheet("""
            QDialog {
                background-color: #ffffff;
                color: #333333;
            }
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QScrollBar:vertical {
                border: none;
                background: #f1f1f1;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #c1c1c1;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #a8a8a8;
            }
        """)
    
    def get_button_style(self):
        """Get standard button style"""
        return """
            QPushButton {
                background-color: #f8f9fa;
                color: #333333;
                border: 1px solid #dee2e6;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
                min-width: 100px;
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
        """Get primary button style"""
        return """
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:pressed {
                background-color: #1565C0;
            }
        """
    
    def get_combobox_style(self):
        """Get combobox style"""
        return """
            QComboBox {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 6px 10px;
                background-color: white;
                min-width: 150px;
            }
            QComboBox:hover {
                border-color: #2196F3;
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
    
    # File dialog methods
    def browse_aster_file(self):
        """Browse for ASTER file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select ASTER L2 HDF File", "",
            "HDF files (*.hdf *.h5 *.hdf5);;All files (*)"
        )
        if file_path:
            self.aster_file_label.setText(os.path.basename(file_path))
            self.aster_file_path = file_path
    
    def browse_s2_directory(self):
        """Browse for Sentinel-2 directory"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Sentinel-2 Product Directory"
        )
        if not dir_path:
            # Try zip file
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Sentinel-2 ZIP file", "",
                "ZIP files (*.zip);;All files (*)"
            )
            if file_path:
                self.s2_file_label.setText(os.path.basename(file_path))
                self.s2_file_path = file_path
        else:
            self.s2_file_label.setText(os.path.basename(dir_path))
            self.s2_file_path = dir_path
    
    def browse_geological_file(self):
        """Browse for geological file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Geological Data", "",
            "Vector files (*.shp *.gpkg *.geojson);;Raster files (*.tif *.tiff);;All files (*)"
        )
        if file_path:
            self.geo_file_label.setText(os.path.basename(file_path))
            self.geo_file_path = file_path
    
    def refresh_layer_list(self):
        """Refresh the layer list with available raster layers"""
        self.layer_list.clear()
        
        project = QgsProject.instance()
        for layer in project.mapLayers().values():
            if isinstance(layer, QgsRasterLayer) and layer.isValid():
                item = QListWidgetItem(layer.name())
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Unchecked)
                item.setData(Qt.UserRole, layer.id())
                self.layer_list.addItem(item)
    
    # Processing methods
    def process_aster_data(self):
        """Process ASTER data"""
        if not hasattr(self, 'aster_file_path'):
            QMessageBox.warning(self, "Warning", "Please select an ASTER file first.")
            return
        
        try:
            from ..processing.aster_processor import AsterProcessor
            
            self.status_label.setText("Processing ASTER data...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            # Create processor
            processor = AsterProcessor(self.iface)
            
            # Start processing in worker thread
            self.worker = ProcessingWorker(processor, 'process_specific_file', self.aster_file_path)
            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.status.connect(self.status_label.setText)
            self.worker.finished.connect(self.on_processing_finished)
            self.worker.start()
            
        except ImportError as e:
            QMessageBox.critical(self, "Error", f"Failed to import ASTER processor: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start ASTER processing: {str(e)}")
    
    def process_s2_data(self):
        """Process Sentinel-2 data"""
        if not hasattr(self, 's2_file_path'):
            QMessageBox.warning(self, "Warning", "Please select a Sentinel-2 product first.")
            return
        
        try:
            from ..processing.sentinel2_processor import Sentinel2Processor
            
            self.status_label.setText("Processing Sentinel-2 data...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            processor = Sentinel2Processor(self.iface)
            
            self.worker = ProcessingWorker(processor, 'process_specific_product', self.s2_file_path)
            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.status.connect(self.status_label.setText)
            self.worker.finished.connect(self.on_processing_finished)
            self.worker.start()
            
        except ImportError as e:
            QMessageBox.critical(self, "Error", f"Failed to import Sentinel-2 processor: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start Sentinel-2 processing: {str(e)}")
    
    def process_geological_data(self):
        """Process geological data"""
        if not hasattr(self, 'geo_file_path'):
            QMessageBox.warning(self, "Warning", "Please select a geological file first.")
            return
        
        try:
            from ..processing.geological_processor import GeologicalProcessor
            
            self.status_label.setText("Processing geological data...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            processor = GeologicalProcessor(self.iface)
            
            # Determine if it's vector or raster
            file_ext = os.path.splitext(self.geo_file_path)[1].lower()
            if file_ext in ['.shp', '.gpkg', '.geojson']:
                method = 'process_vector_geology'
            else:
                method = 'process_raster_geology'
            
            self.worker = ProcessingWorker(processor, method, self.geo_file_path)
            self.worker.progress.connect(self.progress_bar.setValue)
            self.worker.status.connect(self.status_label.setText)
            self.worker.finished.connect(self.on_processing_finished)
            self.worker.start()
            
        except ImportError as e:
            QMessageBox.critical(self, "Error", f"Failed to import geological processor: {str(e)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start geological processing: {str(e)}")
    
    def run_spectral_analysis(self):
        """Run spectral analysis"""
        # Get selected layers
        selected_layers = []
        project = QgsProject.instance()
        
        for layer in project.mapLayers().values():
            if isinstance(layer, QgsRasterLayer) and layer.isValid():
                selected_layers.append(layer)
        
        if not selected_layers:
            QMessageBox.warning(self, "Warning", "No raster layers available for spectral analysis.")
            return
        
        # Get selected minerals
        selected_minerals = []
        for i in range(self.mineral_list.count()):
            item = self.mineral_list.item(i)
            if item.checkState() == Qt.Checked:
                selected_minerals.append(item.text())
        
        if not selected_minerals:
            QMessageBox.warning(self, "Warning", "Please select at least one target mineral.")
            return
        
        try:
            from ..algorithms.spectral_analysis import SpectralAnalyzer
            
            self.status_label.setText("Running spectral analysis...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            analyzer = SpectralAnalyzer()
            method = self.spectral_method.currentText()
            
            # Process first layer for now
            layer = selected_layers[0]
            if analyzer.load_raster(layer.source()):
                results = analyzer.analyze_spectra(method, selected_minerals)
                self.display_results(f"Spectral Analysis Results:\nMethod: {method}\nMinerals: {', '.join(selected_minerals)}\nResults: {len(results)} maps generated")
            else:
                QMessageBox.critical(self, "Error", "Failed to load raster data for spectral analysis")
            
            self.progress_bar.setValue(100)
            self.status_label.setText("Spectral analysis completed")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Spectral analysis failed: {str(e)}")
            self.status_label.setText("Ready")
            self.progress_bar.setVisible(False)
    
    def generate_prospectivity_map(self):
        """Generate prospectivity map"""
        # Get selected layers
        selected_layer_ids = []
        weights = []
        
        for i in range(self.layer_list.count()):
            item = self.layer_list.item(i)
            if item.checkState() == Qt.Checked:
                layer_id = item.data(Qt.UserRole)
                selected_layer_ids.append(layer_id)
                weights.append(1.0)  # Default equal weights
        
        if len(selected_layer_ids) < 2:
            QMessageBox.warning(self, "Warning", "Please select at least 2 layers for prospectivity mapping.")
            return
        
        try:
            from ..algorithms.prospectivity_mapping import ProspectivityMapper
            
            self.status_label.setText("Generating prospectivity map...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            mapper = ProspectivityMapper()
            method = self.prospect_method.currentText().lower().replace(' ', '_')
            
            if mapper.load_layers(selected_layer_ids, weights):
                prospectivity = mapper.compute_prospectivity(method=method)
                self.display_results(f"Prospectivity Mapping Results:\nMethod: {self.prospect_method.currentText()}\nLayers: {len(selected_layer_ids)}\nMap generated successfully")
            else:
                QMessageBox.critical(self, "Error", "Failed to load layers for prospectivity mapping")
            
            self.progress_bar.setValue(100)
            self.status_label.setText("Prospectivity mapping completed")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Prospectivity mapping failed: {str(e)}")
            self.status_label.setText("Ready")
            self.progress_bar.setVisible(False)
    
    def run_data_fusion(self):
        """Run data fusion"""
        # Get selected layers (same as prospectivity)
        selected_layer_ids = []
        weights = []
        
        for i in range(self.layer_list.count()):
            item = self.layer_list.item(i)
            if item.checkState() == Qt.Checked:
                layer_id = item.data(Qt.UserRole)
                selected_layer_ids.append(layer_id)
                weights.append(1.0)
        
        if len(selected_layer_ids) < 2:
            QMessageBox.warning(self, "Warning", "Please select at least 2 layers for data fusion.")
            return
        
        try:
            from ..algorithms.data_fusion import DataFusionProcessor
            
            self.status_label.setText("Running data fusion...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            
            processor = DataFusionProcessor()
            fusion_method = self.fusion_method.currentText().lower().replace(' ', '_')
            norm_method = self.norm_method.currentText().lower().replace('-', '_')
            
            if processor.load_layers(selected_layer_ids, weights):
                processor.normalize_layers(norm_method)
                result = processor.create_preview(fusion_method)
                self.display_results(f"Data Fusion Results:\nMethod: {self.fusion_method.currentText()}\nNormalization: {self.norm_method.currentText()}\nLayers: {len(selected_layer_ids)}\nFusion completed successfully")
            else:
                QMessageBox.critical(self, "Error", "Failed to load layers for data fusion")
            
            self.progress_bar.setValue(100)
            self.status_label.setText("Data fusion completed")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Data fusion failed: {str(e)}")
            self.status_label.setText("Ready")
            self.progress_bar.setVisible(False)
    
    def on_processing_finished(self, success, message):
        """Handle processing completion"""
        self.progress_bar.setVisible(False)
        
        if success:
            self.status_label.setText("Processing completed")
            self.display_results(message)
            QMessageBox.information(self, "Success", message)
        else:
            self.status_label.setText("Processing failed")
            self.display_results(f"ERROR: {message}")
            QMessageBox.critical(self, "Error", message)
        
        if self.worker:
            self.worker.deleteLater()
            self.worker = None
    
    def display_results(self, text):
        """Display results in the results tab"""
        current_text = self.results_text.toPlainText()
        if current_text.strip() == "Processing results will appear here...":
            self.results_text.setPlainText(text)
        else:
            self.results_text.append(f"\n{'-'*50}\n{text}")
        
        # Switch to results tab
        self.tab_widget.setCurrentIndex(2)
    
    def clear_results(self):
        """Clear results text"""
        self.results_text.setPlainText("Processing results will appear here...")
    
    def show_help(self):
        """Show help dialog"""
        QMessageBox.information(self, "Help", 
            "Mineral Prospectivity Mapping Plugin Help\n\n"
            "1. Data Processing: Process ASTER, Sentinel-2, and geological data\n"
            "2. Analysis: Run spectral analysis, prospectivity mapping, and data fusion\n"
            "3. Results: View processing results and export data\n\n"
            "For detailed documentation, visit the plugin website.")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About", 
            "Mineral Prospectivity Mapping Plugin\n\n"
            "Version: 1.0\n"
            "Advanced geological data processing and analysis for mineral exploration\n\n"
            "Features:\n"
            "• ASTER L2 data processing\n"
            "• Sentinel-2 MSI analysis\n"
            "• Spectral unmixing and mineral mapping\n"
            "• Multi-source data fusion\n"
            "• Prospectivity mapping\n"
            "• Geological data analysis")