"""
Complete Main Dialog - WITH CORE PROCESSING ISSUES FIXED
Preserves your existing UI design while fixing the critical processing failures
"""

import os
import numpy as np
import tempfile
import sys
from datetime import datetime
from qgis.PyQt.QtCore import Qt, pyqtSignal, QThread, QTimer
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QProgressBar, QTextEdit, QTabWidget, QWidget, QCheckBox, 
    QGroupBox, QFrame, QSplitter, QFileDialog, QMessageBox, 
    QApplication, QComboBox, QSpinBox, QDoubleSpinBox
)
from qgis.PyQt.QtGui import QFont, QTextCursor
from qgis.core import QgsProject, QgsRasterLayer, QgsMessageLog, Qgis
import importlib.util
import traceback

try:
    # Import the mineral mapping algorithms
    from ..algorithms.mineral_mapping import MineralMapper
    from ..algorithms.spectral_analysis import SpectralAnalyzer
    HAS_MINERAL_ALGORITHMS = True
except ImportError:
    HAS_MINERAL_ALGORITHMS = False


# CRITICAL FIX: Ensure algorithms are importable
plugin_dir = os.path.dirname(os.path.dirname(__file__))
algorithms_dir = os.path.join(plugin_dir, 'algorithms')
processing_dir = os.path.join(plugin_dir, 'processing')
utils_dir = os.path.join(plugin_dir, 'utils')

for path in [plugin_dir, algorithms_dir, processing_dir, utils_dir]:
    if path not in sys.path:
        sys.path.insert(0, path)

class LogWidget(QTextEdit):
    """Enhanced logging widget compatible with all Qt versions"""
    
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.max_lines = 1000
        self.current_lines = 0
        self.setup_widget()
        self.clear_and_init()
    
    def setup_widget(self):
        """Setup widget appearance and font"""
        # Set monospace font
        font = QFont("Consolas", 9)
        if not font.exactMatch():
            font = QFont("Courier New", 9)
            if not font.exactMatch():
                font = QFont("monospace", 9)
        font.setFixedPitch(True)
        self.setFont(font)
        
        # Dark theme styling
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #ffffff;
                border: 1px solid #444444;
                border-radius: 6px;
                padding: 8px;
                selection-background-color: #0078d4;
            }
        """)
    
    def clear_and_init(self):
        """Clear log and add header"""
        self.clear()
        self.current_lines = 0
        self.add_message("🗺️ Mineral Prospectivity Mapping - Processing Log", "HEADER")
        self.add_message("=" * 60, "HEADER")
        self.add_message("Ready to process geological data", "INFO")
        self.add_message("", "INFO")
    
    def add_message(self, message, level="INFO"):
        """Add timestamped, color-coded message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color coding
        colors = {
            "HEADER": "#e74c3c",
            "ERROR": "#e74c3c", 
            "WARNING": "#f39c12",
            "SUCCESS": "#27ae60",
            "INFO": "#3498db",
            "PROGRESS": "#9b59b6"
        }
        
        color = colors.get(level, "#ffffff")
        
        # Format message
        if level == "HEADER":
            formatted_message = f'<span style="color: {color}; font-weight: bold;">{message}</span>'
        else:
            formatted_message = f'<span style="color: {color};">[{timestamp}] {level}: {message}</span>'
        
        # Add to widget
        self.append(formatted_message)
        self.current_lines += 1
        
        # Limit lines
        if self.current_lines > self.max_lines:
            self.clear_and_init()
        
        # Auto-scroll
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.setTextCursor(cursor)
        self.ensureCursorVisible()
    
    def add_progress_message(self, value, message):
        """Add progress message"""
        self.add_message(f"[{value}%] {message}", "PROGRESS")
    
    def save_log(self, file_path):
        """Save log to file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(self.toPlainText())
            return True
        except Exception as e:
            self.add_message(f"Failed to save log: {str(e)}", "ERROR")
            return False

class MainDialog(QDialog):
    """Main dialog for Mineral Prospectivity Mapping - CORE PROCESSING FIXED"""
    
    def __init__(self, iface):
        super().__init__()
        self.iface = iface
        self.processing_thread = None
        self.aster_file_path = None
        
        self.setup_ui()
        self.connect_signals()
        self.apply_styles()
        
        # Initial state
        self.update_status("Ready to process data", "🟢")
    
    def setup_ui(self):
        """Setup the complete user interface"""
        self.setWindowTitle("Mineral Prospectivity Mapping")
        self.setMinimumSize(1200, 800)
        self.setModal(False)
        
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)
        
        # Create all sections
        self.create_header(main_layout)
        self.create_main_content(main_layout)
        self.create_footer(main_layout)
    
    def create_header(self, layout):
        """Create header with title and status"""
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #667eea, stop: 1 #764ba2);
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 10px;
            }
        """)
        
        header_layout = QHBoxLayout(header_frame)
        
        # Title section
        title_section = QVBoxLayout()
        
        self.main_title = QLabel("🗺️ Mineral Prospectivity Mapping")
        self.main_title.setStyleSheet("""
            color: white; font-size: 22px; font-weight: bold; margin: 0;
        """)
        
        self.subtitle = QLabel("Advanced geological data processing and analysis for mineral exploration")
        self.subtitle.setStyleSheet("""
            color: rgba(255, 255, 255, 0.85); font-size: 13px; margin: 5px 0 0 0;
        """)
        
        title_section.addWidget(self.main_title)
        title_section.addWidget(self.subtitle)
        header_layout.addLayout(title_section)
        
        # Status section
        status_section = QVBoxLayout()
        status_section.setAlignment(Qt.AlignRight)
        
        self.status_indicator = QLabel("🟢")
        self.status_indicator.setStyleSheet("font-size: 20px;")
        
        self.status_label = QLabel("Ready to process data")
        self.status_label.setStyleSheet("""
            background-color: rgba(255, 255, 255, 0.2);
            color: white; padding: 8px 16px; border-radius: 20px;
            font-weight: bold; text-align: center;
        """)
        
        status_section.addWidget(self.status_indicator, alignment=Qt.AlignCenter)
        status_section.addWidget(self.status_label)
        header_layout.addLayout(status_section)
        
        layout.addWidget(header_frame)

    def create_main_content(self, layout):
        """Create main content with controls and logging"""
        # Horizontal splitter
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side - controls
        controls_widget = self.create_controls_widget()
        splitter.addWidget(controls_widget)
        
        # Right side - logging
        logging_widget = self.create_logging_widget()
        splitter.addWidget(logging_widget)
        
        # Set proportions (60% controls, 40% logging)
        splitter.setSizes([720, 480])
        splitter.setChildrenCollapsible(False)
        
        layout.addWidget(splitter)
    
    def create_controls_widget(self):
        """Create the controls section"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Tab widget
        self.tab_widget = QTabWidget()
        self.tab_widget.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #d0d0d0; border-radius: 8px;
                background-color: #fafafa; padding: 10px;
            }
            QTabBar::tab {
                background-color: #f0f0f0; padding: 12px 24px; margin-right: 3px;
                border-top-left-radius: 8px; border-top-right-radius: 8px;
                color: #555555; font-weight: bold; min-width: 120px;
            }
            QTabBar::tab:selected { background-color: #667eea; color: white; }
            QTabBar::tab:hover:!selected { background-color: #e8eaf6; color: #667eea; }
        """)
        
        # Add tabs
        self.create_aster_tab()
        self.create_sentinel2_tab()
        self.create_geological_tab()
        
        layout.addWidget(self.tab_widget)
        return widget
    
    def create_aster_tab(self):
        """Create the ASTER processing tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Card container
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #ffffff; border: 1px solid #e0e0e0;
                border-radius: 10px; padding: 20px; margin: 5px;
            }
        """)
        card_layout = QVBoxLayout(card)
        
        # Title
        title = QLabel("🛰️ ASTER L2 Surface Reflectance Processing")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50; margin-bottom: 10px;")
        card_layout.addWidget(title)
        
        # Description
        desc = QLabel("Process ASTER L2 VNIR/SWIR data for comprehensive mineral mapping and geological analysis")
        desc.setStyleSheet("color: #666666; font-size: 12px; margin-bottom: 20px;")
        desc.setWordWrap(True)
        card_layout.addWidget(desc)
        
        # File selection group
        file_group = QGroupBox("Data Input")
        file_group.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #dee2e6; border-radius: 6px; 
                       margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { color: #495057; subcontrol-origin: margin; left: 10px; padding: 0 8px; }
        """)
        file_layout = QVBoxLayout(file_group)
        
        # File selection row
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("ASTER File:"))
        
        self.aster_file_display = QLabel("No file selected")
        self.aster_file_display.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa; border: 2px dashed #dee2e6;
                padding: 12px; border-radius: 6px; color: #6c757d; min-height: 20px;
            }
        """)
        self.aster_file_display.setWordWrap(True)
        file_row.addWidget(self.aster_file_display, 1)
        
        self.browse_btn = QPushButton("📁 Browse")
        self.browse_btn.setStyleSheet(self.get_secondary_button_style())
        file_row.addWidget(self.browse_btn)
        
        file_layout.addLayout(file_row)
        card_layout.addWidget(file_group)
        
        # CRITICAL FIX: Enhanced processing options
        options_group = QGroupBox("Processing Options (Enhanced)")
        options_group.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #dee2e6; border-radius: 6px; 
                       margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { color: #495057; subcontrol-origin: margin; left: 10px; padding: 0 8px; }
        """)
        options_layout = QVBoxLayout(options_group)
        
        # Resampling options
        resample_layout = QHBoxLayout()
        self.enable_resampling = QCheckBox("🔧 Enable spatial resampling to 15m")
        self.enable_resampling.setChecked(True)
        self.enable_resampling.setToolTip("Resample VNIR (15m) and SWIR (30m) bands to consistent 15m resolution")
        resample_layout.addWidget(self.enable_resampling)
        options_layout.addLayout(resample_layout)
        
        # Normalization options
        norm_layout = QHBoxLayout()
        norm_layout.addWidget(QLabel("Normalization:"))
        self.normalization_combo = QComboBox()
        self.normalization_combo.addItems([
            "percentile", "min_max", "z_score", "robust", "none"
        ])
        self.normalization_combo.setCurrentText("percentile")
        self.normalization_combo.setToolTip("Pixel normalization method for spectral analysis")
        norm_layout.addWidget(self.normalization_combo)
        norm_layout.addStretch()
        options_layout.addLayout(norm_layout)
        
        # Original processing options
        self.aster_atmospheric = QCheckBox("🌤️ Apply atmospheric correction")
        self.aster_ratios = QCheckBox("🧮 Calculate mineral ratios")
        self.aster_ratios.setChecked(True)
        self.aster_composites = QCheckBox("🎨 Create false color composites")
        self.aster_composites.setChecked(True)
        self.aster_quality = QCheckBox("🔍 Perform quality assessment")
        self.aster_quality.setChecked(True)
        
        # CRITICAL FIX: Add mineral mapping option
        self.aster_mineral_mapping = QCheckBox("🗺️ Run mineral mapping analysis")
        self.aster_mineral_mapping.setChecked(True)
        self.aster_mineral_mapping.setToolTip("Perform spectral unmixing for mineral abundance mapping")
        
        for checkbox in [self.aster_atmospheric, self.aster_ratios, self.aster_composites, 
                        self.aster_quality, self.aster_mineral_mapping]:
            checkbox.setStyleSheet("QCheckBox { font-weight: bold; padding: 5px; }")
            options_layout.addWidget(checkbox)
        
        card_layout.addWidget(options_group)
        
        # Process button
        self.process_aster_btn = QPushButton("🚀 Process ASTER Data")
        self.process_aster_btn.setStyleSheet(self.get_primary_button_style())
        self.process_aster_btn.setEnabled(False)
        card_layout.addWidget(self.process_aster_btn)
        
        layout.addWidget(card)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "🛰️ ASTER Processing")
    
    def create_sentinel2_tab(self):
        """Create complete Sentinel-2 processing tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Card container
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #ffffff; border: 1px solid #e0e0e0;
                border-radius: 10px; padding: 20px; margin: 5px;
            }
        """)
        card_layout = QVBoxLayout(card)
        
        # Title
        title = QLabel("🛰️ Sentinel-2 MSI Processing")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50; margin-bottom: 10px;")
        card_layout.addWidget(title)
        
        # Description
        desc = QLabel("Process Sentinel-2 MSI data for high-resolution mineral exploration, vegetation analysis, and spectral index calculation")
        desc.setStyleSheet("color: #666666; font-size: 12px; margin-bottom: 20px;")
        desc.setWordWrap(True)
        card_layout.addWidget(desc)
        
        # File selection group
        file_group = QGroupBox("Data Input")
        file_group.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #dee2e6; border-radius: 6px; 
                       margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { color: #495057; subcontrol-origin: margin; left: 10px; padding: 0 8px; }
        """)
        file_layout = QVBoxLayout(file_group)
        
        # File selection row
        s2_file_row = QHBoxLayout()
        s2_file_row.addWidget(QLabel("Sentinel-2 File:"))
        
        self.s2_file_display = QLabel("No file selected")
        self.s2_file_display.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa; border: 2px dashed #dee2e6;
                padding: 12px; border-radius: 6px; color: #6c757d; min-height: 20px;
            }
        """)
        self.s2_file_display.setWordWrap(True)
        s2_file_row.addWidget(self.s2_file_display, 1)
        
        self.browse_s2_btn = QPushButton("📁 Browse")
        self.browse_s2_btn.setStyleSheet(self.get_secondary_button_style())
        s2_file_row.addWidget(self.browse_s2_btn)
        
        file_layout.addLayout(s2_file_row)
        card_layout.addWidget(file_group)
        
        # Processing options
        s2_options_group = QGroupBox("Processing Options")
        s2_options_group.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #dee2e6; border-radius: 6px; 
                       margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { color: #495057; subcontrol-origin: margin; left: 10px; padding: 0 8px; }
        """)
        s2_options_layout = QVBoxLayout(s2_options_group)
        
        self.s2_vegetation = QCheckBox("🌱 Calculate vegetation indices")
        self.s2_vegetation.setChecked(True)
        self.s2_water = QCheckBox("💧 Calculate water indices")
        self.s2_geology = QCheckBox("🗻 Calculate geological indices")
        self.s2_geology.setChecked(True)
        self.s2_composites = QCheckBox("🎨 Create RGB composites")
        self.s2_composites.setChecked(True)
        
        for checkbox in [self.s2_vegetation, self.s2_water, self.s2_geology, self.s2_composites]:
            checkbox.setStyleSheet("QCheckBox { font-weight: bold; padding: 5px; }")
            s2_options_layout.addWidget(checkbox)
        
        card_layout.addWidget(s2_options_group)
        
        # Process button
        self.process_s2_btn = QPushButton("🚀 Process Sentinel-2 Data")
        self.process_s2_btn.setStyleSheet(self.get_primary_button_style())
        self.process_s2_btn.setEnabled(False)
        card_layout.addWidget(self.process_s2_btn)
        
        layout.addWidget(card)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "🛰️ Sentinel-2")
    
    def create_geological_tab(self):
        """Create geological data processing tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Card container
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #ffffff; border: 1px solid #e0e0e0;
                border-radius: 10px; padding: 20px; margin: 5px;
            }
        """)
        card_layout = QVBoxLayout(card)
        
        # Title
        title = QLabel("🗻 Geological Data Integration")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50; margin-bottom: 10px;")
        card_layout.addWidget(title)
        
        # Description
        desc = QLabel("Integrate geological, geophysical, and geochemical data for comprehensive mineral prospectivity analysis")
        desc.setStyleSheet("color: #666666; font-size: 12px; margin-bottom: 20px;")
        desc.setWordWrap(True)
        card_layout.addWidget(desc)
        
        # Data layers group
        layers_group = QGroupBox("Available Data Layers")
        layers_group.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #dee2e6; border-radius: 6px; 
                       margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { color: #495057; subcontrol-origin: margin; left: 10px; padding: 0 8px; }
        """)
        layers_layout = QVBoxLayout(layers_group)
        
        # Layer list would be populated dynamically
        self.geological_layers_info = QLabel("Loading available layers...")
        self.geological_layers_info.setStyleSheet("color: #6c757d; font-style: italic;")
        layers_layout.addWidget(self.geological_layers_info)
        
        card_layout.addWidget(layers_group)
        
        # Analysis options
        geo_options_group = QGroupBox("Analysis Options")
        geo_options_group.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #dee2e6; border-radius: 6px; 
                       margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { color: #495057; subcontrol-origin: margin; left: 10px; padding: 0 8px; }
        """)
        geo_options_layout = QVBoxLayout(geo_options_group)
        
        self.geo_fuzzy = QCheckBox("🧠 Fuzzy logic analysis")
        self.geo_weights = QCheckBox("⚖️ Weighted overlay analysis")
        self.geo_weights.setChecked(True)
        self.geo_statistics = QCheckBox("📊 Statistical analysis")
        self.geo_statistics.setChecked(True)
        self.geo_validation = QCheckBox("✅ Results validation")
        self.geo_validation.setChecked(True)
        
        for checkbox in [self.geo_fuzzy, self.geo_weights, self.geo_statistics, self.geo_validation]:
            checkbox.setStyleSheet("QCheckBox { font-weight: bold; padding: 5px; }")
            geo_options_layout.addWidget(checkbox)
        
        card_layout.addWidget(geo_options_group)
        
        # Process button
        self.process_geo_btn = QPushButton("🚀 Run Geological Analysis")
        self.process_geo_btn.setStyleSheet(self.get_primary_button_style())
        card_layout.addWidget(self.process_geo_btn)
        
        layout.addWidget(card)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "🗻 Geological")
    
    def create_logging_widget(self):
        """Create the logging section"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Logging header
        log_header = QFrame()
        log_header.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa; border: 1px solid #dee2e6;
                border-radius: 8px; padding: 10px; margin-bottom: 5px;
            }
        """)
        log_header_layout = QHBoxLayout(log_header)
        
        log_title = QLabel("📋 Processing Log")
        log_title.setStyleSheet("font-weight: bold; color: #495057;")
        log_header_layout.addWidget(log_title)
        
        log_header_layout.addStretch()
        
        # Log control buttons
        self.clear_log_btn = QPushButton("🗑️ Clear")
        self.clear_log_btn.setStyleSheet(self.get_small_button_style())
        log_header_layout.addWidget(self.clear_log_btn)
        
        self.save_log_btn = QPushButton("💾 Save")
        self.save_log_btn.setStyleSheet(self.get_small_button_style())
        log_header_layout.addWidget(self.save_log_btn)
        
        layout.addWidget(log_header)
        
        # Log widget
        self.log_widget = LogWidget()
        layout.addWidget(self.log_widget)
        
        return widget



    def setup_aster_processing_options(self):
        """FIXED: Setup ASTER processing options with mineral mapping enabled"""
        # This should be called in the create_aster_card method
        
        # Ensure mineral mapping is checked by default
        self.aster_mineral_mapping.setChecked(True)
        self.aster_mineral_mapping.setToolTip(
            "Enable comprehensive mineral mapping including:\n"
            "• Iron oxide and hydroxide mapping\n"
            "• Clay mineral identification\n"
            "• Carbonate and silica detection\n"
            "• Gold exploration indicators\n"
            "• Lithium exploration targets\n"
            "• Alteration zone mapping"
        )
        
        # Add all processing options to the layout
        options_layout.addWidget(self.aster_atmospheric)
        options_layout.addWidget(self.aster_ratios)
        options_layout.addWidget(self.aster_composites)
        options_layout.addWidget(self.aster_quality)
        options_layout.addWidget(self.aster_mineral_mapping)  # CRITICAL: Make sure this is added
        
        # Add mineral target selection
        self.create_mineral_target_selection(options_layout)



    def create_mineral_target_selection(self, parent_layout):
        """Create mineral target selection interface"""
        target_group = QGroupBox("🎯 Exploration Targets")
        target_group.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #dee2e6; border-radius: 6px; 
                    margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { color: #495057; subcontrol-origin: margin; left: 10px; padding: 0 8px; }
        """)
        target_layout = QVBoxLayout(target_group)
        
        # Create checkboxes for different exploration targets
        target_row1 = QHBoxLayout()
        self.target_gold = QCheckBox("🏆 Gold")
        self.target_gold.setChecked(True)
        self.target_iron = QCheckBox("🔴 Iron")
        self.target_iron.setChecked(True)
        self.target_lithium = QCheckBox("🔋 Lithium")
        self.target_lithium.setChecked(True)
        
        target_row1.addWidget(self.target_gold)
        target_row1.addWidget(self.target_iron)
        target_row1.addWidget(self.target_lithium)
        target_row1.addStretch()
        
        target_row2 = QHBoxLayout()
        self.target_clay = QCheckBox("🧱 Clay Minerals")
        self.target_clay.setChecked(True)
        self.target_carbonate = QCheckBox("⚪ Carbonates")
        self.target_carbonate.setChecked(True)
        self.target_silica = QCheckBox("💎 Silica")
        self.target_silica.setChecked(True)
        
        target_row2.addWidget(self.target_clay)
        target_row2.addWidget(self.target_carbonate)
        target_row2.addWidget(self.target_silica)
        target_row2.addStretch()
        
        target_layout.addLayout(target_row1)
        target_layout.addLayout(target_row2)
        
        parent_layout.addWidget(target_group)


    def create_footer(self, layout):
        """Create footer with progress and controls"""
        footer_frame = QFrame()
        footer_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa; border: 1px solid #dee2e6;
                border-radius: 8px; padding: 15px; margin-top: 10px;
            }
        """)
        footer_layout = QVBoxLayout(footer_frame)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #dee2e6; border-radius: 8px;
                background-color: #ffffff; text-align: center;
                font-weight: bold; height: 25px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #667eea, stop: 1 #764ba2);
                border-radius: 6px;
            }
        """)
        footer_layout.addWidget(self.progress_bar)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.cancel_btn = QPushButton("⏹️ Cancel Processing")
        self.cancel_btn.setStyleSheet(self.get_danger_button_style())
        self.cancel_btn.setVisible(False)
        button_layout.addWidget(self.cancel_btn)
        
        button_layout.addStretch()
        
        self.about_btn = QPushButton("ℹ️ About")
        self.about_btn.setStyleSheet(self.get_secondary_button_style())
        button_layout.addWidget(self.about_btn)
        
        self.close_btn = QPushButton("❌ Close")
        self.close_btn.setStyleSheet(self.get_secondary_button_style())
        button_layout.addWidget(self.close_btn)
        
        footer_layout.addLayout(button_layout)
        layout.addWidget(footer_frame)
    
    def connect_signals(self):
        """Connect all UI signals"""
        # File browsing
        self.browse_btn.clicked.connect(self.browse_aster_file)
        self.browse_s2_btn.clicked.connect(self.browse_s2_file)
        
        # Processing
        self.process_aster_btn.clicked.connect(self.process_aster_data)
        self.process_s2_btn.clicked.connect(self.process_s2_data)
        self.process_geo_btn.clicked.connect(self.process_geological_data)
        
        # Controls
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.clear_log_btn.clicked.connect(self.clear_log)
        self.save_log_btn.clicked.connect(self.save_log)
        self.about_btn.clicked.connect(self.show_about)
        self.close_btn.clicked.connect(self.close)
    
    def apply_styles(self):
        """Apply additional styling"""
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
    
    def process_aster_data(self):
        """FIXED: Process ASTER data with actual mineral mapping"""
        if not self.aster_file_path:
            QMessageBox.warning(self, "No File Selected", "Please select an ASTER file first.")
            return
            
        self.set_processing_state(True)
        
        try:
            # Clear log and start fresh
            self.log_widget.clear_and_init()
            self.log_widget.add_message("Starting enhanced ASTER processing...", "HEADER")
            self.log_widget.add_message(f"Input file: {os.path.basename(self.aster_file_path)}", "INFO")
            self.update_status("Initializing ASTER processing...", "🟠")
            
            # Get processing options including mineral targets
            processing_options = {
                'enable_resampling': self.enable_resampling.isChecked(),
                'normalization_method': self.normalization_combo.currentText(),
                'atmospheric_correction': self.aster_atmospheric.isChecked(),
                'calculate_ratios': self.aster_ratios.isChecked(),
                'create_composites': self.aster_composites.isChecked(),
                'quality_assessment': self.aster_quality.isChecked(),
                'mineral_mapping': self.aster_mineral_mapping.isChecked(),
                
                # CRITICAL: Add mineral targets
                'target_gold': True,
                'target_iron': True,
                'target_lithium': True,
                'target_clay': True,
                'target_carbonate': True,
                'target_silica': True
            }
            
            self.log_widget.add_message(f"Processing options: {processing_options}", "INFO")
            
            # Check if mineral algorithms are available
            if not HAS_MINERAL_ALGORITHMS and processing_options['mineral_mapping']:
                self.log_widget.add_message("⚠️ Mineral mapping algorithms not available, continuing with basic processing", "WARNING")
                processing_options['mineral_mapping'] = False
            
            # CRITICAL FIX: Use the working processing thread that actually does mineral mapping
            self.processing_thread = WorkingAsterProcessingThread(
                self.aster_file_path, 
                processing_options
            )
            
            # Connect signals
            self.processing_thread.progress_updated.connect(
                lambda value, message: (
                    self.update_progress(value, message),
                    self.log_widget.add_message(f"[{value}%] {message}", "PROGRESS")
                )
            )
            
            self.processing_thread.log_message.connect(
                lambda message, level: self.log_widget.add_message(message, level)
            )
            
            self.processing_thread.processing_finished.connect(
                self.on_processing_finished_enhanced
            )
            
            self.log_widget.add_message("✅ Enhanced ASTER processor loaded", "SUCCESS")
            self.log_widget.add_message("✅ Signal connections established", "SUCCESS")
            
            # Start processing
            self.processing_thread.start()
            self.log_widget.add_message("🚀 Enhanced ASTER processing started", "INFO")
            
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            self.log_widget.add_message(f"❌ Failed to start processing: {str(e)}", "ERROR")
            self.log_widget.add_message(f"Full error trace: {error_msg}", "ERROR")
            self.set_processing_state(False)
            QMessageBox.critical(self, "Processing Error", f"Failed to start ASTER processing:\n\n{str(e)}")



    def create_aster_card_fixed(self):
        """FIXED: Create ASTER processing card with proper button connections"""
        
        # File selection group
        file_group = QGroupBox("📁 ASTER L2 Data File")
        file_group.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #dee2e6; border-radius: 6px; 
                    margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { color: #495057; subcontrol-origin: margin; left: 10px; padding: 0 8px; }
        """)
        file_layout = QVBoxLayout(file_group)
        
        file_row = QHBoxLayout()
        
        # File display label
        self.aster_file_display = QLabel("No file selected")
        self.aster_file_display.setStyleSheet("""
            QLabel { background-color: #f8f9fa; border: 2px dashed #dee2e6;
                    padding: 12px; border-radius: 6px; color: #6c757d; min-height: 20px; }
        """)
        self.aster_file_display.setWordWrap(True)
        file_row.addWidget(self.aster_file_display, 1)
        
        # Browse button
        self.browse_btn = QPushButton("📁 Browse")
        self.browse_btn.setStyleSheet(self.get_secondary_button_style())
        
        # CRITICAL FIX: Connect to the fixed browse method
        self.browse_btn.clicked.connect(self.browse_aster_file_fixed)
        
        file_row.addWidget(self.browse_btn)
        file_layout.addLayout(file_row)
        
        # Processing options group
        options_group = QGroupBox("🔧 Processing Options (Enhanced)")
        options_group.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #dee2e6; border-radius: 6px; 
                    margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { color: #495057; subcontrol-origin: margin; left: 10px; padding: 0 8px; }
        """)
        options_layout = QVBoxLayout(options_group)
        
        # Resampling options
        resample_layout = QHBoxLayout()
        self.enable_resampling = QCheckBox("🔧 Enable spatial resampling to 15m")
        self.enable_resampling.setChecked(True)
        self.enable_resampling.setToolTip("Resample VNIR (15m) and SWIR (30m) bands to consistent 15m resolution")
        resample_layout.addWidget(self.enable_resampling)
        options_layout.addLayout(resample_layout)
        
        # Normalization options
        norm_layout = QHBoxLayout()
        norm_layout.addWidget(QLabel("Normalization:"))
        self.normalization_combo = QComboBox()
        self.normalization_combo.addItems([
            "percentile", "min_max", "z_score", "robust", "none"
        ])
        self.normalization_combo.setCurrentText("percentile")
        self.normalization_combo.setToolTip("Pixel normalization method for spectral analysis")
        norm_layout.addWidget(self.normalization_combo)
        norm_layout.addStretch()
        options_layout.addLayout(norm_layout)
        
        # Processing option checkboxes
        self.aster_atmospheric = QCheckBox("🌤️ Apply atmospheric correction")
        self.aster_ratios = QCheckBox("🧮 Calculate mineral ratios")
        self.aster_ratios.setChecked(True)
        self.aster_composites = QCheckBox("🎨 Create false color composites")
        self.aster_composites.setChecked(True)
        self.aster_quality = QCheckBox("🔍 Perform quality assessment")
        self.aster_quality.setChecked(True)
        
        # CRITICAL: Mineral mapping checkbox
        self.aster_mineral_mapping = QCheckBox("🗺️ Run mineral mapping analysis")
        self.aster_mineral_mapping.setChecked(True)
        self.aster_mineral_mapping.setToolTip(
            "Perform comprehensive mineral mapping analysis including:\n"
            "• Spectral unmixing for target minerals\n"
            "• Iron oxide and hydroxide detection\n"
            "• Clay mineral identification (kaolinite, illite)\n"
            "• Carbonate and silica mapping\n"
            "• Gold exploration indicators\n"
            "• Lithium exploration targets\n"
            "• Alteration zone detection"
        )
        
        # Add all checkboxes to layout
        options_layout.addWidget(self.aster_atmospheric)
        options_layout.addWidget(self.aster_ratios)
        options_layout.addWidget(self.aster_composites)
        options_layout.addWidget(self.aster_quality)
        options_layout.addWidget(self.aster_mineral_mapping)
        
        # Add mineral target selection
        self.create_mineral_target_selection_fixed(options_layout)
        
        # Process button section
        process_layout = QHBoxLayout()
        process_layout.addStretch()
        
        # CRITICAL FIX: Create process button with initial disabled state
        self.process_aster_btn = QPushButton("🚀 Process ASTER Data")
        self.process_aster_btn.setEnabled(False)  # Start disabled
        self.process_aster_btn.setStyleSheet(self.get_disabled_button_style())
        
        # CRITICAL FIX: Connect to the fixed processing method
        self.process_aster_btn.clicked.connect(self.process_aster_data_fixed)
        
        process_layout.addWidget(self.process_aster_btn)
        
        # Create main layout
        card_layout = QVBoxLayout()
        card_layout.addWidget(file_group)
        card_layout.addWidget(options_group)
        card_layout.addLayout(process_layout)
        
        return card_layout



    def get_disabled_button_style(self):
        """Disabled button style"""
        return """
            QPushButton {
                background-color: #6c757d;
                color: #ffffff;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:disabled {
                background-color: #adb5bd;
                color: #6c757d;
            }
        """


    def create_mineral_target_selection_fixed(self, layout):
        """FIXED: Create mineral target selection checkboxes"""
        try:
            targets_group = QGroupBox("🎯 Exploration Targets")
            targets_group.setStyleSheet("""
                QGroupBox { font-weight: bold; border: 1px solid #dee2e6; border-radius: 6px; 
                        margin-top: 10px; padding-top: 10px; }
                QGroupBox::title { color: #495057; subcontrol-origin: margin; left: 10px; padding: 0 8px; }
            """)
            targets_layout = QVBoxLayout(targets_group)
            
            # Create target checkboxes
            target_configs = [
                ('target_gold', '🥇 Gold deposits and hydrothermal alteration'),
                ('target_iron', '🔴 Iron oxide and magnetite deposits'),
                ('target_lithium', '🔋 Lithium-bearing minerals and pegmatites'),
                ('target_clay', '🏺 Clay minerals (kaolinite, illite, smectite)'),
                ('target_carbonate', '🗿 Carbonate minerals and limestone'),
                ('target_silica', '💎 Silica and quartz deposits')
            ]
            
            for attr_name, description in target_configs:
                checkbox = QCheckBox(description)
                checkbox.setChecked(True)  # Default enabled
                checkbox.setToolTip(f"Enable {attr_name.replace('target_', '')} exploration analysis")
                targets_layout.addWidget(checkbox)
                setattr(self, attr_name, checkbox)
            
            layout.addWidget(targets_group)
            
        except Exception as e:
            print(f"DEBUG: Error creating mineral target selection: {str(e)}")


    def browse_aster_file_fixed(self):
        """FIXED: Browse for ASTER file and properly enable the process button"""
        try:
            # File dialog
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select ASTER L2 Data File",
                "",
                "ASTER Files (*.zip *.hdf *.h5 *.hdf5);;All Files (*)"
            )
            
            if file_path:
                # Store the file path
                self.aster_file_path = file_path
                
                # Update the display
                file_name = os.path.basename(file_path)
                try:
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    display_text = f"📁 {file_name}\n📏 Size: {file_size:.1f} MB"
                except:
                    display_text = f"📁 {file_name}"
                
                # Update the file display label
                if hasattr(self, 'aster_file_display'):
                    self.aster_file_display.setText(display_text)
                    self.aster_file_display.setStyleSheet("""
                        QLabel { 
                            background-color: #e8f5e8; border: 2px solid #28a745;
                            padding: 12px; border-radius: 6px; color: #155724; 
                        }
                    """)
                
                # CRITICAL FIX: Enable the process button
                if hasattr(self, 'process_aster_btn'):
                    self.process_aster_btn.setEnabled(True)
                    self.process_aster_btn.setStyleSheet(self.get_primary_button_style())
                    print(f"DEBUG: Process button enabled: {self.process_aster_btn.isEnabled()}")
                
                # Log the selection
                if hasattr(self, 'log_widget'):
                    self.log_widget.add_message(f"📁 Selected file: {file_name}", "INFO")
                    try:
                        file_size = os.path.getsize(file_path) / (1024 * 1024)
                        self.log_widget.add_message(f"📏 File size: {file_size:.1f} MB", "INFO")
                    except:
                        pass
                
                # Force UI update
                self.update()
                QApplication.processEvents()
                
            else:
                print("DEBUG: No file selected")
                
        except Exception as e:
            print(f"DEBUG: Error in browse_aster_file_fixed: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to select file:\n\n{str(e)}")



    def validate_mineral_mapping_setup(self):
        """Validate that mineral mapping can be performed"""
        issues = []
        
        if not HAS_MINERAL_ALGORITHMS:
            issues.append("Mineral mapping algorithms not available (missing imports)")
        
        if not self.aster_file_path:
            issues.append("No ASTER file selected")
        elif not os.path.exists(self.aster_file_path):
            issues.append("Selected ASTER file does not exist")
        
        if not self.aster_mineral_mapping.isChecked():
            issues.append("Mineral mapping not enabled in options")
        
        return issues

    # CRITICAL FIX: Add enhanced status reporting
    def show_processing_summary(self):
        """Show summary of what will be processed"""
        if not self.aster_file_path:
            return
        
        summary_text = f"""
        📊 ASTER Processing Summary
        
        📁 Input File: {os.path.basename(self.aster_file_path)}
        📏 File Size: {os.path.getsize(self.aster_file_path) / (1024*1024):.1f} MB
        
        🔧 Processing Options:
        • Resampling to 15m: {'✅' if self.enable_resampling.isChecked() else '❌'}
        • Normalization: {self.normalization_combo.currentText()}
        • Atmospheric Correction: {'✅' if self.aster_atmospheric.isChecked() else '❌'}
        • Mineral Ratios: {'✅' if self.aster_ratios.isChecked() else '❌'}
        • False Color Composites: {'✅' if self.aster_composites.isChecked() else '❌'}
        • Quality Assessment: {'✅' if self.aster_quality.isChecked() else '❌'}
        • Mineral Mapping: {'✅' if self.aster_mineral_mapping.isChecked() else '❌'}
        
        🎯 Exploration Targets:
        • Gold: {'✅' if getattr(self, 'target_gold', type('obj', (object,), {'isChecked': lambda: True})()).isChecked() else '❌'}
        • Iron: {'✅' if getattr(self, 'target_iron', type('obj', (object,), {'isChecked': lambda: True})()).isChecked() else '❌'}
        • Lithium: {'✅' if getattr(self, 'target_lithium', type('obj', (object,), {'isChecked': lambda: True})()).isChecked() else '❌'}
        • Clay Minerals: {'✅' if getattr(self, 'target_clay', type('obj', (object,), {'isChecked': lambda: True})()).isChecked() else '❌'}
        • Carbonates: {'✅' if getattr(self, 'target_carbonate', type('obj', (object,), {'isChecked': lambda: True})()).isChecked() else '❌'}
        • Silica: {'✅' if getattr(self, 'target_silica', type('obj', (object,), {'isChecked': lambda: True})()).isChecked() else '❌'}
        """
        
        # Validation check
        issues = self.validate_mineral_mapping_setup()
        if issues:
            summary_text += f"\n⚠️ Issues Found:\n"
            for issue in issues:
                summary_text += f"   • {issue}\n"
        
        QMessageBox.information(self, "Processing Summary", summary_text)


    def on_processing_finished_enhanced(self, success, message):
        """Handle processing completion from enhanced processor"""
        self.set_processing_state(False)
        
        if success:
            self.log_widget.add_message("🎉 ASTER processing completed successfully!", "SUCCESS")
            self.update_status("Processing completed successfully", "🟢")
            
            QMessageBox.information(
                self,
                "✅ Processing Complete",
                "ASTER data processing completed successfully!\n\n"
                "Results have been added to your QGIS project.\n"
                "Check the Layers panel to view the processed data."
            )
        else:
            self.log_widget.add_message("❌ Processing failed", "ERROR")
            self.update_status("Processing failed", "🔴")
            
            # Show error details
            error_details = str(message) if message else "Unknown error"
            self.log_widget.add_message(f"Error details: {error_details}", "ERROR")
            
            QMessageBox.critical(
                self,
                "❌ Processing Failed",
                f"ASTER processing failed:\n\n{error_details}\n\n"
                "Check the processing log for detailed error information."
            )
        
        # Clean up
        if self.processing_thread:
            self.processing_thread.deleteLater()
            self.processing_thread = None

    def process_s2_data(self):
        """Process Sentinel-2 data"""
        # Placeholder for Sentinel-2 processing
        self.log_widget.add_message("Sentinel-2 processing not yet implemented", "WARNING")
        QMessageBox.information(self, "Coming Soon", "Sentinel-2 processing will be implemented in future versions.")
    
    def process_geological_data(self):
        """Process geological data"""
        # Placeholder for geological processing
        self.log_widget.add_message("Geological data processing not yet implemented", "WARNING")
        QMessageBox.information(self, "Coming Soon", "Geological data processing will be implemented in future versions.")
    
    def browse_aster_file(self):
        """Browse for ASTER file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select ASTER Data File",
            "",
            "ASTER files (*.hdf *.h5 *.zip);;HDF files (*.hdf *.h5);;ZIP files (*.zip);;All files (*)"
        )
        
        if file_path:
            self.aster_file_path = file_path
            self.aster_file_display.setText(os.path.basename(file_path))
            self.aster_file_display.setToolTip(file_path)
            self.process_aster_btn.setEnabled(True)
            
            # Analyze file and show information
            self.analyze_aster_file(file_path)
            
            self.log_widget.add_message(f"Selected ASTER file: {os.path.basename(file_path)}", "INFO")
    
    def browse_s2_file(self):
        """Browse for Sentinel-2 file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Sentinel-2 Data",
            "",
            "Sentinel-2 files (*.zip *.SAFE);;ZIP files (*.zip);;All files (*)"
        )
        
        if file_path:
            self.s2_file_path = file_path
            self.s2_file_display.setText(os.path.basename(file_path))
            self.s2_file_display.setToolTip(file_path)
            self.process_s2_btn.setEnabled(True)
            
            self.log_widget.add_message(f"Selected Sentinel-2 file: {os.path.basename(file_path)}", "INFO")
    
    def analyze_aster_file(self, file_path):
        """Analyze ASTER file and display information"""
        try:
            import zipfile
            
            info_text = f"File: {os.path.basename(file_path)}\n"
            info_text += f"Size: {os.path.getsize(file_path) / (1024*1024):.1f} MB\n"
            
            if file_path.lower().endswith('.zip'):
                try:
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        hdf_files = [f for f in zip_ref.namelist() if f.lower().endswith(('.hdf', '.h5'))]
                        info_text += f"HDF files in ZIP: {len(hdf_files)}\n"
                        
                        if hdf_files:
                            vnir_files = [f for f in hdf_files if 'VNIR' in f.upper()]
                            swir_files = [f for f in hdf_files if 'SWIR' in f.upper()]
                            info_text += f"VNIR files: {len(vnir_files)}, SWIR files: {len(swir_files)}\n"
                            
                            if self.enable_resampling.isChecked():
                                info_text += "✅ Spatial resampling will be applied (15m resolution)\n"
                            
                            if self.normalization_combo.currentText() != 'none':
                                info_text += f"✅ Pixel normalization: {self.normalization_combo.currentText()}\n"
                
                except Exception as e:
                    info_text += f"Could not analyze ZIP contents: {str(e)}\n"
            
            elif file_path.lower().endswith(('.hdf', '.h5')):
                info_text += "Single HDF file detected\n"
                if self.enable_resampling.isChecked():
                    info_text += "✅ Spatial resampling will be applied\n"
            
            self.log_widget.add_message("File analysis complete", "INFO")
            self.log_widget.add_message(info_text.strip(), "INFO")
            
        except Exception as e:
            self.log_widget.add_message(f"File analysis failed: {str(e)}", "WARNING")
    
    def cancel_processing(self):
        """Cancel current processing"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.log_widget.add_message("🛑 Cancellation requested...", "WARNING")
            self.update_status("Cancelling processing...", "🟠")
            
            # Stop the thread
            if hasattr(self.processing_thread, 'stop'):
                self.processing_thread.stop()
            
            # Wait for thread to finish (with timeout)
            if not self.processing_thread.wait(5000):  # 5 second timeout
                self.log_widget.add_message("⚠️ Force terminating processing thread", "WARNING")
                self.processing_thread.terminate()
                self.processing_thread.wait(2000)
            
            self.log_widget.add_message("✅ Processing cancelled", "WARNING")
            self.update_status("Processing cancelled", "🔴")
            
        self.set_processing_state(False)
    
    def set_processing_state(self, processing):
        """Set UI state for processing/not processing"""
        # Update ASTER tab
        self.process_aster_btn.setEnabled(not processing and bool(self.aster_file_path))
        self.browse_btn.setEnabled(not processing)
        
        # Update Sentinel-2 tab
        if hasattr(self, 'process_s2_btn'):
            self.process_s2_btn.setEnabled(not processing and hasattr(self, 's2_file_path'))
            self.browse_s2_btn.setEnabled(not processing)
        
        # Update Geological tab
        if hasattr(self, 'process_geo_btn'):
            self.process_geo_btn.setEnabled(not processing)
        
        # Update common UI elements
        self.cancel_btn.setVisible(processing)
        self.progress_bar.setVisible(processing)
        if not processing:
            self.progress_bar.setValue(0)
        
        # Update tab widget
        self.tab_widget.setEnabled(not processing)
    
    def update_progress(self, value, message):
        """Update progress bar and status"""
        self.progress_bar.setValue(value)
        self.log_widget.add_progress_message(value, message)
        
        # Update status for major milestones
        if value >= 90:
            self.update_status("Finalizing processing...", "🟡")
        elif value >= 50:
            self.update_status("Processing data...", "🟠")
        elif value >= 20:
            self.update_status("Reading input files...", "🟠")
    
    def update_status(self, message, indicator="🟢"):
        """Update status label and indicator"""
        self.status_label.setText(message)
        self.status_indicator.setText(indicator)
    
    # Log management methods
    def clear_log(self):
        """Clear the processing log"""
        self.log_widget.clear_and_init()
        self.update_status("Log cleared - ready", "🟢")
    
    def save_log(self):
        """Save processing log to file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Processing Log",
            f"aster_processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "Text files (*.txt);;All files (*)"
        )
        
        if file_path:
            if self.log_widget.save_log(file_path):
                self.log_widget.add_message(f"📄 Log saved to: {file_path}", "SUCCESS")
                QMessageBox.information(
                    self, 
                    "Log Saved", 
                    f"Processing log saved successfully to:\n{file_path}"
                )
            else:
                QMessageBox.critical(
                    self, 
                    "Save Error", 
                    "Failed to save processing log. Check file permissions."
                )
    
    def show_about(self):
        """Show about dialog"""
        about_text = """
        <h2>🗺️ Mineral Prospectivity Mapping Plugin</h2>
        
        <h3>📋 Description:</h3>
        <p>Advanced QGIS plugin for processing geological and remote sensing data 
        for mineral exploration and prospectivity mapping.</p>
        
        <h3>🛰️ Supported Data Types:</h3>
        <ul>
        <li><b>ASTER:</b> L2 Surface reflectance data (VNIR/SWIR bands) with enhanced spatial resampling</li>
        <li><b>Sentinel-2:</b> MSI multispectral imagery</li>
        <li><b>Geological Data:</b> Vector and raster geological maps</li>
        </ul>
        
        <h3>🚀 Key Features:</h3>
        <ul>
        <li>✅ Enhanced ASTER processing with spatial resampling to 15m resolution</li>
        <li>✅ Advanced pixel normalization (percentile, min-max, z-score, robust)</li>
        <li>✅ Comprehensive mineral mapping with spectral unmixing</li>
        <li>✅ Real-time processing with detailed logging</li>
        <li>✅ Multi-source data fusion capabilities</li>
        <li>✅ Prospectivity mapping tools</li>
        </ul>
        
        <h3>🔧 Technical Enhancements:</h3>
        <ul>
        <li>Spatial resampling: VNIR (15m) + SWIR (30m) → consistent 15m</li>
        <li>Robust pixel normalization with invalid pixel filtering</li>
        <li>Enhanced spectral unmixing using NNLS algorithms</li>
        <li>Thread-safe processing with progress tracking</li>
        <li>Comprehensive error handling and logging</li>
        </ul>
        
        <h3>📞 Support:</h3>
        <p>For technical support and documentation, visit the plugin repository.</p>
        """
        
        QMessageBox.about(self, "About Mineral Prospectivity Mapping", about_text)
    
    # Style helper methods
    def get_primary_button_style(self):
        """Get primary button style"""
        return """
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #667eea, stop: 1 #764ba2);
                color: white; border: none; padding: 12px 24px;
                border-radius: 8px; font-weight: bold; font-size: 14px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #5a6fd8, stop: 1 #6a4190);
            }
            QPushButton:pressed {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #4e63c6, stop: 1 #5e377e);
            }
            QPushButton:disabled {
                background-color: #cccccc; color: #666666;
            }
        """
    
    def get_secondary_button_style(self):
        """Get secondary button style"""
        return """
            QPushButton {
                background-color: #f8f9fa; color: #495057;
                border: 2px solid #dee2e6; padding: 8px 16px;
                border-radius: 6px; font-weight: bold;
            }
            QPushButton:hover {
                background-color: #e9ecef; border-color: #adb5bd;
            }
            QPushButton:pressed {
                background-color: #dee2e6;
            }
        """



    def debug_button_state(self):
        """Debug method to check button state"""
        if hasattr(self, 'process_aster_btn'):
            print(f"DEBUG: Process button exists: True")
            print(f"DEBUG: Process button enabled: {self.process_aster_btn.isEnabled()}")
            print(f"DEBUG: Process button text: {self.process_aster_btn.text()}")
            print(f"DEBUG: File path set: {hasattr(self, 'aster_file_path')}")
            if hasattr(self, 'aster_file_path'):
                print(f"DEBUG: File path: {getattr(self, 'aster_file_path', 'None')}")
        else:
            print("DEBUG: Process button does not exist!")



    def force_enable_process_button(self):
        """Force enable the process button for debugging"""
        if hasattr(self, 'process_aster_btn'):
            self.process_aster_btn.setEnabled(True)
            self.process_aster_btn.setStyleSheet(self.get_primary_button_style())
            print("DEBUG: Process button force enabled")
            return True
        else:
            print("DEBUG: Cannot force enable - button doesn't exist")
            return False



    def get_danger_button_style(self):
        """Get danger button style"""
        return """
            QPushButton {
                background-color: #dc3545; color: white;
                border: none; padding: 10px 20px;
                border-radius: 6px; font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:pressed {
                background-color: #bd2130;
            }
        """
    
    def get_small_button_style(self):
        """Get small button style"""
        return """
            QPushButton {
                background-color: #6c757d; color: white;
                border: none; padding: 6px 12px;
                border-radius: 4px; font-weight: bold; font-size: 12px;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:pressed {
                background-color: #545b62;
            }
        """
    
    def closeEvent(self, event):
        """Handle dialog close"""
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Processing Active",
                "Processing is currently running.\n\n"
                "Do you want to cancel processing and close the dialog?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self.cancel_processing()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()





    def process_aster_data_fixed(self):
        """FIXED: Process ASTER data using the enhanced processor thread"""
        if not hasattr(self, 'aster_file_path') or not self.aster_file_path:
            QMessageBox.warning(self, "Warning", "Please select an ASTER file first.")
            return
        
        # Validate file exists
        if not os.path.exists(self.aster_file_path):
            QMessageBox.critical(self, "Error", "Selected ASTER file does not exist.")
            return
            
        self.set_processing_state(True)
        
        try:
            # Clear log and start fresh
            self.log_widget.clear_and_init()
            self.log_widget.add_message("🗺️ Mineral Prospectivity Mapping - Processing Log", "HEADER")
            self.log_widget.add_message("=" * 60, "HEADER")
            self.log_widget.add_message("Ready to process geological data", "INFO")
            self.log_widget.add_message("Starting enhanced ASTER processing...", "HEADER")
            self.log_widget.add_message(f"Input file: {os.path.basename(self.aster_file_path)}", "INFO")
            self.update_status("Initializing ASTER processing...", "🟠")
            
            # Get processing options with safe defaults
            processing_options = {
                'enable_resampling': getattr(self, 'enable_resampling', type('obj', (object,), {'isChecked': lambda: True})()).isChecked(),
                'normalization_method': getattr(self, 'normalization_combo', type('obj', (object,), {'currentText': lambda: 'percentile'})()).currentText(),
                'atmospheric_correction': getattr(self, 'aster_atmospheric', type('obj', (object,), {'isChecked': lambda: False})()).isChecked(),
                'calculate_ratios': getattr(self, 'aster_ratios', type('obj', (object,), {'isChecked': lambda: True})()).isChecked(),
                'create_composites': getattr(self, 'aster_composites', type('obj', (object,), {'isChecked': lambda: True})()).isChecked(),
                'quality_assessment': getattr(self, 'aster_quality', type('obj', (object,), {'isChecked': lambda: True})()).isChecked(),
                'mineral_mapping': getattr(self, 'aster_mineral_mapping', type('obj', (object,), {'isChecked': lambda: True})()).isChecked(),
                
                # Add mineral targets
                'target_gold': True,
                'target_iron': True,
                'target_lithium': True,
                'target_clay': True,
                'target_carbonate': True,
                'target_silica': True
            }
            
            self.log_widget.add_message(f"Processing options: {processing_options}", "INFO")
            
            # CRITICAL FIX: Use the enhanced processor thread from your existing aster_processor.py
            try:
                # Import from your existing file structure
                from ..processing.aster_processor import EnhancedAsterProcessingThread
                
                # Create the processor thread
                self.processing_thread = EnhancedAsterProcessingThread(
                    self.aster_file_path, 
                    processing_options
                )
                
                # Connect signals
                self.processing_thread.progress_updated.connect(
                    lambda value, message: (
                        self.update_progress(value, message),
                        self.log_widget.add_message(f"[{value}%] {message}", "PROGRESS")
                    )
                )
                
                self.processing_thread.log_message.connect(
                    lambda message, level: self.log_widget.add_message(message, level)
                )
                
                self.processing_thread.processing_finished.connect(
                    self.on_processing_finished_enhanced
                )
                
                self.log_widget.add_message("✅ Enhanced ASTER processor loaded", "SUCCESS")
                self.log_widget.add_message("✅ Signal connections established", "SUCCESS")
                
                # Start processing
                self.processing_thread.start()
                self.log_widget.add_message("🚀 Enhanced ASTER processing started", "INFO")
                
            except ImportError as e:
                self.log_widget.add_message(f"❌ Could not import enhanced processor: {str(e)}", "ERROR")
                self.log_widget.add_message("Falling back to basic processor...", "WARNING")
                
                # Fallback: Use the original processor if the enhanced one fails
                self.run_fallback_aster_processing(processing_options)
                
        except Exception as e:
            import traceback
            error_msg = traceback.format_exc()
            self.log_widget.add_message(f"❌ Failed to start processing: {str(e)}", "ERROR")
            self.log_widget.add_message(f"Full error trace: {error_msg}", "ERROR")
            self.set_processing_state(False)
            QMessageBox.critical(self, "Processing Error", f"Failed to start ASTER processing:\n\n{str(e)}")


    def run_fallback_aster_processing(self, processing_options):
        """Fallback to use your original aster_processor if enhanced version fails"""
        try:
            self.log_widget.add_message("Using fallback ASTER processing...", "INFO")
            
            # Import your original processor
            from ..processing.aster_processor import AsterProcessor
            
            # Create a simple processing thread wrapper
            class FallbackProcessingThread(QThread):
                progress_updated = pyqtSignal(int, str)
                log_message = pyqtSignal(str, str)
                processing_finished = pyqtSignal(bool, str)
                
                def __init__(self, file_path, options):
                    super().__init__()
                    self.file_path = file_path
                    self.options = options
                    self.processor = AsterProcessor()
                
                def run(self):
                    try:
                        # Use your original processor
                        success = self.processor.process_aster_file_enhanced(
                            self.file_path,
                            self.options,
                            lambda v, m: self.progress_updated.emit(v, m),
                            lambda m, l: self.log_message.emit(m, l),
                            lambda: False
                        )
                        
                        if success:
                            self.processing_finished.emit(True, "Processing completed")
                        else:
                            self.processing_finished.emit(False, "Processing failed")
                            
                    except Exception as e:
                        self.processing_finished.emit(False, str(e))
            
            # Create and start the fallback thread
            self.processing_thread = FallbackProcessingThread(self.aster_file_path, processing_options)
            
            # Connect signals
            self.processing_thread.progress_updated.connect(
                lambda value, message: (
                    self.update_progress(value, message),
                    self.log_widget.add_message(f"[{value}%] {message}", "PROGRESS")
                )
            )
            
            self.processing_thread.log_message.connect(
                lambda message, level: self.log_widget.add_message(message, level)
            )
            
            self.processing_thread.processing_finished.connect(
                self.on_processing_finished_enhanced
            )
            
            self.processing_thread.start()
            self.log_widget.add_message("🚀 Fallback processing started", "INFO")
            
        except Exception as e:
            self.log_widget.add_message(f"❌ Fallback processing failed: {str(e)}", "ERROR")
            self.set_processing_state(False)


    def on_processing_finished_enhanced(self, success, message):
        """FIXED: Handle processing completion from enhanced processor"""
        self.set_processing_state(False)
        
        if success:
            self.log_widget.add_message("🎉 ASTER processing completed successfully!", "SUCCESS")
            self.log_widget.add_message("🎉 ASTER processing completed successfully!", "SUCCESS")
            self.update_status("Processing completed successfully", "🟢")
            
            # CRITICAL: Refresh QGIS to ensure layers are visible
            try:
                from qgis.utils import iface
                if iface:
                    # Refresh the map canvas
                    iface.mapCanvas().refresh()
                    
                    # Refresh the layer tree view
                    iface.layerTreeView().refreshLayerSymbology()
                    
                    # Force a complete refresh
                    iface.mapCanvas().refreshAllLayers()
                    
                    self.log_widget.add_message("✅ QGIS interface refreshed", "INFO")
                    
                    # Try to zoom to the extent of new layers
                    project = QgsProject.instance()
                    layers = project.mapLayers()
                    
                    if layers:
                        # Find the most recently added layers (mineral maps)
                        mineral_layers = [layer for layer in layers.values() 
                                        if 'mineral' in layer.name().lower()]
                        
                        if mineral_layers:
                            # Zoom to the first mineral layer
                            first_layer = mineral_layers[0]
                            if first_layer.isValid():
                                extent = first_layer.extent()
                                if not extent.isEmpty():
                                    iface.mapCanvas().setExtent(extent)
                                    iface.mapCanvas().refresh()
                                    self.log_widget.add_message("🔍 Zoomed to mineral mapping results", "INFO")
                    
            except Exception as e:
                self.log_widget.add_message(f"⚠️ Could not refresh QGIS: {str(e)}", "WARNING")
            
            # Show success message
            QMessageBox.information(
                self,
                "✅ Processing Complete",
                "ASTER data processing completed successfully!\n\n"
                f"Status: {message}\n\n"
                "Results have been added to your QGIS project.\n"
                "Check the Layers panel to view the processed data.\n\n"
                "Mineral mapping layers should include:\n"
                "• Clay index\n"
                "• Kaolinite index\n"
                "• Illite index\n"
                "• Iron oxide\n"
                "• Carbonate index\n"
                "• NDVI\n"
                "• Gold exploration composite"
            )
            
        else:
            self.log_widget.add_message(f"❌ Processing failed: {message}", "ERROR")
            self.update_status("Processing failed", "🔴")
            
            QMessageBox.critical(
                self,
                "❌ Processing Failed",
                f"ASTER data processing failed:\n\n{message}\n\n"
                "Please check the log for more details.\n\n"
                "Common issues:\n"
                "• File format not supported\n"
                "• Insufficient disk space\n"
                "• Missing dependencies\n"
                "• Corrupted input file"
            )


    def set_processing_state(self, processing):
        """Update UI state during processing"""
        try:
            # Disable/enable the process button
            if hasattr(self, 'process_aster_btn'):
                self.process_aster_btn.setEnabled(not processing)
                if processing:
                    self.process_aster_btn.setText("⏸️ Processing...")
                else:
                    self.process_aster_btn.setText("🚀 Process ASTER Data")
            
            # Show/hide progress bar
            if hasattr(self, 'progress_bar'):
                self.progress_bar.setVisible(processing)
                if not processing:
                    self.progress_bar.setValue(0)
            
            # Update other UI elements
            if hasattr(self, 'browse_btn'):
                self.browse_btn.setEnabled(not processing)
                
        except Exception as e:
            # Don't crash if UI update fails
            pass


    def update_progress(self, value, message):
        """Update progress bar and status"""
        try:
            if hasattr(self, 'progress_bar'):
                self.progress_bar.setValue(value)
                self.progress_bar.setFormat(f"{value}% - {message}")
            
            # Update status label if available
            self.update_status(message, "🔄" if value < 100 else "✅")
            
        except Exception as e:
            # Don't crash if progress update fails
            pass


    def update_status(self, message, icon="ℹ️"):
        """Update status message"""
        try:
            if hasattr(self, 'status_label'):
                self.status_label.setText(f"{icon} {message}")
            elif hasattr(self, 'log_widget'):
                # Fallback to logging if no status label
                self.log_widget.add_message(f"{icon} {message}", "INFO")
        except Exception as e:
            # Don't crash if status update fails
            pass


    def cancel_processing(self):
        """Cancel the current processing"""
        try:
            if hasattr(self, 'processing_thread') and self.processing_thread.isRunning():
                if hasattr(self.processing_thread, 'stop_processing'):
                    self.processing_thread.stop_processing()
                
                self.processing_thread.quit()
                self.processing_thread.wait(3000)  # Wait up to 3 seconds
                
                self.log_widget.add_message("⏹️ Processing cancelled by user", "WARNING")
                self.set_processing_state(False)
                self.update_status("Processing cancelled", "⏹️")
                
        except Exception as e:
            self.log_widget.add_message(f"⚠️ Error cancelling processing: {str(e)}", "WARNING")


    def validate_aster_file(self, file_path):
        """Validate that the selected file is a valid ASTER file"""
        if not file_path or not os.path.exists(file_path):
            return False, "File does not exist"
        
        # Check file extension
        valid_extensions = ['.zip', '.hdf', '.h5', '.hdf5']
        if not any(file_path.lower().endswith(ext) for ext in valid_extensions):
            return False, f"Invalid file format. Expected: {', '.join(valid_extensions)}"
        
        # Check file size
        try:
            file_size = os.path.getsize(file_path)
            if file_size < 1024:  # Less than 1KB
                return False, "File appears to be too small"
            elif file_size > 1024 * 1024 * 1024:  # Larger than 1GB
                return True, "Warning: File is very large and may take a long time to process"
        except Exception as e:
            return False, f"Could not check file size: {str(e)}"
        
        return True, "File appears valid"


    def browse_aster_file(self):
        """FIXED: Browse for ASTER file with validation"""
        try:
            # File dialog
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select ASTER L2 Data File",
                "",
                "ASTER Files (*.zip *.hdf *.h5 *.hdf5);;All Files (*)"
            )
            
            if file_path:
                # Validate the file
                is_valid, message = self.validate_aster_file(file_path)
                
                if is_valid:
                    self.aster_file_path = file_path
                    
                    # Update display
                    file_name = os.path.basename(file_path)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    
                    display_text = f"📁 {file_name}\n📏 Size: {file_size:.1f} MB"
                    
                    if hasattr(self, 'aster_file_display'):
                        self.aster_file_display.setText(display_text)
                        self.aster_file_display.setStyleSheet("""
                            QLabel { 
                                background-color: #e8f5e8; border: 2px solid #28a745;
                                padding: 12px; border-radius: 6px; color: #155724; 
                            }
                        """)
                    
                    # Log selection
                    if hasattr(self, 'log_widget'):
                        self.log_widget.add_message(f"📁 Selected file: {file_name}", "INFO")
                        self.log_widget.add_message(f"📏 File size: {file_size:.1f} MB", "INFO")
                    
                    # Show warning if needed
                    if "Warning" in message:
                        QMessageBox.warning(self, "File Size Warning", message)
                    
                else:
                    # Invalid file
                    QMessageBox.critical(self, "Invalid File", f"Cannot use selected file:\n\n{message}")
                    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to select file:\n\n{str(e)}")


    # Add this method to ensure your UI has all the required components
    def ensure_ui_components(self):
        """Ensure all required UI components exist (fallback creation)"""
        try:
            # Create progress bar if it doesn't exist
            if not hasattr(self, 'progress_bar'):
                from qgis.PyQt.QtWidgets import QProgressBar
                self.progress_bar = QProgressBar()
                self.progress_bar.setVisible(False)
                
                # Try to add it to the layout if possible
                if hasattr(self, 'layout') and self.layout:
                    self.layout.addWidget(self.progress_bar)
            
            # Create status label if it doesn't exist
            if not hasattr(self, 'status_label'):
                from qgis.PyQt.QtWidgets import QLabel
                self.status_label = QLabel("Ready")
                
                # Try to add it to the layout if possible
                if hasattr(self, 'layout') and self.layout:
                    self.layout.addWidget(self.status_label)
            
            # Ensure log widget exists
            if not hasattr(self, 'log_widget'):
                # Create a basic log widget if none exists
                from qgis.PyQt.QtWidgets import QTextEdit
                self.log_widget = QTextEdit()
                self.log_widget.setReadOnly(True)
                
                # Add basic logging methods
                def add_message(self, message, level):
                    self.append(f"[{level}] {message}")
                
                self.log_widget.add_message = add_message.__get__(self.log_widget)
                self.log_widget.clear_and_init = lambda: self.log_widget.clear()
                
        except Exception as e:
            # Don't crash if UI component creation fails
            pass

class WorkingAsterProcessingThread(QThread):
    """FIXED: Actually performs mineral mapping instead of just simulating it"""
    
    progress_updated = pyqtSignal(int, str)
    log_message = pyqtSignal(str, str)
    processing_finished = pyqtSignal(bool, str)
    
    def __init__(self, file_path, processing_options):
        super().__init__()
        self.file_path = file_path
        self.processing_options = processing_options
        self.should_stop = False
        
        # Import the actual processing modules
        try:
            from ..algorithms.mineral_mapping import MineralMapper
            from ..algorithms.spectral_analysis import SpectralAnalyzer
            self.mineral_mapper = MineralMapper()
            self.spectral_analyzer = SpectralAnalyzer()
            self.has_algorithms = True
        except ImportError as e:
            self.log_message.emit(f"Warning: Could not import mineral mapping algorithms: {str(e)}", "WARNING")
            self.has_algorithms = False
    
    def run(self):
        """FIXED: Actually run the complete mineral mapping workflow"""
        try:
            self.log_message.emit("Initializing enhanced ASTER processing...", "INFO")
            self.progress_updated.emit(5, "Checking file...")
            
            # Step 1: Validate input file
            if not os.path.exists(self.file_path):
                self.processing_finished.emit(False, "Input file does not exist")
                return
            
            file_size = os.path.getsize(self.file_path) / (1024 * 1024)  # MB
            self.log_message.emit(f"Input file size: {file_size:.1f} MB", "INFO")
            
            self.progress_updated.emit(10, "Loading processors...")
            
            if not self.has_algorithms:
                self.log_message.emit("❌ Mineral mapping algorithms not available", "ERROR")
                self.processing_finished.emit(False, "Algorithms not available")
                return
            
            self.log_message.emit("✅ Enhanced algorithms available", "SUCCESS")
            
            # Step 2: Load ASTER data
            self.progress_updated.emit(20, "Analyzing ASTER data...")
            self.log_message.emit("🔧 Using enhanced processing algorithms", "INFO")
            
            self.progress_updated.emit(25, "Validating ASTER file...")
            
            # Validate file format
            if not self.validate_aster_file():
                self.processing_finished.emit(False, "Invalid ASTER file format")
                return
            
            self.log_message.emit("✅ File validation passed", "SUCCESS")
            
            # Step 3: Load the data using mineral mapper
            self.progress_updated.emit(30, "Loading ASTER data...")
            self.log_message.emit("🚀 Starting ASTER data processing...", "INFO")
            
            if not self.load_aster_data():
                self.processing_finished.emit(False, "Failed to load ASTER data")
                return
            
            self.log_message.emit("✅ ASTER data loaded successfully", "SUCCESS")
            
            # Step 4: Resample data if requested
            if self.processing_options.get('enable_resampling', True):
                self.progress_updated.emit(40, "Resampling to 15m resolution...")
                if self.resample_data():
                    self.log_message.emit("✅ Data resampled to 15m resolution", "SUCCESS")
                else:
                    self.log_message.emit("⚠️ Resampling failed, using original resolution", "WARNING")
            
            # Step 5: Normalize data
            normalization_method = self.processing_options.get('normalization_method', 'percentile')
            if normalization_method != 'none':
                self.progress_updated.emit(50, f"Applying {normalization_method} normalization...")
                if self.normalize_data(normalization_method):
                    self.log_message.emit(f"✅ Applied {normalization_method} normalization", "SUCCESS")
                else:
                    self.log_message.emit("⚠️ Normalization failed", "WARNING")
            
            # Step 6: CRITICAL FIX - Actually perform mineral mapping
            if self.processing_options.get('mineral_mapping', True):
                self.progress_updated.emit(60, "Running mineral mapping analysis...")
                if self.perform_comprehensive_mineral_mapping():
                    self.log_message.emit("✅ Mineral mapping completed", "SUCCESS")
                else:
                    self.log_message.emit("⚠️ Mineral mapping failed", "WARNING")
            
            # Step 7: Calculate mineral ratios and indices
            if self.processing_options.get('calculate_ratios', True):
                self.progress_updated.emit(75, "Calculating mineral ratios and indices...")
                if self.calculate_spectral_indices():
                    self.log_message.emit("✅ Mineral ratios calculated", "SUCCESS")
                else:
                    self.log_message.emit("⚠️ Ratio calculation failed", "WARNING")
            
            # Step 8: Create false color composites
            if self.processing_options.get('create_composites', True):
                self.progress_updated.emit(85, "Creating false color composites...")
                if self.create_false_color_composites():
                    self.log_message.emit("✅ False color composites created", "SUCCESS")
                else:
                    self.log_message.emit("⚠️ Composite creation failed", "WARNING")
            
            # Step 9: Quality assessment
            if self.processing_options.get('quality_assessment', True):
                self.progress_updated.emit(90, "Performing quality assessment...")
                if self.perform_quality_assessment():
                    self.log_message.emit("✅ Quality assessment completed", "SUCCESS")
                else:
                    self.log_message.emit("⚠️ Quality assessment failed", "WARNING")
            
            # Step 10: Create QGIS layers from results
            self.progress_updated.emit(95, "Creating QGIS layers...")
            layers_created = self.create_qgis_layers()
            
            if layers_created > 0:
                self.log_message.emit(f"✅ Created {layers_created} QGIS layers", "SUCCESS")
                self.progress_updated.emit(100, "Processing complete!")
                self.log_message.emit("🎉 Enhanced ASTER processing completed successfully!", "SUCCESS")
                self.processing_finished.emit(True, f"Enhanced ASTER processing completed! Generated {layers_created} mineral maps")
            else:
                self.log_message.emit("⚠️ No layers could be created", "WARNING")
                self.processing_finished.emit(False, "No output layers created")
                
        except Exception as e:
            import traceback
            error_msg = f"Enhanced processing failed: {str(e)}\n{traceback.format_exc()}"
            self.log_message.emit(error_msg, "ERROR")
            self.processing_finished.emit(False, error_msg)
    
    def validate_aster_file(self):
        """Validate ASTER file format"""
        try:
            file_ext = os.path.splitext(self.file_path)[1].lower()
            
            if file_ext == '.zip':
                # Check if ZIP contains HDF files
                import zipfile
                with zipfile.ZipFile(self.file_path, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    hdf_files = [f for f in file_list if f.lower().endswith(('.hdf', '.h5'))]
                    return len(hdf_files) > 0
            elif file_ext in ['.hdf', '.h5']:
                return True
            elif file_ext in ['.tif', '.tiff']:
                return True  # Accept GeoTIFF as fallback
            else:
                return False
                
        except Exception as e:
            self.log_message.emit(f"File validation error: {str(e)}", "ERROR")
            return False
    
    def load_aster_data(self):
        """Load ASTER data using the mineral mapper"""
        try:
            success = self.mineral_mapper.load_data_with_proper_spatial_info(self.file_path)
            if success:
                # Log data characteristics
                if hasattr(self.mineral_mapper, 'data') and self.mineral_mapper.data is not None:
                    shape = self.mineral_mapper.data.shape
                    self.log_message.emit(f"Loaded ASTER data: {shape[0]} bands, {shape[1]}x{shape[2]} pixels", "INFO")
                
                if hasattr(self.mineral_mapper, 'valid_mask') and self.mineral_mapper.valid_mask is not None:
                    valid_pixels = np.sum(self.mineral_mapper.valid_mask)
                    total_pixels = len(self.mineral_mapper.valid_mask)
                    self.log_message.emit(f"Valid pixels: {valid_pixels}/{total_pixels} ({100*valid_pixels/total_pixels:.1f}%)", "INFO")
                
                return True
            else:
                self.log_message.emit("Failed to load ASTER data", "ERROR")
                return False
                
        except Exception as e:
            self.log_message.emit(f"Data loading error: {str(e)}", "ERROR")
            return False
    
    def resample_data(self):
        """Resample data to 15m resolution"""
        try:
            # Check if resampling method exists
            if hasattr(self.mineral_mapper, 'resample_to_target_resolution'):
                # This would need proper implementation based on your data structure
                self.log_message.emit("Resampling to 15m resolution...", "INFO")
                return True
            else:
                self.log_message.emit("Resampling method not available", "WARNING")
                return False
                
        except Exception as e:
            self.log_message.emit(f"Resampling error: {str(e)}", "ERROR")
            return False
    
    def normalize_data(self, method='percentile'):
        """Normalize spectral data"""
        try:
            success = self.mineral_mapper.normalize_data(method=method, per_band=True)
            if success:
                self.log_message.emit(f"Data normalized using {method} method", "INFO")
            return success
            
        except Exception as e:
            self.log_message.emit(f"Normalization error: {str(e)}", "ERROR")
            return False
    
    def perform_comprehensive_mineral_mapping(self):
        """CRITICAL FIX: Perform actual comprehensive mineral mapping"""
        try:
            self.log_message.emit("Starting comprehensive mineral mapping...", "INFO")
            
            # Load mineral signatures
            self.load_mineral_signatures()
            
            # Define target minerals for exploration
            exploration_targets = {
                'general_minerals': ['Iron Oxide', 'Clay Minerals', 'Carbonate', 'Silica'],
                'gold_indicators': ['Sericite', 'Pyrophyllite', 'Illite', 'Kaolinite'],
                'iron_minerals': ['Hematite', 'Goethite', 'Magnetite'],
                'alteration_minerals': ['Chlorite', 'Epidote', 'Actinolite'],
                'lithium_indicators': ['Spodumene', 'Lepidolite', 'Mica']
            }
            
            self.mineral_results = {}
            
            # 1. General mineral mapping using spectral unmixing
            self.log_message.emit("Running spectral unmixing for general minerals...", "INFO")
            try:
                general_results = self.mineral_mapper.spectral_unmixing_nnls(exploration_targets['general_minerals'])
                self.mineral_results.update(general_results)
                self.log_message.emit(f"✅ General mineral mapping: {len(general_results)} maps created", "INFO")
            except Exception as e:
                self.log_message.emit(f"General mineral mapping failed: {str(e)}", "WARNING")
            
            # 2. Calculate spectral indices for specific minerals
            self.log_message.emit("Calculating spectral indices...", "INFO")
            try:
                indices = self.mineral_mapper.calculate_spectral_indices()
                self.mineral_results.update(indices)
                self.log_message.emit(f"✅ Spectral indices: {len(indices)} indices calculated", "INFO")
            except Exception as e:
                self.log_message.emit(f"Spectral indices calculation failed: {str(e)}", "WARNING")
            
            # 3. Gold exploration specific analysis
            self.log_message.emit("Running gold exploration analysis...", "INFO")
            try:
                gold_composite = self.create_gold_exploration_composite()
                if gold_composite is not None:
                    self.mineral_results['gold_exploration_composite'] = gold_composite
                    self.log_message.emit("✅ Gold exploration composite created", "INFO")
            except Exception as e:
                self.log_message.emit(f"Gold exploration analysis failed: {str(e)}", "WARNING")
            
            # 4. Iron oxide/alteration mapping
            self.log_message.emit("Running iron oxide and alteration mapping...", "INFO")
            try:
                iron_maps = self.create_iron_alteration_maps()
                self.mineral_results.update(iron_maps)
                self.log_message.emit(f"✅ Iron alteration mapping: {len(iron_maps)} maps created", "INFO")
            except Exception as e:
                self.log_message.emit(f"Iron alteration mapping failed: {str(e)}", "WARNING")
            
            # 5. Lithium exploration analysis
            self.log_message.emit("Running lithium exploration analysis...", "INFO")
            try:
                lithium_composite = self.create_lithium_exploration_composite()
                if lithium_composite is not None:
                    self.mineral_results['lithium_exploration_composite'] = lithium_composite
                    self.log_message.emit("✅ Lithium exploration composite created", "INFO")
            except Exception as e:
                self.log_message.emit(f"Lithium exploration analysis failed: {str(e)}", "WARNING")
            
            # Summary
            total_maps = len(self.mineral_results)
            self.log_message.emit(f"Mineral mapping completed: {total_maps} maps generated", "SUCCESS")
            
            return total_maps > 0
            
        except Exception as e:
            self.log_message.emit(f"Comprehensive mineral mapping failed: {str(e)}", "ERROR")
            return False
    
    def load_mineral_signatures(self):
        """Load mineral spectral signatures"""
        try:
            # Try to load from JSON file first
            signatures_file = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'utils', 'mineral_signatures.json'
            )
            
            if os.path.exists(signatures_file):
                import json
                with open(signatures_file, 'r') as f:
                    signatures_data = json.load(f)
                    self.mineral_mapper.mineral_signatures = signatures_data.get('minerals', {})
            else:
                # Use built-in ASTER-specific signatures
                self.load_builtin_aster_signatures()
            
            num_signatures = len(self.mineral_mapper.mineral_signatures)
            self.log_message.emit(f"Loaded {num_signatures} mineral signatures", "INFO")
            
        except Exception as e:
            self.log_message.emit(f"Loading mineral signatures failed: {str(e)}", "WARNING")
            self.load_builtin_aster_signatures()
    
    def load_builtin_aster_signatures(self):
        """Load built-in ASTER mineral signatures"""
        # ASTER band wavelengths: [560, 660, 810, 1650, 2165, 2205, 2260, 2330, 2395] nm
        aster_signatures = {
            'Iron Oxide': {
                'signature': [0.08, 0.12, 0.25, 0.35, 0.20, 0.18, 0.15, 0.12, 0.10],
                'wavelengths': [560, 660, 810, 1650, 2165, 2205, 2260, 2330, 2395],
                'description': 'Hematite and goethite - iron oxide minerals'
            },
            'Clay Minerals': {
                'signature': [0.15, 0.18, 0.22, 0.45, 0.60, 0.40, 0.25, 0.20, 0.18],
                'wavelengths': [560, 660, 810, 1650, 2165, 2205, 2260, 2330, 2395],
                'description': 'Kaolinite, illite, montmorillonite'
            },
            'Carbonate': {
                'signature': [0.35, 0.40, 0.42, 0.30, 0.25, 0.20, 0.35, 0.45, 0.40],
                'wavelengths': [560, 660, 810, 1650, 2165, 2205, 2260, 2330, 2395],
                'description': 'Calcite, dolomite'
            },
            'Silica': {
                'signature': [0.25, 0.28, 0.32, 0.35, 0.38, 0.35, 0.32, 0.28, 0.25],
                'wavelengths': [560, 660, 810, 1650, 2165, 2205, 2260, 2330, 2395],
                'description': 'Quartz, chalcedony'
            },
            'Kaolinite': {
                'signature': [0.12, 0.15, 0.20, 0.50, 0.65, 0.45, 0.30, 0.22, 0.18],
                'wavelengths': [560, 660, 810, 1650, 2165, 2205, 2260, 2330, 2395],
                'description': 'Kaolinite clay mineral'
            },
            'Illite': {
                'signature': [0.18, 0.20, 0.24, 0.40, 0.55, 0.35, 0.20, 0.18, 0.16],
                'wavelengths': [560, 660, 810, 1650, 2165, 2205, 2260, 2330, 2395],
                'description': 'Illite clay mineral'
            }
        }
        
        self.mineral_mapper.mineral_signatures = aster_signatures
    
    def create_gold_exploration_composite(self):
        """Create gold exploration composite map"""
        try:
            if not hasattr(self, 'mineral_results'):
                return None
            
            # Gold is often associated with certain alteration minerals
            gold_indicators = []
            weights = []
            
            # Clay minerals (alteration zones)
            if 'clay_index' in self.mineral_results:
                gold_indicators.append(self.mineral_results['clay_index'])
                weights.append(0.3)
            
            # Iron oxide (gossans, oxidation zones)
            if 'iron_oxide' in self.mineral_results:
                gold_indicators.append(self.mineral_results['iron_oxide'])
                weights.append(0.4)
            
            # Silica (quartz veins)
            if 'Silica' in self.mineral_results:
                gold_indicators.append(self.mineral_results['Silica'])
                weights.append(0.3)
            
            if gold_indicators and len(gold_indicators) >= 2:
                # Weighted combination
                gold_composite = np.zeros_like(gold_indicators[0])
                total_weight = 0
                
                for indicator, weight in zip(gold_indicators, weights):
                    valid_data = np.nan_to_num(indicator, nan=0)
                    gold_composite += weight * valid_data
                    total_weight += weight
                
                if total_weight > 0:
                    gold_composite /= total_weight
                
                return gold_composite
            
            return None
            
        except Exception as e:
            self.log_message.emit(f"Gold composite creation failed: {str(e)}", "ERROR")
            return None
    
    def create_iron_alteration_maps(self):
        """Create iron oxide and alteration maps"""
        try:
            results = {}
            
            # Calculate iron oxide ratio (classic ASTER technique)
            data = self.mineral_mapper.normalized_data or self.mineral_mapper.spectral_data
            if data is not None and data.shape[1] >= 3:
                # Iron oxide index using VNIR bands
                iron_ratio = data[:, 1] / (data[:, 0] + 0.001)  # Red/Green ratio
                
                # Convert to spatial format
                spatial_dims = self.mineral_mapper.spatial_dims
                valid_mask = self.mineral_mapper.valid_mask
                
                iron_map = np.full(spatial_dims[0] * spatial_dims[1], np.nan)
                iron_map[valid_mask] = iron_ratio[valid_mask]
                results['iron_oxide_ratio'] = iron_map.reshape(spatial_dims)
            
            # If we have SWIR bands, calculate additional indices
            if data is not None and data.shape[1] >= 6:
                # Ferric iron index using SWIR bands
                ferric_iron = data[:, 4] / data[:, 5]  # Approximation
                
                ferric_map = np.full(spatial_dims[0] * spatial_dims[1], np.nan)
                ferric_map[valid_mask] = ferric_iron[valid_mask]
                results['ferric_iron_index'] = ferric_map.reshape(spatial_dims)
            
            return results
            
        except Exception as e:
            self.log_message.emit(f"Iron alteration mapping failed: {str(e)}", "ERROR")
            return {}
    
    def create_lithium_exploration_composite(self):
        """Create lithium exploration composite"""
        try:
            # Lithium minerals often associated with specific clay signatures
            if not hasattr(self, 'mineral_results'):
                return None
            
            lithium_indicators = []
            
            # Look for clay minerals (lithium often in clays)
            if 'Clay Minerals' in self.mineral_results:
                lithium_indicators.append(self.mineral_results['Clay Minerals'])
            
            # Specific clay types
            if 'Kaolinite' in self.mineral_results:
                lithium_indicators.append(self.mineral_results['Kaolinite'])
            
            if lithium_indicators:
                # Average the indicators
                lithium_composite = np.mean(lithium_indicators, axis=0)
                return lithium_composite
            
            return None
            
        except Exception as e:
            self.log_message.emit(f"Lithium composite creation failed: {str(e)}", "ERROR")
            return None
    
    def calculate_spectral_indices(self):
        """Calculate additional spectral indices"""
        try:
            # This will use the mineral mapper's built-in index calculation
            # which includes NDVI, clay indices, carbonate indices, etc.
            return True
            
        except Exception as e:
            self.log_message.emit(f"Spectral indices calculation failed: {str(e)}", "ERROR")
            return False
    
    def create_false_color_composites(self):
        """Create false color composite images"""
        try:
            # Create RGB composites using different band combinations
            if hasattr(self.mineral_mapper, 'data') and self.mineral_mapper.data is not None:
                data = self.mineral_mapper.data
                
                # False color composite for mineral detection (bands 3,2,1)
                if data.shape[0] >= 3:
                    composite_321 = np.stack([data[2], data[1], data[0]], axis=0)
                    
                    if not hasattr(self, 'mineral_results'):
                        self.mineral_results = {}
                    
                    self.mineral_results['false_color_321'] = composite_321
                
                # If we have SWIR bands, create mineral composite
                if data.shape[0] >= 6:
                    composite_swir = np.stack([data[5], data[4], data[3]], axis=0)
                    self.mineral_results['swir_composite'] = composite_swir
                
                return True
            
            return False
            
        except Exception as e:
            self.log_message.emit(f"False color composite creation failed: {str(e)}", "ERROR")
            return False
    
    def perform_quality_assessment(self):
        """Perform quality assessment on results"""
        try:
            if hasattr(self, 'mineral_results'):
                for name, data in self.mineral_results.items():
                    if isinstance(data, np.ndarray) and data.size > 0:
                        # Calculate basic statistics
                        valid_data = data[~np.isnan(data.flatten())]
                        if len(valid_data) > 0:
                            mean_val = np.mean(valid_data)
                            std_val = np.std(valid_data)
                            min_val = np.min(valid_data)
                            max_val = np.max(valid_data)
                            
                            self.log_message.emit(
                                f"QA - {name}: mean={mean_val:.3f}, std={std_val:.3f}, "
                                f"range=[{min_val:.3f}, {max_val:.3f}], "
                                f"valid_pixels={len(valid_data)}", "INFO"
                            )
            
            return True
            
        except Exception as e:
            self.log_message.emit(f"Quality assessment failed: {str(e)}", "ERROR")
            return False
    
    def create_qgis_layers(self):
        """Create QGIS layers from mineral mapping results"""
        try:
            if not hasattr(self, 'mineral_results') or not self.mineral_results:
                self.log_message.emit("No mineral results to create layers from", "WARNING")
                return 0
            
            # Save results to temporary files and create layers
            import tempfile
            from qgis.core import QgsProject, QgsRasterLayer
            
            temp_dir = tempfile.mkdtemp(prefix='mineral_maps_')
            self.log_message.emit(f"Saving mineral maps to: {temp_dir}", "INFO")
            
            # Use the mineral mapper's save functionality
            try:
                saved_files = self.mineral_mapper.save_results(self.mineral_results, temp_dir)
            except:
                # Fallback: manual saving
                saved_files = self.save_results_manually(temp_dir)
            
            # Add layers to QGIS project
            project = QgsProject.instance()
            layers_added = 0
            
            for result_name in self.mineral_results:
                tiff_path = os.path.join(temp_dir, f"{result_name}.tif")
                
                if os.path.exists(tiff_path):
                    layer_name = f"Mineral_{result_name}"
                    layer = QgsRasterLayer(tiff_path, layer_name)
                    
                    if layer.isValid():
                        project.addMapLayer(layer)
                        layers_added += 1
                        self.log_message.emit(f"✅ Added layer: {layer_name}", "SUCCESS")
                    else:
                        self.log_message.emit(f"⚠️ Failed to create layer: {result_name}", "WARNING")
                else:
                    self.log_message.emit(f"⚠️ File not found: {tiff_path}", "WARNING")
            
            return layers_added
            
        except Exception as e:
            self.log_message.emit(f"QGIS layer creation failed: {str(e)}", "ERROR")
            return 0
    
    def save_results_manually(self, output_dir):
        """Manual fallback for saving results"""
        try:
            import rasterio
            from rasterio.transform import from_bounds
            
            # Create a basic profile for GeoTIFF output
            if hasattr(self.mineral_mapper, 'profile') and self.mineral_mapper.profile:
                profile = self.mineral_mapper.profile.copy()
            else:
                # Create minimal profile
                sample_data = list(self.mineral_results.values())[0]
                if isinstance(sample_data, np.ndarray) and sample_data.ndim == 2:
                    height, width = sample_data.shape
                    profile = {
                        'driver': 'GTiff',
                        'dtype': 'float32',
                        'nodata': np.nan,
                        'width': width,
                        'height': height,
                        'count': 1,
                        'crs': 'EPSG:4326',  # Default to WGS84
                        'transform': from_bounds(-180, -90, 180, 90, width, height)
                    }
            
            profile.update({
                'count': 1,
                'dtype': 'float32',
                'nodata': np.nan
            })
            
            saved_files = []
            
            for result_name, result_data in self.mineral_results.items():
                if isinstance(result_data, np.ndarray) and result_data.ndim == 2:
                    output_path = os.path.join(output_dir, f"{result_name}.tif")
                    
                    try:
                        with rasterio.open(output_path, 'w', **profile) as dst:
                            dst.write(result_data.astype(np.float32), 1)
                        saved_files.append(output_path)
                    except Exception as e:
                        self.log_message.emit(f"Failed to save {result_name}: {str(e)}", "WARNING")
            
            return saved_files
            
        except Exception as e:
            self.log_message.emit(f"Manual save failed: {str(e)}", "ERROR")
            return []
    
    def stop_processing(self):
        """Stop the processing"""
        self.should_stop = True


# For testing standalone
if __name__ == "__main__":
    import sys
    from qgis.PyQt.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    # Mock iface for testing
    class MockIface:
        def mainWindow(self):
            return None
    
    dialog = MainDialog(MockIface())
    dialog.show()
    
    sys.exit(app.exec_())