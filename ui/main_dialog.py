"""
Complete Main Dialog - Rewritten from scratch with proper signals and clean code
This completely replaces your existing main_dialog.py
"""

import os
from datetime import datetime
from qgis.PyQt.QtCore import Qt, pyqtSignal, QThread, QTimer
from qgis.PyQt.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QProgressBar, QTextEdit, QTabWidget, QWidget, QCheckBox, 
    QGroupBox, QFrame, QSplitter, QFileDialog, QMessageBox, 
    QApplication, QComboBox  
)
from qgis.PyQt.QtGui import QFont, QTextCursor
from qgis.core import QgsProject, QgsRasterLayer, QgsMessageLog, Qgis


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
        """Add timestamped, color-coded message - THREAD SAFE"""
        # Ensure this runs on the main thread
        if QThread.currentThread() != QApplication.instance().thread():
            # If called from background thread, use a signal/slot mechanism
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color scheme
        colors = {
            "HEADER": "#00d4ff", "INFO": "#00ff00", "SUCCESS": "#00ff88",
            "WARNING": "#ffaa00", "ERROR": "#ff4444", "DEBUG": "#888888",
            "PROGRESS": "#ffff00"
        }
        
        # Icons
        icons = {
            "HEADER": "🎯", "INFO": "ℹ️", "SUCCESS": "✅",
            "WARNING": "⚠️", "ERROR": "❌", "DEBUG": "🔧", "PROGRESS": "📊"
        }
        
        color = colors.get(level, "#ffffff")
        icon = icons.get(level, "")
        
        # Format message
        if level == "HEADER":
            formatted = f'<span style="color: {color}; font-weight: bold;">{icon} {message}</span>'
        else:
            formatted = (
                f'<span style="color: #888888;">[{timestamp}]</span> '
                f'<span style="color: {color};">{icon} {message}</span>'
            )
        
        # Manage line count
        if self.current_lines >= self.max_lines:
            self.trim_old_lines()
        
        self.append(formatted)
        self.current_lines += 1
        self.scroll_to_bottom()
        
        # Remove QApplication.processEvents() - this was causing issues
        # QApplication.processEvents()
    
    def add_progress_message(self, progress_value, message):
        """Add progress message with visual bar"""
        progress_bar = "█" * (progress_value // 5) + "░" * (20 - progress_value // 5)
        full_message = f"[{progress_value:3d}%] {progress_bar} {message}"
        self.add_message(full_message, "PROGRESS")
    
    def trim_old_lines(self):
        """Remove old lines to stay within limit"""
        try:
            current_text = self.toPlainText()
            lines = current_text.split('\n')
            keep_lines = int(self.max_lines * 0.8)
            
            if len(lines) > keep_lines:
                new_lines = lines[-keep_lines:]
                self.clear()
                self.current_lines = 0
                self.append('<span style="color: #888888;">[...previous logs trimmed...]</span>')
                self.current_lines += 1
                
                for line in new_lines:
                    if line.strip():
                        self.append(line)
                        self.current_lines += 1
        except Exception:
            self.clear_and_init()
    
    def scroll_to_bottom(self):
        """Scroll to bottom of text"""
        try:
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.setTextCursor(cursor)
            scrollbar = self.verticalScrollBar()
            if scrollbar:
                scrollbar.setValue(scrollbar.maximum())
        except Exception:
            pass
    
    def save_log(self, file_path):
        """Save log to file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("Mineral Prospectivity Mapping - Processing Log\n")
                f.write("=" * 60 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(self.toPlainText())
            return True
        except Exception as e:
            self.add_message(f"Failed to save log: {str(e)}", "ERROR")
            return False


class ProcessingThread(QThread):
    """Thread for ASTER processing"""
    
    progress_updated = pyqtSignal(int, str)
    log_message = pyqtSignal(str, str)
    processing_finished = pyqtSignal(bool, str)
    
    def __init__(self, processor_class, method_name, *args, **kwargs):
        super().__init__()
        self.processor_class = processor_class
        self.method_name = method_name
        self.args = args
        self.kwargs = kwargs
        self.should_stop = False
    
    def stop(self):
        """Stop processing"""
        self.should_stop = True
        self.log_message.emit("Processing cancellation requested...", "WARNING")
    
    def run(self):
        """Run the processing - THREAD SAFE VERSION"""
        try:
            self.log_message.emit("Initializing processor...", "INFO")
            
            # Create processor instance
            if len(self.args) > 0:
                iface = self.args[0]
                processor = self.processor_class(iface)
                args = self.args[1:]
            else:
                processor = self.processor_class()
                args = self.args
            
            self.log_message.emit(f"Starting {self.method_name}...", "INFO")
            
            # Check method exists
            if not hasattr(processor, self.method_name):
                raise AttributeError(f"Processor missing method '{self.method_name}'")
            
            method = getattr(processor, self.method_name)
            
            # Create thread-safe callbacks
            def progress_callback(value, message):
                self.progress_updated.emit(value, message)
            
            def log_callback(message, level="INFO"):
                self.log_message.emit(message, level)
            
            def should_stop_callback():
                return self.should_stop
            
            # Execute with thread-safe callbacks
            if self.method_name == 'process_specific_file':
                # For process_specific_file, we need to override the processor's callbacks
                result = processor.process_aster_file_threaded(
                    args[0],  # file_path
                    progress_callback,
                    log_callback, 
                    should_stop_callback
                )
            else:
                # For other methods, call normally
                result = method(*args, **self.kwargs)
            
            if self.should_stop:
                self.log_message.emit("Processing was cancelled", "WARNING")
                self.processing_finished.emit(False, "Processing cancelled by user")
            elif result:
                self.log_message.emit("Processing completed successfully!", "SUCCESS")
                self.processing_finished.emit(True, "Processing completed successfully!")
            else:
                self.log_message.emit("Processing returned no result", "WARNING")
                self.processing_finished.emit(False, "Processing returned no result")
                
        except Exception as e:
            import traceback
            error_msg = f"Processing failed: {str(e)}\n{traceback.format_exc()}"
            self.log_message.emit(error_msg, "ERROR")
            self.processing_finished.emit(False, error_msg)

class MainDialog(QDialog):
    """Main dialog for Mineral Prospectivity Mapping"""
    
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
        self.create_aster_tab(),
        self.create_sentinel2_tab(),
        #self.create_placeholder_tabs()
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
        
        # Processing options group
        options_group = QGroupBox("Processing Options")
        options_group.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #dee2e6; border-radius: 6px; 
                       margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { color: #495057; subcontrol-origin: margin; left: 10px; padding: 0 8px; }
        """)
        options_layout = QVBoxLayout(options_group)
        
        self.aster_atmospheric = QCheckBox("🌤️ Apply atmospheric correction")
        self.aster_ratios = QCheckBox("🧮 Calculate mineral ratios")
        self.aster_ratios.setChecked(True)
        self.aster_composites = QCheckBox("🎨 Create false color composites")
        self.aster_composites.setChecked(True)
        self.aster_quality = QCheckBox("🔍 Perform quality assessment")
        self.aster_quality.setChecked(True)
        
        for checkbox in [self.aster_atmospheric, self.aster_ratios, self.aster_composites, self.aster_quality]:
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
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Sentinel-2 Product:"))
        
        self.s2_file_display = QLabel("No product selected")
        self.s2_file_display.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa; border: 2px dashed #dee2e6;
                padding: 12px; border-radius: 6px; color: #6c757d; min-height: 20px;
            }
        """)
        self.s2_file_display.setWordWrap(True)
        file_row.addWidget(self.s2_file_display, 1)
        
        self.browse_s2_btn = QPushButton("📁 Browse")
        self.browse_s2_btn.setStyleSheet(self.get_secondary_button_style())
        file_row.addWidget(self.browse_s2_btn)
        
        file_layout.addLayout(file_row)
        
        # Product type selection
        product_row = QHBoxLayout()
        product_row.addWidget(QLabel("Product Type:"))
        
        self.s2_product_type = QComboBox()
        self.s2_product_type.addItems(["Auto-detect", "L1C (Top of Atmosphere)", "L2A (Bottom of Atmosphere)"])
        self.s2_product_type.setStyleSheet("""
            QComboBox {
                border: 1px solid #e0e0e0; border-radius: 4px; padding: 6px 10px;
                background-color: white; min-width: 150px;
            }
            QComboBox:hover { border-color: #2196F3; }
            QComboBox::drop-down { border: none; width: 20px; }
        """)
        product_row.addWidget(self.s2_product_type)
        product_row.addStretch()
        
        file_layout.addLayout(product_row)
        card_layout.addWidget(file_group)
        
        # Processing options group
        options_group = QGroupBox("Processing Options")
        options_group.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #dee2e6; border-radius: 6px; 
                    margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { color: #495057; subcontrol-origin: margin; left: 10px; padding: 0 8px; }
        """)
        options_layout = QVBoxLayout(options_group)
        
        # Basic processing options
        self.s2_resample = QCheckBox("📐 Resample all bands to 10m resolution")
        self.s2_resample.setChecked(True)
        self.s2_cloud_mask = QCheckBox("☁️ Apply cloud masking (L2A only)")
        self.s2_cloud_mask.setChecked(True)
        self.s2_atmospheric = QCheckBox("🌤️ Apply atmospheric correction (L1C only)")
        
        # Spectral indices
        self.s2_indices = QCheckBox("🧮 Calculate spectral indices")
        self.s2_indices.setChecked(True)
        self.s2_composites = QCheckBox("🎨 Create composite images")
        self.s2_composites.setChecked(True)
        
        for checkbox in [self.s2_resample, self.s2_cloud_mask, self.s2_atmospheric, 
                        self.s2_indices, self.s2_composites]:
            checkbox.setStyleSheet("QCheckBox { font-weight: bold; padding: 5px; }")
            options_layout.addWidget(checkbox)
        
        card_layout.addWidget(options_group)
        
        # Spectral indices selection
        indices_group = QGroupBox("Spectral Indices to Calculate")
        indices_group.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #dee2e6; border-radius: 6px; 
                    margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { color: #495057; subcontrol-origin: margin; left: 10px; padding: 0 8px; }
        """)
        indices_layout = QVBoxLayout(indices_group)
        
        # Create checkboxes for different indices
        indices_row1 = QHBoxLayout()
        self.s2_ndvi = QCheckBox("NDVI (Vegetation)")
        self.s2_ndvi.setChecked(True)
        self.s2_ndwi = QCheckBox("NDWI (Water)")
        self.s2_ndwi.setChecked(True)
        self.s2_iron_oxide = QCheckBox("Iron Oxide Index")
        self.s2_iron_oxide.setChecked(True)
        
        indices_row1.addWidget(self.s2_ndvi)
        indices_row1.addWidget(self.s2_ndwi)
        indices_row1.addWidget(self.s2_iron_oxide)
        indices_layout.addLayout(indices_row1)
        
        indices_row2 = QHBoxLayout()
        self.s2_clay = QCheckBox("Clay Minerals")
        self.s2_clay.setChecked(True)
        self.s2_carbonate = QCheckBox("Carbonate Index")
        self.s2_carbonate.setChecked(True)
        self.s2_alteration = QCheckBox("Alteration Index")
        self.s2_alteration.setChecked(True)
        
        indices_row2.addWidget(self.s2_clay)
        indices_row2.addWidget(self.s2_carbonate)
        indices_row2.addWidget(self.s2_alteration)
        indices_layout.addLayout(indices_row2)
        
        for checkbox in [self.s2_ndvi, self.s2_ndwi, self.s2_iron_oxide, 
                        self.s2_clay, self.s2_carbonate, self.s2_alteration]:
            checkbox.setStyleSheet("QCheckBox { font-weight: bold; padding: 3px; }")
        
        card_layout.addWidget(indices_group)
        
        # Process button
        self.process_s2_btn = QPushButton("🚀 Process Sentinel-2 Data")
        self.process_s2_btn.setStyleSheet(self.get_primary_button_style())
        self.process_s2_btn.setEnabled(False)
        card_layout.addWidget(self.process_s2_btn)
        
        layout.addWidget(card)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "🛰️ Sentinel-2")

    def create_geological_tab(self):
        """Create complete geological processing tab"""
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
        title = QLabel("🗺️ Geological Data Processing")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50; margin-bottom: 10px;")
        card_layout.addWidget(title)
        
        # Description
        desc = QLabel("Process geological maps, structural data, and lithological information for mineral prospectivity mapping and geological analysis")
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
        file_row.addWidget(QLabel("Geological Data:"))
        
        self.geo_file_display = QLabel("No geological data selected")
        self.geo_file_display.setStyleSheet("""
            QLabel {
                background-color: #f8f9fa; border: 2px dashed #dee2e6;
                padding: 12px; border-radius: 6px; color: #6c757d; min-height: 20px;
            }
        """)
        self.geo_file_display.setWordWrap(True)
        file_row.addWidget(self.geo_file_display, 1)
        
        self.browse_geo_btn = QPushButton("📁 Browse")
        self.browse_geo_btn.setStyleSheet(self.get_secondary_button_style())
        file_row.addWidget(self.browse_geo_btn)
        
        file_layout.addLayout(file_row)
        
        # Data type selection
        data_type_row = QHBoxLayout()
        data_type_row.addWidget(QLabel("Data Type:"))
        
        self.geo_data_type = QComboBox()
        self.geo_data_type.addItems([
            "Auto-detect", 
            "Vector (Shapefile/GeoPackage)", 
            "Raster (GeoTIFF)", 
            "Geological Map",
            "Structural Data",
            "Geochemical Data"
        ])
        self.geo_data_type.setStyleSheet("""
            QComboBox {
                border: 1px solid #e0e0e0; border-radius: 4px; padding: 6px 10px;
                background-color: white; min-width: 150px;
            }
            QComboBox:hover { border-color: #2196F3; }
            QComboBox::drop-down { border: none; width: 20px; }
        """)
        data_type_row.addWidget(self.geo_data_type)
        data_type_row.addStretch()
        
        file_layout.addLayout(data_type_row)
        card_layout.addWidget(file_group)
        
        # Processing options group
        options_group = QGroupBox("Analysis Options")
        options_group.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #dee2e6; border-radius: 6px; 
                    margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { color: #495057; subcontrol-origin: margin; left: 10px; padding: 0 8px; }
        """)
        options_layout = QVBoxLayout(options_group)
        
        # Basic analysis options
        self.geo_favorability = QCheckBox("⭐ Calculate geological favorability")
        self.geo_favorability.setChecked(True)
        self.geo_structural = QCheckBox("🔍 Analyze structural patterns")
        self.geo_structural.setChecked(True)
        self.geo_lithology = QCheckBox("🗿 Perform lithological analysis")
        self.geo_lithology.setChecked(True)
        self.geo_lineaments = QCheckBox("📏 Extract lineament features")
        self.geo_lineaments.setChecked(True)
        
        for checkbox in [self.geo_favorability, self.geo_structural, 
                        self.geo_lithology, self.geo_lineaments]:
            checkbox.setStyleSheet("QCheckBox { font-weight: bold; padding: 5px; }")
            options_layout.addWidget(checkbox)
        
        card_layout.addWidget(options_group)
        
        # Analysis parameters group
        params_group = QGroupBox("Analysis Parameters")
        params_group.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #dee2e6; border-radius: 6px; 
                    margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { color: #495057; subcontrol-origin: margin; left: 10px; padding: 0 8px; }
        """)
        params_layout = QVBoxLayout(params_group)
        
        # Target minerals selection
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target Minerals:"))
        
        self.geo_target_minerals = QComboBox()
        self.geo_target_minerals.addItems([
            "Gold", "Copper", "Iron", "Lead-Zinc", "Silver", 
            "Lithium", "Rare Earth Elements", "Uranium", "All Minerals"
        ])
        self.geo_target_minerals.setCurrentText("Gold")
        self.geo_target_minerals.setStyleSheet("""
            QComboBox {
                border: 1px solid #e0e0e0; border-radius: 4px; padding: 6px 10px;
                background-color: white; min-width: 120px;
            }
            QComboBox:hover { border-color: #2196F3; }
        """)
        target_row.addWidget(self.geo_target_minerals)
        target_row.addStretch()
        
        params_layout.addLayout(target_row)
        
        # Buffer distance for analysis
        buffer_row = QHBoxLayout()
        buffer_row.addWidget(QLabel("Analysis Buffer (m):"))
        
        self.geo_buffer_distance = QComboBox()
        self.geo_buffer_distance.addItems(["100", "500", "1000", "2000", "5000", "10000"])
        self.geo_buffer_distance.setCurrentText("1000")
        self.geo_buffer_distance.setEditable(True)
        self.geo_buffer_distance.setStyleSheet("""
            QComboBox {
                border: 1px solid #e0e0e0; border-radius: 4px; padding: 6px 10px;
                background-color: white; min-width: 100px;
            }
            QComboBox:hover { border-color: #2196F3; }
        """)
        buffer_row.addWidget(self.geo_buffer_distance)
        buffer_row.addStretch()
        
        params_layout.addLayout(buffer_row)
        card_layout.addWidget(params_group)
        
        # Output options group
        output_group = QGroupBox("Output Options")
        output_group.setStyleSheet("""
            QGroupBox { font-weight: bold; border: 1px solid #dee2e6; border-radius: 6px; 
                    margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { color: #495057; subcontrol-origin: margin; left: 10px; padding: 0 8px; }
        """)
        output_layout = QVBoxLayout(output_group)
        
        output_row1 = QHBoxLayout()
        self.geo_create_maps = QCheckBox("🗺️ Create favorability maps")
        self.geo_create_maps.setChecked(True)
        self.geo_export_vectors = QCheckBox("📊 Export analysis vectors")
        self.geo_export_vectors.setChecked(True)
        
        output_row1.addWidget(self.geo_create_maps)
        output_row1.addWidget(self.geo_export_vectors)
        output_layout.addLayout(output_row1)
        
        output_row2 = QHBoxLayout()
        self.geo_generate_report = QCheckBox("📄 Generate analysis report")
        self.geo_generate_report.setChecked(True)
        self.geo_create_statistics = QCheckBox("📈 Calculate statistics")
        self.geo_create_statistics.setChecked(True)
        
        output_row2.addWidget(self.geo_generate_report)
        output_row2.addWidget(self.geo_create_statistics)
        output_layout.addLayout(output_row2)
        
        for checkbox in [self.geo_create_maps, self.geo_export_vectors, 
                        self.geo_generate_report, self.geo_create_statistics]:
            checkbox.setStyleSheet("QCheckBox { font-weight: bold; padding: 3px; }")
        
        card_layout.addWidget(output_group)
        
        # Process button
        self.process_geo_btn = QPushButton("🚀 Process Geological Data")
        self.process_geo_btn.setStyleSheet(self.get_primary_button_style())
        self.process_geo_btn.setEnabled(False)
        card_layout.addWidget(self.process_geo_btn)
        
        layout.addWidget(card)
        layout.addStretch()
        
        self.tab_widget.addTab(tab, "🗺️ Geological Data")
    
    def create_logging_widget(self):
        """Create the logging section"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(10)
        
        # Header
        header_layout = QHBoxLayout()
        
        log_title = QLabel("📋 Real-time Processing Log")
        log_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #2c3e50; padding: 8px;")
        header_layout.addWidget(log_title)
        header_layout.addStretch()
        
        # Log controls
        self.clear_log_btn = QPushButton("🗑️ Clear")
        self.clear_log_btn.setStyleSheet(self.get_small_button_style())
        
        self.save_log_btn = QPushButton("💾 Save")
        self.save_log_btn.setStyleSheet(self.get_small_button_style())
        
        header_layout.addWidget(self.clear_log_btn)
        header_layout.addWidget(self.save_log_btn)
        
        layout.addLayout(header_layout)
        
        # Log widget
        self.log_widget = LogWidget()
        layout.addWidget(self.log_widget, 1)
        
        return widget
    
    def create_footer(self, layout):
        """Create footer with progress and buttons"""
        footer_frame = QFrame()
        footer_frame.setStyleSheet("""
            QFrame {
                background-color: #f8f9fa; border-top: 1px solid #dee2e6;
                border-radius: 8px; padding: 15px; margin-top: 10px;
            }
        """)
        footer_layout = QVBoxLayout(footer_frame)
        
        # Progress section
        progress_layout = QHBoxLayout()
        
        progress_label = QLabel("Progress:")
        progress_label.setStyleSheet("font-weight: bold; color: #495057;")
        progress_layout.addWidget(progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #ced4da; border-radius: 8px; text-align: center;
                background-color: #f8f9fa; font-weight: bold; color: #495057;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #667eea, stop: 1 #764ba2); border-radius: 7px;
            }
        """)
        progress_layout.addWidget(self.progress_bar, 1)
        footer_layout.addLayout(progress_layout)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.help_btn = QPushButton("❓ Help")
        self.help_btn.setStyleSheet(self.get_secondary_button_style())
        
        self.about_btn = QPushButton("ℹ️ About")
        self.about_btn.setStyleSheet(self.get_secondary_button_style())
        
        button_layout.addWidget(self.help_btn)
        button_layout.addWidget(self.about_btn)
        button_layout.addStretch()
        
        self.cancel_btn = QPushButton("⏹️ Cancel Processing")
        self.cancel_btn.setVisible(False)
        self.cancel_btn.setStyleSheet("""
            QPushButton {
                background-color: #dc3545; color: white; border: none;
                padding: 10px 20px; border-radius: 6px; font-weight: bold;
            }
            QPushButton:hover { background-color: #c82333; }
        """)
        
        self.close_btn = QPushButton("🚪 Close")
        self.close_btn.setStyleSheet(self.get_secondary_button_style())
        
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.close_btn)
        
        footer_layout.addLayout(button_layout)
        layout.addWidget(footer_frame)
    
    def connect_signals(self):
        """Connect all signals and slots"""
        # File browser
        self.browse_btn.clicked.connect(self.browse_aster_file)
        
        # Processing
        self.process_aster_btn.clicked.connect(self.process_aster_data)
        self.cancel_btn.clicked.connect(self.cancel_processing)



        # Sentinel-2 signals
        self.browse_s2_btn.clicked.connect(self.browse_s2_file)
        self.process_s2_btn.clicked.connect(self.process_s2_data)
        self.s2_indices.toggled.connect(self.toggle_s2_indices)
        
        # Geological signals
        self.browse_geo_btn.clicked.connect(self.browse_geo_file)
        self.process_geo_btn.clicked.connect(self.process_geo_data)
        self.geo_data_type.currentTextChanged.connect(self.on_geo_data_type_changed)


        
        # Log controls
        self.clear_log_btn.clicked.connect(self.clear_log)
        self.save_log_btn.clicked.connect(self.save_log)
        
        # Dialog controls
        self.help_btn.clicked.connect(self.show_help)
        self.about_btn.clicked.connect(self.show_about)
        self.close_btn.clicked.connect(self.close)
    
    def apply_styles(self):
        """Apply overall dialog styling"""
        self.setStyleSheet("""
            QDialog {
                background-color: #ffffff; color: #333333;
            }
            QScrollArea {
                border: none; background-color: transparent;
            }
            QScrollBar:vertical {
                border: none; background: #f1f3f4; width: 12px; border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #dadce0; border-radius: 6px; min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #c1c3c7;
            }
        """)
    
    def get_primary_button_style(self):
        """Primary button styling"""
        return """
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #667eea, stop: 1 #764ba2);
                color: white; border: none; padding: 12px 24px; border-radius: 8px;
                font-weight: bold; font-size: 13px; min-width: 150px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #5a6fd8, stop: 1 #6a4190);
            }
            QPushButton:disabled {
                background-color: #e9ecef; color: #6c757d;
            }
        """
    
    def get_secondary_button_style(self):
        """Secondary button styling"""
        return """
            QPushButton {
                background-color: #f8f9fa; color: #495057; border: 1px solid #dee2e6;
                padding: 8px 16px; border-radius: 6px; font-weight: bold; min-width: 80px;
            }
            QPushButton:hover {
                background-color: #e9ecef; border-color: #adb5bd; color: #212529;
            }
        """
    
    def get_small_button_style(self):
        """Small button styling"""
        return """
            QPushButton {
                background-color: #6c757d; color: white; border: none;
                padding: 6px 12px; border-radius: 4px; font-weight: bold;
                font-size: 11px; min-width: 60px;
            }
            QPushButton:hover { background-color: #5a6268; }
        """



    def browse_s2_file(self):
        """Browse for Sentinel-2 product"""
        # Try directory first (for .SAFE products)
        directory = QFileDialog.getExistingDirectory(
            self, "Select Sentinel-2 Product Directory (.SAFE)", ""
        )
        
        if directory:
            self.s2_file_path = directory
            self.update_s2_file_display(directory)
        else:
            # Try ZIP file
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Sentinel-2 ZIP File", "",
                "Sentinel-2 files (*.zip);;All files (*)"
            )
            if file_path:
                self.s2_file_path = file_path
                self.update_s2_file_display(file_path)

    def update_s2_file_display(self, file_path):
        """Update Sentinel-2 file display"""
        file_name = os.path.basename(file_path)
        self.s2_file_display.setText(f"✅ {file_name}")
        self.s2_file_display.setStyleSheet("""
            QLabel {
                background-color: #d4edda; border: 2px solid #c3e6cb;
                padding: 12px; border-radius: 6px; color: #155724;
                min-height: 20px; font-weight: bold;
            }
        """)
        
        self.process_s2_btn.setEnabled(True)
        self.log_widget.add_message(f"Selected Sentinel-2 product: {file_name}", "INFO")
        self.update_status("Sentinel-2 product selected", "🟡")
        
        # Detect product type
        if "L1C" in file_name.upper():
            self.s2_product_type.setCurrentText("L1C (Top of Atmosphere)")
            self.s2_atmospheric.setEnabled(True)
            self.s2_cloud_mask.setEnabled(False)
        elif "L2A" in file_name.upper():
            self.s2_product_type.setCurrentText("L2A (Bottom of Atmosphere)")
            self.s2_atmospheric.setEnabled(False)
            self.s2_cloud_mask.setEnabled(True)


    def browse_geo_file(self):
        """Browse for geological data file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Geological Data", "",
            "Vector files (*.shp *.gpkg *.geojson);;Raster files (*.tif *.tiff);;All files (*)"
        )
        
        if file_path:
            self.geo_file_path = file_path
            file_name = os.path.basename(file_path)
            self.geo_file_display.setText(f"✅ {file_name}")
            self.geo_file_display.setStyleSheet("""
                QLabel {
                    background-color: #d4edda; border: 2px solid #c3e6cb;
                    padding: 12px; border-radius: 6px; color: #155724;
                    min-height: 20px; font-weight: bold;
                }
            """)
            
            self.process_geo_btn.setEnabled(True)
            self.log_widget.add_message(f"Selected geological data: {file_name}", "INFO")
            self.update_status("Geological data selected", "🟡")
            
            # Auto-detect data type
            if file_path.lower().endswith(('.shp', '.gpkg', '.geojson')):
                self.geo_data_type.setCurrentText("Vector (Shapefile/GeoPackage)")
            elif file_path.lower().endswith(('.tif', '.tiff')):
                self.geo_data_type.setCurrentText("Raster (GeoTIFF)")

    def toggle_s2_indices(self, enabled):
        """Toggle Sentinel-2 indices checkboxes"""
        indices_checkboxes = [
            self.s2_ndvi, self.s2_ndwi, self.s2_iron_oxide,
            self.s2_clay, self.s2_carbonate, self.s2_alteration
        ]
        
        for checkbox in indices_checkboxes:
            checkbox.setEnabled(enabled)

    def on_geo_data_type_changed(self, data_type):
        """Handle geological data type change"""
        if "Vector" in data_type:
            self.geo_lineaments.setEnabled(True)
            self.geo_structural.setEnabled(True)
        elif "Raster" in data_type:
            self.geo_lineaments.setEnabled(False)
            self.geo_structural.setEnabled(True)

    def process_s2_data(self):
        """Process Sentinel-2 data"""
        if not hasattr(self, 's2_file_path'):
            QMessageBox.warning(self, "No File Selected", "Please select a Sentinel-2 product first.")
            return
        
        self.log_widget.add_message("🛰️ Sentinel-2 processing will be implemented with the Sentinel-2 processor", "INFO")
        QMessageBox.information(self, "Feature Coming Soon", "Sentinel-2 processing will be implemented soon!")

    def process_geo_data(self):
        """Process geological data"""
        if not hasattr(self, 'geo_file_path'):
            QMessageBox.warning(self, "No File Selected", "Please select geological data first.")
            return
        
        self.log_widget.add_message("🗺️ Geological processing will be implemented with the geological processor", "INFO")
        QMessageBox.information(self, "Feature Coming Soon", "Geological processing will be implemented soon!")

        
    # Event handlers
    def browse_aster_file(self):
        """Browse for ASTER file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select ASTER L2 File", "",
            "ASTER files (*.zip *.hdf *.h5 *.hdf5);;ZIP files (*.zip);;HDF files (*.hdf *.h5 *.hdf5);;All files (*)"
        )
        
        if file_path:
            # Update UI
            file_name = os.path.basename(file_path)
            self.aster_file_display.setText(f"✅ {file_name}")
            self.aster_file_display.setStyleSheet("""
                QLabel {
                    background-color: #d4edda; border: 2px solid #c3e6cb;
                    padding: 12px; border-radius: 6px; color: #155724;
                    min-height: 20px; font-weight: bold;
                }
            """)
            
            # Store path and enable processing
            self.aster_file_path = file_path
            self.process_aster_btn.setEnabled(True)
            
            # Update log and status
            self.log_widget.add_message(f"Selected ASTER file: {file_name}", "INFO")
            self.update_status("File selected - ready to process", "🟡")
            
            # File info
            file_size = os.path.getsize(file_path)
            self.log_widget.add_message(f"File size: {file_size / (1024*1024):.1f} MB", "DEBUG")
            
            if file_path.lower().endswith('.zip'):
                self.log_widget.add_message("Detected ZIP archive format", "DEBUG")
            elif file_path.lower().endswith(('.hdf', '.h5', '.hdf5')):
                self.log_widget.add_message("Detected HDF format", "DEBUG")
    
    def process_aster_data(self):
        """Process ASTER data - THREAD SAFE VERSION"""
        if not self.aster_file_path:
            QMessageBox.warning(self, "No File Selected", "Please select an ASTER file first.")
            return
        
        # Set processing state
        self.set_processing_state(True)
        
        try:
            # Clear log and start fresh
            self.log_widget.clear_and_init()
            self.log_widget.add_message("Starting ASTER data processing...", "HEADER")
            self.log_widget.add_message(f"Input file: {os.path.basename(self.aster_file_path)}", "INFO")
            
            self.update_status("Initializing ASTER processing...", "🟠")
            
            # Import processor
            try:
                from ..processing.aster_processor import AsterProcessor
                self.log_widget.add_message("✅ ASTER processor imported successfully", "SUCCESS")
            except ImportError:
                try:
                    from processing.aster_processor import AsterProcessor
                    self.log_widget.add_message("✅ ASTER processor imported (alternative path)", "SUCCESS")
                except ImportError:
                    raise ImportError("Cannot import ASTER processor. Check plugin installation.")
            
            # Create processing thread - FIXED
            self.processing_thread = ProcessingThread(
                AsterProcessor, 'process_specific_file',
                self.iface, self.aster_file_path
            )
            
            # Connect thread signals - THREAD SAFE
            self.processing_thread.progress_updated.connect(self.update_progress)
            self.processing_thread.log_message.connect(self.add_log_message)  # This is thread safe
            self.processing_thread.processing_finished.connect(self.on_processing_finished)
            
            # Start processing
            self.processing_thread.start()
            self.log_widget.add_message("🚀 Processing thread started", "INFO")
            
        except Exception as e:
            self.log_widget.add_message(f"Failed to start processing: {str(e)}", "ERROR")
            self.set_processing_state(False)
            QMessageBox.critical(self, "Processing Error", f"Failed to start ASTER processing:\n\n{str(e)}")
    
    def cancel_processing(self):
        """Cancel current processing"""
        if self.processing_thread and self.processing_thread.isRunning():
            self.log_widget.add_message("🛑 Cancellation requested...", "WARNING")
            self.update_status("Cancelling processing...", "🟠")
            
            # Stop the thread
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
            self.process_geo_btn.setEnabled(not processing and hasattr(self, 'geo_file_path'))
            self.browse_geo_btn.setEnabled(not processing)
        
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
    
    def add_log_message(self, message, level):
        """Add message to log widget"""
        self.log_widget.add_message(message, level)
    
    def update_status(self, message, indicator="🟢"):
        """Update status label and indicator"""
        self.status_label.setText(message)
        self.status_indicator.setText(indicator)
    
    def on_processing_finished(self, success, message):
        """Handle processing completion"""
        self.set_processing_state(False)
        
        if success:
            self.log_widget.add_message("🎉 ASTER processing completed successfully!", "SUCCESS")
            self.log_widget.add_message("Results have been added to QGIS project", "INFO")
            self.update_status("Processing completed successfully", "🟢")
            
            # Show success dialog
            QMessageBox.information(
                self,
                "✅ Processing Complete",
                "ASTER data processing completed successfully!\n\n"
                "All layers have been added to your QGIS project.\n"
                "Check the Layers panel to view the results."
            )
            
        else:
            self.log_widget.add_message(f"❌ Processing failed: {message}", "ERROR")
            self.update_status("Processing failed", "🔴")
            
            # Show error dialog
            QMessageBox.critical(
                self,
                "❌ Processing Failed",
                f"ASTER processing failed:\n\n{message}\n\n"
                f"Check the processing log for more details."
            )
        
        # Clean up thread
        if self.processing_thread:
            self.processing_thread.deleteLater()
            self.processing_thread = None
    
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
    
    # Dialog methods
    def show_help(self):
        """Show comprehensive help dialog"""
        help_text = """
        <h2>🗺️ Mineral Prospectivity Mapping - Help</h2>
        
        <h3>🛰️ ASTER Processing:</h3>
        <ul>
        <li><b>File Input:</b> Select ASTER L2 ZIP or HDF files</li>
        <li><b>Atmospheric Correction:</b> Apply atmospheric corrections to surface reflectance data</li>
        <li><b>Mineral Ratios:</b> Calculate spectral ratios for mineral identification</li>
        <li><b>Composites:</b> Create false color composite images for visualization</li>
        <li><b>Quality Assessment:</b> Perform data quality checks and validation</li>
        </ul>
        
        <h3>🧮 Calculated Mineral Ratios:</h3>
        <ul>
        <li><b>Iron Oxide Index:</b> NIR/Red ratio for iron oxide detection</li>
        <li><b>Clay Minerals Index:</b> SWIR band ratios for clay mineral mapping</li>
        <li><b>Carbonate Index:</b> Carbonate mineral detection using SWIR bands</li>
        <li><b>Silicate Index:</b> Silicate mineral identification</li>
        <li><b>Alteration Index:</b> Hydrothermal alteration zone mapping</li>
        <li><b>Gossan Index:</b> Iron-rich oxidized zone detection</li>
        </ul>
        
        <h3>📋 Processing Log:</h3>
        <ul>
        <li><b>Real-time Updates:</b> Monitor processing progress with detailed logs</li>
        <li><b>Color Coding:</b> Different colors for info, warnings, and errors</li>
        <li><b>Save/Clear:</b> Export logs for documentation or clear for new processing</li>
        <li><b>Auto-scroll:</b> Automatically shows latest log entries</li>
        </ul>
        
        <h3>🎯 Tips for Best Results:</h3>
        <ul>
        <li>Use ASTER L2 surface reflectance products for best accuracy</li>
        <li>Ensure input files are not corrupted (check file size)</li>
        <li>Enable all processing options for comprehensive analysis</li>
        <li>Monitor the log for any warnings or processing issues</li>
        <li>Save logs for documentation and troubleshooting</li>
        </ul>
        """
        
        QMessageBox.information(self, "Help", help_text)
    
    def show_about(self):
        """Show enhanced about dialog"""
        about_text = """
        <h2>🗺️ Mineral Prospectivity Mapping Plugin</h2>
        <p><b>Version:</b> 2.0.0 (Enhanced Edition)</p>
        <p><b>Author:</b> Geological Survey Team</p>
        <p><b>Build Date:</b> 2024</p>
        
        <h3>🎯 Purpose:</h3>
        <p>Advanced geological data processing and analysis for mineral exploration using multi-source remote sensing data.</p>
        
        <h3>📊 Supported Data Types:</h3>
        <ul>
        <li><b>ASTER L2:</b> Surface reflectance data (VNIR/SWIR bands)</li>
        <li><b>Sentinel-2:</b> MSI multispectral imagery</li>
        <li><b>Geological Data:</b> Vector and raster geological maps</li>
        </ul>
        
        <h3>🚀 Key Features:</h3>
        <ul>
        <li>Real-time processing with detailed logging</li>
        <li>Comprehensive mineral mapping capabilities</li>
        <li>Spectral analysis and ratio calculations</li>
        <li>Multi-source data fusion</li>
        <li>Prospectivity mapping tools</li>
        <li>Enhanced user interface with progress tracking</li>
        </ul>
        
        <h3>🔧 Technical Requirements:</h3>
        <p><b>Dependencies:</b> QGIS 3.x, numpy, rasterio/GDAL, scikit-learn (optional)</p>
        
        <h3>📞 Support:</h3>
        <p>For technical support and documentation, visit the plugin repository or contact the development team.</p>
        """
        
        QMessageBox.about(self, "About", about_text)
    
    def closeEvent(self, event):
        """Handle dialog close with processing check"""
        if self.processing_thread and self.processing_thread.isRunning():
            reply = QMessageBox.question(
                self,
                "Processing Active",
                "ASTER processing is currently running.\n\n"
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