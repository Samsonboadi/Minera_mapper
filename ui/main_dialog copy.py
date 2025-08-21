"""
Complete Main Dialog with fixed LogWidget and enhanced ASTER processor integration
This replaces your existing main_dialog.py
"""

import os
from datetime import datetime
from qgis.PyQt.QtCore import Qt, pyqtSignal, QThread, QTimer
from qgis.PyQt.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
                                QLabel, QPushButton, QProgressBar, QTextEdit,
                                QTabWidget, QWidget, QComboBox, QSpinBox,
                                QDoubleSpinBox, QCheckBox, QGroupBox, QFrame,
                                QScrollArea, QSplitter, QFileDialog, QMessageBox,
                                QListWidget, QListWidgetItem, QSlider, QApplication)
from qgis.PyQt.QtGui import QFont, QPixmap, QIcon, QPalette, QColor, QTextCursor
from qgis.core import QgsProject, QgsRasterLayer, QgsVectorLayer, QgsMessageLog, Qgis


class LogWidget(QTextEdit):
    """Enhanced logging widget compatible with all Qt versions"""
    
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        
        # Manual line limit since setMaximumBlockCount may not be available
        self.max_lines = 1000
        self.current_lines = 0
        
        # Set font
        font = QFont("Consolas", 9)
        if not font.exactMatch():
            font = QFont("Courier New", 9)
            if not font.exactMatch():
                font = QFont("monospace", 9)
        font.setFixedPitch(True)
        self.setFont(font)
        
        # Style the widget
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
        
        # Initialize with welcome message
        self.clear_and_init()
    
    def clear_and_init(self):
        """Clear log and add initialization message"""
        self.clear()
        self.current_lines = 0
        self.add_message("🚀 Mineral Prospectivity Mapping - Processing Log", "HEADER")
        self.add_message("=" * 60, "HEADER")
        self.add_message("Ready to process geological data", "INFO")
        self.add_message("", "INFO")  # Empty line
    
    def add_message(self, message, level="INFO", auto_scroll=True):
        """Add a timestamped and color-coded message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color scheme for different log levels
        colors = {
            "HEADER": "#00d4ff",    # Bright cyan
            "INFO": "#00ff00",      # Green
            "SUCCESS": "#00ff88",   # Light green  
            "WARNING": "#ffaa00",   # Orange
            "ERROR": "#ff4444",     # Red
            "DEBUG": "#888888",     # Gray
            "PROGRESS": "#ffff00"   # Yellow
        }
        
        # Icons for different levels
        icons = {
            "HEADER": "🎯",
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "DEBUG": "🔧",
            "PROGRESS": "📊"
        }
        
        color = colors.get(level, "#ffffff")
        icon = icons.get(level, "")
        
        # Format message
        if level == "HEADER":
            formatted_message = f'<span style="color: {color}; font-weight: bold;">{icon} {message}</span>'
        else:
            formatted_message = (
                f'<span style="color: #888888;">[{timestamp}]</span> '
                f'<span style="color: {color};">{icon} {message}</span>'
            )
        
        # Check line limit and trim if necessary
        if self.current_lines >= self.max_lines:
            self.trim_old_lines()
        
        # Add to text edit
        self.append(formatted_message)
        self.current_lines += 1
        
        # Auto-scroll to bottom if requested
        if auto_scroll:
            self.scroll_to_bottom()
        
        # Update application to show changes immediately
        QApplication.processEvents()
    
    def trim_old_lines(self):
        """Remove old lines to keep within limit"""
        try:
            # Get current text
            current_text = self.toPlainText()
            lines = current_text.split('\n')
            
            # Keep only the last 80% of max_lines
            keep_lines = int(self.max_lines * 0.8)
            if len(lines) > keep_lines:
                new_lines = lines[-keep_lines:]
                
                # Clear and reset
                self.clear()
                self.current_lines = 0
                
                # Add trimmed indicator
                self.append('<span style="color: #888888;">[...previous logs trimmed...]</span>')
                self.current_lines += 1
                
                # Add remaining lines
                for line in new_lines:
                    if line.strip():  # Skip empty lines
                        self.append(line)
                        self.current_lines += 1
                        
        except Exception as e:
            # If trimming fails, just clear and restart
            self.clear_and_init()
    
    def scroll_to_bottom(self):
        """Scroll to bottom of text"""
        try:
            # Move cursor to end
            cursor = self.textCursor()
            cursor.movePosition(QTextCursor.End)
            self.setTextCursor(cursor)
            
            # Scroll to bottom
            scrollbar = self.verticalScrollBar()
            if scrollbar:
                scrollbar.setValue(scrollbar.maximum())
                
        except Exception:
            # Fallback method
            pass
    
    def add_progress_message(self, progress_value, message):
        """Add a progress message with percentage"""
        progress_bar = "█" * (progress_value // 5) + "░" * (20 - progress_value // 5)
        full_message = f"[{progress_value:3d}%] {progress_bar} {message}"
        self.add_message(full_message, "PROGRESS")
    
    def save_log(self, file_path):
        """Save log to file"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # Write plain text version
                f.write("Mineral Prospectivity Mapping - Processing Log\n")
                f.write("=" * 60 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(self.toPlainText())
            return True
        except Exception as e:
            self.add_message(f"Failed to save log: {str(e)}", "ERROR")
            return False


class ModernCard(QFrame):
    """Enhanced modern card widget"""
    
    def __init__(self, title="", icon="", parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel)
        self.setStyleSheet("""
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                padding: 5px;
                margin: 8px;
            }
            QFrame:hover {
                border: 1px solid #2196F3;
                box-shadow: 0 4px 12px rgba(33, 150, 243, 0.15);
            }
        """)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)
        self.layout.setSpacing(15)
        
        if title:
            header_layout = QHBoxLayout()
            
            if icon:
                icon_label = QLabel(icon)
                icon_label.setStyleSheet("font-size: 20px; margin-right: 10px;")
                header_layout.addWidget(icon_label)
            
            title_label = QLabel(title)
            title_label.setFont(QFont("Arial", 14, QFont.Bold))
            title_label.setStyleSheet("color: #2c3e50; margin-bottom: 5px;")
            header_layout.addWidget(title_label)
            header_layout.addStretch()
            
            self.layout.addLayout(header_layout)


class ProcessingThread(QThread):
    """Enhanced processing thread with detailed progress reporting"""
    
    progress_updated = pyqtSignal(int, str)
    log_message = pyqtSignal(str, str)  # message, level
    processing_finished = pyqtSignal(bool, str)  # success, message
    
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
        """Run the processing"""
        try:
            self.log_message.emit("Initializing processor...", "INFO")
            self.progress_updated.emit(5, "Initializing...")
            
            # Create processor instance
            if hasattr(self.args[0], 'iface'):  # Check if first arg is iface
                processor = self.processor_class(self.args[0])
                args = self.args[1:]
            else:
                processor = self.processor_class()
                args = self.args
            
            self.log_message.emit(f"Starting {self.method_name}...", "INFO")
            self.progress_updated.emit(10, "Starting processing...")
            
            # Check if processor has the method
            if not hasattr(processor, self.method_name):
                raise AttributeError(f"Processor does not have method '{self.method_name}'")
            
            # Get the method
            method = getattr(processor, self.method_name)
            
            # Add progress callbacks if supported
            if 'progress_callback' in self.kwargs:
                self.kwargs['progress_callback'] = self.progress_callback
            if 'log_callback' in self.kwargs:
                self.kwargs['log_callback'] = self.log_callback
            
            # Execute the method
            result = method(*args, **self.kwargs)
            
            if self.should_stop:
                self.log_message.emit("Processing was cancelled", "WARNING")
                self.processing_finished.emit(False, "Processing cancelled by user")
            elif result:
                self.log_message.emit("Processing completed successfully!", "SUCCESS")
                self.processing_finished.emit(True, "Processing completed successfully!")
            else:
                self.log_message.emit("Processing completed but returned no result", "WARNING")
                self.processing_finished.emit(False, "Processing completed but returned no result")
                
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            self.log_message.emit(error_msg, "ERROR")
            self.processing_finished.emit(False, error_msg)
    
    def progress_callback(self, value, message):
        """Progress callback for the processor"""
        if not self.should_stop:
            self.progress_updated.emit(value, message)
    
    def log_callback(self, message):
        """Log callback for the processor"""
        self.log_message.emit(message, "INFO")


class MainDialog(QDialog):
    """Enhanced main dialog with integrated logging"""
    
    def __init__(self, iface):
        super().__init__()
        self.iface = iface
        self.processing_thread = None
        
        self.setWindowTitle("Mineral Prospectivity Mapping")
        self.setMinimumSize(1200, 800)
        self.setModal(False)
        
        self.setup_ui()
        self.apply