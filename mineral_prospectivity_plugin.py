"""
Main plugin class for Mineral Prospectivity Mapping - FIXED VERSION
This replaces your existing mineral_prospectivity_plugin.py file
"""

import os
import sys
from qgis.PyQt.QtCore import QSettings, QTranslator, QCoreApplication, Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMenu, QMessageBox
from qgis.core import QgsProject, QgsMessageLog, Qgis, QgsRasterLayer, QgsVectorLayer

# Add plugin directory to Python path
plugin_dir = os.path.dirname(__file__)
if plugin_dir not in sys.path:
    sys.path.insert(0, plugin_dir)


class MineralProspectivityPlugin:
    """Fixed main plugin class with comprehensive error handling"""
    
    def __init__(self, iface):
        """Constructor"""
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        
        # Initialize locale
        try:
            locale = QSettings().value('locale/userLocale')[0:2]
            locale_path = os.path.join(
                self.plugin_dir, 'i18n', f'MineralProspectivityPlugin_{locale}.qm'
            )
            
            if os.path.exists(locale_path):
                self.translator = QTranslator()
                self.translator.load(locale_path)
                QCoreApplication.installTranslator(self.translator)
        except Exception as e:
            self.log_message(f"Locale initialization failed: {str(e)}", Qgis.Warning)
        
        # Plugin variables
        self.actions = []
        self.menu = self.tr(u'&Mineral Prospectivity')
        
        # Initialize toolbar with error handling
        try:
            self.toolbar = self.iface.addToolBar(u'MineralProspectivityPlugin')
            self.toolbar.setObjectName(u'MineralProspectivityPlugin')
        except Exception as e:
            self.log_message(f"Toolbar initialization failed: {str(e)}", Qgis.Warning)
            self.toolbar = None
        
        # Dialogs - will be created on demand
        self.main_dlg = None
        self.fusion_dlg = None
        self.correlation_dlg = None
        
        # Check dependencies on startup
        self.dependency_issues = self.check_dependencies()
        
        self.log_message("Plugin initialized successfully", Qgis.Info)
    
    def tr(self, message):
        """Get translation for string"""
        return QCoreApplication.translate('MineralProspectivityPlugin', message)
    
    def check_dependencies(self):
        """Check for required dependencies and return list of issues"""
        issues = []
        
        # Check for essential packages
        try:
            import numpy
            self.log_message("NumPy: Available", Qgis.Info)
        except ImportError:
            issues.append("NumPy is required but not installed")
            self.log_message("NumPy: Missing", Qgis.Warning)
        
        # Check for raster I/O capabilities
        has_raster_io = False
        try:
            import rasterio
            has_raster_io = True
            self.log_message("Rasterio: Available", Qgis.Info)
        except ImportError:
            try:
                from osgeo import gdal
                has_raster_io = True
                self.log_message("GDAL: Available (rasterio not found)", Qgis.Info)
            except ImportError:
                issues.append("Either rasterio or GDAL is required but neither is installed")
                self.log_message("Raster I/O: No suitable library found", Qgis.Critical)
        
        # Check for optional but recommended packages
        try:
            import sklearn
            self.log_message("scikit-learn: Available", Qgis.Info)
        except ImportError:
            issues.append("scikit-learn is recommended for advanced analysis")
            self.log_message("scikit-learn: Missing (some features will be unavailable)", Qgis.Warning)
        
        try:
            from scipy import ndimage
            self.log_message("SciPy: Available", Qgis.Info)
        except ImportError:
            issues.append("SciPy is recommended for advanced processing")
            self.log_message("SciPy: Missing (some features will be unavailable)", Qgis.Warning)
        
        try:
            import geopandas
            self.log_message("GeoPandas: Available", Qgis.Info)
        except ImportError:
            self.log_message("GeoPandas: Missing (vector processing will be limited)", Qgis.Warning)
        
        return issues
    
    def show_dependency_warning(self):
        """Show warning about missing dependencies"""
        if self.dependency_issues:
            critical_issues = [issue for issue in self.dependency_issues 
                             if "required" in issue.lower()]
            
            if critical_issues:
                warning_msg = "Critical dependencies are missing:\n\n"
                warning_msg += "\n".join(f"• {issue}" for issue in critical_issues)
                warning_msg += "\n\nThe plugin may not function properly."
                warning_msg += "\n\nPlease install the missing packages:"
                warning_msg += "\n\npip install numpy rasterio"
                
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Critical Dependencies Missing",
                    warning_msg
                )
            elif len(self.dependency_issues) > 2:  # Only show if several optional deps missing
                warning_msg = "Some optional dependencies are missing:\n\n"
                warning_msg += "\n".join(f"• {issue}" for issue in self.dependency_issues)
                warning_msg += "\n\nFor full functionality, install:"
                warning_msg += "\n\npip install numpy rasterio scikit-learn scipy geopandas"
                
                QMessageBox.information(
                    self.iface.mainWindow(),
                    "Optional Dependencies",
                    warning_msg
                )
    
    def add_action(self, icon_path, text, callback, enabled_flag=True, 
                   add_to_menu=True, add_to_toolbar=True, status_tip=None,
                   whats_this=None, parent=None):
        """Add a toolbar icon to the toolbar with error handling"""
        
        try:
            # Handle missing icon files gracefully
            if not os.path.exists(icon_path):
                # Use default QGIS icon if custom icon is missing
                icon = self.iface.mainWindow().style().standardIcon(
                    self.iface.mainWindow().style().SP_FileIcon
                )
                self.log_message(f"Icon not found: {icon_path}, using default", Qgis.Warning)
            else:
                icon = QIcon(icon_path)
            
            action = QAction(icon, text, parent)
            action.triggered.connect(callback)
            action.setEnabled(enabled_flag)
            
            if status_tip is not None:
                action.setStatusTip(status_tip)
            
            if whats_this is not None:
                action.setWhatsThis(whats_this)
            
            if add_to_toolbar and self.toolbar is not None:
                self.toolbar.addAction(action)
            
            if add_to_menu:
                self.iface.addPluginToMenu(self.menu, action)
            
            self.actions.append(action)
            return action
            
        except Exception as e:
            self.log_message(f"Failed to add action '{text}': {str(e)}", Qgis.Critical)
            return None
    
    def initGui(self):
        """Create the menu entries and toolbar icons inside the QGIS GUI"""
        
        try:
            # Show dependency warning if needed (delayed to allow QGIS to fully load)
            if self.dependency_issues:
                from qgis.PyQt.QtCore import QTimer
                QTimer.singleShot(3000, self.show_dependency_warning)
            
            # Main plugin dialog
            icon_path = os.path.join(self.plugin_dir, 'icons', 'plugin_icon.svg')
            self.add_action(
                icon_path,
                text=self.tr(u'Mineral Prospectivity Mapping'),
                callback=self.run_main,
                status_tip=self.tr(u'Open Mineral Prospectivity Mapping tool'),
                parent=self.iface.mainWindow()
            )
            
            # Data Fusion dialog
            fusion_icon = os.path.join(self.plugin_dir, 'icons', 'fusion.svg')
            self.add_action(
                fusion_icon,
                text=self.tr(u'Multi-Source Data Fusion'),
                callback=self.run_fusion,
                status_tip=self.tr(u'Open Multi-Source Data Fusion tool'),
                parent=self.iface.mainWindow()
            )
            
            # Correlation Analysis dialog
            corr_icon = os.path.join(self.plugin_dir, 'icons', 'correlation.svg')
            self.add_action(
                corr_icon,
                text=self.tr(u'Correlation Analysis'),
                callback=self.run_correlation,
                status_tip=self.tr(u'Open Correlation Analysis tool'),
                parent=self.iface.mainWindow()
            )
            
            # Add separator to toolbar
            if self.toolbar is not None:
                self.toolbar.addSeparator()
            
            # Mineral Mapping submenu
            self.mineral_menu = QMenu(self.tr(u'Mineral Mapping'), self.iface.mainWindow())
            
            # ASTER processing
            self.add_aster_action = QAction(
                self.tr(u'Process ASTER L2 Data'), self.iface.mainWindow()
            )
            self.add_aster_action.triggered.connect(self.process_aster)
            self.mineral_menu.addAction(self.add_aster_action)
            
            # Sentinel-2 processing
            self.add_s2_action = QAction(
                self.tr(u'Process Sentinel-2 Data'), self.iface.mainWindow()
            )
            self.add_s2_action.triggered.connect(self.process_sentinel2)
            self.mineral_menu.addAction(self.add_s2_action)
            
            # Geological processing
            self.add_geo_action = QAction(
                self.tr(u'Process Geological Data'), self.iface.mainWindow()
            )
            self.add_geo_action.triggered.connect(self.process_geological)
            self.mineral_menu.addAction(self.add_geo_action)
            
            # Add submenu to main menu
            self.iface.pluginMenu().addMenu(self.mineral_menu)
            
            self.log_message("GUI initialized successfully", Qgis.Info)
            
        except Exception as e:
            self.log_message(f"GUI initialization failed: {str(e)}", Qgis.Critical)
            QMessageBox.critical(
                self.iface.mainWindow(),
                "Plugin Initialization Error",
                f"Failed to initialize plugin GUI:\n{str(e)}"
            )
    
    def unload(self):
        """Remove the plugin menu item and icon from QGIS GUI"""
        try:
            # Remove actions
            for action in self.actions:
                self.iface.removePluginMenu(self.tr(u'&Mineral Prospectivity'), action)
                self.iface.removeToolBarIcon(action)
            
            # Remove toolbar
            if hasattr(self, 'toolbar') and self.toolbar is not None:
                del self.toolbar
            
            # Remove mineral menu
            if hasattr(self, 'mineral_menu'):
                self.iface.pluginMenu().removeAction(self.mineral_menu.menuAction())
            
            # Clean up dialogs
            if self.main_dlg:
                self.main_dlg.close()
                self.main_dlg = None
            if self.fusion_dlg:
                self.fusion_dlg.close()
                self.fusion_dlg = None
            if self.correlation_dlg:
                self.correlation_dlg.close()
                self.correlation_dlg = None
            
            self.log_message("Plugin unloaded successfully", Qgis.Info)
            
        except Exception as e:
            self.log_message(f"Error during unload: {str(e)}", Qgis.Warning)
    
    def run_main(self):
        """Run main plugin dialog with comprehensive error handling"""
        try:
            # Check critical dependencies first
            critical_missing = [issue for issue in self.dependency_issues 
                              if "required" in issue.lower()]
            if critical_missing:
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Missing Dependencies",
                    "Cannot open main dialog due to missing critical dependencies.\n\n"
                    + "\n".join(critical_missing)
                )
                return
            
            # Import here to avoid import errors on plugin load
            try:
                from .ui.main_dialog import MainDialog
            except ImportError:
                # Try alternative import path
                from ui.main_dialog import MainDialog
            
            if self.main_dlg is None:
                self.main_dlg = MainDialog(self.iface)
            
            self.main_dlg.show()
            result = self.main_dlg.exec_()
            
            if result:
                self.log_message("Main processing completed", Qgis.Info)
                
        except ImportError as e:
            self.show_import_error("Main Dialog", str(e))
        except Exception as e:
            self.show_generic_error("Main Dialog", str(e))
    
    def run_fusion(self):
        """Run data fusion dialog with error handling"""
        try:
            try:
                from .ui.data_fusion_dialog import DataFusionDialog
            except ImportError:
                from ui.data_fusion_dialog import DataFusionDialog
            
            if self.fusion_dlg is None:
                self.fusion_dlg = DataFusionDialog(self.iface)
            
            self.fusion_dlg.show()
            result = self.fusion_dlg.exec_()
            
            if result:
                self.log_message("Data fusion completed", Qgis.Info)
                
        except ImportError as e:
            self.show_import_error("Data Fusion Dialog", str(e))
        except Exception as e:
            self.show_generic_error("Data Fusion Dialog", str(e))
    
    def run_correlation(self):
        """Run correlation analysis dialog with error handling"""
        try:
            try:
                from .ui.correlation_dialog import CorrelationDialog
            except ImportError:
                from ui.correlation_dialog import CorrelationDialog
            
            if self.correlation_dlg is None:
                self.correlation_dlg = CorrelationDialog(self.iface)
            
            self.correlation_dlg.show()
            result = self.correlation_dlg.exec_()
            
            if result:
                self.log_message("Correlation analysis completed", Qgis.Info)
                
        except ImportError as e:
            self.show_import_error("Correlation Dialog", str(e))
        except Exception as e:
            self.show_generic_error("Correlation Dialog", str(e))
    
    def process_aster(self):
        """Process ASTER L2 data with comprehensive error handling"""
        try:
            # Check for required dependencies
            missing_deps = []
            try:
                import numpy
            except ImportError:
                missing_deps.append("numpy")
            
            has_io = False
            try:
                import rasterio
                has_io = True
            except ImportError:
                try:
                    from osgeo import gdal
                    has_io = True
                except ImportError:
                    missing_deps.append("rasterio or GDAL")
            
            if missing_deps:
                QMessageBox.critical(
                    self.iface.mainWindow(),
                    "Missing Dependencies",
                    f"ASTER processing requires: {', '.join(missing_deps)}\n\n"
                    f"Install with: pip install {' '.join(missing_deps)}"
                )
                return
            
            try:
                from .processing.aster_processor import AsterProcessor
            except ImportError:
                from processing.aster_processor import AsterProcessor
            
            processor = AsterProcessor(self.iface)
            success = processor.process_data()
            
            if success:
                self.log_message("ASTER processing completed successfully", Qgis.Info)
                QMessageBox.information(
                    self.iface.mainWindow(),
                    "Processing Complete",
                    "ASTER data processing completed successfully!"
                )
            else:
                self.log_message("ASTER processing failed or was cancelled", Qgis.Warning)
                
        except ImportError as e:
            self.show_import_error("ASTER Processor", str(e))
        except Exception as e:
            self.show_generic_error("ASTER Processing", str(e))
    
    def process_sentinel2(self):
        """Process Sentinel-2 data with error handling"""
        try:
            try:
                from .processing.sentinel2_processor import Sentinel2Processor
            except ImportError:
                from processing.sentinel2_processor import Sentinel2Processor
            
            processor = Sentinel2Processor(self.iface)
            processor.process_data()
            self.log_message("Sentinel-2 processing completed", Qgis.Info)
            
        except ImportError as e:
            self.show_import_error("Sentinel-2 Processor", str(e))
        except Exception as e:
            self.show_generic_error("Sentinel-2 Processing", str(e))
    
    def process_geological(self):
        """Process geological data with error handling"""
        try:
            try:
                from .processing.geological_processor import GeologicalProcessor
            except ImportError:
                from processing.geological_processor import GeologicalProcessor
            
            processor = GeologicalProcessor(self.iface)
            result = processor.process_geological_map()
            
            if result:
                self.log_message("Geological processing completed", Qgis.Info)
            else:
                self.log_message("Geological processing cancelled", Qgis.Info)
                
        except ImportError as e:
            self.show_import_error("Geological Processor", str(e))
        except Exception as e:
            self.show_generic_error("Geological Processing", str(e))
    
    def show_import_error(self, component, error):
        """Show user-friendly import error message"""
        self.log_message(f"{component} import error: {error}", Qgis.Critical)
        
        # Try to provide helpful suggestions
        suggestions = []
        if "numpy" in error:
            suggestions.append("pip install numpy")
        if "rasterio" in error:
            suggestions.append("pip install rasterio")
        if "sklearn" in error:
            suggestions.append("pip install scikit-learn")
        if "scipy" in error:
            suggestions.append("pip install scipy")
        
        suggestion_text = ""
        if suggestions:
            suggestion_text = f"\n\nTry installing:\n{chr(10).join(suggestions)}"
        
        QMessageBox.critical(
            self.iface.mainWindow(),
            f"{component} - Import Error",
            f"Failed to load {component}.\n\n"
            f"This may be due to missing dependencies.\n"
            f"Technical details: {error}"
            f"{suggestion_text}"
        )
    
    def show_generic_error(self, component, error):
        """Show generic error message"""
        self.log_message(f"{component} error: {error}", Qgis.Critical)
        
        QMessageBox.critical(
            self.iface.mainWindow(),
            f"{component} - Error",
            f"An error occurred in {component}:\n\n{error}\n\n"
            f"Check the QGIS message log for more details."
        )
    
    def log_message(self, message, level=Qgis.Info):
        """Log message to QGIS message log"""
        try:
            QgsMessageLog.logMessage(message, 'Mineral Prospectivity', level)
        except Exception:
            # Fallback if logging fails
            print(f"Mineral Prospectivity: {message}")
    
    def get_plugin_info(self):
        """Get plugin information for debugging"""
        return {
            'name': 'Mineral Prospectivity Mapping',
            'version': '1.0.0',
            'description': 'Advanced geological data processing and analysis for mineral exploration',
            'author': 'Geological Survey Team',
            'plugin_dir': self.plugin_dir,
            'dependencies_checked': True,
            'dependency_issues': self.dependency_issues,
            'actions_count': len(self.actions),
            'toolbar_available': self.toolbar is not None
        }
    
    def run_diagnostic(self):
        """Run diagnostic check for troubleshooting"""
        diagnostic_info = []
        
        # Check plugin directory
        diagnostic_info.append(f"Plugin directory: {self.plugin_dir}")
        diagnostic_info.append(f"Plugin directory exists: {os.path.exists(self.plugin_dir)}")
        
        # Check key files
        key_files = [
            '__init__.py',
            'mineral_prospectivity_plugin.py',
            'ui/__init__.py',
            'ui/main_dialog.py',
            'processing/aster_processor.py'
        ]
        
        for file_path in key_files:
            full_path = os.path.join(self.plugin_dir, file_path)
            exists = os.path.exists(full_path)
            diagnostic_info.append(f"{file_path}: {'✅' if exists else '❌'}")
        
        # Check dependencies
        diagnostic_info.append("\nDependency Status:")
        dependencies = ['numpy', 'rasterio', 'sklearn', 'scipy', 'geopandas']
        for dep in dependencies:
            try:
                if dep == 'sklearn':
                    import sklearn
                else:
                    __import__(dep)
                diagnostic_info.append(f"{dep}: ✅")
            except ImportError:
                diagnostic_info.append(f"{dep}: ❌")
        
        # Check QGIS integration
        diagnostic_info.append(f"\nQGIS Integration:")
        diagnostic_info.append(f"Toolbar created: {'✅' if self.toolbar else '❌'}")
        diagnostic_info.append(f"Actions registered: {len(self.actions)}")
        diagnostic_info.append(f"Menu items: {'✅' if hasattr(self, 'mineral_menu') else '❌'}")
        
        return "\n".join(diagnostic_info)


# Helper functions for standalone use
def check_plugin_health():
    """Standalone function to check plugin health"""
    issues = []
    
    # Check essential imports
    try:
        import numpy
    except ImportError:
        issues.append("NumPy not available")
    
    try:
        import rasterio
    except ImportError:
        try:
            from osgeo import gdal
        except ImportError:
            issues.append("No raster I/O library (rasterio or GDAL)")
    
    try:
        from qgis.core import QgsProject, QgsRasterLayer
        from qgis.PyQt.QtWidgets import QDialog
    except ImportError:
        issues.append("QGIS integration error")
    
    if not issues:
        return "✅ Plugin is healthy and ready to use!"
    else:
        return f"❌ Issues found:\n" + "\n".join(f"• {issue}" for issue in issues)


def create_dependency_report():
    """Create a comprehensive dependency status report"""
    report = ["Mineral Prospectivity Plugin - Dependency Report"]
    report.append("=" * 50)
    report.append("")
    
    # Essential dependencies
    report.append("Essential Dependencies:")
    essential = {
        'numpy': 'NumPy - Numerical computing',
        'rasterio': 'Rasterio - Raster I/O',
        'gdal': 'GDAL - Geospatial data abstraction (alternative to rasterio)'
    }
    
    for package, description in essential.items():
        try:
            if package == 'gdal':
                from osgeo import gdal
            else:
                __import__(package)
            report.append(f"✅ {description}")
        except ImportError:
            report.append(f"❌ {description}")
    
    report.append("")
    report.append("Optional Dependencies:")
    optional = {
        'sklearn': 'scikit-learn - Machine learning',
        'scipy': 'SciPy - Scientific computing',
        'geopandas': 'GeoPandas - Vector data processing'
    }
    
    for package, description in optional.items():
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            report.append(f"✅ {description}")
        except ImportError:
            report.append(f"⚠️ {description} (optional)")
    
    report.append("")
    
    # Installation commands
    missing_essential = []
    missing_optional = []
    
    for package in essential.keys():
        try:
            if package == 'gdal':
                from osgeo import gdal
            else:
                __import__(package)
        except ImportError:
            if package == 'gdal':
                # Only add GDAL if rasterio is also missing
                try:
                    import rasterio
                except ImportError:
                    missing_essential.append('gdal')
            else:
                missing_essential.append(package)
    
    for package in optional.keys():
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
        except ImportError:
            missing_optional.append(package)
    
    if missing_essential:
        report.append("Install essential packages:")
        install_cmd = missing_essential.copy()
        if 'gdal' in install_cmd:
            install_cmd.remove('gdal')
            install_cmd.append('rasterio')  # Prefer rasterio over GDAL
        report.append(f"pip install {' '.join(install_cmd)}")
        report.append("")
    
    if missing_optional:
        report.append("Install optional packages for full functionality:")
        report.append(f"pip install {' '.join(missing_optional)}")
        report.append("")
    
    if not missing_essential and not missing_optional:
        report.append("🎉 All dependencies are satisfied!")
    
    return "\n".join(report)