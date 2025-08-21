"""
Mineral Prospectivity Mapping Plugin
Advanced QGIS plugin for mineral exploration and geological mapping
"""

def classFactory(iface):
    """Load MineralProspectivityPlugin class from file mineral_prospectivity_plugin"""
    from .mineral_prospectivity_plugin import MineralProspectivityPlugin
    return MineralProspectivityPlugin(iface)
