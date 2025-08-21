"""
Dependency checker for QGIS Mineral Prospectivity Plugin
"""

def check_dependencies():
    """Check which dependencies are available and provide installation instructions"""
    missing_packages = []
    available_packages = []
    
    # Check core dependencies
    dependencies = {
        'numpy': 'numpy',
        'scipy': 'scipy', 
        'pandas': 'pandas',
        'scikit-learn': 'sklearn',
        'matplotlib': 'matplotlib',
        'rasterio': 'rasterio',
        'scikit-fuzzy': 'skfuzzy',
        'spectral': 'spectral',
        'opencv-python': 'cv2',
        # 'h5py': 'h5py',  # Removed - using GDAL's HDF5 support instead
        'netcdf4': 'netCDF4',
        'scikit-image': 'skimage',
        'geopandas': 'geopandas',
        'Pillow': 'PIL'
    }
    
    for package_name, import_name in dependencies.items():
        try:
            __import__(import_name)
            available_packages.append(package_name)
        except ImportError:
            missing_packages.append(package_name)
    
    return available_packages, missing_packages

def get_installation_command(missing_packages):
    """Get pip installation command for missing packages"""
    if not missing_packages:
        return "All dependencies are installed!"
    
    return f"pip install {' '.join(missing_packages)}"

if __name__ == "__main__":
    print("QGIS Mineral Prospectivity Plugin - Dependency Check")
    print("=" * 55)
    
    available, missing = check_dependencies()
    
    print(f"✅ Available packages ({len(available)}):")
    for pkg in available:
        print(f"   • {pkg}")
    
    if missing:
        print(f"\n❌ Missing packages ({len(missing)}):")
        for pkg in missing:
            print(f"   • {pkg}")
        
        print(f"\n📦 To install missing dependencies, run:")
        print(f"   {get_installation_command(missing)}")
    else:
        print("\n🎉 All dependencies are available!")