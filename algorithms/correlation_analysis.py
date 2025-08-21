"""
Statistical correlation analysis for geological datasets
"""

import numpy as np
import pandas as pd
import rasterio
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import json
import os

class CorrelationAnalyzer:
    """Advanced correlation analysis for geological data"""
    
    def __init__(self):
        self.data_arrays = {}
        self.layer_names = {}
        self.correlation_matrix = None
        self.p_values_matrix = None
        self.advanced_stats = {}
        
    def load_layers_data(self, layer_ids):
        """Load data from multiple layers"""
        from qgis.core import QgsProject
        
        project = QgsProject.instance()
        loaded_data = {}
        layer_names = {}
        
        # Reference properties
        reference_shape = None
        reference_transform = None
        reference_crs = None
        
        for layer_id in layer_ids:
            layer = project.mapLayer(layer_id)
            if layer is None:
                continue
                
            layer_path = layer.source()
            layer_name = layer.name()
            
            try:
                with rasterio.open(layer_path) as src:
                    data = src.read(1)  # Read first band
                    
                    if reference_shape is None:
                        reference_shape = data.shape
                        reference_transform = src.transform
                        reference_crs = src.crs
                    else:
                        # Ensure consistent dimensions
                        if data.shape != reference_shape:
                            from rasterio.warp import reproject, Resampling
                            aligned_data = np.empty(reference_shape, dtype=np.float32)
                            reproject(
                                source=data,
                                destination=aligned_data,
                                src_transform=src.transform,
                                src_crs=src.crs,
                                dst_transform=reference_transform,
                                dst_crs=reference_crs,
                                resampling=Resampling.bilinear
                            )
                            data = aligned_data
                    
                    loaded_data[layer_id] = data.flatten()
                    layer_names[layer_id] = layer_name
                    
            except Exception as e:
                print(f"Error loading layer {layer_id}: {str(e)}")
                continue
        
        if not loaded_data:
            raise ValueError("No layers could be loaded")
        
        self.data_arrays = loaded_data
        self.layer_names = layer_names
        
        return loaded_data
    
    def get_layer_name(self, layer_id):
        """Get layer name by ID"""
        return self.layer_names.get(layer_id, f"Layer_{layer_id}")
    
    def calculate_correlation_matrix(self, data_arrays=None, method='pearson'):
        """Calculate correlation matrix between layers"""
        if data_arrays is None:
            data_arrays = self.data_arrays
        
        if not data_arrays:
            raise ValueError("No data available for correlation analysis")
        
        # Create data matrix
        layer_ids = list(data_arrays.keys())
        n_layers = len(layer_ids)
        
        # Find common valid pixels
        data_matrix = np.column_stack([data_arrays[lid] for lid in layer_ids])
        valid_mask = np.all(np.isfinite(data_matrix), axis=1) & np.all(data_matrix != 0, axis=1)
        
        if valid_mask.sum() == 0:
            raise ValueError("No valid pixels found for correlation analysis")
        
        valid_data = data_matrix[valid_mask]
        
        # Calculate correlation matrix
        correlation_matrix = np.full((n_layers, n_layers), np.nan)
        
        for i in range(n_layers):
            for j in range(n_layers):
                if i == j:
                    correlation_matrix[i, j] = 1.0
                else:
                    x = valid_data[:, i]
                    y = valid_data[:, j]
                    
                    if method == 'pearson':
                        corr, _ = stats.pearsonr(x, y)
                    elif method == 'spearman':
                        corr, _ = stats.spearmanr(x, y)
                    elif method == 'kendall':
                        corr, _ = stats.kendalltau(x, y)
                    else:
                        corr, _ = stats.pearsonr(x, y)
                    
                    correlation_matrix[i, j] = corr
        
        self.correlation_matrix = correlation_matrix
        return correlation_matrix
    
    def calculate_p_values(self, data_arrays=None, method='pearson'):
        """Calculate p-values for correlation coefficients"""
        if data_arrays is None:
            data_arrays = self.data_arrays
        
        if not data_arrays:
            raise ValueError("No data available for p-value calculation")
        
        # Create data matrix
        layer_ids = list(data_arrays.keys())
        n_layers = len(layer_ids)
        
        data_matrix = np.column_stack([data_arrays[lid] for lid in layer_ids])
        valid_mask = np.all(np.isfinite(data_matrix), axis=1) & np.all(data_matrix != 0, axis=1)
        valid_data = data_matrix[valid_mask]
        
        # Calculate p-values matrix
        p_values_matrix = np.full((n_layers, n_layers), np.nan)
        
        for i in range(n_layers):
            for j in range(n_layers):
                if i == j:
                    p_values_matrix[i, j] = 0.0
                else:
                    x = valid_data[:, i]
                    y = valid_data[:, j]
                    
                    if method == 'pearson':
                        _, p_val = stats.pearsonr(x, y)
                    elif method == 'spearman':
                        _, p_val = stats.spearmanr(x, y)
                    elif method == 'kendall':
                        _, p_val = stats.kendalltau(x, y)
                    else:
                        _, p_val = stats.pearsonr(x, y)
                    
                    p_values_matrix[i, j] = p_val
        
        self.p_values_matrix = p_values_matrix
        return p_values_matrix
    
    def calculate_partial_correlation(self, controlling_variables=None):
        """Calculate partial correlation coefficients"""
        if not self.data_arrays:
            raise ValueError("No data available for partial correlation analysis")
        
        # Create data matrix
        layer_ids = list(self.data_arrays.keys())
        data_matrix = np.column_stack([self.data_arrays[lid] for lid in layer_ids])
        valid_mask = np.all(np.isfinite(data_matrix), axis=1) & np.all(data_matrix != 0, axis=1)
        valid_data = data_matrix[valid_mask]
        
        if valid_data.shape[0] < 10:
            raise ValueError("Insufficient valid data for partial correlation")
        
        # Calculate partial correlations
        n_layers = len(layer_ids)
        partial_corr_matrix = np.full((n_layers, n_layers), np.nan)
        
        # Calculate precision matrix (inverse of covariance matrix)
        try:
            cov_matrix = np.cov(valid_data.T)
            precision_matrix = np.linalg.inv(cov_matrix)
            
            # Partial correlations from precision matrix
            for i in range(n_layers):
                for j in range(n_layers):
                    if i == j:
                        partial_corr_matrix[i, j] = 1.0
                    else:
                        partial_corr = -precision_matrix[i, j] / np.sqrt(
                            precision_matrix[i, i] * precision_matrix[j, j]
                        )
                        partial_corr_matrix[i, j] = partial_corr
                        
        except np.linalg.LinAlgError:
            # If matrix is singular, use pairwise partial correlations
            for i in range(n_layers):
                for j in range(i + 1, n_layers):
                    # Control for all other variables
                    control_vars = [k for k in range(n_layers) if k != i and k != j]
                    
                    if control_vars:
                        x = valid_data[:, i]
                        y = valid_data[:, j]
                        z = valid_data[:, control_vars]
                        
                        partial_corr = self._partial_correlation(x, y, z)
                        partial_corr_matrix[i, j] = partial_corr
                        partial_corr_matrix[j, i] = partial_corr
                    else:
                        corr, _ = stats.pearsonr(valid_data[:, i], valid_data[:, j])
                        partial_corr_matrix[i, j] = corr
                        partial_corr_matrix[j, i] = corr
        
        return partial_corr_matrix
    
    def _partial_correlation(self, x, y, z):
        """Calculate partial correlation between x and y controlling for z"""
        if z.shape[1] == 0:
            corr, _ = stats.pearsonr(x, y)
            return corr
        
        # Regress x on z
        from sklearn.linear_model import LinearRegression
        reg_x = LinearRegression().fit(z, x)
        x_residuals = x - reg_x.predict(z)
        
        # Regress y on z
        reg_y = LinearRegression().fit(z, y)
        y_residuals = y - reg_y.predict(z)
        
        # Correlation of residuals
        if len(x_residuals) > 1 and len(y_residuals) > 1:
            corr, _ = stats.pearsonr(x_residuals, y_residuals)
            return corr
        else:
            return 0.0
    
    def calculate_mutual_information(self):
        """Calculate mutual information between layers"""
        if not self.data_arrays:
            raise ValueError("No data available for mutual information analysis")
        
        # Create data matrix
        layer_ids = list(self.data_arrays.keys())
        data_matrix = np.column_stack([self.data_arrays[lid] for lid in layer_ids])
        valid_mask = np.all(np.isfinite(data_matrix), axis=1) & np.all(data_matrix != 0, axis=1)
        valid_data = data_matrix[valid_mask]
        
        n_layers = len(layer_ids)
        mi_matrix = np.full((n_layers, n_layers), np.nan)
        
        for i in range(n_layers):
            for j in range(n_layers):
                if i == j:
                    mi_matrix[i, j] = 0.0  # Self MI is entropy, set to 0 for normalization
                else:
                    x = valid_data[:, i].reshape(-1, 1)
                    y = valid_data[:, j]
                    
                    try:
                        mi = mutual_info_regression(x, y, random_state=42)[0]
                        mi_matrix[i, j] = mi
                    except:
                        mi_matrix[i, j] = 0.0
        
        return mi_matrix
    
    def calculate_distance_correlation(self):
        """Calculate distance correlation between layers"""
        if not self.data_arrays:
            raise ValueError("No data available for distance correlation analysis")
        
        # Create data matrix
        layer_ids = list(self.data_arrays.keys())
        data_matrix = np.column_stack([self.data_arrays[lid] for lid in layer_ids])
        valid_mask = np.all(np.isfinite(data_matrix), axis=1) & np.all(data_matrix != 0, axis=1)
        valid_data = data_matrix[valid_mask]
        
        # Subsample if too many points (for computational efficiency)
        if len(valid_data) > 10000:
            indices = np.random.choice(len(valid_data), 10000, replace=False)
            valid_data = valid_data[indices]
        
        n_layers = len(layer_ids)
        dcorr_matrix = np.full((n_layers, n_layers), np.nan)
        
        for i in range(n_layers):
            for j in range(n_layers):
                if i == j:
                    dcorr_matrix[i, j] = 1.0
                else:
                    x = valid_data[:, i]
                    y = valid_data[:, j]
                    
                    dcorr = self._distance_correlation(x, y)
                    dcorr_matrix[i, j] = dcorr
        
        return dcorr_matrix
    
    def _distance_correlation(self, x, y):
        """Calculate distance correlation between two variables"""
        n = len(x)
        if n < 2:
            return 0.0
        
        # Calculate distance matrices
        x_dist = pdist(x.reshape(-1, 1))
        y_dist = pdist(y.reshape(-1, 1))
        
        x_dist_matrix = squareform(x_dist)
        y_dist_matrix = squareform(y_dist)
        
        # Center the distance matrices
        x_centered = x_dist_matrix - np.mean(x_dist_matrix, axis=0) - np.mean(x_dist_matrix, axis=1)[:, np.newaxis] + np.mean(x_dist_matrix)
        y_centered = y_dist_matrix - np.mean(y_dist_matrix, axis=0) - np.mean(y_dist_matrix, axis=1)[:, np.newaxis] + np.mean(y_dist_matrix)
        
        # Calculate distance covariance and variances
        dcov_xy = np.sqrt(np.mean(x_centered * y_centered))
        dcov_xx = np.sqrt(np.mean(x_centered * x_centered))
        dcov_yy = np.sqrt(np.mean(y_centered * y_centered))
        
        # Calculate distance correlation
        if dcov_xx > 0 and dcov_yy > 0:
            dcorr = dcov_xy / np.sqrt(dcov_xx * dcov_yy)
        else:
            dcorr = 0.0
        
        return dcorr
    
    def compute_advanced_statistics(self, data_arrays=None):
        """Compute advanced statistical measures"""
        if data_arrays is None:
            data_arrays = self.data_arrays
        
        if not data_arrays:
            raise ValueError("No data available for advanced statistics")
        
        # Create data matrix
        layer_ids = list(data_arrays.keys())
        data_matrix = np.column_stack([data_arrays[lid] for lid in layer_ids])
        valid_mask = np.all(np.isfinite(data_matrix), axis=1) & np.all(data_matrix != 0, axis=1)
        valid_data = data_matrix[valid_mask]
        
        advanced_stats = {}
        
        # Mutual information matrix
        try:
            mi_matrix = self.calculate_mutual_information()
            advanced_stats['mutual_information'] = mi_matrix
        except Exception as e:
            print(f"Error calculating mutual information: {str(e)}")
        
        # Partial correlation matrix
        try:
            partial_corr_matrix = self.calculate_partial_correlation()
            advanced_stats['partial_correlation'] = partial_corr_matrix
        except Exception as e:
            print(f"Error calculating partial correlation: {str(e)}")
        
        # Distance correlation matrix
        try:
            dcorr_matrix = self.calculate_distance_correlation()
            advanced_stats['distance_correlation'] = dcorr_matrix
        except Exception as e:
            print(f"Error calculating distance correlation: {str(e)}")
        
        # Principal component analysis
        try:
            from sklearn.decomposition import PCA
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(valid_data)
            
            pca = PCA()
            pca.fit(scaled_data)
            
            advanced_stats['pca'] = {
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'components': pca.components_.tolist(),
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_).tolist()
            }
        except Exception as e:
            print(f"Error calculating PCA: {str(e)}")
        
        # Cluster analysis
        try:
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
            
            silhouette_scores = []
            for n_clusters in range(2, min(10, len(layer_ids))):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(valid_data)
                
                if len(np.unique(cluster_labels)) > 1:
                    silhouette_avg = silhouette_score(valid_data, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
                else:
                    silhouette_scores.append(0)
            
            advanced_stats['clustering'] = {
                'silhouette_scores': silhouette_scores,
                'optimal_clusters': np.argmax(silhouette_scores) + 2 if silhouette_scores else 2
            }
        except Exception as e:
            print(f"Error in cluster analysis: {str(e)}")
        
        self.advanced_stats = advanced_stats
        return advanced_stats
    
    def calculate_layer_correlations(self, layers):
        """Calculate correlations for QGIS layers"""
        layer_data = {}
        
        for layer in layers:
            layer_path = layer.source()
            layer_name = layer.name()
            
            try:
                with rasterio.open(layer_path) as src:
                    data = src.read(1).flatten()
                    layer_data[layer_name] = data
            except Exception as e:
                print(f"Error loading layer {layer_name}: {str(e)}")
                continue
        
        if len(layer_data) < 2:
            return []
        
        # Calculate pairwise correlations
        correlations = []
        layer_names = list(layer_data.keys())
        
        for i in range(len(layer_names)):
            for j in range(i + 1, len(layer_names)):
                name1, name2 = layer_names[i], layer_names[j]
                data1, data2 = layer_data[name1], layer_data[name2]
                
                # Find common valid pixels
                valid_mask = np.isfinite(data1) & np.isfinite(data2) & (data1 != 0) & (data2 != 0)
                
                if valid_mask.sum() > 10:  # Need minimum pixels
                    valid_data1 = data1[valid_mask]
                    valid_data2 = data2[valid_mask]
                    
                    try:
                        corr, p_val = stats.pearsonr(valid_data1, valid_data2)
                        correlations.append((name1, name2, corr, p_val))
                    except:
                        correlations.append((name1, name2, 0.0, 1.0))
        
        return correlations
    
    def export_correlation_results(self, output_dir):
        """Export correlation analysis results"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Export correlation matrix
        if self.correlation_matrix is not None:
            layer_names = [self.get_layer_name(lid) for lid in self.data_arrays.keys()]
            
            corr_df = pd.DataFrame(
                self.correlation_matrix,
                index=layer_names,
                columns=layer_names
            )
            corr_df.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))
        
        # Export p-values matrix
        if self.p_values_matrix is not None:
            p_val_df = pd.DataFrame(
                self.p_values_matrix,
                index=layer_names,
                columns=layer_names
            )
            p_val_df.to_csv(os.path.join(output_dir, 'p_values_matrix.csv'))
        
        # Export advanced statistics
        if self.advanced_stats:
            with open(os.path.join(output_dir, 'advanced_statistics.json'), 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                stats_for_json = {}
                for key, value in self.advanced_stats.items():
                    if isinstance(value, np.ndarray):
                        stats_for_json[key] = value.tolist()
                    else:
                        stats_for_json[key] = value
                
                json.dump(stats_for_json, f, indent=2)
        
        return output_dir

class StatisticalAnalyzer:
    """Additional statistical analysis tools"""
    
    def __init__(self):
        pass
    
    def multivariate_normality_test(self, data):
        """Test for multivariate normality"""
        from scipy.stats import normaltest
        
        results = {}
        for i, column in enumerate(data.T):
            stat, p_val = normaltest(column[np.isfinite(column)])
            results[f'variable_{i}'] = {'statistic': stat, 'p_value': p_val}
        
        return results
    
    def outlier_detection(self, data, method='iqr'):
        """Detect outliers in multivariate data"""
        if method == 'iqr':
            outliers = np.zeros(data.shape[0], dtype=bool)
            
            for i in range(data.shape[1]):
                column = data[:, i]
                valid_data = column[np.isfinite(column)]
                
                if len(valid_data) > 0:
                    Q1 = np.percentile(valid_data, 25)
                    Q3 = np.percentile(valid_data, 75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    column_outliers = (column < lower_bound) | (column > upper_bound)
                    outliers |= column_outliers
            
            return outliers
        
        elif method == 'mahalanobis':
            from scipy.spatial.distance import mahalanobis
            
            valid_mask = np.all(np.isfinite(data), axis=1)
            valid_data = data[valid_mask]
            
            if len(valid_data) < data.shape[1] + 1:
                return np.zeros(data.shape[0], dtype=bool)
            
            try:
                mean = np.mean(valid_data, axis=0)
                cov = np.cov(valid_data.T)
                inv_cov = np.linalg.inv(cov)
                
                distances = np.array([
                    mahalanobis(row, mean, inv_cov) if np.all(np.isfinite(row)) else 0
                    for row in data
                ])
                
                threshold = np.percentile(distances[distances > 0], 95)
                outliers = distances > threshold
                
                return outliers
                
            except np.linalg.LinAlgError:
                return np.zeros(data.shape[0], dtype=bool)
