"""
Machine Learning city analysis module for NeuroUrban system.
"""

import logging
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import joblib
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from src.config.settings import Config
from src.data.data_collector import CityDataCollector
from src.utils.gpu_utils import get_gpu_manager

# Try to import GPU-accelerated libraries (only if PyTorch is allowed)
CUML_AVAILABLE = False
if not os.environ.get('NEUROURBAN_NO_PYTORCH', False):
    try:
        import cupy as cp
        import cuml
        from cuml.cluster import KMeans as cuKMeans
        from cuml.decomposition import PCA as cuPCA
        from cuml.preprocessing import StandardScaler as cuStandardScaler
        from cuml.ensemble import RandomForestRegressor as cuRandomForestRegressor
        CUML_AVAILABLE = True
    except ImportError:
        CUML_AVAILABLE = False

class CityAnalyzer:
    """Analyzes cities using machine learning techniques."""

    def __init__(self, config: Config):
        """
        Initialize the city analyzer with GPU acceleration support.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize GPU manager
        self.gpu_manager = get_gpu_manager(config)
        self.use_gpu = self.gpu_manager.is_gpu_available() and CUML_AVAILABLE

        # ML models and data (GPU or CPU versions)
        if self.use_gpu:
            self.logger.info("🚀 Using GPU-accelerated ML algorithms (cuML)")
            self.scaler = cuStandardScaler()
            self.pca = cuPCA()
            self.kmeans = cuKMeans()
            self.rf_model = cuRandomForestRegressor()
        else:
            self.logger.info("🖥️ Using CPU-based ML algorithms (scikit-learn)")
            self.scaler = StandardScaler()
            self.pca = PCA()
            self.kmeans = KMeans()
            self.rf_model = RandomForestRegressor()

        # Analysis results
        self.city_features = None
        self.clusters = None
        self.rankings = None
        self.feature_importance = None

        # Performance tracking
        self.analysis_stats = {
            'processing_times': {},
            'memory_usage': {},
            'gpu_utilization': []
        }

    def analyze_all_cities(self) -> Dict:
        """
        Perform comprehensive analysis of all cities with GPU acceleration.

        Returns:
            Dictionary containing analysis results
        """
        start_time = time.time()
        self.logger.info("🚀 Starting GPU-accelerated city analysis...")

        # Load city data
        data_collector = CityDataCollector(self.config)
        city_data = data_collector.load_collected_data()

        if not city_data:
            self.logger.warning("No city data found. Collecting data first...")
            city_data = data_collector.collect_all_data()

        # Prepare features with performance tracking
        prep_start = time.time()
        self.city_features = self._prepare_features(city_data)
        self.analysis_stats['processing_times']['feature_preparation'] = time.time() - prep_start

        # Perform clustering analysis
        cluster_start = time.time()
        self.clusters = self._perform_clustering()
        self.analysis_stats['processing_times']['clustering'] = time.time() - cluster_start

        # Generate city rankings (can be parallelized)
        ranking_start = time.time()
        self.rankings = self._generate_rankings_parallel()
        self.analysis_stats['processing_times']['ranking'] = time.time() - ranking_start

        # Analyze feature importance
        importance_start = time.time()
        self.feature_importance = self._analyze_feature_importance()
        self.analysis_stats['processing_times']['feature_importance'] = time.time() - importance_start

        # Generate insights
        insights = self._generate_insights()

        # Save results
        self._save_analysis_results()

        # Calculate total time and log performance
        total_time = time.time() - start_time
        self.analysis_stats['processing_times']['total'] = total_time

        self._log_performance_stats()

        results = {
            "clusters": self.clusters,
            "rankings": self.rankings,
            "feature_importance": self.feature_importance,
            "insights": insights,
            "performance_stats": self.analysis_stats
        }

        self.logger.info(f"✅ GPU-accelerated city analysis completed in {total_time:.2f} seconds")
        return results

    def _prepare_features(self, city_data: Dict) -> pd.DataFrame:
        """
        Prepare feature matrix from city data.

        Args:
            city_data: Raw city data

        Returns:
            DataFrame with prepared features
        """
        self.logger.info("Preparing feature matrix...")

        features = []
        city_names = []

        for city, data in city_data.items():
            if data is None:
                continue

            city_names.append(city)
            feature_row = {}

            # Extract numerical features from all categories
            for category, values in data.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        if isinstance(value, (int, float)):
                            feature_row[f"{category}_{key}"] = value
                        elif isinstance(value, dict):
                            for subkey, subvalue in value.items():
                                if isinstance(subvalue, (int, float)):
                                    feature_row[f"{category}_{key}_{subkey}"] = subvalue

            features.append(feature_row)

        df = pd.DataFrame(features, index=city_names)

        # Handle missing values
        df = df.fillna(df.mean())

        self.logger.info(f"Prepared {df.shape[0]} cities with {df.shape[1]} features")
        return df

    def _perform_clustering(self) -> Dict:
        """
        Perform GPU-accelerated clustering analysis on cities.

        Returns:
            Dictionary containing clustering results
        """
        self.logger.info("🔄 Performing GPU-accelerated clustering analysis...")

        # Convert to appropriate format for GPU processing
        if self.use_gpu:
            # Convert to cupy array for GPU processing
            X = cp.asarray(self.city_features.values, dtype=cp.float32)
        else:
            X = self.city_features.values

        # Standardize features
        X_scaled = self.scaler.fit_transform(X)

        # Determine optimal number of clusters
        optimal_k = self._find_optimal_clusters(X_scaled)

        # Perform K-means clustering with optimized parameters
        if self.use_gpu:
            self.kmeans = cuKMeans(
                n_clusters=optimal_k,
                random_state=42,
                max_iter=300,
                tol=1e-4
            )
        else:
            self.kmeans = KMeans(
                n_clusters=optimal_k,
                random_state=42,
                max_iter=300,
                tol=1e-4
            )

        cluster_labels = self.kmeans.fit_predict(X_scaled)

        # Perform PCA for visualization
        if self.use_gpu:
            self.pca = cuPCA(n_components=2)
        else:
            self.pca = PCA(n_components=2)

        X_pca = self.pca.fit_transform(X_scaled)

        # Convert results back to CPU if needed
        if self.use_gpu:
            cluster_labels = cp.asnumpy(cluster_labels)
            X_pca = cp.asnumpy(X_pca)
            cluster_centers = cp.asnumpy(self.kmeans.cluster_centers_)
        else:
            cluster_centers = self.kmeans.cluster_centers_

        # Calculate silhouette score
        if self.use_gpu:
            # Use cuML silhouette score if available
            try:
                from cuml.metrics import silhouette_score as cu_silhouette_score
                sil_score = cu_silhouette_score(X_scaled, cluster_labels)
                if hasattr(sil_score, 'item'):
                    sil_score = sil_score.item()
                elif isinstance(sil_score, cp.ndarray):
                    sil_score = float(cp.asnumpy(sil_score))
            except:
                # Fallback to CPU calculation
                X_scaled_cpu = cp.asnumpy(X_scaled) if self.use_gpu else X_scaled
                sil_score = silhouette_score(X_scaled_cpu, cluster_labels)
        else:
            sil_score = silhouette_score(X_scaled, cluster_labels)

        # Create clustering results
        clustering_results = {
            "optimal_k": optimal_k,
            "cluster_labels": cluster_labels.tolist(),
            "city_clusters": dict(zip(self.city_features.index, cluster_labels)),
            "pca_coordinates": X_pca.tolist(),
            "silhouette_score": float(sil_score),
            "cluster_centers": cluster_centers.tolist(),
            "gpu_accelerated": self.use_gpu
        }

        # Analyze cluster characteristics
        clustering_results["cluster_characteristics"] = self._analyze_cluster_characteristics(cluster_labels)

        acceleration_type = "GPU" if self.use_gpu else "CPU"
        self.logger.info(f"✅ {acceleration_type}-accelerated clustering completed with {optimal_k} clusters (silhouette score: {sil_score:.3f})")
        return clustering_results

    def _find_optimal_clusters(self, X: np.ndarray, max_k: int = 10) -> int:
        """
        Find optimal number of clusters using elbow method and silhouette analysis.

        Args:
            X: Feature matrix
            max_k: Maximum number of clusters to test

        Returns:
            Optimal number of clusters
        """
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, len(X)))

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, labels))

        # Find elbow point
        optimal_k = k_range[np.argmax(silhouette_scores)]

        return optimal_k

    def _analyze_cluster_characteristics(self, cluster_labels: np.ndarray) -> Dict:
        """
        Analyze characteristics of each cluster.

        Args:
            cluster_labels: Cluster assignments for each city

        Returns:
            Dictionary containing cluster characteristics
        """
        characteristics = {}

        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_cities = self.city_features.index[cluster_mask].tolist()
            cluster_features = self.city_features[cluster_mask]

            # Calculate cluster statistics
            characteristics[f"cluster_{cluster_id}"] = {
                "cities": cluster_cities,
                "size": len(cluster_cities),
                "mean_features": cluster_features.mean().to_dict(),
                "top_features": self._get_top_cluster_features(cluster_features)
            }

        return characteristics

    def _get_top_cluster_features(self, cluster_features: pd.DataFrame, top_n: int = 5) -> Dict:
        """
        Get top distinguishing features for a cluster.

        Args:
            cluster_features: Features for cities in the cluster
            top_n: Number of top features to return

        Returns:
            Dictionary of top features
        """
        # Calculate z-scores relative to all cities
        global_mean = self.city_features.mean()
        global_std = self.city_features.std()
        cluster_mean = cluster_features.mean()

        z_scores = (cluster_mean - global_mean) / global_std

        # Get top positive and negative features
        top_positive = z_scores.nlargest(top_n).to_dict()
        top_negative = z_scores.nsmallest(top_n).to_dict()

        return {
            "strengths": top_positive,
            "weaknesses": top_negative
        }

    def _generate_rankings(self) -> Dict:
        """
        Generate city rankings across different categories.

        Returns:
            Dictionary containing city rankings
        """
        self.logger.info("Generating city rankings...")

        rankings = {}

        # Define ranking categories
        categories = {
            "overall_livability": ["quality_of_life_livability_index", "quality_of_life_happiness_index"],
            "sustainability": ["environment_green_space_percentage", "environment_renewable_energy_percentage"],
            "innovation": ["economy_innovation_index", "infrastructure_smart_city_index"],
            "transportation": ["transportation_public_transport_coverage", "transportation_walkability_score"],
            "safety": ["safety_safety_index", "safety_crime_rate_per_100k"],
            "education": ["education_literacy_rate", "education_university_ranking_avg"],
            "healthcare": ["healthcare_healthcare_index", "healthcare_life_expectancy"],
            "economy": ["economy_gdp_per_capita", "economy_business_environment_score"]
        }

        for category, features in categories.items():
            # Calculate composite score for category
            available_features = [f for f in features if f in self.city_features.columns]

            if available_features:
                # Normalize features (handle inverse features like crime rate)
                category_scores = pd.Series(index=self.city_features.index, dtype=float)

                for feature in available_features:
                    feature_values = self.city_features[feature]

                    # Inverse features (lower is better)
                    if "crime_rate" in feature or "unemployment" in feature:
                        normalized = (feature_values.max() - feature_values) / (feature_values.max() - feature_values.min())
                    else:
                        normalized = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min())

                    if category_scores.isna().all():
                        category_scores = normalized
                    else:
                        category_scores += normalized

                # Average the scores
                category_scores /= len(available_features)

                # Create ranking
                rankings[category] = category_scores.sort_values(ascending=False).to_dict()

        return rankings

    def _generate_rankings_parallel(self) -> Dict:
        """
        Generate city rankings using parallel processing for better performance.

        Returns:
            Dictionary containing city rankings
        """
        self.logger.info("🔄 Generating city rankings with parallel processing...")

        # Define ranking categories
        categories = {
            "overall_livability": ["quality_of_life_livability_index", "quality_of_life_happiness_index"],
            "sustainability": ["environment_green_space_percentage", "environment_renewable_energy_percentage"],
            "innovation": ["economy_innovation_index", "infrastructure_smart_city_index"],
            "transportation": ["transportation_public_transport_coverage", "transportation_walkability_score"],
            "safety": ["safety_safety_index", "safety_crime_rate_per_100k"],
            "education": ["education_literacy_rate", "education_university_ranking_avg"],
            "healthcare": ["healthcare_healthcare_index", "healthcare_life_expectancy"],
            "economy": ["economy_gdp_per_capita", "economy_business_environment_score"]
        }

        # Use parallel processing for ranking calculation
        def calculate_category_ranking(category_data):
            category, features = category_data
            return category, self._calculate_single_category_ranking(category, features)

        # Use sequential processing to avoid pickling issues with local functions
        rankings = {}

        for category, features in categories.items():
            try:
                ranking = self._calculate_single_category_ranking(category, features)
                if ranking:
                    rankings[category] = ranking
                    self.logger.info(f"✅ Generated ranking for {category}")
            except Exception as e:
                self.logger.error(f"❌ Error generating ranking for {category}: {str(e)}")
                # Continue with other categories

        self.logger.info(f"✅ Generated rankings for {len(rankings)} categories using sequential processing")
        return rankings

    def _calculate_single_category_ranking(self, category: str, features: List[str]) -> Dict:
        """Calculate ranking for a single category."""
        # Find available features
        available_features = [f for f in features if f in self.city_features.columns]

        if not available_features:
            return {}

        # Calculate composite score for category
        category_scores = pd.Series(index=self.city_features.index, dtype=float)

        for feature in available_features:
            feature_values = self.city_features[feature]

            # Inverse features (lower is better)
            if "crime_rate" in feature or "unemployment" in feature:
                normalized = (feature_values.max() - feature_values) / (feature_values.max() - feature_values.min())
            else:
                normalized = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min())

            if category_scores.isna().all():
                category_scores = normalized
            else:
                category_scores += normalized

        # Average the scores
        category_scores /= len(available_features)

        # Create ranking
        return category_scores.sort_values(ascending=False).to_dict()

    def _log_performance_stats(self):
        """Log detailed performance statistics."""
        self.logger.info("📊 Performance Statistics:")

        for operation, time_taken in self.analysis_stats['processing_times'].items():
            self.logger.info(f"  {operation}: {time_taken:.3f} seconds")

        if self.use_gpu and self.gpu_manager:
            memory_info = self.gpu_manager.get_memory_info()
            self.logger.info(f"  GPU Memory Used: {memory_info['memory_used']:.2f} GB")
            self.logger.info(f"  GPU Memory Total: {memory_info['memory_total']:.2f} GB")

        # Calculate speedup if we have comparison data
        total_time = self.analysis_stats['processing_times'].get('total', 0)
        if total_time > 0:
            cities_per_second = len(self.city_features) / total_time
            self.logger.info(f"  Processing Rate: {cities_per_second:.1f} cities/second")

        acceleration_type = "GPU" if self.use_gpu else "CPU"
        self.logger.info(f"  Acceleration: {acceleration_type}")

        if CUML_AVAILABLE and self.use_gpu:
            self.logger.info("  cuML GPU acceleration: ✅ Active")
        elif CUML_AVAILABLE and not self.use_gpu:
            self.logger.info("  cuML available but using CPU")
        else:
            self.logger.info("  cuML not available - using CPU only")

    def _analyze_feature_importance(self) -> Dict:
        """
        Analyze feature importance for predicting city success.

        Returns:
            Dictionary containing feature importance analysis
        """
        self.logger.info("Analyzing feature importance...")

        # Use overall livability as target variable
        if "quality_of_life_livability_index" in self.city_features.columns:
            target = self.city_features["quality_of_life_livability_index"]
            features = self.city_features.drop("quality_of_life_livability_index", axis=1)
        else:
            # Create composite target from available quality metrics
            quality_features = [col for col in self.city_features.columns if "quality_of_life" in col]
            if quality_features:
                target = self.city_features[quality_features].mean(axis=1)
                features = self.city_features.drop(quality_features, axis=1)
            else:
                self.logger.warning("No quality of life features found for importance analysis")
                return {}

        # Train Random Forest model
        self.rf_model.fit(features, target)

        # Get feature importance
        importance_scores = dict(zip(features.columns, self.rf_model.feature_importances_))

        # Sort by importance
        sorted_importance = dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))

        return {
            "feature_importance": sorted_importance,
            "top_10_features": dict(list(sorted_importance.items())[:10]),
            "model_score": self.rf_model.score(features, target)
        }

    def _generate_insights(self) -> Dict:
        """
        Generate insights from the analysis.

        Returns:
            Dictionary containing insights
        """
        insights = {
            "summary": f"Analyzed {len(self.city_features)} cities across {len(self.city_features.columns)} features",
            "best_overall_cities": [],
            "cluster_insights": [],
            "key_success_factors": []
        }

        # Best overall cities
        if "overall_livability" in self.rankings:
            top_cities = list(self.rankings["overall_livability"].keys())[:5]
            insights["best_overall_cities"] = top_cities

        # Cluster insights
        if self.clusters and "cluster_characteristics" in self.clusters:
            for cluster_id, characteristics in self.clusters["cluster_characteristics"].items():
                insight = f"Cluster {cluster_id}: {characteristics['size']} cities including {', '.join(characteristics['cities'][:3])}"
                insights["cluster_insights"].append(insight)

        # Key success factors
        if self.feature_importance and "top_10_features" in self.feature_importance:
            top_features = list(self.feature_importance["top_10_features"].keys())[:5]
            insights["key_success_factors"] = top_features

        return insights

    def _save_analysis_results(self):
        """Save analysis results to files."""
        import json

        # Save clustering results
        if self.clusters:
            cluster_path = self.config.get_data_path("clustering_results.json", "processed")
            with open(cluster_path, 'w') as f:
                json.dump(self.clusters, f, indent=2, default=str)

        # Save rankings
        if self.rankings:
            rankings_path = self.config.get_data_path("city_rankings.json", "processed")
            with open(rankings_path, 'w') as f:
                json.dump(self.rankings, f, indent=2, default=str)

        # Save feature importance
        if self.feature_importance:
            importance_path = self.config.get_data_path("feature_importance.json", "processed")
            with open(importance_path, 'w') as f:
                json.dump(self.feature_importance, f, indent=2, default=str)

        # Save models
        if hasattr(self, 'scaler'):
            scaler_path = self.config.get_model_path("scaler.joblib")
            joblib.dump(self.scaler, scaler_path)

        if hasattr(self, 'kmeans'):
            kmeans_path = self.config.get_model_path("kmeans.joblib")
            joblib.dump(self.kmeans, kmeans_path)

        self.logger.info("💾 Analysis results saved successfully")

