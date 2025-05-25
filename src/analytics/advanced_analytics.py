"""
Advanced analytics module for NeuroUrban.
Provides deep insights, predictive analytics, and trend analysis.
"""

import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
from pathlib import Path

try:
    from sklearn.ensemble import RandomForestRegressor, IsolationForest
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from src.config.settings import Config

class AdvancedAnalytics:
    """Advanced analytics engine for city data analysis."""
    
    def __init__(self, config: Config):
        """
        Initialize the advanced analytics engine.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Analytics models
        self.trend_predictor = None
        self.anomaly_detector = None
        self.feature_importance_model = None
        
        # Initialize ML models if available
        if SKLEARN_AVAILABLE:
            self._initialize_models()
        
        self.logger.info("📊 Advanced analytics engine initialized")
    
    def generate_comprehensive_report(self, cities_data: List[Dict]) -> Dict:
        """
        Generate a comprehensive analytics report for all cities.
        
        Args:
            cities_data: List of city data dictionaries
            
        Returns:
            Comprehensive analytics report
        """
        self.logger.info("📈 Generating comprehensive analytics report")
        
        # Convert to DataFrame for analysis
        df = self._prepare_dataframe(cities_data)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary_statistics': self._calculate_summary_statistics(df),
            'trend_analysis': self._analyze_trends(df),
            'correlation_analysis': self._analyze_correlations(df),
            'outlier_detection': self._detect_outliers(df),
            'predictive_insights': self._generate_predictions(df),
            'benchmarking': self._perform_benchmarking(df),
            'recommendations': self._generate_analytics_recommendations(df),
            'visualizations': self._create_advanced_visualizations(df)
        }
        
        self.logger.info("✅ Comprehensive analytics report generated")
        return report
    
    def analyze_city_performance_over_time(self, city_data: Dict, historical_data: List[Dict] = None) -> Dict:
        """
        Analyze city performance trends over time.
        
        Args:
            city_data: Current city data
            historical_data: Optional historical data points
            
        Returns:
            Time-series analysis results
        """
        analysis = {
            'city_name': city_data.get('name', 'Unknown'),
            'current_performance': self._calculate_performance_score(city_data),
            'trend_direction': 'stable',
            'growth_rate': 0.0,
            'forecasts': {},
            'key_insights': []
        }
        
        if historical_data:
            # Analyze historical trends
            performance_history = [self._calculate_performance_score(data) for data in historical_data]
            
            if len(performance_history) > 1:
                # Calculate growth rate
                analysis['growth_rate'] = (performance_history[-1] - performance_history[0]) / len(performance_history)
                
                # Determine trend direction
                if analysis['growth_rate'] > 2:
                    analysis['trend_direction'] = 'improving'
                elif analysis['growth_rate'] < -2:
                    analysis['trend_direction'] = 'declining'
                
                # Generate forecasts
                analysis['forecasts'] = self._forecast_performance(performance_history)
        
        return analysis
    
    def compare_cities_advanced(self, cities_data: List[Dict], comparison_metrics: List[str] = None) -> Dict:
        """
        Perform advanced city comparison with statistical analysis.
        
        Args:
            cities_data: List of city data dictionaries
            comparison_metrics: Optional list of specific metrics to compare
            
        Returns:
            Advanced comparison results
        """
        df = self._prepare_dataframe(cities_data)
        
        if comparison_metrics:
            # Filter to specific metrics
            available_metrics = [m for m in comparison_metrics if m in df.columns]
            df = df[['city'] + available_metrics]
        
        comparison = {
            'cities_analyzed': len(cities_data),
            'metrics_compared': len(df.columns) - 1,
            'statistical_summary': df.describe().to_dict(),
            'rankings': self._create_comprehensive_rankings(df),
            'clusters': self._identify_city_clusters(df),
            'performance_gaps': self._analyze_performance_gaps(df),
            'best_practices': self._identify_best_practices(df),
            'improvement_opportunities': self._identify_improvement_opportunities(df)
        }
        
        return comparison
    
    def detect_data_quality_issues(self, cities_data: List[Dict]) -> Dict:
        """
        Detect data quality issues in city datasets.
        
        Args:
            cities_data: List of city data dictionaries
            
        Returns:
            Data quality assessment
        """
        df = self._prepare_dataframe(cities_data)
        
        quality_report = {
            'overall_quality_score': 0,
            'completeness': {},
            'consistency': {},
            'outliers': {},
            'recommendations': []
        }
        
        # Completeness analysis
        missing_data = df.isnull().sum()
        total_cells = len(df) * len(df.columns)
        missing_percentage = (missing_data.sum() / total_cells) * 100
        
        quality_report['completeness'] = {
            'overall_completeness': 100 - missing_percentage,
            'missing_by_column': missing_data.to_dict(),
            'most_incomplete_columns': missing_data.nlargest(5).to_dict()
        }
        
        # Consistency analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        consistency_scores = {}
        
        for col in numeric_columns:
            if col != 'city':
                # Check for reasonable ranges
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                consistency_scores[col] = max(0, 100 - (len(outliers) / len(df)) * 100)
        
        quality_report['consistency'] = consistency_scores
        
        # Overall quality score
        completeness_score = quality_report['completeness']['overall_completeness']
        consistency_score = np.mean(list(consistency_scores.values())) if consistency_scores else 100
        quality_report['overall_quality_score'] = (completeness_score + consistency_score) / 2
        
        # Generate recommendations
        if missing_percentage > 10:
            quality_report['recommendations'].append("High missing data rate detected. Consider data collection improvements.")
        
        if consistency_score < 80:
            quality_report['recommendations'].append("Data consistency issues detected. Review data validation processes.")
        
        return quality_report
    
    def _initialize_models(self):
        """Initialize machine learning models for analytics."""
        try:
            self.trend_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
            self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
            self.feature_importance_model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.logger.info("✅ Analytics ML models initialized")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not initialize ML models: {e}")
    
    def _prepare_dataframe(self, cities_data: List[Dict]) -> pd.DataFrame:
        """Prepare DataFrame from cities data."""
        rows = []
        
        for city_data in cities_data:
            row = {'city': city_data.get('name', 'Unknown')}
            
            # Flatten nested data
            for category, data in city_data.items():
                if isinstance(data, dict):
                    for key, value in data.items():
                        if isinstance(value, (int, float)):
                            row[f"{category}_{key}"] = value
                elif isinstance(data, (int, float)):
                    row[category] = data
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _calculate_summary_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate summary statistics for the dataset."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        return {
            'total_cities': len(df),
            'total_metrics': len(numeric_df.columns),
            'mean_values': numeric_df.mean().to_dict(),
            'median_values': numeric_df.median().to_dict(),
            'std_values': numeric_df.std().to_dict(),
            'min_values': numeric_df.min().to_dict(),
            'max_values': numeric_df.max().to_dict()
        }
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict:
        """Analyze trends in the data."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        trends = {}
        for column in numeric_df.columns:
            values = numeric_df[column].dropna()
            if len(values) > 1:
                # Simple trend analysis using correlation with index
                correlation = np.corrcoef(range(len(values)), values)[0, 1]
                
                if correlation > 0.3:
                    trend = 'increasing'
                elif correlation < -0.3:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
                
                trends[column] = {
                    'trend': trend,
                    'correlation': correlation,
                    'volatility': values.std() / values.mean() if values.mean() != 0 else 0
                }
        
        return trends
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict:
        """Analyze correlations between metrics."""
        numeric_df = df.select_dtypes(include=[np.number])
        correlation_matrix = numeric_df.corr()
        
        # Find strongest correlations
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # Strong correlation threshold
                    strong_correlations.append({
                        'metric1': correlation_matrix.columns[i],
                        'metric2': correlation_matrix.columns[j],
                        'correlation': corr_value,
                        'strength': 'strong' if abs(corr_value) > 0.8 else 'moderate'
                    })
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'strong_correlations': strong_correlations,
            'most_correlated_pairs': sorted(strong_correlations, 
                                          key=lambda x: abs(x['correlation']), 
                                          reverse=True)[:5]
        }
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict:
        """Detect outliers in the data."""
        numeric_df = df.select_dtypes(include=[np.number])
        outliers = {}
        
        for column in numeric_df.columns:
            values = numeric_df[column].dropna()
            if len(values) > 0:
                q1, q3 = values.quantile([0.25, 0.75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outlier_indices = numeric_df[(numeric_df[column] < lower_bound) | 
                                           (numeric_df[column] > upper_bound)].index
                
                if len(outlier_indices) > 0:
                    outlier_cities = df.loc[outlier_indices, 'city'].tolist()
                    outliers[column] = {
                        'count': len(outlier_indices),
                        'cities': outlier_cities,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
        
        return outliers
    
    def _generate_predictions(self, df: pd.DataFrame) -> Dict:
        """Generate predictive insights."""
        predictions = {
            'performance_forecasts': {},
            'risk_assessments': {},
            'growth_potential': {}
        }
        
        # Simple predictions based on current trends
        numeric_df = df.select_dtypes(include=[np.number])
        
        for column in numeric_df.columns:
            values = numeric_df[column].dropna()
            if len(values) > 2:
                # Simple linear trend prediction
                trend = np.polyfit(range(len(values)), values, 1)[0]
                current_avg = values.mean()
                
                predictions['performance_forecasts'][column] = {
                    'current_average': current_avg,
                    'predicted_change_1year': trend * 12,  # Assuming monthly data
                    'confidence': 'medium'
                }
        
        return predictions
    
    def _perform_benchmarking(self, df: pd.DataFrame) -> Dict:
        """Perform benchmarking analysis."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        benchmarks = {}
        for column in numeric_df.columns:
            values = numeric_df[column].dropna()
            if len(values) > 0:
                benchmarks[column] = {
                    'top_10_percent': values.quantile(0.9),
                    'median': values.median(),
                    'bottom_10_percent': values.quantile(0.1),
                    'best_performer': df.loc[values.idxmax(), 'city'] if not values.empty else None,
                    'worst_performer': df.loc[values.idxmin(), 'city'] if not values.empty else None
                }
        
        return benchmarks
    
    def _calculate_performance_score(self, city_data: Dict) -> float:
        """Calculate overall performance score for a city."""
        # Simplified scoring based on key metrics
        scores = []
        
        # Extract key performance indicators
        indicators = [
            'economy_gdp_per_capita',
            'environment_air_quality_index',
            'infrastructure_internet_speed_mbps',
            'transportation_walkability_score',
            'safety_safety_index'
        ]
        
        for indicator in indicators:
            value = self._extract_nested_value(city_data, indicator)
            if value is not None:
                # Normalize to 0-100 scale (simplified)
                normalized = min(100, max(0, value))
                scores.append(normalized)
        
        return np.mean(scores) if scores else 50.0
    
    def _extract_nested_value(self, data: Dict, key: str) -> Optional[float]:
        """Extract value from nested dictionary structure."""
        parts = key.split('_')
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current if isinstance(current, (int, float)) else None
    
    def _forecast_performance(self, performance_history: List[float]) -> Dict:
        """Forecast future performance based on historical data."""
        if len(performance_history) < 3:
            return {'error': 'Insufficient historical data'}
        
        # Simple linear regression forecast
        x = np.array(range(len(performance_history)))
        y = np.array(performance_history)
        
        # Fit linear trend
        coeffs = np.polyfit(x, y, 1)
        
        # Forecast next 3 periods
        future_x = np.array([len(performance_history) + i for i in range(1, 4)])
        forecasts = np.polyval(coeffs, future_x)
        
        return {
            'next_period': forecasts[0],
            'next_2_periods': forecasts[1],
            'next_3_periods': forecasts[2],
            'trend_slope': coeffs[0],
            'confidence': 'medium'
        }
    
    def _create_comprehensive_rankings(self, df: pd.DataFrame) -> Dict:
        """Create comprehensive city rankings."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate composite score
        normalized_df = (numeric_df - numeric_df.min()) / (numeric_df.max() - numeric_df.min())
        composite_scores = normalized_df.mean(axis=1)
        
        # Create rankings
        rankings_df = pd.DataFrame({
            'city': df['city'],
            'composite_score': composite_scores
        }).sort_values('composite_score', ascending=False)
        
        return {
            'overall_ranking': rankings_df.to_dict('records'),
            'top_performers': rankings_df.head(5).to_dict('records'),
            'bottom_performers': rankings_df.tail(5).to_dict('records')
        }
    
    def _identify_city_clusters(self, df: pd.DataFrame) -> Dict:
        """Identify clusters of similar cities."""
        if not SKLEARN_AVAILABLE:
            return {'error': 'Clustering requires scikit-learn'}
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df) < 3:
            return {'error': 'Insufficient data for clustering'}
        
        # Standardize data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df.fillna(0))
        
        # Perform clustering
        clustering = DBSCAN(eps=0.5, min_samples=2)
        cluster_labels = clustering.fit_predict(scaled_data)
        
        # Organize results
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(df.iloc[i]['city'])
        
        return {
            'num_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'clusters': clusters,
            'outliers': clusters.get(-1, [])
        }
    
    def _analyze_performance_gaps(self, df: pd.DataFrame) -> Dict:
        """Analyze performance gaps between cities."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        gaps = {}
        for column in numeric_df.columns:
            values = numeric_df[column].dropna()
            if len(values) > 1:
                gap = values.max() - values.min()
                relative_gap = gap / values.mean() if values.mean() != 0 else 0
                
                gaps[column] = {
                    'absolute_gap': gap,
                    'relative_gap': relative_gap,
                    'coefficient_of_variation': values.std() / values.mean() if values.mean() != 0 else 0
                }
        
        return gaps
    
    def _identify_best_practices(self, df: pd.DataFrame) -> List[Dict]:
        """Identify best practices from top-performing cities."""
        numeric_df = df.select_dtypes(include=[np.number])
        best_practices = []
        
        for column in numeric_df.columns:
            values = numeric_df[column].dropna()
            if len(values) > 0:
                top_performer_idx = values.idxmax()
                top_performer = df.loc[top_performer_idx, 'city']
                top_value = values.max()
                
                best_practices.append({
                    'metric': column,
                    'best_performer': top_performer,
                    'value': top_value,
                    'practice_area': column.split('_')[0] if '_' in column else 'general'
                })
        
        return best_practices
    
    def _identify_improvement_opportunities(self, df: pd.DataFrame) -> List[Dict]:
        """Identify improvement opportunities for cities."""
        numeric_df = df.select_dtypes(include=[np.number])
        opportunities = []
        
        for column in numeric_df.columns:
            values = numeric_df[column].dropna()
            if len(values) > 1:
                median_value = values.median()
                below_median = df[numeric_df[column] < median_value]
                
                for idx, row in below_median.iterrows():
                    gap = median_value - numeric_df.loc[idx, column]
                    if gap > 0:
                        opportunities.append({
                            'city': row['city'],
                            'metric': column,
                            'current_value': numeric_df.loc[idx, column],
                            'median_value': median_value,
                            'improvement_gap': gap,
                            'improvement_potential': (gap / median_value) * 100 if median_value != 0 else 0
                        })
        
        # Sort by improvement potential
        opportunities.sort(key=lambda x: x['improvement_potential'], reverse=True)
        
        return opportunities[:20]  # Top 20 opportunities
    
    def _generate_analytics_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate recommendations based on analytics."""
        recommendations = []
        
        # Data quality recommendations
        missing_data = df.isnull().sum().sum()
        if missing_data > 0:
            recommendations.append(f"Address {missing_data} missing data points to improve analysis accuracy")
        
        # Performance recommendations
        numeric_df = df.select_dtypes(include=[np.number])
        low_variance_cols = [col for col in numeric_df.columns 
                           if numeric_df[col].std() / numeric_df[col].mean() < 0.1 
                           if numeric_df[col].mean() != 0]
        
        if low_variance_cols:
            recommendations.append(f"Consider additional metrics for {', '.join(low_variance_cols[:3])} as current data shows low variation")
        
        # Outlier recommendations
        outliers = self._detect_outliers(df)
        if outliers:
            recommendations.append(f"Investigate outliers in {len(outliers)} metrics for data validation")
        
        return recommendations
    
    def _create_advanced_visualizations(self, df: pd.DataFrame) -> Dict:
        """Create advanced visualization configurations."""
        # Return configuration for visualizations rather than actual plots
        # This allows the UI to render them appropriately
        
        visualizations = {
            'correlation_heatmap': {
                'type': 'heatmap',
                'data': df.select_dtypes(include=[np.number]).corr().to_dict(),
                'title': 'Metric Correlation Matrix'
            },
            'performance_radar': {
                'type': 'radar',
                'metrics': list(df.select_dtypes(include=[np.number]).columns)[:8],
                'title': 'City Performance Radar'
            },
            'trend_analysis': {
                'type': 'line',
                'title': 'Performance Trends Over Time'
            },
            'outlier_detection': {
                'type': 'scatter',
                'title': 'Outlier Detection Analysis'
            }
        }
        
        return visualizations
