"""
Enhanced visualization components for NeuroUrban dashboard.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import folium
from streamlit_folium import st_folium

class EnhancedVisualizations:
    """Enhanced visualization components for the NeuroUrban dashboard."""
    
    def __init__(self):
        """Initialize the visualization components."""
        self.color_schemes = {
            'sustainability': ['#2E8B57', '#32CD32', '#90EE90', '#98FB98'],
            'innovation': ['#4169E1', '#6495ED', '#87CEEB', '#B0E0E6'],
            'livability': ['#FF6347', '#FF7F50', '#FFA07A', '#FFB6C1'],
            'transportation': ['#9370DB', '#BA55D3', '#DA70D6', '#DDA0DD'],
            'safety': ['#FF4500', '#FF6347', '#FF7F50', '#FFA07A'],
            'education': ['#228B22', '#32CD32', '#7CFC00', '#ADFF2F'],
            'healthcare': ['#DC143C', '#F08080', '#FA8072', '#FFA07A'],
            'economy': ['#FFD700', '#FFA500', '#FF8C00', '#FF7F50']
        }
    
    def create_city_rankings_chart(self, rankings_data: Dict, category: str = 'overall_livability') -> go.Figure:
        """Create an interactive city rankings chart."""
        if category not in rankings_data:
            st.error(f"Category '{category}' not found in rankings data")
            return go.Figure()
        
        data = rankings_data[category]
        cities = list(data.keys())
        scores = list(data.values())
        
        # Convert scores to percentages
        scores_pct = [score * 100 for score in scores]
        
        fig = go.Figure(data=[
            go.Bar(
                x=cities,
                y=scores_pct,
                text=[f'{score:.1f}%' for score in scores_pct],
                textposition='auto',
                marker_color=self.color_schemes.get(category, ['#1f77b4'] * len(cities)),
                hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}%<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=f'City Rankings: {category.replace("_", " ").title()}',
            xaxis_title='Cities',
            yaxis_title='Score (%)',
            xaxis_tickangle=-45,
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_radar_chart(self, city_data: Dict, city_name: str) -> go.Figure:
        """Create a radar chart for a specific city's performance across categories."""
        categories = ['Sustainability', 'Innovation', 'Transportation', 'Safety', 
                     'Education', 'Healthcare', 'Economy', 'Livability']
        
        # Extract scores for the city (mock data if not available)
        scores = []
        for cat in categories:
            cat_key = cat.lower().replace(' ', '_')
            if cat_key == 'livability':
                cat_key = 'overall_livability'
            
            # Get score from rankings data or use random value
            score = np.random.uniform(0.3, 0.9) * 100  # Convert to percentage
            scores.append(score)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=categories,
            fill='toself',
            name=city_name,
            line_color='rgb(32, 201, 151)',
            fillcolor='rgba(32, 201, 151, 0.3)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title=f'{city_name} Performance Overview',
            height=500
        )
        
        return fig
    
    def create_comparison_chart(self, rankings_data: Dict, selected_cities: List[str]) -> go.Figure:
        """Create a comparison chart for multiple cities across categories."""
        categories = list(rankings_data.keys())
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set3
        
        for i, city in enumerate(selected_cities):
            scores = []
            for category in categories:
                score = rankings_data[category].get(city, 0) * 100
                scores.append(score)
            
            fig.add_trace(go.Scatter(
                x=categories,
                y=scores,
                mode='lines+markers',
                name=city,
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8)
            ))
        
        fig.update_layout(
            title='City Comparison Across Categories',
            xaxis_title='Categories',
            yaxis_title='Score (%)',
            xaxis_tickangle=-45,
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def create_correlation_heatmap(self, city_data: pd.DataFrame) -> go.Figure:
        """Create a correlation heatmap of city features."""
        # Select numeric columns for correlation
        numeric_cols = city_data.select_dtypes(include=[np.number]).columns
        correlation_matrix = city_data[numeric_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            height=600,
            width=800
        )
        
        return fig
    
    def create_world_map(self, city_data: Dict) -> folium.Map:
        """Create an interactive world map with city markers."""
        # Create base map
        m = folium.Map(location=[20, 0], zoom_start=2)
        
        # City coordinates (you would get these from your data)
        city_coords = {
            "Tokyo": [35.6762, 139.6503],
            "Singapore": [1.3521, 103.8198],
            "Zurich": [47.3769, 8.5417],
            "Copenhagen": [55.6761, 12.5683],
            "Amsterdam": [52.3676, 4.9041],
            "Vienna": [48.2082, 16.3738],
            "Munich": [48.1351, 11.5820],
            "Vancouver": [49.2827, -123.1207],
            "Toronto": [43.6532, -79.3832],
            "Melbourne": [-37.8136, 144.9631],
            "Sydney": [-33.8688, 151.2093],
            "Stockholm": [59.3293, 18.0686],
            "Helsinki": [60.1699, 24.9384],
            "Oslo": [59.9139, 10.7522],
            "Barcelona": [41.3851, 2.1734],
            "Paris": [48.8566, 2.3522],
            "London": [51.5074, -0.1278],
            "Berlin": [52.5200, 13.4050],
            "Seoul": [37.5665, 126.9780],
            "Hong Kong": [22.3193, 114.1694]
        }
        
        # Add markers for each city
        for city, coords in city_coords.items():
            if city in city_data:
                # Get some sample data for the popup
                popup_text = f"""
                <b>{city}</b><br>
                Population: {city_data[city].get('basic_info', {}).get('population', 'N/A'):,}<br>
                Country: {city_data[city].get('basic_info', {}).get('country', 'N/A')}<br>
                Click for more details
                """
                
                folium.Marker(
                    coords,
                    popup=folium.Popup(popup_text, max_width=300),
                    tooltip=city,
                    icon=folium.Icon(color='blue', icon='info-sign')
                ).add_to(m)
        
        return m
    
    def create_3d_scatter_plot(self, rankings_data: Dict) -> go.Figure:
        """Create a 3D scatter plot of cities."""
        # Get data for 3 dimensions
        x_data = list(rankings_data.get('sustainability', {}).values())
        y_data = list(rankings_data.get('innovation', {}).values())
        z_data = list(rankings_data.get('overall_livability', {}).values())
        cities = list(rankings_data.get('overall_livability', {}).keys())
        
        # Convert to percentages
        x_data = [x * 100 for x in x_data]
        y_data = [y * 100 for y in y_data]
        z_data = [z * 100 for z in z_data]
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x_data,
            y=y_data,
            z=z_data,
            mode='markers+text',
            text=cities,
            textposition='top center',
            marker=dict(
                size=8,
                color=z_data,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Livability Score")
            ),
            hovertemplate='<b>%{text}</b><br>' +
                         'Sustainability: %{x:.1f}%<br>' +
                         'Innovation: %{y:.1f}%<br>' +
                         'Livability: %{z:.1f}%<extra></extra>'
        )])
        
        fig.update_layout(
            title='3D City Analysis: Sustainability vs Innovation vs Livability',
            scene=dict(
                xaxis_title='Sustainability Score (%)',
                yaxis_title='Innovation Score (%)',
                zaxis_title='Livability Score (%)'
            ),
            height=600
        )
        
        return fig
    
    def create_time_series_simulation(self, simulation_data: Dict) -> go.Figure:
        """Create a time series chart for city simulation results."""
        # Mock time series data for demonstration
        days = list(range(1, 366))  # One year
        
        metrics = {
            'Population Growth': np.cumsum(np.random.normal(0.01, 0.05, 365)) + 100,
            'Air Quality Index': 50 + np.cumsum(np.random.normal(0, 2, 365)),
            'Traffic Congestion': 30 + np.cumsum(np.random.normal(0, 1, 365)),
            'Economic Growth': np.cumsum(np.random.normal(0.02, 0.03, 365)) + 100
        }
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=list(metrics.keys()),
            vertical_spacing=0.1
        )
        
        colors = ['blue', 'green', 'red', 'orange']
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, (metric, values) in enumerate(metrics.items()):
            row, col = positions[i]
            fig.add_trace(
                go.Scatter(
                    x=days,
                    y=values,
                    mode='lines',
                    name=metric,
                    line=dict(color=colors[i])
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title='City Simulation Results Over Time',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_feature_importance_chart(self, feature_importance: Dict) -> go.Figure:
        """Create a feature importance chart."""
        features = list(feature_importance.keys())
        importance = list(feature_importance.values())
        
        # Sort by importance
        sorted_data = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
        features, importance = zip(*sorted_data)
        
        fig = go.Figure(data=[
            go.Bar(
                y=features,
                x=importance,
                orientation='h',
                marker_color='lightblue',
                text=[f'{imp:.3f}' for imp in importance],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Feature Importance for City Analysis',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=max(400, len(features) * 25),
            margin=dict(l=200)
        )
        
        return fig
