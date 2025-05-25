"""
Simple blueprint generator for Streamlit interface (without PyTorch).
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import io
import base64

from src.config.settings import Config

class SimpleBlueprintGenerator:
    """Simple blueprint generator using matplotlib and PIL."""
    
    def __init__(self, config: Config):
        """Initialize the simple blueprint generator."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # City zones and their colors
        self.zone_colors = {
            'residential': '#90EE90',  # Light green
            'commercial': '#FFB6C1',   # Light pink
            'industrial': '#D3D3D3',   # Light gray
            'parks': '#228B22',        # Forest green
            'transportation': '#696969', # Dim gray
            'water': '#87CEEB',        # Sky blue
            'education': '#DDA0DD',    # Plum
            'healthcare': '#F0E68C',   # Khaki
            'government': '#B0C4DE'    # Light steel blue
        }
        
        # Default city parameters
        self.default_params = {
            'population': 2000000,
            'area_km2': 500,
            'sustainability_focus': True,
            'tech_focus': False,
            'cultural_focus': True,
            'density': 'medium'
        }
    
    def generate_blueprint(self, city_params: Optional[Dict] = None) -> Dict:
        """Generate a simple city blueprint."""
        self.logger.info("🏗️ Generating simple city blueprint...")
        
        # Use default parameters if none provided
        if city_params is None:
            city_params = self.default_params.copy()
        else:
            # Merge with defaults
            params = self.default_params.copy()
            params.update(city_params)
            city_params = params
        
        # Generate zone layout
        zone_layout = self._generate_zone_layout(city_params)
        
        # Create visual blueprint
        blueprint_image = self._create_blueprint_image(zone_layout, city_params)
        
        # Generate statistics
        city_stats = self._calculate_city_statistics(zone_layout, city_params)
        
        # Create blueprint data
        blueprint_data = {
            'timestamp': datetime.now().isoformat(),
            'generator_type': 'simple_matplotlib',
            'city_parameters': city_params,
            'zone_layout': zone_layout,
            'city_statistics': city_stats,
            'blueprint_image': blueprint_image,
            'recommendations': self._generate_recommendations(city_params, city_stats)
        }
        
        self.logger.info("✅ Simple blueprint generation completed")
        return blueprint_data
    
    def _generate_zone_layout(self, city_params: Dict) -> Dict:
        """Generate zone layout based on city parameters."""
        # Calculate zone percentages based on city focus
        zone_percentages = {
            'residential': 35,
            'commercial': 15,
            'industrial': 10,
            'parks': 20,
            'transportation': 8,
            'water': 5,
            'education': 3,
            'healthcare': 2,
            'government': 2
        }
        
        # Adjust based on focus areas
        if city_params.get('sustainability_focus', False):
            zone_percentages['parks'] += 5
            zone_percentages['industrial'] -= 3
            zone_percentages['commercial'] -= 2
        
        if city_params.get('tech_focus', False):
            zone_percentages['commercial'] += 3
            zone_percentages['education'] += 2
            zone_percentages['residential'] -= 5
        
        if city_params.get('cultural_focus', False):
            zone_percentages['parks'] += 2
            zone_percentages['government'] += 1
            zone_percentages['industrial'] -= 3
        
        # Adjust for density
        density = city_params.get('density', 'medium')
        if density == 'high':
            zone_percentages['residential'] += 5
            zone_percentages['parks'] -= 3
            zone_percentages['transportation'] -= 2
        elif density == 'low':
            zone_percentages['residential'] -= 5
            zone_percentages['parks'] += 3
            zone_percentages['transportation'] += 2
        
        # Ensure percentages sum to 100
        total = sum(zone_percentages.values())
        zone_percentages = {k: (v / total) * 100 for k, v in zone_percentages.items()}
        
        return zone_percentages
    
    def _create_blueprint_image(self, zone_layout: Dict, city_params: Dict) -> str:
        """Create a visual blueprint image."""
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Create a grid-based city layout
        grid_size = 20
        city_grid = np.zeros((grid_size, grid_size), dtype=object)
        
        # Fill grid based on zone percentages
        total_cells = grid_size * grid_size
        zone_cells = {}
        
        for zone, percentage in zone_layout.items():
            num_cells = int((percentage / 100) * total_cells)
            zone_cells[zone] = num_cells
        
        # Place zones in grid
        cell_count = 0
        for zone, num_cells in zone_cells.items():
            for _ in range(num_cells):
                if cell_count < total_cells:
                    row = cell_count // grid_size
                    col = cell_count % grid_size
                    city_grid[row, col] = zone
                    cell_count += 1
        
        # Create the visual representation
        for i in range(grid_size):
            for j in range(grid_size):
                zone = city_grid[i, j]
                if zone and zone in self.zone_colors:
                    color = self.zone_colors[zone]
                    rect = patches.Rectangle((j, grid_size-1-i), 1, 1, 
                                           linewidth=0.5, edgecolor='black', 
                                           facecolor=color, alpha=0.8)
                    ax.add_patch(rect)
        
        # Add title and labels
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect('equal')
        ax.set_title(f'City Blueprint - Population: {city_params["population"]:,}', 
                    fontsize=16, fontweight='bold')
        
        # Add legend
        legend_elements = [patches.Patch(facecolor=color, label=zone.title()) 
                          for zone, color in self.zone_colors.items() 
                          if zone in zone_layout]
        ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save to base64 string
        buffer = io.BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        
        # Convert to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close(fig)
        
        return image_base64
    
    def _calculate_city_statistics(self, zone_layout: Dict, city_params: Dict) -> Dict:
        """Calculate city statistics based on layout."""
        population = city_params['population']
        area_km2 = city_params['area_km2']
        
        # Calculate derived statistics
        population_density = population / area_km2
        
        # Estimate green space per capita
        parks_percentage = zone_layout.get('parks', 20)
        green_space_per_capita = (parks_percentage / 100) * area_km2 * 1000000 / population  # m² per person
        
        # Estimate sustainability score
        sustainability_score = (
            parks_percentage * 0.3 +
            (100 - zone_layout.get('industrial', 10)) * 0.2 +
            zone_layout.get('transportation', 8) * 0.1 +
            50  # Base score
        )
        sustainability_score = min(100, max(0, sustainability_score))
        
        # Estimate livability score
        livability_score = (
            zone_layout.get('parks', 20) * 0.25 +
            zone_layout.get('education', 3) * 5 +
            zone_layout.get('healthcare', 2) * 7.5 +
            (100 - zone_layout.get('industrial', 10)) * 0.15 +
            40  # Base score
        )
        livability_score = min(100, max(0, livability_score))
        
        return {
            'population_density': round(population_density, 2),
            'green_space_per_capita_m2': round(green_space_per_capita, 2),
            'sustainability_score': round(sustainability_score, 1),
            'livability_score': round(livability_score, 1),
            'total_area_km2': area_km2,
            'residential_area_km2': round((zone_layout.get('residential', 35) / 100) * area_km2, 2),
            'commercial_area_km2': round((zone_layout.get('commercial', 15) / 100) * area_km2, 2),
            'industrial_area_km2': round((zone_layout.get('industrial', 10) / 100) * area_km2, 2),
            'parks_area_km2': round((zone_layout.get('parks', 20) / 100) * area_km2, 2)
        }
    
    def _generate_recommendations(self, city_params: Dict, city_stats: Dict) -> List[str]:
        """Generate recommendations based on city design."""
        recommendations = []
        
        # Population density recommendations
        if city_stats['population_density'] > 8000:
            recommendations.append("Consider increasing public transportation and parks for high density areas")
        elif city_stats['population_density'] < 2000:
            recommendations.append("Low density allows for more green spaces and suburban development")
        
        # Green space recommendations
        if city_stats['green_space_per_capita_m2'] < 9:
            recommendations.append("Increase park areas - WHO recommends minimum 9m² green space per person")
        elif city_stats['green_space_per_capita_m2'] > 20:
            recommendations.append("Excellent green space allocation promotes health and wellbeing")
        
        # Sustainability recommendations
        if city_stats['sustainability_score'] < 60:
            recommendations.append("Consider reducing industrial zones and increasing renewable energy")
        elif city_stats['sustainability_score'] > 80:
            recommendations.append("Strong sustainability focus - consider this as a model for other cities")
        
        # Focus-based recommendations
        if city_params.get('tech_focus', False):
            recommendations.append("Tech focus: Ensure high-speed internet infrastructure in commercial zones")
        
        if city_params.get('sustainability_focus', False):
            recommendations.append("Sustainability focus: Consider implementing circular economy principles")
        
        if city_params.get('cultural_focus', False):
            recommendations.append("Cultural focus: Preserve historical areas and promote local arts")
        
        return recommendations
    
    def save_blueprint(self, blueprint_data: Dict, filename: Optional[str] = None) -> str:
        """Save blueprint data to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"simple_blueprint_{timestamp}.json"
        
        # Remove image data for JSON serialization
        save_data = blueprint_data.copy()
        if 'blueprint_image' in save_data:
            save_data['blueprint_image'] = f"<base64_image_{len(save_data['blueprint_image'])}_chars>"
        
        # Save to output directory
        output_path = self.config.get_output_path(filename)
        with open(output_path, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        self.logger.info(f"💾 Blueprint saved to {output_path}")
        return str(output_path)
