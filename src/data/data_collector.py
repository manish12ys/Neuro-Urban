"""
City data collection module for NeuroUrban system.
"""

import logging
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import time
from datetime import datetime

from src.config.settings import Config

class CityDataCollector:
    """Collects data about cities from various sources."""
    
    def __init__(self, config: Config):
        """
        Initialize the data collector.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.city_data = {}
        self.collected_data = {}
        
    def collect_all_data(self) -> Dict:
        """
        Collect data for all target cities.
        
        Returns:
            Dictionary containing collected data for all cities
        """
        self.logger.info(f"Starting data collection for {len(self.config.data.target_cities)} cities")
        
        for city in self.config.data.target_cities:
            self.logger.info(f"Collecting data for {city}...")
            try:
                city_data = self.collect_city_data(city)
                self.collected_data[city] = city_data
                self.logger.info(f"✅ Data collection completed for {city}")
            except Exception as e:
                self.logger.error(f"❌ Failed to collect data for {city}: {str(e)}")
                self.collected_data[city] = None
        
        # Save collected data
        self._save_collected_data()
        
        return self.collected_data
    
    def collect_city_data(self, city_name: str) -> Dict:
        """
        Collect comprehensive data for a specific city.
        
        Args:
            city_name: Name of the city
            
        Returns:
            Dictionary containing city data
        """
        city_data = {
            "name": city_name,
            "timestamp": datetime.now().isoformat(),
            "basic_info": {},
            "demographics": {},
            "infrastructure": {},
            "economy": {},
            "environment": {},
            "quality_of_life": {},
            "transportation": {},
            "safety": {},
            "education": {},
            "healthcare": {},
            "geospatial": {}
        }
        
        # Collect basic city information
        city_data["basic_info"] = self._get_basic_city_info(city_name)
        
        # Collect demographic data
        city_data["demographics"] = self._get_demographic_data(city_name)
        
        # Collect infrastructure data
        city_data["infrastructure"] = self._get_infrastructure_data(city_name)
        
        # Collect economic data
        city_data["economy"] = self._get_economic_data(city_name)
        
        # Collect environmental data
        city_data["environment"] = self._get_environmental_data(city_name)
        
        # Collect quality of life data
        city_data["quality_of_life"] = self._get_quality_of_life_data(city_name)
        
        # Collect transportation data
        city_data["transportation"] = self._get_transportation_data(city_name)
        
        # Collect safety data
        city_data["safety"] = self._get_safety_data(city_name)
        
        # Collect education data
        city_data["education"] = self._get_education_data(city_name)
        
        # Collect healthcare data
        city_data["healthcare"] = self._get_healthcare_data(city_name)
        
        # Collect geospatial data
        city_data["geospatial"] = self._get_geospatial_data(city_name)
        
        return city_data
    
    def _get_basic_city_info(self, city_name: str) -> Dict:
        """Get basic city information."""
        # Mock data for demonstration - replace with real API calls
        basic_info = {
            "country": self._get_country_for_city(city_name),
            "population": np.random.randint(500000, 10000000),
            "area_km2": np.random.randint(100, 2000),
            "founded_year": np.random.randint(800, 1900),
            "timezone": "UTC+0",
            "coordinates": {
                "latitude": np.random.uniform(-90, 90),
                "longitude": np.random.uniform(-180, 180)
            }
        }
        return basic_info
    
    def _get_demographic_data(self, city_name: str) -> Dict:
        """Get demographic data."""
        demographics = {
            "population_density": np.random.uniform(1000, 15000),
            "age_distribution": {
                "0-14": np.random.uniform(10, 25),
                "15-64": np.random.uniform(60, 75),
                "65+": np.random.uniform(5, 20)
            },
            "education_level": {
                "primary": np.random.uniform(20, 40),
                "secondary": np.random.uniform(30, 50),
                "tertiary": np.random.uniform(20, 40)
            },
            "employment_rate": np.random.uniform(60, 95),
            "diversity_index": np.random.uniform(0.3, 0.9)
        }
        return demographics
    
    def _get_infrastructure_data(self, city_name: str) -> Dict:
        """Get infrastructure data."""
        infrastructure = {
            "internet_speed_mbps": np.random.uniform(20, 200),
            "electricity_reliability": np.random.uniform(85, 99.9),
            "water_quality_index": np.random.uniform(70, 100),
            "waste_management_efficiency": np.random.uniform(60, 95),
            "public_wifi_coverage": np.random.uniform(40, 90),
            "smart_city_index": np.random.uniform(30, 90)
        }
        return infrastructure
    
    def _get_economic_data(self, city_name: str) -> Dict:
        """Get economic data."""
        economy = {
            "gdp_per_capita": np.random.uniform(20000, 80000),
            "cost_of_living_index": np.random.uniform(50, 150),
            "unemployment_rate": np.random.uniform(2, 15),
            "business_environment_score": np.random.uniform(60, 95),
            "innovation_index": np.random.uniform(40, 90),
            "startup_density": np.random.uniform(10, 100)
        }
        return economy
    
    def _get_environmental_data(self, city_name: str) -> Dict:
        """Get environmental data."""
        environment = {
            "air_quality_index": np.random.uniform(20, 150),
            "green_space_percentage": np.random.uniform(10, 50),
            "carbon_emissions_per_capita": np.random.uniform(3, 20),
            "renewable_energy_percentage": np.random.uniform(10, 80),
            "water_consumption_per_capita": np.random.uniform(100, 400),
            "noise_pollution_level": np.random.uniform(40, 80)
        }
        return environment
    
    def _get_quality_of_life_data(self, city_name: str) -> Dict:
        """Get quality of life data."""
        quality_of_life = {
            "livability_index": np.random.uniform(60, 95),
            "happiness_index": np.random.uniform(5, 8),
            "cultural_diversity_score": np.random.uniform(40, 90),
            "recreational_facilities_score": np.random.uniform(50, 95),
            "housing_affordability": np.random.uniform(30, 80),
            "work_life_balance_score": np.random.uniform(60, 90)
        }
        return quality_of_life
    
    def _get_transportation_data(self, city_name: str) -> Dict:
        """Get transportation data."""
        transportation = {
            "public_transport_coverage": np.random.uniform(60, 95),
            "traffic_congestion_index": np.random.uniform(20, 80),
            "cycling_infrastructure_score": np.random.uniform(30, 90),
            "walkability_score": np.random.uniform(40, 95),
            "electric_vehicle_adoption": np.random.uniform(5, 40),
            "average_commute_time": np.random.uniform(20, 60)
        }
        return transportation
    
    def _get_safety_data(self, city_name: str) -> Dict:
        """Get safety data."""
        safety = {
            "crime_rate_per_100k": np.random.uniform(200, 2000),
            "safety_index": np.random.uniform(60, 95),
            "emergency_response_time": np.random.uniform(5, 15),
            "natural_disaster_risk": np.random.uniform(10, 80),
            "cybersecurity_index": np.random.uniform(50, 90)
        }
        return safety
    
    def _get_education_data(self, city_name: str) -> Dict:
        """Get education data."""
        education = {
            "literacy_rate": np.random.uniform(85, 99),
            "university_ranking_avg": np.random.uniform(100, 500),
            "education_expenditure_per_capita": np.random.uniform(1000, 8000),
            "student_teacher_ratio": np.random.uniform(10, 25),
            "digital_education_score": np.random.uniform(40, 90)
        }
        return education
    
    def _get_healthcare_data(self, city_name: str) -> Dict:
        """Get healthcare data."""
        healthcare = {
            "healthcare_index": np.random.uniform(60, 95),
            "life_expectancy": np.random.uniform(75, 85),
            "hospital_beds_per_1000": np.random.uniform(2, 8),
            "doctors_per_1000": np.random.uniform(1, 5),
            "healthcare_expenditure_per_capita": np.random.uniform(2000, 10000),
            "mental_health_support_score": np.random.uniform(40, 90)
        }
        return healthcare
    
    def _get_geospatial_data(self, city_name: str) -> Dict:
        """Get geospatial data."""
        geospatial = {
            "elevation_avg": np.random.uniform(0, 1000),
            "climate_zone": np.random.choice(["tropical", "temperate", "continental", "polar"]),
            "annual_rainfall_mm": np.random.uniform(300, 2000),
            "average_temperature": np.random.uniform(5, 30),
            "coastal_city": np.random.choice([True, False]),
            "river_access": np.random.choice([True, False])
        }
        return geospatial
    
    def _get_country_for_city(self, city_name: str) -> str:
        """Get country for a given city."""
        city_country_map = {
            "Tokyo": "Japan", "Singapore": "Singapore", "Zurich": "Switzerland",
            "Copenhagen": "Denmark", "Amsterdam": "Netherlands", "Vienna": "Austria",
            "Munich": "Germany", "Vancouver": "Canada", "Toronto": "Canada",
            "Melbourne": "Australia", "Sydney": "Australia", "Stockholm": "Sweden",
            "Helsinki": "Finland", "Oslo": "Norway", "Barcelona": "Spain",
            "Paris": "France", "London": "United Kingdom", "Berlin": "Germany",
            "Seoul": "South Korea", "Hong Kong": "Hong Kong"
        }
        return city_country_map.get(city_name, "Unknown")
    
    def _save_collected_data(self):
        """Save collected data to files."""
        # Save as JSON
        json_path = self.config.get_data_path("city_data.json", "raw")
        with open(json_path, 'w') as f:
            json.dump(self.collected_data, f, indent=2, default=str)
        
        # Save as CSV for easy analysis
        csv_data = []
        for city, data in self.collected_data.items():
            if data is not None:
                row = {"city": city}
                # Flatten nested dictionaries
                for category, values in data.items():
                    if isinstance(values, dict):
                        for key, value in values.items():
                            if isinstance(value, dict):
                                for subkey, subvalue in value.items():
                                    row[f"{category}_{key}_{subkey}"] = subvalue
                            else:
                                row[f"{category}_{key}"] = value
                    else:
                        row[category] = values
                csv_data.append(row)
        
        df = pd.DataFrame(csv_data)
        csv_path = self.config.get_data_path("city_data.csv", "raw")
        df.to_csv(csv_path, index=False)
        
        self.logger.info(f"💾 Data saved to {json_path} and {csv_path}")
    
    def load_collected_data(self) -> Dict:
        """Load previously collected data."""
        json_path = self.config.get_data_path("city_data.json", "raw")
        if json_path.exists():
            with open(json_path, 'r') as f:
                self.collected_data = json.load(f)
            self.logger.info(f"📂 Loaded data for {len(self.collected_data)} cities")
            return self.collected_data
        else:
            self.logger.warning("No previously collected data found")
            return {}
