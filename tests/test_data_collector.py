"""
Unit tests for data collector module.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from src.config.settings import Config
from src.data.data_collector import CityDataCollector

class TestCityDataCollector:
    """Test cases for CityDataCollector class."""
    
    @pytest.fixture
    def config(self):
        """Create a test configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = Config()
            config.project_root = Path(temp_dir)
            config.data.raw_data_dir = "test_raw"
            config.data.processed_data_dir = "test_processed"
            config.data.target_cities = ["TestCity1", "TestCity2"]
            
            # Create directories
            (Path(temp_dir) / "test_raw").mkdir(exist_ok=True)
            (Path(temp_dir) / "test_processed").mkdir(exist_ok=True)
            
            yield config
    
    @pytest.fixture
    def collector(self, config):
        """Create a CityDataCollector instance."""
        return CityDataCollector(config)
    
    def test_initialization(self, collector, config):
        """Test collector initialization."""
        assert collector.config == config
        assert collector.collected_data == {}
        assert hasattr(collector, 'logger')
    
    def test_collect_city_data(self, collector):
        """Test collecting data for a single city."""
        city_data = collector.collect_city_data("TestCity")
        
        # Check required fields
        assert "name" in city_data
        assert "timestamp" in city_data
        assert "basic_info" in city_data
        assert "demographics" in city_data
        assert "infrastructure" in city_data
        
        # Check data types
        assert isinstance(city_data["basic_info"], dict)
        assert isinstance(city_data["demographics"], dict)
        assert city_data["name"] == "TestCity"
    
    def test_get_basic_city_info(self, collector):
        """Test basic city info collection."""
        info = collector._get_basic_city_info("TestCity")
        
        assert "country" in info
        assert "population" in info
        assert "area_km2" in info
        assert "coordinates" in info
        
        # Check data ranges
        assert 500000 <= info["population"] <= 10000000
        assert 100 <= info["area_km2"] <= 2000
    
    def test_get_demographic_data(self, collector):
        """Test demographic data collection."""
        demographics = collector._get_demographic_data("TestCity")
        
        assert "population_density" in demographics
        assert "age_distribution" in demographics
        assert "employment_rate" in demographics
        
        # Check age distribution sums to reasonable total
        age_dist = demographics["age_distribution"]
        total_age = sum(age_dist.values())
        assert 90 <= total_age <= 110  # Should be close to 100%
    
    def test_save_collected_data(self, collector, config):
        """Test saving collected data."""
        # Add some test data
        collector.collected_data = {
            "TestCity": {
                "name": "TestCity",
                "basic_info": {"population": 1000000}
            }
        }
        
        collector._save_collected_data()
        
        # Check JSON file was created
        json_path = config.get_data_path("city_data.json", "raw")
        assert json_path.exists()
        
        # Check CSV file was created
        csv_path = config.get_data_path("city_data.csv", "raw")
        assert csv_path.exists()
        
        # Verify JSON content
        with open(json_path, 'r') as f:
            saved_data = json.load(f)
        
        assert "TestCity" in saved_data
        assert saved_data["TestCity"]["name"] == "TestCity"
    
    def test_load_collected_data(self, collector, config):
        """Test loading previously collected data."""
        # Create test data file
        test_data = {
            "TestCity": {
                "name": "TestCity",
                "basic_info": {"population": 1000000}
            }
        }
        
        json_path = config.get_data_path("city_data.json", "raw")
        with open(json_path, 'w') as f:
            json.dump(test_data, f)
        
        # Load data
        loaded_data = collector.load_collected_data()
        
        assert "TestCity" in loaded_data
        assert loaded_data["TestCity"]["name"] == "TestCity"
        assert collector.collected_data == loaded_data
    
    def test_collect_all_data(self, collector):
        """Test collecting data for all target cities."""
        results = collector.collect_all_data()
        
        # Should have data for all target cities
        assert len(results) == len(collector.config.data.target_cities)
        
        # Check each city has data
        for city in collector.config.data.target_cities:
            assert city in results
            if results[city] is not None:  # Some might fail in mock environment
                assert "name" in results[city]
                assert results[city]["name"] == city

if __name__ == "__main__":
    pytest.main([__file__])
