# 🌍 Real-world Data Integration - Q2 2024 Enhancement

## Overview

NeuroUrban now supports real-world data collection from live APIs, replacing mock data with actual city metrics from authoritative sources. This enhancement significantly improves the accuracy and credibility of city analysis and recommendations.

## 🔄 Implementation Status

### ✅ Phase 1A: World Bank Open Data API Integration (COMPLETED)
- **World Bank API**: Economic, demographic, education, and healthcare data
- **OpenStreetMap Overpass API**: Infrastructure and transportation data
- **OpenWeatherMap API**: Environmental and air quality data
- **Enhanced Data Processing**: Intelligent estimates based on real data

### 🔄 Phase 1B-1E: Additional APIs (IN PROGRESS)
- Government open data portals
- Real-time traffic data
- Additional environmental data sources

## 🚀 Features

### 🆓 Free Real-world Data Sources

| Data Source | Type | Coverage | API Key Required | Cost |
|-------------|------|----------|------------------|------|
| **World Bank Open Data** | Economic, Demographics, Health, Education | Global | ❌ No | 🆓 Completely Free |
| **OpenStreetMap Overpass** | Infrastructure, Transportation | Global | ❌ No | 🆓 Completely Free |
| **REST Countries** | Country info, Languages, Currencies | Global | ❌ No | 🆓 Completely Free |
| **OpenWeatherMap** | Environmental, Air Quality | Global | ✅ Yes | 🆓 Free (1,000 calls/day) |
| **Government Open Data** | Various city-specific data | Regional | ❌ No | 🆓 Completely Free |

**💡 Key Benefits:**
- **No subscription fees** - All APIs are free to use
- **No credit card required** - Only OpenWeatherMap needs registration
- **High rate limits** - Sufficient for city analysis needs
- **Reliable sources** - Government and institutional data

### Data Categories Collected

#### 🏛️ **Demographics & Economy**
- Population statistics (World Bank + REST Countries)
- GDP per capita (World Bank)
- Unemployment rates (World Bank)
- Urban population percentage (World Bank)
- Dependency ratios (World Bank)
- Country area and population density (REST Countries)

#### 🌐 **Country & Regional Data**
- Official country names and capitals (REST Countries)
- Languages and currencies (REST Countries)
- Geographic regions and subregions (REST Countries)
- Neighboring countries and borders (REST Countries)
- Timezone information (REST Countries)
- Country flags and basic info (REST Countries)

#### 🌍 **Environment & Sustainability**
- Air quality index (real-time)
- CO2 emissions per capita
- Forest area percentage
- Current weather conditions
- PM2.5 and PM10 levels

#### 🏥 **Healthcare & Education**
- Life expectancy
- Hospital beds per 1000 people
- Literacy rates
- Education expenditure

#### 🚇 **Infrastructure & Transportation**
- Hospital count (from OSM)
- School count (from OSM)
- Public transport stops
- Parks and green spaces
- Cycling infrastructure

## 🛠️ Setup Instructions

### 1. Install Required Dependencies

```bash
pip install wbdata overpy python-dotenv
```

### 2. Configure API Keys

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```env
# OpenWeatherMap API (for environmental data)
OPENWEATHER_API_KEY=your_api_key_here

# Optional: Additional API keys for enhanced data
GOOGLE_MAPS_API_KEY=your_google_maps_key
HERE_API_KEY=your_here_api_key
```

### 3. Get Free API Keys

#### OpenWeatherMap (Recommended)
1. Visit: https://openweathermap.org/api
2. Sign up for free account
3. Get API key (1000 calls/day free)
4. Add to `.env` file

#### Google Maps (Optional)
1. Visit: https://developers.google.com/maps/documentation
2. Enable Maps JavaScript API
3. Get API key
4. Add to `.env` file

## 📊 Usage

### Via Dashboard
1. Launch NeuroUrban dashboard: `python -m streamlit run streamlit_app.py`
2. Navigate to "📊 Data Collection"
3. Select "🌍 Real-world APIs" option
4. Click "🔄 Collect All Data"

### Via Python API
```python
from src.config.settings import Config
from src.data.real_world_data_collector import RealWorldDataCollector

# Initialize
config = Config()
collector = RealWorldDataCollector(config)

# Collect data for all cities
data = collector.collect_all_data()

# Or collect for specific city
singapore_data = collector.collect_city_data("Singapore")
```

## 📈 Data Quality Features

### Intelligent Data Enhancement
- **GDP-based Estimates**: Cost of living and innovation indices derived from real GDP data
- **Regional Intelligence**: European cities get higher walkability scores based on known patterns
- **Safety Estimates**: Based on country development indices and regional data
- **Population Scaling**: City-specific population estimates from country data

### Data Validation
- **Completeness Metrics**: Track percentage of real vs estimated data
- **API Status Monitoring**: Real-time status of data source availability
- **Freshness Indicators**: Timestamp tracking for data collection
- **Error Handling**: Graceful fallback to estimates when APIs are unavailable

## 🔍 Data Comparison: Real vs Mock

### Real-world Data Advantages
- ✅ **Accuracy**: Actual economic and demographic data from World Bank
- ✅ **Credibility**: Authoritative sources increase analysis reliability
- ✅ **Timeliness**: Recent data (typically 1-3 years old for economic data)
- ✅ **Consistency**: Standardized methodologies across countries
- ✅ **Verifiability**: Data sources are publicly documented

### Mock Data Use Cases
- ✅ **Speed**: Instant data generation for testing
- ✅ **Completeness**: All fields always populated
- ✅ **Offline**: No internet connection required
- ✅ **Development**: Ideal for algorithm development and testing

## 🚨 Troubleshooting

### Common Issues

#### "No World Bank data found"
- **Cause**: Country code not recognized or no recent data
- **Solution**: Check city coordinates mapping in `real_world_data_collector.py`

#### "OpenWeather API key not found"
- **Cause**: Missing or incorrect API key in `.env` file
- **Solution**: Verify `.env` file exists and contains valid `OPENWEATHER_API_KEY`

#### "OSM query failed"
- **Cause**: Overpass API rate limiting or network issues
- **Solution**: Wait a few minutes and retry; queries include automatic rate limiting

#### Import errors for wbdata/overpy
- **Cause**: Missing dependencies
- **Solution**: Run `pip install wbdata overpy`

### Performance Optimization

#### Rate Limiting
- World Bank API: Built-in caching and backoff
- OpenStreetMap: 0.5-second delays between queries
- OpenWeatherMap: Respects API rate limits

#### Caching
- World Bank data cached locally for 24 hours
- Environmental data refreshed on each collection
- Infrastructure data cached for 7 days

## 📋 Testing

### Run Test Suite
```bash
python test_real_world_data.py
```

### Expected Output
```
🌍 NeuroUrban Real-world Data Collection Test
✅ World Bank API library available
✅ OpenStreetMap Overpass API library available
✅ Data collected for Singapore
🎉 Real-world data collection test PASSED!
```

## 🔮 Future Enhancements

### Phase 1B: Government Open Data (Planned)
- US Census Bureau API
- European Union Open Data Portal
- City-specific open data APIs

### Phase 1C: Real-time Traffic Data (Planned)
- Google Maps Traffic API
- HERE Traffic API
- TomTom Traffic API

### Phase 1D: Enhanced Environmental Data (Planned)
- AirVisual API for detailed air quality
- NASA Earth Data for satellite imagery
- Climate data APIs

## 📞 Support

For issues with real-world data integration:
1. Check the troubleshooting section above
2. Verify API key configuration
3. Test with `python test_real_world_data.py`
4. Create an issue on GitHub with error details

---

**🌟 Real-world data integration brings NeuroUrban closer to actual city planning scenarios, enabling more accurate and actionable insights for urban development.**
