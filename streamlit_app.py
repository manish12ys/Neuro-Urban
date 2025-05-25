#!/usr/bin/env python3
"""
Streamlit web application for NeuroUrban.
This is a separate entry point to avoid import conflicts.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Apply compatibility fixes first
try:
    from src.utils.compatibility_fixes import setup_safe_environment
    setup_safe_environment()
except ImportError:
    pass

# Import with error handling
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import json
    from PIL import Image
    import logging

    # Import NeuroUrban components with lazy loading
    from src.config.settings import Config
    from src.utils.logger import setup_logging, safe_emoji_text

except ImportError as e:
    st.error(f"Import error: {str(e)}")
    st.error("Please install missing dependencies: pip install -r requirements.txt")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="NeuroUrban: AI City Planner",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

@st.cache_resource
def initialize_app():
    """Initialize the NeuroUrban application with caching."""
    try:
        config = Config()

        # Import components only when needed to avoid PyTorch conflicts
        from src.data.data_collector import CityDataCollector
        from src.ml.city_analyzer import CityAnalyzer

        # Initialize basic components (avoid PyTorch for now)
        data_collector = CityDataCollector(config)
        city_analyzer = CityAnalyzer(config)

        return {
            'config': config,
            'data_collector': data_collector,
            'city_analyzer': city_analyzer
        }
    except Exception as e:
        st.error(f"Failed to initialize application: {str(e)}")
        return None

def show_home_page():
    """Show home page."""
    st.title("🏙️ NeuroUrban: AI-Powered City Planning System")
    st.markdown("*Designing the cities of tomorrow using AI and data science*")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Project Overview")
        st.markdown("""
        NeuroUrban is an AI-powered city planning system that:

        - **Analyzes** successful global cities using ML
        - **Generates** optimal city blueprints using Deep Learning
        - **Simulates** city dynamics using Reinforcement Learning
        - **Visualizes** results through interactive dashboards

        Our goal is to design sustainable, livable, and efficient cities
        that balance tradition with innovation.
        """)

    with col2:
        st.subheader("🚀 Quick Start")

        app_components = initialize_app()
        if app_components is None:
            st.error("Application failed to initialize")
            return

        if st.button("🔄 Collect City Data", use_container_width=True):
            with st.spinner("Collecting city data..."):
                try:
                    app_components['data_collector'].collect_all_data()
                    st.success("✅ Data collection completed!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        if st.button("🔍 Analyze Cities", use_container_width=True):
            with st.spinner("Analyzing cities..."):
                try:
                    results = app_components['city_analyzer'].analyze_all_cities()
                    st.success("✅ City analysis completed!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

        # Note about blueprint generation
        st.info("🏗️ Blueprint generation requires PyTorch. Use CLI mode for full functionality.")

def show_data_collection_page():
    """Show data collection page."""
    st.header("📊 City Data Collection")

    app_components = initialize_app()
    if app_components is None:
        return

    config = app_components['config']

    st.subheader("Target Cities")
    cities_df = pd.DataFrame({
        "City": config.data.target_cities,
        "Status": ["✅ Ready" for _ in config.data.target_cities]
    })
    st.dataframe(cities_df, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Data Sources")
        for source, url in config.data.data_sources.items():
            st.write(f"- **{source.title()}**: {url}")

    with col2:
        st.subheader("Collection Controls")

        if st.button("🔄 Collect All Data", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                cities = config.data.target_cities
                for i, city in enumerate(cities):
                    status_text.text(f"Collecting data for {city}...")
                    progress_bar.progress((i + 1) / len(cities))

                app_components['data_collector'].collect_all_data()
                st.success("✅ Data collection completed!")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Show collected data if available
    show_collected_data_preview(config)

def show_collected_data_preview(config):
    """Show preview of collected data."""
    data_path = config.get_data_path("city_data.csv", "raw")

    if data_path.exists():
        st.subheader("📋 Data Preview")
        df = pd.read_csv(data_path)
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("📈 Data Statistics")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Cities", len(df))
        with col2:
            st.metric("Features", len(df.columns) - 1)
        with col3:
            st.metric("Data Points", len(df) * (len(df.columns) - 1))

def show_analysis_page():
    """Show city analysis page."""
    st.header("🔍 City Analysis")

    app_components = initialize_app()
    if app_components is None:
        return

    config = app_components['config']

    # Load analysis results if available
    rankings_path = config.get_data_path("city_rankings.json", "processed")

    if rankings_path.exists():
        with open(rankings_path, 'r') as f:
            rankings = json.load(f)
        show_city_rankings(rankings)
    else:
        st.info("No analysis results found. Run analysis first.")

    # Analysis controls
    st.subheader("🔧 Analysis Controls")
    if st.button("🔍 Run Analysis", use_container_width=True):
        with st.spinner("Running city analysis..."):
            try:
                results = app_components['city_analyzer'].analyze_all_cities()
                st.success("✅ Analysis completed!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

def show_city_rankings(rankings: dict):
    """Show city rankings."""
    st.subheader("🏆 City Rankings")

    # Category selection
    category = st.selectbox(
        "Select Category:",
        list(rankings.keys())
    )

    if category in rankings:
        ranking_data = rankings[category]

        # Create ranking dataframe
        df = pd.DataFrame([
            {"City": city, "Score": score, "Rank": i+1}
            for i, (city, score) in enumerate(ranking_data.items())
        ])

        # Display ranking
        col1, col2 = st.columns([2, 1])

        with col1:
            fig = px.bar(
                df.head(10),
                x="Score",
                y="City",
                orientation="h",
                title=f"Top 10 Cities - {category.replace('_', ' ').title()}"
            )
            fig.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.dataframe(df, use_container_width=True)

def show_blueprint_page():
    """Show blueprint generation page with simple generator."""
    st.header("🏗️ City Blueprint Generation")

    app_components = initialize_app()
    if app_components is None:
        return

    # Show configuration options
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🎛️ Blueprint Configuration")

        population = st.slider("Population (millions)", 0.5, 10.0, 2.0, 0.1)
        area = st.slider("Area (km²)", 100, 2000, 500, 50)

        st.write("**Focus Areas:**")
        sustainability_focus = st.checkbox("Sustainability Focus", value=True)
        tech_focus = st.checkbox("Technology Focus")
        cultural_focus = st.checkbox("Cultural Preservation", value=True)

        density = st.select_slider(
            "City Density",
            options=["low", "medium", "high"],
            value="medium"
        )

        if st.button("🏗️ Generate Blueprint", use_container_width=True):
            with st.spinner("Generating city blueprint..."):
                try:
                    # Import simple blueprint generator
                    from src.dl.simple_blueprint_generator import SimpleBlueprintGenerator

                    generator = SimpleBlueprintGenerator(app_components['config'])

                    # Prepare parameters
                    city_params = {
                        'population': int(population * 1000000),
                        'area_km2': area,
                        'sustainability_focus': sustainability_focus,
                        'tech_focus': tech_focus,
                        'cultural_focus': cultural_focus,
                        'density': density
                    }

                    # Generate blueprint
                    blueprint_data = generator.generate_blueprint(city_params)

                    # Store in session state
                    st.session_state['blueprint_data'] = blueprint_data
                    st.success("✅ Blueprint generated successfully!")

                except Exception as e:
                    st.error(f"❌ Error generating blueprint: {str(e)}")

    with col2:
        st.subheader("🖼️ Generated Blueprint")

        if 'blueprint_data' in st.session_state:
            blueprint_data = st.session_state['blueprint_data']

            # Display blueprint image
            if 'blueprint_image' in blueprint_data:
                import base64
                from io import BytesIO

                # Decode base64 image
                image_data = base64.b64decode(blueprint_data['blueprint_image'])
                st.image(image_data, caption="Generated City Blueprint", use_container_width=True)

            # Display statistics
            st.subheader("📊 City Statistics")
            stats = blueprint_data.get('city_statistics', {})

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Population Density", f"{stats.get('population_density', 0):.0f} people/km²")
                st.metric("Sustainability Score", f"{stats.get('sustainability_score', 0):.1f}/100")
                st.metric("Green Space", f"{stats.get('green_space_per_capita_m2', 0):.1f} m²/person")

            with col_b:
                st.metric("Livability Score", f"{stats.get('livability_score', 0):.1f}/100")
                st.metric("Parks Area", f"{stats.get('parks_area_km2', 0):.1f} km²")
                st.metric("Residential Area", f"{stats.get('residential_area_km2', 0):.1f} km²")

            # Display recommendations
            st.subheader("💡 Recommendations")
            recommendations = blueprint_data.get('recommendations', [])
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")

            # Download option
            if st.button("💾 Save Blueprint"):
                try:
                    from src.dl.simple_blueprint_generator import SimpleBlueprintGenerator
                    generator = SimpleBlueprintGenerator(app_components['config'])
                    saved_path = generator.save_blueprint(blueprint_data)
                    st.success(f"Blueprint saved to: {saved_path}")
                except Exception as e:
                    st.error(f"Error saving blueprint: {str(e)}")

        else:
            st.info("Configure parameters and click 'Generate Blueprint' to create a city design")

            # Show sample zone colors
            st.subheader("🎨 Zone Color Legend")
            zone_info = {
                "Residential": "🏠 Housing and neighborhoods",
                "Commercial": "🏢 Business and shopping areas",
                "Industrial": "🏭 Manufacturing and logistics",
                "Parks": "🌳 Green spaces and recreation",
                "Transportation": "🚗 Roads and transit",
                "Water": "💧 Rivers, lakes, and waterways",
                "Education": "🎓 Schools and universities",
                "Healthcare": "🏥 Hospitals and clinics",
                "Government": "🏛️ Administrative buildings"
            }

            for zone, description in zone_info.items():
                st.write(f"**{zone}**: {description}")

def show_settings_page():
    """Show settings page."""
    st.header("⚙️ Settings")

    app_components = initialize_app()
    if app_components is None:
        return

    config = app_components['config']

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎨 Display Settings")
        theme = st.selectbox("Theme", ["Dark", "Light"], index=0)

        st.subheader("🔧 Model Settings")
        st.write("**Current Settings:**")
        st.write(f"- GAN Image Size: {config.model.gan_image_size}")
        st.write(f"- GAN Epochs: {config.model.gan_epochs}")
        st.write(f"- RL Episodes: {config.model.rl_episodes}")

    with col2:
        st.subheader("📁 Data Settings")
        st.write("**Data Directories:**")
        st.write(f"- Raw Data: {config.data.raw_data_dir}")
        st.write(f"- Processed Data: {config.data.processed_data_dir}")
        st.write(f"- Models: {config.data.models_dir}")
        st.write(f"- Output: {config.data.output_dir}")

def main():
    """Main Streamlit application."""

    # Sidebar navigation
    page = st.sidebar.selectbox(
        "Navigate to:",
        ["🏠 Home", "📊 Data Collection", "🔍 City Analysis", "🏗️ Blueprint Generation", "⚙️ Settings"]
    )

    # Route to appropriate page
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Data Collection":
        show_data_collection_page()
    elif page == "🔍 City Analysis":
        show_analysis_page()
    elif page == "🏗️ Blueprint Generation":
        show_blueprint_page()
    elif page == "⚙️ Settings":
        show_settings_page()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**NeuroUrban v1.0**")
    st.sidebar.markdown("AI-Powered City Planning")

if __name__ == "__main__":
    main()
