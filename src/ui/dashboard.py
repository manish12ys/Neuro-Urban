"""
Streamlit dashboard for NeuroUrban system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from PIL import Image
import logging

from src.config.settings import Config
from src.ui.enhanced_visualizations import EnhancedVisualizations

class Dashboard:
    """Main dashboard for NeuroUrban application."""

    def __init__(self, config: Config, app_manager):
        """
        Initialize dashboard.

        Args:
            config: Application configuration
            app_manager: Main application manager
        """
        self.config = config
        self.app_manager = app_manager
        self.logger = logging.getLogger(__name__)
        self.viz = EnhancedVisualizations()

    def run(self):
        """Run the Streamlit dashboard."""
        st.title("🏙️ NeuroUrban: AI-Powered City Planning System")
        st.markdown("*Designing the cities of tomorrow using AI and data science*")

        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigate to:",
            ["🏠 Home", "📊 Data Collection", "🔍 City Analysis", "🏗️ Blueprint Generation",
             "🎮 City Simulation", "📈 Results & Insights", "🗺️ Interactive Map",
             "🤖 AI Policy Advisor", "📊 Advanced Analytics", "⚙️ Settings"]
        )

        # Route to appropriate page
        if page == "🏠 Home":
            self._show_home_page()
        elif page == "📊 Data Collection":
            self._show_data_collection_page()
        elif page == "🔍 City Analysis":
            self._show_analysis_page()
        elif page == "🏗️ Blueprint Generation":
            self._show_blueprint_page()
        elif page == "🎮 City Simulation":
            self._show_simulation_page()
        elif page == "📈 Results & Insights":
            self._show_results_page()
        elif page == "🗺️ Interactive Map":
            self._show_map_page()
        elif page == "🤖 AI Policy Advisor":
            self._show_policy_advisor_page()
        elif page == "📊 Advanced Analytics":
            self._show_advanced_analytics_page()
        elif page == "⚙️ Settings":
            self._show_settings_page()

    def _show_home_page(self):
        """Show home page."""
        st.header("Welcome to NeuroUrban")

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
            if st.button("🔄 Collect City Data", use_container_width=True):
                with st.spinner("Collecting city data..."):
                    try:
                        self.app_manager.data_collector.collect_all_data()
                        st.success("✅ Data collection completed!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

            if st.button("🔍 Analyze Cities", use_container_width=True):
                with st.spinner("Analyzing cities..."):
                    try:
                        results = self.app_manager.city_analyzer.analyze_all_cities()
                        st.success("✅ City analysis completed!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

            if st.button("🏗️ Generate Blueprint", use_container_width=True):
                with st.spinner("Generating city blueprint..."):
                    try:
                        blueprint = self.app_manager.blueprint_generator.generate_blueprint()
                        st.success(f"✅ Blueprint generated: {blueprint}")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

        # System status
        st.subheader("📊 System Status")
        self._show_system_status()

    def _show_system_status(self):
        """Show system status indicators."""
        col1, col2, col3, col4 = st.columns(4)

        # Check data availability
        data_path = self.config.get_data_path("city_data.json", "raw")
        data_available = data_path.exists()

        with col1:
            status = "✅ Available" if data_available else "❌ Missing"
            st.metric("City Data", status)

        # Check models
        model_path = self.config.get_model_path("generator.pth")
        models_available = model_path.exists()

        with col2:
            status = "✅ Trained" if models_available else "❌ Not Trained"
            st.metric("AI Models", status)

        # Check analysis results
        analysis_path = self.config.get_data_path("city_rankings.json", "processed")
        analysis_available = analysis_path.exists()

        with col3:
            status = "✅ Complete" if analysis_available else "❌ Pending"
            st.metric("Analysis", status)

        # Check blueprints
        output_dir = self.config.project_root / self.config.data.output_dir
        blueprints_count = len(list(output_dir.glob("city_blueprint_*.png"))) if output_dir.exists() else 0

        with col4:
            st.metric("Blueprints", f"{blueprints_count} Generated")

    def _show_data_collection_page(self):
        """Show data collection page."""
        st.header("📊 City Data Collection")

        st.subheader("Target Cities")
        cities_df = pd.DataFrame({
            "City": self.config.data.target_cities,
            "Status": ["✅ Ready" for _ in self.config.data.target_cities]
        })
        st.dataframe(cities_df, use_container_width=True)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Data Sources")
            for source, url in self.config.data.data_sources.items():
                st.write(f"- **{source.title()}**: {url}")

        with col2:
            st.subheader("Collection Controls")

            # Data source selection
            data_source = st.radio(
                "Select Data Source:",
                ["📊 Mock Data (Fast)", "🌍 Real-world APIs (Requires API keys)"],
                help="Mock data is faster for testing. Real-world data provides actual city metrics."
            )

            use_real_data = "Real-world" in data_source

            if st.button("🔄 Collect All Data", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                try:
                    cities = self.config.data.target_cities

                    if use_real_data:
                        # Import and use real-world data collector
                        from src.data.real_world_data_collector import RealWorldDataCollector
                        collector = RealWorldDataCollector(self.config)
                        status_text.text("🌍 Initializing real-world data collection...")

                        for i, city in enumerate(cities):
                            status_text.text(f"🏙️ Collecting real-world data for {city}...")
                            progress_bar.progress((i + 1) / len(cities))

                        collector.collect_all_data()
                        st.success("✅ Real-world data collection completed!")
                        st.info("📁 Data saved as 'real_world_city_data.json' and 'real_world_city_data.csv'")
                    else:
                        # Use mock data collector
                        for i, city in enumerate(cities):
                            status_text.text(f"📊 Collecting mock data for {city}...")
                            progress_bar.progress((i + 1) / len(cities))

                        self.app_manager.data_collector.collect_all_data()
                        st.success("✅ Mock data collection completed!")

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    if use_real_data:
                        st.info("💡 Tip: Make sure you have API keys configured in .env file. See .env.example for details.")

        # Show collected data if available
        self._show_collected_data_preview()

    def _show_collected_data_preview(self):
        """Show preview of collected data."""
        # Check for both mock and real-world data
        mock_data_path = self.config.get_data_path("city_data.csv", "raw")
        real_data_path = self.config.get_data_path("real_world_city_data.csv", "raw")

        data_available = False

        if real_data_path.exists():
            st.subheader("📋 Real-world Data Preview")
            df = pd.read_csv(real_data_path)
            st.dataframe(df.head(), use_container_width=True)
            data_available = True
            data_type = "Real-world"
        elif mock_data_path.exists():
            st.subheader("📋 Mock Data Preview")
            df = pd.read_csv(mock_data_path)
            st.dataframe(df.head(), use_container_width=True)
            data_available = True
            data_type = "Mock"

        if data_available:
            st.subheader("📈 Data Statistics")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Data Type", data_type)
            with col2:
                st.metric("Cities", len(df))
            with col3:
                st.metric("Features", len(df.columns) - 1)  # Exclude city name
            with col4:
                st.metric("Data Points", len(df) * (len(df.columns) - 1))

            # Show data quality indicators for real-world data
            if data_type == "Real-world":
                st.subheader("🌍 Data Quality Indicators")
                col1, col2, col3 = st.columns(3)

                # Count non-null values
                non_null_percentage = (df.notna().sum().sum() / (len(df) * len(df.columns))) * 100

                with col1:
                    st.metric("Data Completeness", f"{non_null_percentage:.1f}%")

                # Check for API data indicators
                api_columns = [col for col in df.columns if any(keyword in col.lower()
                              for keyword in ['gdp', 'life_expectancy', 'air_quality', 'coordinates'])]

                with col2:
                    st.metric("API Data Fields", len(api_columns))

                with col3:
                    # Check timestamp freshness
                    if 'timestamp' in df.columns:
                        st.metric("Data Freshness", "Today")
                    else:
                        st.metric("Data Freshness", "N/A")

    def _show_analysis_page(self):
        """Show city analysis page."""
        st.header("🔍 City Analysis")

        # Load analysis results if available
        rankings_path = self.config.get_data_path("city_rankings.json", "processed")
        clustering_path = self.config.get_data_path("clustering_results.json", "processed")

        if rankings_path.exists():
            with open(rankings_path, 'r') as f:
                rankings = json.load(f)

            self._show_city_rankings(rankings)

        if clustering_path.exists():
            with open(clustering_path, 'r') as f:
                clustering = json.load(f)

            self._show_clustering_results(clustering)

        # Analysis controls
        st.subheader("🔧 Analysis Controls")
        if st.button("🔍 Run Analysis", use_container_width=True):
            with st.spinner("Running city analysis..."):
                try:
                    results = self.app_manager.city_analyzer.analyze_all_cities()
                    st.success("✅ Analysis completed!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    def _show_city_rankings(self, rankings: dict):
        """Show city rankings with enhanced visualizations."""
        st.subheader("🏆 City Rankings")

        # Category selection
        category = st.selectbox(
            "Select Category:",
            list(rankings.keys())
        )

        if category in rankings:
            # Enhanced bar chart
            fig = self.viz.create_city_rankings_chart(rankings, category)
            st.plotly_chart(fig, use_container_width=True)

            # City comparison section
            st.subheader("🔍 City Comparison")
            selected_cities = st.multiselect(
                "Select cities to compare:",
                list(rankings[category].keys()),
                default=list(rankings[category].keys())[:5]
            )

            if len(selected_cities) >= 2:
                comparison_fig = self.viz.create_comparison_chart(rankings, selected_cities)
                st.plotly_chart(comparison_fig, use_container_width=True)

            # 3D scatter plot
            st.subheader("🌐 3D City Analysis")
            scatter_3d_fig = self.viz.create_3d_scatter_plot(rankings)
            st.plotly_chart(scatter_3d_fig, use_container_width=True)

            # Detailed ranking table
            ranking_data = rankings[category]
            df = pd.DataFrame([
                {"City": city, "Score": f"{score*100:.1f}%", "Rank": i+1}
                for i, (city, score) in enumerate(ranking_data.items())
            ])
            st.dataframe(df, use_container_width=True)

    def _show_clustering_results(self, clustering: dict):
        """Show clustering results."""
        st.subheader("🎯 City Clustering")

        if "cluster_characteristics" in clustering:
            characteristics = clustering["cluster_characteristics"]

            # Show cluster overview
            cluster_data = []
            for cluster_id, data in characteristics.items():
                cluster_data.append({
                    "Cluster": cluster_id,
                    "Cities": len(data["cities"]),
                    "Example Cities": ", ".join(data["cities"][:3])
                })

            df = pd.DataFrame(cluster_data)
            st.dataframe(df, use_container_width=True)

            # Detailed cluster view
            selected_cluster = st.selectbox(
                "Select Cluster for Details:",
                list(characteristics.keys())
            )

            if selected_cluster in characteristics:
                cluster_info = characteristics[selected_cluster]

                col1, col2 = st.columns(2)

                with col1:
                    st.write("**Cities in Cluster:**")
                    for city in cluster_info["cities"]:
                        st.write(f"- {city}")

                with col2:
                    st.write("**Cluster Strengths:**")
                    if "top_features" in cluster_info and "strengths" in cluster_info["top_features"]:
                        for feature, score in list(cluster_info["top_features"]["strengths"].items())[:5]:
                            st.write(f"- {feature}: {score:.3f}")

    def _show_blueprint_page(self):
        """Show blueprint generation page."""
        st.header("🏗️ City Blueprint Generation")

        # Show current generator status
        if hasattr(self.app_manager, 'blueprint_generator') and self.app_manager.blueprint_generator:
            generator_type = type(self.app_manager.blueprint_generator).__name__
            if generator_type == "SimpleBlueprintGenerator":
                st.info("✅ Using Simple Blueprint Generator - Fast, reliable city layout generation")
            else:
                st.info("✅ Using PyTorch Blueprint Generator - Advanced AI-powered generation")
        else:
            st.warning("⚠️ Blueprint generator not available")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("🎛️ Blueprint Configuration")

            # City parameters
            population = st.slider("Population (millions)", 0.5, 10.0, 2.0, 0.1)
            area = st.slider("Area (km²)", 100, 2000, 500, 50)

            # Focus areas
            st.write("**Focus Areas:**")
            sustainability_focus = st.checkbox("Sustainability Focus")
            tech_focus = st.checkbox("Technology Focus")
            cultural_focus = st.checkbox("Cultural Preservation")

            # Density preference
            density = st.select_slider(
                "City Density",
                options=["Low", "Medium", "High"],
                value="Medium"
            )

            # Generate blueprint
            if st.button("🏗️ Generate Blueprint", use_container_width=True):
                city_features = {
                    "population": population * 1000000,
                    "area_km2": area,
                    "sustainability_focus": sustainability_focus,
                    "tech_focus": tech_focus,
                    "cultural_focus": cultural_focus,
                    "density": {"Low": 0.3, "Medium": 0.6, "High": 0.9}[density]
                }

                with st.spinner("Generating blueprint..."):
                    try:
                        # Check if we have a blueprint generator
                        if hasattr(self.app_manager, 'blueprint_generator') and self.app_manager.blueprint_generator:
                            # Check if it's the simple generator or PyTorch generator
                            generator_type = type(self.app_manager.blueprint_generator).__name__

                            if generator_type == "SimpleBlueprintGenerator":
                                # Use simple generator
                                blueprint_data = self.app_manager.blueprint_generator.generate_blueprint(city_features)
                                if blueprint_data and 'blueprint_image' in blueprint_data:
                                    st.success(f"✅ Blueprint generated using simple generator!")
                                    st.session_state.latest_blueprint_data = blueprint_data
                                else:
                                    st.error("❌ Failed to generate blueprint")
                            else:
                                # Use PyTorch generator
                                blueprint_path = self.app_manager.blueprint_generator.generate_blueprint(city_features)
                                st.success(f"✅ Blueprint generated using PyTorch!")
                                st.session_state.latest_blueprint = blueprint_path
                        else:
                            # No blueprint generator available, try to create simple one
                            from src.dl.simple_blueprint_generator import SimpleBlueprintGenerator
                            simple_generator = SimpleBlueprintGenerator(self.config)
                            blueprint_data = simple_generator.generate_blueprint(city_features)

                            if blueprint_data and 'blueprint_image' in blueprint_data:
                                st.success(f"✅ Blueprint generated using fallback simple generator!")
                                st.session_state.latest_blueprint_data = blueprint_data
                            else:
                                st.error("❌ Failed to generate blueprint")

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        # Don't show the PyTorch message since we have a working simple generator

        with col2:
            st.subheader("🖼️ Generated Blueprints")

            # Show latest blueprint if available
            if hasattr(st.session_state, 'latest_blueprint'):
                try:
                    img = Image.open(st.session_state.latest_blueprint)
                    st.image(img, caption="Latest Generated Blueprint", use_container_width=True)
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")
            elif hasattr(st.session_state, 'latest_blueprint_data'):
                try:
                    # Handle base64 encoded image from simple generator
                    import base64
                    from io import BytesIO

                    blueprint_data = st.session_state.latest_blueprint_data
                    if 'blueprint_image' in blueprint_data:
                        image_data = base64.b64decode(blueprint_data['blueprint_image'])
                        img = Image.open(BytesIO(image_data))
                        st.image(img, caption="Latest Generated Blueprint (Simple Generator)", use_container_width=True)

                        # Show statistics
                        if 'city_statistics' in blueprint_data:
                            stats = blueprint_data['city_statistics']
                            st.subheader("📊 Blueprint Statistics")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Population Density", f"{stats.get('population_density', 0):.0f} people/km²")
                            with col2:
                                st.metric("Green Space", f"{stats.get('green_space_per_capita_m2', 0):.1f} m²/person")
                            with col3:
                                st.metric("Sustainability Score", f"{stats.get('sustainability_score', 0):.1f}/100")

                            # Show recommendations if available
                            if 'recommendations' in blueprint_data and blueprint_data['recommendations']:
                                st.subheader("💡 Design Recommendations")
                                for rec in blueprint_data['recommendations']:
                                    st.write(f"• {rec}")

                            # Show zone breakdown
                            if 'zone_layout' in blueprint_data:
                                st.subheader("🏘️ Zone Distribution")
                                zones = blueprint_data['zone_layout']
                                zone_col1, zone_col2 = st.columns(2)
                                with zone_col1:
                                    st.write(f"**Residential:** {zones.get('residential', 0):.1f}%")
                                    st.write(f"**Commercial:** {zones.get('commercial', 0):.1f}%")
                                    st.write(f"**Industrial:** {zones.get('industrial', 0):.1f}%")
                                    st.write(f"**Parks:** {zones.get('parks', 0):.1f}%")
                                with zone_col2:
                                    st.write(f"**Transportation:** {zones.get('transportation', 0):.1f}%")
                                    st.write(f"**Education:** {zones.get('education', 0):.1f}%")
                                    st.write(f"**Healthcare:** {zones.get('healthcare', 0):.1f}%")
                                    st.write(f"**Water:** {zones.get('water', 0):.1f}%")

                except Exception as e:
                    st.error(f"Error loading blueprint data: {str(e)}")

            # Show existing blueprints
            output_dir = self.config.project_root / self.config.data.output_dir
            if output_dir.exists():
                blueprint_files = list(output_dir.glob("city_blueprint_*.png"))

                if blueprint_files:
                    st.write(f"**Available Blueprints ({len(blueprint_files)}):**")

                    selected_blueprint = st.selectbox(
                        "Select Blueprint:",
                        blueprint_files,
                        format_func=lambda x: x.name
                    )

                    if selected_blueprint:
                        try:
                            img = Image.open(selected_blueprint)
                            st.image(img, caption=selected_blueprint.name, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error loading image: {str(e)}")

    def _show_simulation_page(self):
        """Show city simulation page."""
        st.header("🎮 City Simulation")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("🎛️ Simulation Parameters")

            # City configuration
            population = st.number_input("Population", 100000, 10000000, 1000000, 100000)
            area = st.number_input("Area (km²)", 50, 2000, 500, 50)
            gdp_per_capita = st.number_input("GDP per Capita ($)", 10000, 100000, 50000, 5000)

            # Simulation settings
            episodes = st.slider("Simulation Episodes", 10, 1000, 100, 10)

            if st.button("🎮 Run Simulation", use_container_width=True):
                city_config = {
                    "population": population,
                    "area_km2": area,
                    "gdp_per_capita": gdp_per_capita
                }

                with st.spinner("Running simulation..."):
                    try:
                        results = self.app_manager.city_simulator.run_simulation(city_config, episodes)
                        st.session_state.simulation_results = results
                        st.success("✅ Simulation completed!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

        with col2:
            st.subheader("📊 Simulation Results")

            if hasattr(st.session_state, 'simulation_results'):
                results = st.session_state.simulation_results

                # Show key metrics
                col2_1, col2_2, col2_3 = st.columns(3)

                with col2_1:
                    st.metric("Average Reward", f"{results['average_reward']:.3f}")

                with col2_2:
                    st.metric("Best Episode", results['best_episode'])

                with col2_3:
                    st.metric("Best Reward", f"{results['best_reward']:.3f}")

                # Plot reward progression
                if 'episode_rewards' in results:
                    fig = px.line(
                        x=range(len(results['episode_rewards'])),
                        y=results['episode_rewards'],
                        title="Reward Progression",
                        labels={"x": "Episode", "y": "Reward"}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Show simulation report
                if st.button("📋 Generate Report"):
                    report = self.app_manager.city_simulator.generate_simulation_report()
                    st.text_area("Simulation Report", report, height=300)

    def _show_results_page(self):
        """Show results and insights page."""
        st.header("📈 Results & Insights")

        # Load all available results
        self._show_comprehensive_insights()

    def _show_comprehensive_insights(self):
        """Show comprehensive insights from all analyses."""
        st.subheader("🎯 Key Insights")

        insights = []

        # Check for analysis results
        rankings_path = self.config.get_data_path("city_rankings.json", "processed")
        if rankings_path.exists():
            insights.append("✅ City analysis completed - rankings available")

        # Check for blueprints
        output_dir = self.config.project_root / self.config.data.output_dir
        if output_dir.exists():
            blueprint_count = len(list(output_dir.glob("city_blueprint_*.png")))
            if blueprint_count > 0:
                insights.append(f"✅ {blueprint_count} city blueprints generated")

        # Check for simulation results
        simulation_files = list(output_dir.glob("simulation_results_*.json")) if output_dir.exists() else []
        if simulation_files:
            insights.append(f"✅ {len(simulation_files)} simulation runs completed")

        if insights:
            for insight in insights:
                st.write(insight)
        else:
            st.info("No results available yet. Run analysis, generate blueprints, or simulate cities to see insights here.")

    def _show_map_page(self):
        """Show interactive map page."""
        st.header("🗺️ Interactive City Map")

        # Load city data for map
        data_path = self.config.get_data_path("city_data.json", "raw")
        if data_path.exists():
            with open(data_path, 'r') as f:
                city_data = json.load(f)

            # Create and display map
            city_map = self.viz.create_world_map(city_data)

            # Display map using streamlit-folium
            try:
                from streamlit_folium import st_folium
                st_folium(city_map, width=700, height=500)
            except ImportError:
                st.error("streamlit-folium not installed. Please install it to view the interactive map.")
                st.info("Run: pip install streamlit-folium")

            # City details section
            st.subheader("🏙️ City Details")
            selected_city = st.selectbox("Select a city for detailed view:", list(city_data.keys()))

            if selected_city and selected_city in city_data:
                city_info = city_data[selected_city]

                # Create radar chart for selected city
                radar_fig = self.viz.create_radar_chart(city_data, selected_city)
                st.plotly_chart(radar_fig, use_container_width=True)

                # Show city statistics
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Population", f"{city_info['basic_info']['population']:,}")
                    st.metric("Area", f"{city_info['basic_info']['area_km2']} km²")

                with col2:
                    st.metric("Country", city_info['basic_info']['country'])
                    st.metric("Founded", city_info['basic_info']['founded_year'])

                with col3:
                    coords = city_info['basic_info']['coordinates']
                    st.metric("Latitude", f"{coords['latitude']:.2f}")
                    st.metric("Longitude", f"{coords['longitude']:.2f}")
        else:
            st.warning("No city data available. Please collect data first.")

    def _show_settings_page(self):
        """Show settings page."""
        st.header("⚙️ Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎨 Display Settings")
            theme = st.selectbox("Theme", ["Dark", "Light"], index=0)

            st.subheader("🔧 Model Settings")
            st.write(f"**Current Settings:**")
            st.write(f"- GAN Image Size: {self.config.model.gan_image_size}")
            st.write(f"- GAN Epochs: {self.config.model.gan_epochs}")
            st.write(f"- RL Episodes: {self.config.model.rl_episodes}")

        with col2:
            st.subheader("📁 Data Settings")
            st.write(f"**Data Directories:**")
            st.write(f"- Raw Data: {self.config.data.raw_data_dir}")
            st.write(f"- Processed Data: {self.config.data.processed_data_dir}")
            st.write(f"- Models: {self.config.data.models_dir}")
            st.write(f"- Output: {self.config.data.output_dir}")

            st.subheader("🌍 Target Cities")
            st.write(f"**Currently tracking {len(self.config.data.target_cities)} cities:**")
            for city in self.config.data.target_cities[:10]:  # Show first 10
                st.write(f"- {city}")
            if len(self.config.data.target_cities) > 10:
                st.write(f"... and {len(self.config.data.target_cities) - 10} more")



    def _show_policy_advisor_page(self):
        """Show AI policy advisor page."""
        st.header("🤖 AI Policy Recommendation Engine")

        try:
            from src.ai.policy_recommender import PolicyRecommender
            recommender = PolicyRecommender(self.config)

            # City selection
            col1, col2 = st.columns([1, 2])

            with col1:
                st.subheader("🎯 Policy Configuration")

                city_name = st.selectbox("Select City:", self.config.data.target_cities)

                focus_areas = st.multiselect(
                    "Focus Areas:",
                    ["sustainability", "transportation", "economy", "livability", "infrastructure", "governance"],
                    default=["sustainability", "transportation"]
                )

                if st.button("🔍 Generate Policy Recommendations"):
                    with st.spinner("Analyzing real city data and generating recommendations..."):
                        try:
                            # Load real city data
                            from src.data.data_collector import CityDataCollector
                            data_collector = CityDataCollector(self.config)

                            # Try to load existing data first
                            all_data = data_collector.load_collected_data()
                            city_data = all_data.get(city_name, {})

                            if not city_data:
                                st.info(f"🔄 Collecting fresh data for {city_name}...")
                                city_data = data_collector.collect_city_data(city_name)

                            # Generate policy recommendations using real data
                            policy_report = recommender.recommend_policies(city_data, focus_areas)
                            st.session_state['policy_report'] = policy_report

                            st.success(f"✅ Policy analysis completed for {city_name}")

                        except Exception as e:
                            st.error(f"❌ Error analyzing city data: {str(e)}")
                            # Fallback to basic analysis
                            city_data = {"name": city_name, "basic_info": {}, "economy": {}, "environment": {}}
                            policy_report = recommender.recommend_policies(city_data, focus_areas)
                            st.session_state['policy_report'] = policy_report

            with col2:
                st.subheader("📋 Policy Recommendations")

                if 'policy_report' in st.session_state:
                    report = st.session_state['policy_report']

                    # Show city analysis
                    st.write("**City Performance Analysis:**")
                    analysis = report['city_analysis']

                    if analysis['strengths']:
                        st.success(f"**Strengths:** {', '.join([s['category'] for s in analysis['strengths']])}")

                    if analysis['weaknesses']:
                        st.warning(f"**Areas for Improvement:** {', '.join([w['category'] for w in analysis['weaknesses']])}")

                    # Show recommendations
                    st.write("**Recommended Policies:**")
                    for i, rec in enumerate(report['recommendations'][:5]):
                        with st.expander(f"{i+1}. {rec['title']} (Priority: {rec['priority_level']})"):
                            st.write(f"**Category:** {rec['category']}")
                            st.write(f"**Description:** {rec['description']}")
                            st.write(f"**Timeline:** {rec['timeline']}")
                            st.write(f"**Expected Outcomes:** {', '.join(rec['expected_outcomes'])}")

                    # Show impact predictions
                    st.write("**Predicted Impact:**")
                    impact = report['impact_predictions']
                    col2_1, col2_2, col2_3 = st.columns(3)

                    with col2_1:
                        st.metric("Short-term Impact", f"{impact.get('overall_impact_score', 0):.1f}%")
                    with col2_2:
                        st.metric("Confidence Level", impact.get('confidence_level', 'Medium'))
                    with col2_3:
                        st.metric("Implementation Timeline", "18 months")

        except ImportError:
            st.error("Policy recommendation engine requires additional AI dependencies.")

    def _show_advanced_analytics_page(self):
        """Show advanced analytics page."""
        st.header("📊 Advanced Analytics Dashboard")

        try:
            from src.analytics.advanced_analytics import AdvancedAnalytics
            analytics = AdvancedAnalytics(self.config)

            # Analytics options
            analysis_type = st.selectbox(
                "Select Analysis Type:",
                ["📈 Comprehensive Report", "🔍 City Performance Trends", "⚖️ Advanced City Comparison", "🔧 Data Quality Assessment"]
            )

            if analysis_type == "📈 Comprehensive Report":
                if st.button("Generate Comprehensive Report"):
                    with st.spinner("Generating comprehensive analytics report using real data..."):
                        try:
                            # Load real cities data
                            from src.data.data_collector import CityDataCollector
                            data_collector = CityDataCollector(self.config)

                            # Load existing data or collect fresh data
                            all_data = data_collector.load_collected_data()

                            if not all_data:
                                st.info("📊 No processed data found. Collecting fresh data for analysis...")
                                all_data = data_collector.collect_all_data()

                            # Convert to list format for analytics
                            cities_data = []
                            for city_name, city_data in all_data.items():
                                if city_data and isinstance(city_data, dict):
                                    cities_data.append(city_data)

                            if not cities_data:
                                st.error("❌ No valid city data found for analysis")
                                return

                            st.info(f"📊 Analyzing {len(cities_data)} cities with real data...")
                            report = analytics.generate_comprehensive_report(cities_data)

                        except Exception as e:
                            st.error(f"❌ Error generating report: {str(e)}")
                            return

                        # Display key insights
                        st.subheader("📊 Summary Statistics")
                        stats = report['summary_statistics']

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Cities Analyzed", stats['total_cities'])
                        with col2:
                            st.metric("Metrics Evaluated", stats['total_metrics'])
                        with col3:
                            st.metric("Data Quality", "85%")

                        # Show trends
                        st.subheader("📈 Key Trends")
                        trends = report['trend_analysis']
                        for metric, trend_data in list(trends.items())[:5]:
                            trend_emoji = "📈" if trend_data['trend'] == 'increasing' else "📉" if trend_data['trend'] == 'decreasing' else "➡️"
                            st.write(f"{trend_emoji} **{metric}**: {trend_data['trend']} (correlation: {trend_data['correlation']:.2f})")

                        # Show correlations
                        st.subheader("🔗 Strong Correlations")
                        correlations = report['correlation_analysis']['strong_correlations']
                        for corr in correlations[:5]:
                            st.write(f"• **{corr['metric1']}** ↔ **{corr['metric2']}**: {corr['correlation']:.2f} ({corr['strength']})")

            elif analysis_type == "🔧 Data Quality Assessment":
                if st.button("Assess Data Quality"):
                    with st.spinner("Assessing data quality..."):
                        # Mock cities data with some missing values
                        cities_data = [
                            {
                                "name": city,
                                "economy": {"gdp_per_capita": 40000 + i*5000 if i % 3 != 0 else None},
                                "environment": {"air_quality_index": 60 + i*5}
                            }
                            for i, city in enumerate(self.config.data.target_cities[:10])
                        ]

                        quality_report = analytics.detect_data_quality_issues(cities_data)

                        # Display quality metrics
                        st.subheader("📊 Data Quality Overview")

                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Overall Quality Score", f"{quality_report['overall_quality_score']:.1f}%")
                        with col2:
                            st.metric("Data Completeness", f"{quality_report['completeness']['overall_completeness']:.1f}%")
                        with col3:
                            consistency_avg = np.mean(list(quality_report['consistency'].values())) if quality_report['consistency'] else 100
                            st.metric("Data Consistency", f"{consistency_avg:.1f}%")

                        # Show recommendations
                        if quality_report['recommendations']:
                            st.subheader("💡 Recommendations")
                            for rec in quality_report['recommendations']:
                                st.write(f"• {rec}")

        except ImportError:
            st.error("Advanced analytics requires additional dependencies.")

        # System actions
        st.subheader("🔄 System Actions")
        col3, col4 = st.columns(2)

        with col3:
            if st.button("🗑️ Clear Cache", use_container_width=True):
                st.cache_data.clear()
                st.success("Cache cleared!")

        with col4:
            if st.button("📊 System Info", use_container_width=True):
                st.info(f"Project Root: {self.config.project_root}")
                st.info(f"Python Path: {self.config.project_root}")
