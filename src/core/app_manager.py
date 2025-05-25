"""
Main application manager for NeuroUrban system.
"""

import logging
from typing import Optional
import streamlit as st

from src.config.settings import Config
from src.data.data_collector import CityDataCollector
from src.ml.city_analyzer import CityAnalyzer
from src.dl.blueprint_generator import BlueprintGenerator
from src.simulation.city_simulator import CitySimulator
from src.ui.dashboard import Dashboard
from src.utils.logger import safe_emoji_text
from src.utils.gpu_utils import setup_gpu_environment

class NeuroUrbanApp:
    """Main application class for NeuroUrban system."""

    def __init__(self, config: Config):
        """
        Initialize the NeuroUrban application.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize GPU environment
        self.gpu_manager = setup_gpu_environment(config)

        # Initialize components
        self.data_collector = None
        self.city_analyzer = None
        self.blueprint_generator = None
        self.city_simulator = None
        self.dashboard = None

        self._initialize_components()

    def _initialize_components(self):
        """Initialize all application components."""
        self.logger.info("Initializing NeuroUrban components...")

        try:
            # Data collection component
            self.data_collector = CityDataCollector(self.config)
            self.logger.info(safe_emoji_text("✅ Data collector initialized", "Data collector initialized"))

            # ML analysis component
            self.city_analyzer = CityAnalyzer(self.config)
            self.logger.info(safe_emoji_text("✅ City analyzer initialized", "City analyzer initialized"))

            # DL blueprint generation component - try PyTorch first, fallback to simple
            try:
                self.blueprint_generator = BlueprintGenerator(self.config)
                self.logger.info(safe_emoji_text("✅ PyTorch blueprint generator initialized", "PyTorch blueprint generator initialized"))
            except Exception as e:
                self.logger.warning(f"PyTorch blueprint generator failed: {str(e)}")
                self.logger.info("Falling back to simple blueprint generator...")
                try:
                    from src.dl.simple_blueprint_generator import SimpleBlueprintGenerator
                    self.blueprint_generator = SimpleBlueprintGenerator(self.config)
                    self.logger.info(safe_emoji_text("✅ Simple blueprint generator initialized", "Simple blueprint generator initialized"))
                except Exception as e2:
                    self.logger.error(f"Simple blueprint generator also failed: {str(e2)}")
                    self.blueprint_generator = None

            # Simulation component
            self.city_simulator = CitySimulator(self.config)
            self.logger.info(safe_emoji_text("✅ City simulator initialized", "City simulator initialized"))

            # Dashboard component
            self.dashboard = Dashboard(self.config, self)
            self.logger.info(safe_emoji_text("✅ Dashboard initialized", "Dashboard initialized"))

        except Exception as e:
            self.logger.error(safe_emoji_text(f"❌ Failed to initialize components: {str(e)}",
                                             f"Failed to initialize components: {str(e)}"))
            raise

    def run(self):
        """Run the main application."""
        self.logger.info(safe_emoji_text("🚀 Starting NeuroUrban application...",
                                        "Starting NeuroUrban application..."))

        # Check if we're in a Streamlit environment
        try:
            # Check if streamlit is already running
            import streamlit as st

            # Try to detect if we're in Streamlit context
            if hasattr(st, '_is_running_with_streamlit') or 'streamlit' in str(type(st)):
                # We're likely in Streamlit context
                self._run_streamlit()
            else:
                # Not in Streamlit, run CLI
                self._run_cli()

        except Exception as e:
            self.logger.info(f"Streamlit not available or error: {str(e)}")
            # Fall back to CLI version
            self._run_cli()

    def _run_streamlit(self):
        """Run Streamlit version with error handling."""
        try:
            import streamlit as st

            # Set page config with error handling
            try:
                st.set_page_config(
                    page_title=self.config.app.page_title,
                    page_icon=self.config.app.page_icon,
                    layout="wide",
                    initial_sidebar_state="expanded"
                )
            except Exception as e:
                self.logger.warning(f"Could not set page config: {str(e)}")

            # Run dashboard
            self.dashboard.run()

        except Exception as e:
            self.logger.error(f"Streamlit error: {str(e)}")
            self.logger.info("Falling back to CLI mode...")
            self._run_cli()

    def _run_cli(self):
        """Run command-line interface version."""
        self.logger.info("Running NeuroUrban in CLI mode...")

        print(safe_emoji_text("\n🏙️ Welcome to NeuroUrban: AI-Powered City Planning System",
                              "\nWelcome to NeuroUrban: AI-Powered City Planning System"))
        print("=" * 60)

        while True:
            print("\nAvailable options:")
            print("1. Collect city data")
            print("2. Analyze cities")
            print("3. Generate city blueprint")
            print("4. Run city simulation")
            print("5. Launch web dashboard")
            print("6. Exit")

            choice = input("\nEnter your choice (1-6): ").strip()

            if choice == "1":
                self._collect_data_cli()
            elif choice == "2":
                self._analyze_cities_cli()
            elif choice == "3":
                self._generate_blueprint_cli()
            elif choice == "4":
                self._run_simulation_cli()
            elif choice == "5":
                self._launch_dashboard()
            elif choice == "6":
                print(safe_emoji_text("👋 Goodbye!", "Goodbye!"))
                break
            else:
                print(safe_emoji_text("❌ Invalid choice. Please try again.",
                                     "Invalid choice. Please try again."))

    def _collect_data_cli(self):
        """Collect city data via CLI."""
        print(safe_emoji_text("\n📊 Collecting city data...", "\nCollecting city data..."))
        try:
            self.data_collector.collect_all_data()
            print(safe_emoji_text("✅ Data collection completed!", "Data collection completed!"))
        except Exception as e:
            print(safe_emoji_text(f"❌ Data collection failed: {str(e)}",
                                 f"Data collection failed: {str(e)}"))

    def _analyze_cities_cli(self):
        """Analyze cities via CLI."""
        print(safe_emoji_text("\n🔍 Analyzing cities...", "\nAnalyzing cities..."))
        try:
            results = self.city_analyzer.analyze_all_cities()
            print(safe_emoji_text("✅ City analysis completed!", "City analysis completed!"))
            print(safe_emoji_text(f"📈 Analyzed {len(results)} cities",
                                 f"Analyzed {len(results)} cities"))
        except Exception as e:
            print(safe_emoji_text(f"❌ City analysis failed: {str(e)}",
                                 f"City analysis failed: {str(e)}"))

    def _generate_blueprint_cli(self):
        """Generate city blueprint via CLI."""
        print(safe_emoji_text("\n🏗️ Generating city blueprint...", "\nGenerating city blueprint..."))
        try:
            blueprint = self.blueprint_generator.generate_blueprint()
            print(safe_emoji_text("✅ Blueprint generation completed!", "Blueprint generation completed!"))
            print(safe_emoji_text(f"💾 Blueprint saved to: {blueprint}",
                                 f"Blueprint saved to: {blueprint}"))
        except Exception as e:
            print(safe_emoji_text(f"❌ Blueprint generation failed: {str(e)}",
                                 f"Blueprint generation failed: {str(e)}"))

    def _run_simulation_cli(self):
        """Run city simulation via CLI."""
        print(safe_emoji_text("\n🎮 Running city simulation...", "\nRunning city simulation..."))
        try:
            results = self.city_simulator.run_simulation()
            print(safe_emoji_text("✅ Simulation completed!", "Simulation completed!"))
            print(safe_emoji_text(f"📊 Simulation results: {results}",
                                 f"Simulation results: {results}"))
        except Exception as e:
            print(safe_emoji_text(f"❌ Simulation failed: {str(e)}",
                                 f"Simulation failed: {str(e)}"))

    def _launch_dashboard(self):
        """Launch web dashboard."""
        print(safe_emoji_text("\n🌐 Launching web dashboard...", "\nLaunching web dashboard..."))
        print(safe_emoji_text(f"🔗 Open your browser to: http://{self.config.app.host}:{self.config.app.port}",
                             f"Open your browser to: http://{self.config.app.host}:{self.config.app.port}"))

        import subprocess
        import sys

        try:
            # Launch Streamlit
            subprocess.run([
                sys.executable, "-m", "streamlit", "run", "main.py",
                "--server.port", str(self.config.app.port),
                "--server.address", self.config.app.host
            ])
        except Exception as e:
            print(safe_emoji_text(f"❌ Failed to launch dashboard: {str(e)}",
                                 f"Failed to launch dashboard: {str(e)}"))

    def get_component(self, component_name: str):
        """Get a specific component by name."""
        components = {
            "data_collector": self.data_collector,
            "city_analyzer": self.city_analyzer,
            "blueprint_generator": self.blueprint_generator,
            "city_simulator": self.city_simulator,
            "dashboard": self.dashboard
        }
        return components.get(component_name)
