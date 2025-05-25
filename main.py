#!/usr/bin/env python3
"""
NeuroUrban: AI-Powered Ideal City Blueprint System
Main application entry point for the city planning AI system.

This system analyzes successful global cities and generates optimal city blueprints
using Machine Learning and Deep Learning techniques.
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.app_manager import NeuroUrbanApp
from src.utils.logger import setup_logging, safe_emoji_text
from src.config.settings import Config

def main():
    """Main entry point for NeuroUrban application."""

    # Check if we're being run by Streamlit
    if len(sys.argv) > 1 and 'streamlit' in ' '.join(sys.argv):
        # We're being run by Streamlit, don't initialize here
        return

    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info(safe_emoji_text("🏙️ Starting NeuroUrban: AI-Powered City Planning System",
                                "Starting NeuroUrban: AI-Powered City Planning System"))
    logger.info("=" * 60)

    try:
        # Initialize configuration
        config = Config()

        # Create and run the main application
        app = NeuroUrbanApp(config)
        app.run()

    except KeyboardInterrupt:
        logger.info(safe_emoji_text("\n🛑 Application interrupted by user",
                                   "\nApplication interrupted by user"))
    except Exception as e:
        logger.error(safe_emoji_text(f"❌ Application error: {str(e)}",
                                    f"Application error: {str(e)}"))
        raise
    finally:
        logger.info(safe_emoji_text("🏁 NeuroUrban application finished",
                                   "NeuroUrban application finished"))

if __name__ == "__main__":
    main()