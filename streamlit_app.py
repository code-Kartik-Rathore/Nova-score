"""
Streamlit entry point for Nova Score application.
This file is required for Streamlit Cloud deployment.
"""
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

# Import and run the main application
from app import score

if __name__ == "__main__":
    score.main()
