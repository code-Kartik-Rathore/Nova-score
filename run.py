#!/usr/bin/env python3
"""
Nova Score - Main entry point

This script provides a command-line interface to run different components
of the Nova Score system.
"""
import os
import sys
import subprocess
import signal
import time
from pathlib import Path
import argparse
import webbrowser
from typing import List, Optional

# Project directories
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

# Ensure required directories exist
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

def run_command(command: List[str], cwd: Optional[str] = None) -> int:
    """Run a shell command and return the exit code."""
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd or BASE_DIR,
            stdout=sys.stdout,
            stderr=subprocess.PIPE,
            text=True,
            shell=sys.platform == 'win32'
        )
        return process.wait()
    except KeyboardInterrupt:
        return 0
    except Exception as e:
        print(f"Error running command: {' '.join(command)}")
        print(f"Error: {e}")
        return 1

def generate_data() -> bool:
    """Generate synthetic partner data."""
    print("\n=== Generating Synthetic Data ===")
    data_file = DATA_DIR / "partners.csv"
    
    if data_file.exists():
        print(f"Data file already exists at {data_file}")
        if input("Regenerate data? (y/n): ").lower() != 'y':
            return True
    
    cmd = [sys.executable, "-m", "data.simulate"]
    return run_command(cmd) == 0

def train_model() -> bool:
    """Train the credit scoring model."""
    print("\n=== Training Model ===")
    cmd = [sys.executable, "-m", "src.train"]
    return run_command(cmd) == 0

def start_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = True) -> int:
    """Start the FastAPI server."""
    print("\n=== Starting API Server ===")
    cmd = [
        "uvicorn",
        "src.service:app",
        f"--host={host}",
        f"--port={port}",
    ]
    if reload:
        cmd.append("--reload")
    
    return run_command(cmd)

def start_ui(port: int = 8501) -> int:
    """Start the Streamlit UI."""
    print("\n=== Starting Web UI ===")
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    cmd = ["streamlit", "run", "app/score.py"]
    return run_command(cmd)

def open_browser(port: int, path: str = "") -> None:
    """Open a web browser to the specified URL."""
    url = f"http://localhost:{port}/{path}"
    print(f"Opening {url} in your default web browser...")
    webbrowser.open(url)

def check_requirements() -> bool:
    """Check if all required packages are installed."""
    print("\n=== Checking Requirements ===")
    cmd = [sys.executable, "-m", "pip", "check"]
    return run_command(cmd) == 0

def install_requirements() -> bool:
    """Install required packages."""
    print("\n=== Installing Requirements ===")
    cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    return run_command(cmd) == 0

def run_system():
    """Run the entire Nova Score system."""
    parser = argparse.ArgumentParser(description="Nova Score - Equitable Credit Scoring for Grab Partners")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Data generation command
    data_parser = subparsers.add_parser("data", help="Generate synthetic data")
    
    # Model training command
    train_parser = subparsers.add_parser("train", help="Train the model")
    
    # API server command
    api_parser = subparsers.add_parser("api", help="Start the API server")
    api_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    api_parser.add_argument("--port", type=int, default=8000, help="Port to run the API on")
    api_parser.add_argument("--no-reload", action="store_false", dest="reload", help="Disable auto-reload")
    
    # UI command
    ui_parser = subparsers.add_parser("ui", help="Start the web UI")
    ui_parser.add_argument("--port", type=int, default=8501, help="Port to run the UI on")
    
    # Run all command
    run_parser = subparsers.add_parser("run", help="Run the entire system")
    run_parser.add_argument("--api-port", type=int, default=8000, help="Port for the API server")
    run_parser.add_argument("--ui-port", type=int, default=8501, help="Port for the web UI")
    run_parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    
    # Install command
    install_parser = subparsers.add_parser("install", help="Install requirements")
    
    # Check command
    check_parser = subparsers.add_parser("check", help="Check requirements")
    
    args = parser.parse_args()
    
    if args.command == "data":
        sys.exit(0 if generate_data() else 1)
    
    elif args.command == "train":
        sys.exit(0 if train_model() else 1)
    
    elif args.command == "api":
        sys.exit(start_api(args.host, args.port, args.reload))
    
    elif args.command == "ui":
        sys.exit(start_ui(args.port))
    
    elif args.command == "install":
        sys.exit(0 if install_requirements() else 1)
    
    elif args.command == "check":
        sys.exit(0 if check_requirements() else 1)
    
    elif args.command == "run" or not hasattr(args, 'command'):
        # Default command: run the entire system
        print("=== Nova Score - Equitable Credit Scoring ===\n")
        
        # Check requirements
        if not check_requirements():
            print("\nSome requirements are not satisfied. Installing...")
            if not install_requirements():
                print("Failed to install requirements. Please check your Python environment.")
                sys.exit(1)
        
        # Generate data if needed
        if not generate_data():
            print("Failed to generate data.")
            sys.exit(1)
        
        # Check if model exists, otherwise train one
        model_file = MODEL_DIR / "xgboost.joblib"
        if not model_file.exists():
            print("\nNo trained model found. Training a new model...")
            if not train_model():
                print("Failed to train model.")
                sys.exit(1)
        
        # Start API server in a separate process
        import threading
        api_thread = threading.Thread(
            target=start_api,
            kwargs={"host": "0.0.0.0", "port": args.api_port, "reload": False},
            daemon=True
        )
        api_thread.start()
        
        # Give API server a moment to start
        time.sleep(2)
        
        # Open browser to UI
        if not args.no_browser:
            open_browser(args.ui_port)
        
        # Start UI (this will block)
        sys.exit(start_ui(args.ui_port))
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    run_system()
