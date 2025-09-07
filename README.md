# Nova Score

An equitable credit scoring system for Grab partners (drivers/merchants) who lack traditional credit histories.

## Overview

Nova Score is a machine learning-based credit scoring system that evaluates partners based on their in-app behavioral signals such as:

- Trip history and reliability
- Earnings patterns
- Customer ratings
- Platform tenure
- And more...

The system provides:
- **Fair Credit Assessment**: Uses statistical techniques to ensure fairness across different demographic groups
- **Explainable Decisions**: Provides clear reasons for credit decisions
- **Fast API**: Real-time scoring with low latency
- **Interactive Demo**: Streamlit UI for testing and demonstration

## Project Structure

```
nova-score/
├── data/                   # Data directory
│   ├── __init__.py
│   └── simulate.py         # Script to generate synthetic partner data
├── notebooks/              # Jupyter notebooks for EDA and analysis
├── src/                    # Source code
│   ├── __init__.py
│   ├── train.py            # Model training script
│   ├── service.py          # FastAPI service
│   └── fairness.py         # Fairness evaluation utilities
├── app/
│   └── score.py            # Streamlit demo UI
├── tests/                  # Test files
├── models/                 # Trained models and metadata (created at runtime)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/nova-score.git
   cd nova-score
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

The Nova Score application has two main components:

### 1. Streamlit UI (Demo Frontend)

To run the interactive Streamlit demo:

```bash
cd app
streamlit run app.py
```

This will start the Streamlit development server and open your default web browser to `http://localhost:8501`.

### 2. FastAPI Backend (Optional)

If you want to run the API server separately:

```bash
cd src
uvicorn service:app --reload
```

The API will be available at `http://127.0.0.1:8000`

## Application Flow

1. **Home Page** (`/`)
   - Quick overview and navigation
   - Check eligibility for partners
   - Access to Fairness Dashboard and Ops Console

2. **Partner Scoring** (`/partner`)
   - Input partner details or use example data
   - Real-time score preview
   - Submit for full evaluation

3. **Result Page** (`/result`)
   - Visual score representation
   - Approval/Review/Decline decision
   - Key factors affecting the score
   - Actionable insights for improvement

4. **Fairness Dashboard** (`/fairness`)
   - Model fairness metrics
   - Approval rates by segment
   - What-if analysis for threshold tuning

5. **Ops Console** (`/ops`)
   - Review partner applications
   - View detailed application information
   - Make manual decisions
   - Request additional information

6. **About** (`/about`)
   - Model information and metrics
   - Fairness and ethics statement
   - Contact information

## Development

### Project Structure

```
nova-score/
├── app/                    # Streamlit application
│   ├── pages/              # Application pages
│   │   ├── partner.py      # Partner scoring form
│   │   ├── result.py       # Score results
│   │   ├── fairness.py     # Fairness dashboard
│   │   ├── ops.py          # Operations console
│   │   └── about.py        # About page
│   └── app.py              # Main application entry point
├── data/                   # Data and simulation
│   ├── __init__.py
│   └── simulate.py
├── models/                 # Trained models
├── src/                    # Backend services
│   ├── __init__.py
│   └── service.py          # FastAPI service
├── requirements.txt        # Python dependencies
└── README.md
```

### Adding New Features

1. Create a new file in `app/pages/` for new pages
2. Add navigation in `app/app.py`
3. Update the requirements if adding new dependencies

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd nova-score
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Generate Synthetic Data

```bash
# Generate synthetic partner data (saved to data/partners.csv)
python -m data.simulate
```

### 2. Train the Model

```bash
# Train models and save to models/ directory
python -m src.train
```

### 3. Start the API Server

```bash
# Start the FastAPI server
uvicorn src.service:app --reload
```

The API will be available at `http://localhost:8000`

### 4. Run the Streamlit Demo

```bash
# In a new terminal
streamlit run app/score.py
```

Access the demo at `http://localhost:8501`

## API Endpoints

- `GET /`: API information
- `GET /health`: Health check
- `GET /models`: List available models
- `POST /score`: Get credit score for a partner

### Example API Request

```bash
curl -X 'POST' \
  'http://localhost:8000/score' \
  -H 'Content-Type: application/json' \
  -d '{
    "months_on_platform": 12,
    "weekly_trips": 25,
    "cancel_rate": 0.1,
    "on_time_rate": 0.95,
    "avg_rating": 4.5,
    "earnings_volatility": 0.15,
    "region": "North"
  }'
```

## Model Fairness

The system includes fairness evaluation tools to ensure equitable treatment across different demographic groups. Key fairness metrics include:

- Equal Opportunity Difference
- Demographic Parity
- Disparate Impact Ratio

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Grab for the opportunity to work on financial inclusion
- Open-source community for the amazing tools and libraries
- Research in fair machine learning and credit scoring
