CreditGuard: Taiwan Credit Default Prediction System
An end-to-end machine learning system designed to predict the probability of credit card default for customers in Taiwan. This project integrates real-time inference, model explainability, and automated monitoring.

Business Context
Problem: Financial institutions in Taiwan face significant losses due to credit card defaults. Manual risk assessment is slow and prone to human error. Solution: CreditGuard provides an automated risk score (0.0 to 1.0).
Target: Predict if a client will default next month.
KPIs: Reduce Non-Performing Loans (NPLs) by identifying high-risk clients before credit expansion.
Threshold: The system uses a calibrated threshold of 0.45 to balance precision and recall.

Data Science & Modeling
The model is built on the UCI Taiwan Credit Card dataset containing 30,000 records.

Technical Stack
Model: XGBoost Classifier.
Explainability (SHAP): Every prediction includes a waterfall plot explaining why a specific user was flagged as high or low risk.
Monitoring (Evidently AI): Automated data drift detection to ensure the model remains accurate as financial trends shift.

Feature Engineering
X1: Amount of given credit (NT dollar).
X6-X11: History of past payments (Sept - April).
X12-X23: Bill amounts and previous payment amounts.

MLOps & Infrastructure
This project follows modern DevOps practices to ensure scalability and reliability.
Architecture
Backend: FastAPI (Python 3.12.7) serving the model.
Frontend: Streamlit for the business user dashboard.
Tracking: MLflow for experiment tracking and model versioning.
Containerization: Multi-container setup via Docker Compose.
CI/CD: GitHub Actions automatically builds and pushes images to DockerHub and AWS ECR using OIDC authentication.

 Getting Started
1. Local Development (Docker Compose)
To run the entire system (Frontend + Backend + MLflow) locally:

Bash
git clone https://github.com/your-username/taiwan-credit.git
cd taiwan-credit
docker-compose up --build
Streamlit UI: http://localhost:8501

FastAPI Docs: http://localhost:8000/docs

2. Environment Variables
Ensure you create a .env file in the root directory:

Code snippet

BACKEND_URL=http://backend:8000
MLFLOW_TRACKING_URI=http://mlflow_server:5000

 Deployment Pipeline
Our GitHub Actions workflow handles the production lifecycle:
Lint & Test: Validates Python code quality.
Build Images: Creates Docker images for Frontend and Backend.
DockerHub Push: Pushes to public registry for portfolio visibility.
AWS ECR Push: Securely pushes to AWS for production deployment using OIDC.

Monitoring & Observability
We use Evidently AI to monitor the "Health" of our model.
Data Drift: Checks if the input features have changed statistically compared to training.
Prediction Drift: Checks if the model is suddenly predicting significantly more defaults than usual.

