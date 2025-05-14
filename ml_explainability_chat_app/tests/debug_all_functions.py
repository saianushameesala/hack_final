"""
Debug and test all functions in ML Explainability Chat App (step-by-step, app.py-like flow).
This script demonstrates the core pipeline (excluding Streamlit UI) using test data.
"""
import os
import sys
import logging
import json
import pandas as pd
import pickle as pkl
import traceback
import numpy as np

# Add parent directory to path so we can import modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Improve logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(parent_dir, "logs", "debug.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("debug")

# Add logs directory if it doesn't exist
os.makedirs(os.path.join(parent_dir, "logs"), exist_ok=True)

# Import app modules (no Streamlit)
from utils.llm_connector import OpenAIConnector
from utils.llm_explainer import LLMExplainer
from utils.executor import execute_action
from utils.vector_store import initialize_vector_store, get_similar_docs
from utils.explainers import get_shap_explanation, get_lime_explanation

# --- Test Data Setup (simulate a project selection as in app.py) ---
PROJECT_DATA = {
    "Machine Learning Models": {
        "Random Forest": {
            "model_path": os.path.join(parent_dir, "models", "rf_model.pkl"),
            "data_path": os.path.join(parent_dir, "data", "breast_cancer.csv"),
            "description": "Random Forest classifier for breast cancer prediction with high accuracy."
        },
        "XGBoost": {
            "model_path": os.path.join(parent_dir, "models", "xgb_model.pkl"),
            "data_path": os.path.join(parent_dir, "data", "diabetes.csv"),
            "description": "Gradient boosting model for diabetes prediction using XGBoost."
        }
    }
}

def load_model_and_data(project_type, project_name):
    logger.info(f"Loading model and data for {project_name}")
    project_info = PROJECT_DATA[project_type][project_name]
    model_path = project_info["model_path"]
    data_path = project_info["data_path"]
    
    # Check if model and data files exist
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        logger.warning("Model or data file not found. Creating test data and model...")
        
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        # Create test data and model
        from sklearn.datasets import load_breast_cancer, load_diabetes
        from sklearn.ensemble import RandomForestClassifier
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        
        if project_name == "Random Forest":
            # Create a simple Random Forest model and test data
            dataset = load_breast_cancer()
            X, y = dataset.data, dataset.target
            
            # Split data to save a portion as sample data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create DataFrame with feature names
            data = pd.DataFrame(np.vstack([X_train, X_test]), columns=dataset.feature_names)
            data['target'] = np.hstack([y_train, y_test])
            
            # Add sample indicator column
            data['is_sample'] = 0
            data.loc[len(X_train):, 'is_sample'] = 1
            
            # Create and fit a simple RF model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Save model and data
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            data.to_csv(data_path, index=False)
            
        elif project_name == "XGBoost":
            # Create a simple XGBoost model and test data
            dataset = load_diabetes()
            X, y = dataset.data, dataset.target
            
            # Split data to save a portion as sample data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Create DataFrame with feature names
            data = pd.DataFrame(np.vstack([X_train, X_test]), columns=dataset.feature_names)
            data['target'] = np.hstack([y_train, y_test])
            
            # Add sample indicator column
            data['is_sample'] = 0
            data.loc[len(X_train):, 'is_sample'] = 1
            
            # Create and fit a simple XGBoost model
            model = xgb.XGBRegressor(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            
            # Save model and data
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            data.to_csv(data_path, index=False)
            
        logger.info(f"Created and saved test model and data for {project_name}")
    else:
        # Load existing model and data
        with open(model_path, 'rb') as f:
            model = pkl.load(f)
        data = pd.read_csv(data_path)
        
    return model, data, project_info

def main():
    # --- Step 1: Simulate project selection ---
    project_type = "Machine Learning Models"
    project_name = "Random Forest"
    logger.info(f"Selected project: {project_name} from {project_type}")
    model, data, project_info = load_model_and_data(project_type, project_name)
    if model is None or data is None:
        print("Model or data not found. Exiting.")
        return

    print(f"\nSelected Project: {project_name}")
    print(f"Description: {project_info['description']}")
    print(f"Model type: {type(model).__name__}")
    print(f"Data shape: {data.shape}")

    # --- Step 2: Initialize vector store ---
    logger.info("Initializing vector store...")
    print("\n[Step] Initializing vector store...")
    initialize_vector_store(PROJECT_DATA)
    print("Vector store initialized.")

    # --- Step 3: Get context documents for a test query ---
    test_query = "Explain SHAP values for this model"
    logger.info(f"Retrieving context docs for query: '{test_query}'")
    print(f"\n[Step] Retrieving context docs for query: '{test_query}'")
    context_docs = get_similar_docs(test_query, top_k=2)
    for i, doc in enumerate(context_docs):
        print(f"Context Doc {i+1}: {doc['content'][:80]}...")

    # --- Step 4: Execute ML actions directly (simulate user requests) ---
    logger.info("Executing ML actions (SHAP, LIME, Feature Importance, etc.)")
    print("\n[Step] Executing ML actions (SHAP, LIME, Feature Importance, Prediction, General Info)...")

    # SHAP
    shap_result = execute_action("shap", model, data, project_info, context_docs)
    print(f"\nSHAP Result: {list(shap_result.keys())}")

    # LIME (for sample 0)
    lime_result = execute_action("lime:0", model, data, project_info, context_docs)
    print(f"\nLIME Result: {list(lime_result.keys())}")

    # Feature Importance
    fi_result = execute_action("feature_importance", model, data, project_info, context_docs)
    print(f"\nFeature Importance Result: {list(fi_result.keys())}")

    # Prediction
    pred_result = execute_action("prediction", model, data, project_info, context_docs)
    print(f"\nPrediction Result: {list(pred_result.keys())}")

    # General Info
    general_result = execute_action("general", model, data, project_info, context_docs)
    print(f"\nGeneral Info Result: {list(general_result.keys())}")

    # --- Step 5: Initialize LLM connector and explainer ---
    logger.info("Initializing LLM connector and explainer...")
    print("\n[Step] Initializing LLM connector and explainer...")
    llm_connector = OpenAIConnector(model_name="gpt-4.1-nano")
    if not llm_connector.is_initialized:
        print("OpenAI connector not initialized. Check your API key.")
        return
    explainer = LLMExplainer(llm_connector)
    print("LLM connector and explainer initialized.")

    # --- Step 6: Generate LLM explanations for each ML result ---
    logger.info("Generating LLM explanations for each ML result...")
    print("\n[Step] Generating LLM explanations for each ML result...")

    for action_type, result in [
        # ("shap", shap_result),
        # ("lime", lime_result),  # This line is handling LIME explanation results
        ("feature_importance", fi_result),
        # ("prediction", pred_result),
        # ("general", general_result)
    ]:
        print(f"\n--- LLM Explanation for {action_type.upper()} ---")
        try:
            # Pass user_query parameter to store Q&A in vector store
            explanation = explainer.explain_results(
                result_data=result,
                action_type=action_type,
                model_type=type(model).__name__,
                model=model,  # Pass model for more context
                project_info=project_info,  # Pass project info for better context
                user_query=f"Explain {action_type} for this model"  # Sample query
            )
            print(explanation[:500] + ("..." if len(explanation) > 500 else ""))
        except Exception as e:
            print(f"Error generating explanation for {action_type}: {e}")
            traceback.print_exc()

    logger.info("All core functions tested successfully")
    print("\n[Done] All core functions tested in app.py-like flow (excluding Streamlit UI).")

if __name__ == "__main__":
    main()
