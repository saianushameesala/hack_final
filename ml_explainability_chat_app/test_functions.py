"""
Test module for ML Explainability Chat App functions.
Run this before starting the Streamlit app to verify that all components work.
"""
import argparse
import logging
import sys
import os
import pickle as pkl
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import traceback

# Setup logging
logging.basicConfig(level=logging.INFO,
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_functions")

# Import application modules
from utils.llm_connector import get_available_llm, OpenAIConnector
from utils.llm_explainer import LLMExplainer
from utils.parser import parse_user_input, extract_sample_details
from utils.executor import execute_action
from utils.vector_store import initialize_vector_store, get_similar_docs
from utils.explainers import get_shap_explanation, get_lime_explanation

# Project data (same as in app.py)
project_data = {
    "Machine Learning Models": {
        "Random Forest": {
            "model_path": "models/rf_model.pkl",
            "data_path": "data/breast_cancer.csv",
            "description": "Random Forest classifier for breast cancer prediction with high accuracy."
        },
        "XGBoost": {
            "model_path": "models/xgb_model.pkl",
            "data_path": "data/diabetes.csv",
            "description": "Gradient boosting model for diabetes prediction using XGBoost."
        }
    }
}

def load_model_and_data(model_path: str, data_path: str) -> tuple:
    """
    Load a model and dataset from files
    
    Args:
        model_path: Path to the pickled model file
        data_path: Path to the CSV data file
        
    Returns:
        Tuple of (model, data) or (None, None) on error
    """
    model, data = None, None
    
    # Load model
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pkl.load(f)
            logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
    else:
        logger.error(f"Model file not found: {model_path}")
    
    # Load data
    if os.path.exists(data_path):
        try:
            data = pd.read_csv(data_path)
            logger.info(f"Loaded data from {data_path} with shape {data.shape}")
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {str(e)}")
    else:
        logger.error(f"Data file not found: {data_path}")
    
    return model, data

def test_parser():
    """Test the parser functions"""
    logger.info("Testing parser...")
    
    test_queries = [
        "Show me SHAP values for this model",
        "What are the most important features?",
        "Generate LIME explanation for sample #5",
        "How does this model make predictions?",
        "Explain the model's results for the first data point"
    ]
    
    for query in test_queries:
        action = parse_user_input(query)
        print(f"Query: '{query}' -> Action: '{action}'")
        
        # Test sample extraction if applicable
        if "sample" in query or "instance" in query:
            details = extract_sample_details(query)
            print(f"    Sample details: {details}")
    
    print("Parser test complete!")

def test_executor(model, data, project_info):
    """Test the executor functions"""
    logger.info("Testing executor...")
    if model is None or data is None:
        logger.error("Cannot test executor: model or data is None")
        return
        
    # Test SHAP explanation
    try:
        print("\nTesting SHAP explanation...")
        shap_result = execute_action("shap", model, data, project_info)
        print(f"SHAP result keys: {shap_result.keys()}")
        if "feature_importance" in shap_result:
            top_features = sorted(zip(shap_result["feature_names"], shap_result["feature_importance"]), 
                                key=lambda x: abs(x[1]), reverse=True)[:5]
            print(f"Top 5 features by SHAP: {top_features}")
    except Exception as e:
        print(f"Error in SHAP test: {str(e)}")
    
    # Test LIME explanation
    try:
        print("\nTesting LIME explanation...")
        lime_result = execute_action("lime", model, data, project_info)
        print(f"LIME result keys: {lime_result.keys()}")
        if "explanation" in lime_result:
            print(f"LIME explanation: {lime_result['explanation']}")
    except Exception as e:
        print(f"Error in LIME test: {str(e)}")
    
    # Test feature importance
    try:
        print("\nTesting feature importance...")
        fi_result = execute_action("feature_importance", model, data, project_info)
        print(f"Feature importance result keys: {fi_result.keys()}")
        if "sorted_importance" in fi_result:
            top_features = fi_result["sorted_importance"][:5]
            print(f"Top 5 features by importance: {top_features}")
    except Exception as e:
        print(f"Error in feature importance test: {str(e)}")
    
    # Test prediction
    try:
        print("\nTesting prediction...")
        pred_result = execute_action("prediction", model, data, project_info)
        print(f"Prediction result keys: {pred_result.keys()}")
        if "prediction" in pred_result:
            print(f"Prediction: {pred_result['prediction']}")
            if "probability" in pred_result:
                print(f"Probability: {pred_result['probability']}")
    except Exception as e:
        print(f"Error in prediction test: {str(e)}")
    
    print("Executor tests complete!")

def test_llm_connector():
    """Test LLM connector"""
    logger.info("Testing LLM connector...")
    
    try:
        # Try to get available LLM
        llm = OpenAIConnector()
        print(f"Using LLM connector: {type(llm).__name__}")
        
        if not llm.is_initialized:
            print("OpenAI connector not initialized. Check API key.")
            return None
        
        # Test text generation
        system_prompt = "You are a helpful AI assistant that explains machine learning concepts."
        user_prompt = "Explain SHAP values in simple terms."
        
        print("Generating text from OpenAI...")
        response = llm.generate_text(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.7
        )
        
        print(f"\nOpenAI Response:\n{'-' * 40}\n{response}\n{'-' * 40}")
        print("OpenAI connector test complete!")
        
        return llm
        
    except Exception as e:
        print(f"Error testing OpenAI connector: {str(e)}")
        traceback.print_exc()
        return None

def test_llm_explainer(llm, model, data, project_info):
    """Test LLM explainer with different types of ML results"""
    logger.info("Testing LLM explainer...")
    
    if llm is None:
        logger.error("Cannot test LLM explainer: no LLM connector available")
        return
    
    if model is None or data is None:
        logger.error("Cannot test LLM explainer: model or data is None")
        return
        
    # Initialize LLM explainer
    explainer = LLMExplainer(llm)
    
    try:
        # First generate some ML results
        print("\nGenerating feature importance results...")
        fi_result = execute_action("feature_importance", model, data, project_info)
        
        # Test explanation of feature importance
        print("Generating explanation for feature importance...")
        fi_explanation = explainer.explain_results(
            fi_result, 
            action_type="feature_importance", 
            model_type=type(model).__name__
        )
        
        print(f"\nFeature Importance Explanation:\n{'-' * 40}\n{fi_explanation}\n{'-' * 40}")
        
    except Exception as e:
        print(f"Error testing LLM explainer: {str(e)}")
        traceback.print_exc()
    
    print("LLM explainer tests complete!")

def test_vector_store():
    """Test vector store initialization and retrieval"""
    logger.info("Testing vector store...")
    
    try:
        # Initialize vector store
        print("Initializing vector store...")
        initialize_vector_store(project_data)
        
        # Test retrieval
        test_queries = [
            "How does the XGBoost model work?",
            "What are SHAP values for Random Forest?",
            "Show me feature importance"
        ]
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            docs = get_similar_docs(query, top_k=2)
            print(f"Found {len(docs)} relevant documents:")
            for i, doc in enumerate(docs):
                print(f"  {i+1}. {doc.get('title', 'Untitled')} (score: {doc.get('score', 0):.4f})")
                print(f"     {doc.get('content', '')[:100]}...")
    
    except Exception as e:
        print(f"Error testing vector store: {str(e)}")
        traceback.print_exc()
    
    print("Vector store tests complete!")

def main():
    """Run tests for all components"""
    parser = argparse.ArgumentParser(description='Test ML Explainability Chat App functions')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--parser', action='store_true', help='Test parser functions')
    parser.add_argument('--executor', action='store_true', help='Test executor functions')
    parser.add_argument('--llm', action='store_true', help='Test LLM connector')
    parser.add_argument('--explainer', action='store_true', help='Test LLM explainer')
    parser.add_argument('--vector-store', action='store_true', help='Test vector store')
    parser.add_argument('--project', choices=['Random Forest', 'XGBoost'], default='XGBoost',
                      help='Project to use for testing')
    args = parser.parse_args()
    
    # If no specific tests selected, show help
    if not any([args.all, args.parser, args.executor, args.llm, args.explainer, args.vector_store]):
        parser.print_help()
        return
    
    print("=" * 60)
    print(" ML Explainability Chat App Function Tests")
    print("=" * 60)
    
    # Load model and data for testing
    project_category = "Machine Learning Models"
    project_name = args.project
    project_info = project_data[project_category][project_name]
    
    model, data = None, None
    if args.all or args.executor or args.explainer:
        model, data = load_model_and_data(project_info["model_path"], project_info["data_path"])
        # Add name to project info for better output
        project_info["name"] = project_name
    
    # Run selected tests
    if args.all or args.parser:
        print("\n--- Testing Parser ---")
        test_parser()
    
    llm = None
    if args.all or args.llm:
        print("\n--- Testing LLM Connector ---")
        llm = test_llm_connector()
    
    if args.all or args.executor:
        print("\n--- Testing Executor ---")
        test_executor(model, data, project_info)
    
    if args.all or args.explainer:
        print("\n--- Testing LLM Explainer ---")
        if llm is None and args.all:
            # Try to get LLM if not already tested but running all tests
            llm = test_llm_connector()
        test_llm_explainer(llm, model, data, project_info)
    
    if args.all or args.vector_store:
        print("\n--- Testing Vector Store ---")
        test_vector_store()
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()
