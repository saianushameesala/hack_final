"""
Executor module for the ML Explainability Chat App.
Executes ML explanation actions based on user requests.
"""
import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
import re
import os
import traceback
from utils.explainers import get_shap_explanation, get_lime_explanation

logger = logging.getLogger("executor")

def execute_action(action: str, model, data, project_info: Dict[str, Any], context_docs: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute an ML explanation action
    """
    if model is None:
        logger.error("No model available for execution")
        return {"error": "No model available for execution", "type": "error"}
        
    if data is None or data.empty:
        logger.error("No data available for execution")
        return {"error": "No data available for execution", "type": "error"}
        
    logger.info(f"Executing action: {action}")
    
    # Check for specific sample in SHAP and LIME requests
    sample_index = None
    if action.startswith("lime:"):
        parts = action.split(":")
        action = "lime"
        try:
            sample_index = int(parts[1])
            logger.info(f"LIME requested for specific sample index: {sample_index}")
        except (IndexError, ValueError):
            logger.warning("Invalid sample index format, defaulting to sample 0")
            sample_index = 0  # Default to first sample
    elif action.startswith("shap:"):
        parts = action.split(":")
        action = "shap"
        try:
            sample_index = int(parts[1])
            logger.info(f"SHAP requested for specific sample index: {sample_index}")
        except (IndexError, ValueError):
            logger.warning("Invalid sample index format, using global SHAP")
            sample_index = None  # Use global SHAP explanation
    
    # Calculate basic data statistics for LLM context
    features_only = data.copy()
    if 'target' in features_only.columns:
        features_only.drop('target', axis=1, inplace=True)
    if 'is_sample' in features_only.columns:
        features_only.drop('is_sample', axis=1, inplace=True)
            
    data_stats = {
        "n_samples": len(features_only),
        "n_features": len(features_only.columns),
        "column_names": features_only.columns.tolist(),
        "column_types": {col: str(features_only[col].dtype) for col in features_only.columns},
        "has_missing_values": features_only.isna().any().any(),
        "sample_data": features_only.head(3).to_dict(orient='records')  # Include some sample data
    }
    
    # Execute based on action type
    try:
        result = None
        
        if action == "shap":
            logger.info(f"Executing SHAP explanation {'for specific sample' if sample_index is not None else 'globally'}")
            result = execute_shap(model, data, sample_index=sample_index)
            # Add data stats for LLM context
            if result and isinstance(result, dict):
                result["data_stats"] = data_stats
            
        elif action == "lime":
            logger.info(f"Executing LIME explanation for sample {sample_index}")
            result = execute_lime(model, data, sample_index=sample_index)
            # Add data stats for LLM context
            if result and isinstance(result, dict):
                result["data_stats"] = data_stats
            
        elif action == "feature_importance":
            logger.info("Executing feature importance analysis")
            result = execute_feature_importance(model, data)
            # Add data stats for LLM context
            if result and isinstance(result, dict):
                result["data_stats"] = data_stats
            
        elif action == "prediction":
            logger.info("Executing prediction")
            result = execute_prediction(model, data, sample_index)
            # Add data stats for LLM context
            if result and isinstance(result, dict):
                result["data_stats"] = data_stats
            
        else:
            logger.info("Executing general model information")
            result = execute_general_info(model, data, project_info)
            # Add data stats for LLM context
            if result and isinstance(result, dict):
                result["data_stats"] = data_stats
            
        # Add model metadata if available
        if result and isinstance(result, dict) and model is not None:
            model_metadata = {}
            if hasattr(model, "get_params"):
                model_metadata["params"] = model.get_params()
            if hasattr(model, "feature_names_in_"):
                model_metadata["feature_names"] = list(model.feature_names_in_)
            if hasattr(model, "classes_"):
                model_metadata["classes"] = list(map(str, model.classes_))
                
            # Add feature importance if available in model
            if hasattr(model, "feature_importances_"):
                feature_names = features_only.columns.tolist()
                feature_importances = model.feature_importances_
                model_metadata["feature_importance"] = {
                    feature: float(importance) 
                    for feature, importance in zip(feature_names, feature_importances)
                }
            elif hasattr(model, "coef_"):
                feature_names = features_only.columns.tolist()
                coefficients = model.coef_
                if coefficients.ndim > 1:
                    coefficients = coefficients[0] # Take first class for multiclass
                model_metadata["coefficients"] = {
                    feature: float(coef) 
                    for feature, coef in zip(feature_names, coefficients)
                }
                
            result["model_metadata"] = model_metadata
            
        # Add project context if available
        if result and isinstance(result, dict) and project_info:
            result["project_info"] = {
                "name": project_info.get("name", "Unknown"),
                "description": project_info.get("description", "")
            }
            
        # Add context documents if available
        if result and isinstance(result, dict) and context_docs:
            result["context"] = [
                {"content": doc.get("content", ""), "score": doc.get("score", 0)}
                for doc in context_docs[:3]  # Top 3 context documents
            ]
            
        logger.info(f"Action '{action}' execution successful")
        return result
            
    except Exception as e:
        logger.error(f"Error executing {action}: {str(e)}")
        import traceback
        tb = traceback.format_exc()
        logger.debug(tb)
        return {
            "error": f"Error executing {action}: {str(e)}",
            "traceback": tb,
            "type": "error",
            "data_stats": data_stats  # Include data stats even in error case
        }

def execute_shap(model, data, sample_index: Optional[int] = None) -> Dict[str, Any]:
    """Execute SHAP explanation"""
    from utils.explainers import get_shap_explanation
    logger.info(f"Getting SHAP explanation {'for specific sample' if sample_index is not None else 'globally'}")
    
    try:
        # Split features and target
        if 'target' in data.columns:
            X = data.drop(['target'], axis=1)
            if 'is_sample' in X.columns:
                X = X.drop(['is_sample'], axis=1)
        else:
            X = data.iloc[:, :-1]  # All columns except last
        
        # Get SHAP values with proper error handling
        try:
            result = get_shap_explanation(model, X, instance_index=sample_index)
            return result
        except Exception as e:
            logger.error(f"Error in SHAP explanation: {str(e)}")
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Traceback: {tb}")
            return {
                "error": f"Error generating SHAP explanation: {str(e)}",
                "traceback": tb,
                "type": "error"
            }
    except Exception as e:
        logger.error(f"Error preparing data for SHAP explanation: {str(e)}")
        import traceback
        return {
            "error": f"Error preparing data for SHAP explanation: {str(e)}",
            "traceback": traceback.format_exc(),
            "type": "error"
        }

def execute_lime(model, data, sample_index: Optional[int] = None) -> Dict[str, Any]:
    """Execute LIME explanation"""
    from utils.explainers import get_lime_explanation
    
    # Split features and target
    X = data.iloc[:, :-1]  # All columns except last
    y_col = data.columns[-1]  # Target column name
    
    # Default to first sample if not specified
    if sample_index is None or sample_index >= len(data):
        sample_index = 0
        
    logger.info(f"Getting LIME explanation for sample {sample_index}")
    
    try:
        # Get LIME explanation
        result = get_lime_explanation(model, X, sample_index)
        # Add the actual sample data to the result for better LLM context
        if isinstance(result, dict):
            result["sample_data"] = X.iloc[sample_index].to_dict()
            # If we can make a prediction, add it
            try:
                prediction = model.predict(X.iloc[[sample_index]])[0]
                result["prediction"] = prediction
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X.iloc[[sample_index]])[0]
                    result["prediction_proba"] = proba.tolist() 
            except Exception as e:
                logger.warning(f"Could not add prediction to LIME result: {e}")
        return result
    except Exception as e:
        logger.error(f"Error in LIME explanation: {str(e)}")
        import traceback
        return {
            "error": f"Error generating LIME explanation: {str(e)}",
            "traceback": traceback.format_exc(),
            "type": "error"
        }

def execute_feature_importance(model, data) -> Dict[str, Any]:
    """Execute feature importance analysis"""
    logger.info("Getting feature importance")
    
    # Split features and target
    X = data.iloc[:, :-1]  # All columns except last
    feature_names = X.columns
    
    try:
        # Get feature importance based on model type
        if hasattr(model, "feature_importances_"):
            # For tree-based models (Random Forest, XGBoost, etc.)
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            # For linear models (Linear/Logistic Regression, etc.)
            importance = model.coef_
            if importance.ndim > 1:
                importance = importance[0]  # Get first row for multi-class
                
            # Take absolute value for linear models
            importance = np.abs(importance)
        else:
            return {
                "error": f"Model type {type(model).__name__} doesn't provide feature importance",
                "type": "error"
            }
            
        # Create dictionary mapping feature names to importance values
        importance_dict = {name: float(imp) for name, imp in zip(feature_names, importance)}
        
        # Sort features by importance (descending)
        sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Add additional information for LLM context
        result = {
            "feature_importance": importance_dict,
            "sorted_importance": sorted_importance,
            "model_type": type(model).__name__,
            "type": "feature_importance"
        }
        
        # Add feature statistics if possible
        try:
            feature_stats = {
                feature: {
                    "mean": float(X[feature].mean()),
                    "std": float(X[feature].std()),
                    "min": float(X[feature].min()),
                    "max": float(X[feature].max())
                }
                for feature in X.columns[:10]  # Top 10 features for simplicity
            }
            result["feature_stats"] = feature_stats
        except Exception as e:
            logger.warning(f"Could not add feature stats: {e}")
            
        return result
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        import traceback
        return {
            "error": f"Error getting feature importance: {str(e)}",
            "traceback": traceback.format_exc(),
            "type": "error"
        }

def execute_prediction(model, data, sample_index: Optional[int] = None) -> Dict[str, Any]:
    """Execute prediction on sample data"""
    logger.info("Making prediction")
    
    # Use first row by default or specified index
    X = data.iloc[:, :-1]  # All columns except last
    
    if sample_index is not None and 0 <= sample_index < len(X):
        sample = X.iloc[[sample_index]]  # Use specified sample
        sample_idx = sample_index
    else:
        sample = X.iloc[0:1]  # First row
        sample_idx = 0
    
    try:
        # Make prediction
        if hasattr(model, "predict_proba"):
            pred_proba = model.predict_proba(sample)
            prediction = model.predict(sample)
            
            result = {
                "prediction": prediction[0],
                "probability": float(np.max(pred_proba[0])),
                "probabilities": pred_proba[0].tolist() if hasattr(pred_proba[0], "tolist") else pred_proba[0],
                "input_features": sample.iloc[0].to_dict(),
                "sample_index": sample_idx,
                "type": "prediction"
            }
            
            # If possible, add class labels
            if hasattr(model, "classes_"):
                result["classes"] = model.classes_.tolist() if hasattr(model.classes_, "tolist") else [str(c) for c in model.classes_]
                
            return result
        else:
            prediction = model.predict(sample)
            return {
                "prediction": float(prediction[0]) if isinstance(prediction[0], (np.number, float, int)) else prediction[0],
                "input_features": sample.iloc[0].to_dict(),
                "sample_index": sample_idx,
                "type": "prediction"
            }
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        import traceback
        return {
            "error": f"Error making prediction: {str(e)}",
            "traceback": traceback.format_exc(),
            "type": "error"
        }

def execute_general_info(model, data, project_info: Dict[str, Any]) -> Dict[str, Any]:
    """Execute general information about the model"""
    logger.info("Getting general model information")
    
    model_type = type(model).__name__
    
    # Basic information
    result = {
        "model_type": model_type,
        "project_name": project_info.get("name", "Unknown"),
        "description": project_info.get("description", "No description available"),
        "type": "general_info"
    }
    
    # Data information
    if data is not None:
        result.update({
            "data_shape": data.shape,
            "feature_count": data.shape[1] - 1,  # Excluding target
            "sample_count": data.shape[0],
            "feature_names": list(data.columns[:-1])
        })
        
        # Add data statistics for better LLM context
        try:
            # Select only numeric columns
            numeric_cols = data.select_dtypes(include=["number"]).columns
            stats = {}
            for col in numeric_cols:
                stats[col] = {
                    "mean": float(data[col].mean()),
                    "std": float(data[col].std()),
                    "min": float(data[col].min()),
                    "max": float(data[col].max())
                }
            result["column_stats"] = stats
        except Exception as e:
            logger.warning(f"Could not add column statistics: {e}")
    
    # Add model parameters if available
    if hasattr(model, "get_params"):
        result["model_params"] = model.get_params()
        
    # Add feature importance if available
    if hasattr(model, "feature_importances_") and data is not None:
        feature_names = list(data.columns[:-1])
        feature_importances = model.feature_importances_
        
        # Ensure lengths match
        if len(feature_names) == len(feature_importances):
            sorted_importance = sorted(zip(feature_names, feature_importances), key=lambda x: x[1], reverse=True)
            result["feature_importance"] = sorted_importance
    
    return result
