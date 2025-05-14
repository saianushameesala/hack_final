"""
Module for comprehensive model explanations by combining multiple data sources.
"""

import os
import json
import pandas as pd
from openai import OpenAI
import logging
import numpy as np
from utils.explainers import get_shap_explanation, get_lime_explanation
from utils.executor import execute_action

logger = logging.getLogger("comprehensive_explainer")

def gather_comprehensive_data(model, data, project_info, sample_index=0):
    """
    Gather comprehensive data about the model from multiple sources.
    
    Args:
        model: The trained model
        data: The dataset
        project_info: Information about the project
        sample_index: Index of sample to use for instance-specific explanations
    
    Returns:
        Dictionary containing data from multiple analysis sources
    """
    try:
        # Determine target column name based on the dataset
        target_column = identify_target_column(data, project_info)
        logger.info(f"Identified target column: {target_column}")
        
        # Get full instance data including both features and target
        instance_data = data.iloc[sample_index].to_dict() if sample_index < len(data) else None
        actual_target_value = None
        if instance_data and target_column in instance_data:
            actual_target_value = instance_data[target_column]
        
        comprehensive_data = {
            "model_type": type(model).__name__,
            "project_info": project_info,
            "data_stats": {
                "shape": data.shape,
                "columns": list(data.columns),
                "target_column": target_column,
                "sample_rows": data.head(3).to_dict(orient="records")
            },
            "instance_full_data": instance_data,  # Full instance data including target
            "actual_value": actual_target_value,  # The actual target value for this instance
            "sample_index": sample_index  # Store the sample index
        }
        
        # Get feature importance
        logger.info("Gathering feature importance data")
        try:
            fi_result = execute_action("feature_importance", model, data, project_info)
            comprehensive_data["feature_importance"] = fi_result
        except Exception as e:
            logger.warning(f"Could not get feature importance: {e}")
        
        # Get SHAP global explanation
        logger.info("Gathering SHAP global explanation")
        try:
            # Use a sample of data for efficiency
            sample_size = min(100, len(data))
            sample_indices = np.random.choice(len(data), sample_size, replace=False)
            data_sample = data.iloc[sample_indices]
            
            shap_global = get_shap_explanation(model, data_sample)
            comprehensive_data["shap_global"] = shap_global
        except Exception as e:
            logger.warning(f"Could not get SHAP global explanation: {e}")
        
        # Get SHAP explanation for specific instance
        logger.info(f"Gathering SHAP explanation for instance {sample_index}")
        try:
            # Make SHAP explanation for a specific instance
            X = get_features_only(data, target_column)
                
            shap_instance = get_shap_explanation(model, X, instance_index=sample_index)
            comprehensive_data["shap_instance"] = shap_instance
        except Exception as e:
            logger.warning(f"Could not get SHAP instance explanation: {e}")
        
        # Get LIME explanation
        logger.info(f"Gathering LIME explanation for instance {sample_index}")
        try:
            # Get features only (excluding target)
            X = get_features_only(data, target_column)
            
            lime_result = get_lime_explanation(model, X, sample_index)
            comprehensive_data["lime"] = lime_result
        except Exception as e:
            logger.warning(f"Could not get LIME explanation: {e}")
            
        # Get prediction
        logger.info(f"Getting prediction for instance {sample_index}")
        try:
            prediction_result = execute_action(f"prediction:{sample_index}", model, data, project_info)
            comprehensive_data["prediction"] = prediction_result
        except Exception as e:
            logger.warning(f"Could not get prediction: {e}")
        
        # Directly include predicted vs actual comparison
        try:
            if "prediction" in comprehensive_data and actual_target_value is not None:
                pred_value = comprehensive_data["prediction"].get("prediction")
                comprehensive_data["prediction_vs_actual"] = {
                    "predicted": pred_value,
                    "actual": actual_target_value,
                    "match": str(pred_value) == str(actual_target_value)
                }
        except Exception as e:
            logger.warning(f"Could not compare prediction vs actual: {e}")
        
        return comprehensive_data
        
    except Exception as e:
        logger.error(f"Error gathering comprehensive data: {e}")
        return {"error": str(e)}

def identify_target_column(data, project_info):
    """
    Identify the target column name based on the dataset and project info.
    """
    # Try to find target column from common names
    common_target_names = ["target", "y", "Loan_Status", "SeriousDlqin2yrs", "label", "Class"]
    
    # First check if any of the common names exist in the data
    for name in common_target_names:
        if name in data.columns:
            return name
    
    # Check if any column is obviously the target by checking project description
    # For loan eligibility model, Loan_Status is the target (case-insensitive)
    project_desc = project_info.get('description', '').lower()
    if 'loan' in project_desc and 'eligibility' in project_desc:
        for col in data.columns:
            if 'loan' in col.lower() and 'status' in col.lower():
                return col
    
    # For credit risk model, SeriousDlqin2yrs is the target (case-insensitive)
    if 'credit' in project_desc and 'risk' in project_desc:
        for col in data.columns:
            if 'dlq' in col.lower() or 'default' in col.lower():
                return col
    
    # If no common name is found, assume the last column is the target
    # This is a common convention in many ML datasets
    if len(data.columns) > 0:
        return data.columns[-1]
    
    # If all else fails, return None
    return None

def get_features_only(data, target_column):
    """
    Get features only by removing target column and other special columns.
    """
    features_only = data.copy()
    
    # Remove target column if it exists
    if target_column in features_only.columns:
        features_only = features_only.drop(target_column, axis=1)
    
    # Remove other special columns that shouldn't be used as features
    columns_to_remove = ["is_sample", "id", "ID", "Id", "index", "Index"]
    for col in columns_to_remove:
        if col in features_only.columns:
            features_only = features_only.drop(col, axis=1)
    
    return features_only

def openai_comprehensive_explanation(user_request, model, data, project_info, sample_index=0):
    """
    Generate a comprehensive explanation using multiple data sources.
    
    Args:
        user_request: User's query string
        model: The trained model
        data: The dataset
        project_info: Information about the project
        sample_index: Index of sample to use for instance-specific explanations
    
    Returns:
        The model's explanation drawing from all data sources
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    # Get path to general prompt
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "general_prompt.txt")
    
    client = OpenAI(api_key=api_key)

    # Read prompt instructions
    system_prompt = None
    if os.path.exists(prompt_path):
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                system_prompt = f.read().strip()
            logger.info(f"Loaded general prompt from {prompt_path}")
        except Exception as e:
            logger.error(f"Error reading general prompt: {e}")
    
    # Gather comprehensive data from multiple sources
    comprehensive_data = gather_comprehensive_data(model, data, project_info, sample_index)
    
    # Extract and format the most important elements for the prompt
    prompt_parts = [f"User Question: {user_request}\n"]
    
    # Add model and project info
    prompt_parts.append(f"Model Type: {comprehensive_data.get('model_type', 'Unknown')}")
    if "project_info" in comprehensive_data:
        project = comprehensive_data["project_info"]
        prompt_parts.append(f"Project: {project.get('name', 'Unknown')}")
        prompt_parts.append(f"Description: {project.get('description', '')}\n")
    
    # Add data statistics and target column info
    if "data_stats" in comprehensive_data:
        stats = comprehensive_data["data_stats"]
        prompt_parts.append(f"Dataset Shape: {stats.get('shape', 'Unknown')}")
        target_column = stats.get('target_column', 'Unknown')
        prompt_parts.append(f"Target Column: {target_column} (this is what the model predicts)")
    
    # Add information about the specific instance being analyzed
    prompt_parts.append(f"\n## INSTANCE DATA (Sample #{comprehensive_data.get('sample_index', sample_index)})")
    if "instance_full_data" in comprehensive_data:
        instance = comprehensive_data["instance_full_data"]
        prompt_parts.append("Feature values for this instance:")
        for feature, value in instance.items():
            prompt_parts.append(f"- {feature}: {value}")
    
    # Add prediction vs actual comparison
    if "prediction_vs_actual" in comprehensive_data:
        comp = comprehensive_data["prediction_vs_actual"]
        prompt_parts.append(f"\n## PREDICTION VS ACTUAL")
        prompt_parts.append(f"Predicted: {comp.get('predicted', 'N/A')}")
        prompt_parts.append(f"Actual: {comp.get('actual', 'N/A')}")
        prompt_parts.append(f"Match: {comp.get('match', 'N/A')}")
    
    # Add feature importance information
    if "feature_importance" in comprehensive_data:
        fi_data = comprehensive_data["feature_importance"]
        prompt_parts.append("\n## FEATURE IMPORTANCE")
        if "sorted_importance" in fi_data:
            sorted_imp = fi_data["sorted_importance"]
            prompt_parts.append("Top features by importance:")
            for i, (feature, importance) in enumerate(sorted_imp[:10], 1):
                prompt_parts.append(f"{i}. {feature}: {importance:.4f}")
    
    # Add SHAP global information
    if "shap_global" in comprehensive_data:
        shap_global = comprehensive_data["shap_global"]
        prompt_parts.append("\n## GLOBAL SHAP VALUES")
        if "feature_importance" in shap_global:
            fi = shap_global["feature_importance"]
            sorted_fi = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)
            prompt_parts.append("Top features by SHAP impact:")
            for i, (feature, impact) in enumerate(sorted_fi[:10], 1):
                direction = "positive" if impact > 0 else "negative"
                prompt_parts.append(f"{i}. {feature}: {impact:.4f} ({direction} impact)")
    
    # Add SHAP instance information
    if "shap_instance" in comprehensive_data:
        shap_instance = comprehensive_data["shap_instance"]
        prompt_parts.append(f"\n## SHAP VALUES FOR INSTANCE #{sample_index}")
        if "feature_importance" in shap_instance and "feature_values" in shap_instance:
            fi = shap_instance["feature_importance"]
            fv = shap_instance["feature_values"]
            sorted_features = sorted([(f, fv[f], fi[f]) for f in fi.keys()], 
                                   key=lambda x: abs(x[2]), reverse=True)
            prompt_parts.append("Top feature impacts for this instance:")
            for i, (feature, value, impact) in enumerate(sorted_features[:10], 1):
                direction = "positive" if impact > 0 else "negative"
                prompt_parts.append(f"{i}. {feature}: value={value}, impact={impact:.4f} ({direction})")
    
    # Add LIME information
    if "lime" in comprehensive_data:
        lime_data = comprehensive_data["lime"]
        prompt_parts.append(f"\n## LIME EXPLANATION FOR INSTANCE #{sample_index}")
        if "explanation_obj" in lime_data and lime_data["explanation_obj"] is not None:
            try:
                lime_exp = lime_data["explanation_obj"]
                if hasattr(lime_exp, "as_list"):
                    lime_list = lime_exp.as_list()
                    prompt_parts.append("Feature weights from LIME:")
                    for i, (feature, weight) in enumerate(lime_list[:10], 1):
                        direction = "positive" if weight > 0 else "negative"
                        prompt_parts.append(f"{i}. {feature}: {weight:.4f} ({direction})")
            except Exception as e:
                logger.warning(f"Could not format LIME data: {e}")
    
    # Add prediction information
    if "prediction" in comprehensive_data:
        pred_data = comprehensive_data["prediction"]
        prompt_parts.append(f"\n## PREDICTION FOR INSTANCE #{sample_index}")
        if "prediction" in pred_data:
            prompt_parts.append(f"Prediction: {pred_data['prediction']}")
        if "probability" in pred_data:
            prompt_parts.append(f"Confidence: {pred_data['probability']:.4f}")
        if "input_features" in pred_data:
            prompt_parts.append("Input feature values:")
            for feature, value in list(pred_data["input_features"].items())[:10]:
                prompt_parts.append(f"- {feature}: {value}")
    
    # Combine all parts into a complete prompt
    full_prompt = "\n".join(prompt_parts)
    
    # Make the API call
    if system_prompt:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=1500
        )
    else:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=1500
        )
    
    result = response.choices[0].message.content.strip()
    return result
