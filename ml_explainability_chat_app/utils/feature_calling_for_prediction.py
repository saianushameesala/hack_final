"""
Module for OpenAI to explain model predictions with reasoning based on feature importance data.
"""

import os
import json
import pandas as pd
from openai import OpenAI
import numpy as np

def openai_explain_prediction(user_request, instance_data, prediction_result, explanation_data=None):
    """
    Use OpenAI to generate an explanation for why a model made a specific prediction.
    
    Args:
        user_request: User's query string
        instance_data: DataFrame containing the instance data being predicted (features and values)
        prediction_result: Model's prediction and confidence
        explanation_data: Optional dictionary with SHAP/LIME/feature importance values
    
    Returns:
        The model's explanation of the prediction
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    # Get path to prediction prompt
    prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "predict_prompt.txt")
    
    client = OpenAI(api_key=api_key)

    # Read prediction prompt instructions if available
    system_prompt = None
    if os.path.exists(prompt_path):
        try:
            with open(prompt_path, "r", encoding="utf-8") as f:
                system_prompt = f.read().strip()
        except Exception as e:
            print(f"Error reading prediction prompt: {e}")
    
    # Format instance data as a table
    if isinstance(instance_data, pd.DataFrame):
        df_table = instance_data.to_markdown(index=False)
    elif isinstance(instance_data, dict):
        df_table = pd.DataFrame([instance_data]).to_markdown(index=False)
    else:
        df_table = "Instance data not available in expected format."
    
    # Format prediction result information
    prediction_info = ""
    if isinstance(prediction_result, dict):
        prediction_info += f"Prediction: {prediction_result.get('prediction', 'N/A')}\n"
        
        if 'probability' in prediction_result:
            prediction_info += f"Confidence: {prediction_result.get('probability', 0) * 100:.2f}%\n"
        elif 'probabilities' in prediction_result and 'classes' in prediction_result:
            probs = prediction_result.get('probabilities', [])
            classes = prediction_result.get('classes', [])
            if len(probs) == len(classes):
                prediction_info += "Class probabilities:\n"
                for cls, prob in zip(classes, probs):
                    prediction_info += f"- {cls}: {prob * 100:.2f}%\n"
    
    # Format explanation data from SHAP, LIME, or feature importance
    explanation_info = ""
    if explanation_data:
        # Handle SHAP values
        if "feature_importance" in explanation_data and "feature_values" in explanation_data:
            fi = explanation_data["feature_importance"]
            fv = explanation_data["feature_values"]
            sorted_features = sorted([(f, fv.get(f, "N/A"), fi.get(f, 0)) for f in fi.keys()], 
                                    key=lambda x: abs(x[2]), reverse=True)
            
            explanation_info += "\nFeature contributions (SHAP values):\n"
            explanation_info += "| Feature | Value | Contribution |\n"
            explanation_info += "|---------|-------|-------------|\n"
            for feat, val, imp in sorted_features[:10]:  # Top 10 features
                direction = "+" if imp > 0 else "-"
                explanation_info += f"| {feat} | {val} | {direction}{abs(imp):.4f} |\n"
        
        # Handle LIME explanation
        elif "explanation_obj" in explanation_data and hasattr(explanation_data["explanation_obj"], "as_list"):
            try:
                lime_list = explanation_data["explanation_obj"].as_list()
                explanation_info += "\nFeature contributions (LIME):\n"
                explanation_info += "| Feature | Contribution |\n"
                explanation_info += "|---------|-------------|\n"
                for feat, weight in lime_list[:10]:  # Top 10 features
                    direction = "+" if weight > 0 else "-"
                    explanation_info += f"| {feat} | {direction}{abs(weight):.4f} |\n"
            except:
                pass
                
        # Handle feature importance
        elif "sorted_importance" in explanation_data:
            sorted_imp = explanation_data["sorted_importance"]
            explanation_info += "\nFeature importance:\n"
            explanation_info += "| Feature | Importance |\n"
            explanation_info += "|---------|------------|\n"
            for feat, imp in sorted_imp[:10]:  # Top 10 features
                explanation_info += f"| {feat} | {imp:.4f} |\n"
    
    # Prepare prompt with user request and all the gathered information
    prompt = f"""
{user_request}

Instance data:
{df_table}

{prediction_info}

{explanation_info}
"""

    # Use system prompt if available
    if system_prompt:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
    else:
        response = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
    
    result = response.choices[0].message.content.strip()
    return result

def test_openai_explain_prediction():
    """
    Test function for openai_explain_prediction with sample data.
    """
    # Create fake instance data
    instance_data = {
        "income": 65000,
        "debt_to_income": 0.52,
        "credit_score": 680, 
        "credit_history_length": 48,
        "age": 35,
        "num_credit_lines": 4
    }
    
    # Create fake prediction result
    prediction_result = {
        "prediction": "Approved",
        "probability": 0.78,
        "classes": ["Rejected", "Approved"],
        "probabilities": [0.22, 0.78]
    }
    
    # Create fake SHAP explanation data
    explanation_data = {
        "feature_importance": {
            "income": 0.35,
            "debt_to_income": -0.15,
            "credit_score": 0.40,
            "credit_history_length": 0.22,
            "age": 0.05,
            "num_credit_lines": -0.08
        },
        "feature_values": {
            "income": 65000,
            "debt_to_income": 0.52,
            "credit_score": 680,
            "credit_history_length": 48,
            "age": 35,
            "num_credit_lines": 4
        }
    }
    
    # Test request
    test_request = "Why was this loan application approved? What factors most influenced the decision?"
    
    print("Testing prediction explanation...")
    try:
        response = openai_explain_prediction(test_request, instance_data, prediction_result, explanation_data)
        print(f"Generated explanation:\n{response}")
    except Exception as e:
        print(f"Error: {e}")

# if __name__ == "__main__":
#     test_openai_explain_prediction()
