"""
LLM Explainer for ML Explainability Chat App.
Converts technical ML results to natural language explanations.
"""
import json
import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
import os
from utils.llm_connector import LLMConnector
from utils.vector_store import documents  # Import global documents list for storing Q&A

logger = logging.getLogger("llm_explainer")

class LLMExplainer:
    """Uses LLM to generate human-readable explanations of ML results"""
    
    def __init__(self, llm_connector: LLMConnector):
        """Initialize the LLM explainer"""
        self.llm = llm_connector
        if not self.llm:
            logger.error("No LLM connector provided!")
        elif not self.llm.is_available():
            logger.error(f"LLM connector {type(self.llm).__name__} is not initialized!")
        else:
            logger.info(f"Initialized LLM explainer with connector: {type(self.llm).__name__}")
            
    def explain_results(self, result_data: Any, action_type: str, model_type: str = None, model=None, project_info=None, user_query=None) -> str:
        """
        Generate a natural language explanation of ML results.
        Passes model and project_info for richer context.
        Also stores the user query and LLM result in the vector store.
        Includes SHAP/LIME values as DataFrame if available.
        """
        if not self.llm or not self.llm.is_available():
            logger.error("Cannot generate explanation: No LLM connector available")
            return "Cannot generate explanation: No LLM connector available."
    
        logger.info(f"Generating explanation for {action_type} results on {model_type}")
        
        try:
            # Format the result data for the LLM, including model and project info
            formatted_prompt = self._format_results_for_prompt(
                result_data, action_type, model_type, model=model, project_info=project_info
            )
            logger.info(f"Formatted prompt length: {len(formatted_prompt)} chars")
            
            # Add SHAP/LIME raw values to the prompt if available, as DataFrame
            if action_type.lower() == "shap" and isinstance(result_data, dict):
                try:
                    # Add feature values and their SHAP impact for specific instance
                    if "feature_importance" in result_data and "feature_values" in result_data:
                        fi = result_data["feature_importance"]
                        fv = result_data["feature_values"]
                        sorted_features = sorted([(f, fv[f], fi[f]) for f in fi.keys()], 
                                               key=lambda x: abs(x[2]), reverse=True)
                        
                        formatted_prompt += "\n\nFeature values and their SHAP impact (sorted by importance):\n"
                        formatted_prompt += "| Feature | Value | SHAP Impact |\n"
                        formatted_prompt += "|---------|-------|------------|\n"
                        for feat, val, imp in sorted_features[:15]:  # Top 15 features
                            formatted_prompt += f"| {feat} | {val} | {imp:.4f} |\n"
                            
                        if "sample_index" in result_data:
                            formatted_prompt += f"\nThis is for instance #{result_data['sample_index']}"
                            
                        if "base_value" in result_data:
                            formatted_prompt += f"\nBase value: {result_data['base_value']:.4f}"
                    
                    # Add global SHAP values if available
                    elif "feature_importance" in result_data:
                        formatted_prompt += "\n\nGlobal SHAP values (feature importance):\n"
                        fi = result_data["feature_importance"]
                        sorted_fi = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)
                        formatted_prompt += "| Feature | SHAP Value |\n"
                        formatted_prompt += "|---------|------------|\n"
                        for feat, val in sorted_fi[:15]:  # Top 15 features
                            formatted_prompt += f"| {feat} | {val:.4f} |\n"
                    
                except Exception as e:
                    logger.warning(f"Could not add SHAP DataFrame: {e}")
                    
            elif action_type.lower() == "lime" and isinstance(result_data, dict) and "explanation_obj" in result_data:
                try:
                    lime_exp = result_data.get("explanation_obj")
                    if lime_exp is not None and hasattr(lime_exp, "as_list"):
                        lime_list = lime_exp.as_list()
                        if lime_list:  # Check if lime_list is not empty
                            formatted_prompt += "\n\nLIME explanation weights:\n"
                            formatted_prompt += "| Feature | Weight |\n"
                            formatted_prompt += "|---------|--------|\n"
                            for feat, weight in lime_list:
                                formatted_prompt += f"| {feat} | {weight:.4f} |\n"
                        else:
                            formatted_prompt += "\n\nLIME explanation available but contains no feature weights."
                    else:
                        formatted_prompt += "\n\nLIME explanation object is not available or doesn't have the expected structure."
                except Exception as e:
                    logger.warning(f"Could not add LIME DataFrame: {e}")
                    formatted_prompt += "\n\nCould not format LIME explanation data due to an error."

            # Create appropriate system prompt based on action type
            system_prompt = self._get_system_prompt(action_type)
            
            # Generate explanation with increased max_tokens
            logger.info("Sending prompt to LLM")
            explanation = self.llm.generate_text(
                prompt=formatted_prompt,
                system_prompt=system_prompt,
                temperature=0.7,
                max_tokens=2000  # Further increased token limit for richer explanations
            )
            logger.info(f"Generated explanation (length: {len(explanation)} chars)")
            
            # Check if explanation appears too technical or contains too much raw data
            contains_table_markers = "| --" in explanation or "+---" in explanation
            contains_json_like_structure = "{" in explanation and "}" in explanation and ":" in explanation
            contains_raw_data_patterns = any(pattern in explanation for pattern in ["array(", "dtype=", "[0."])
            
            # If explanation seems too technical, request a simplified version
            if contains_table_markers or contains_json_like_structure or contains_raw_data_patterns:
                logger.info("Detected technical explanation, requesting a more human-friendly version")
                simplify_prompt = (
                    "The above explanation contains too much technical detail and raw data. "
                    "Please rewrite it in plain English with a focus on insights rather than numbers. "
                    "Provide a conversational explanation that a non-technical person would understand. "
                    "Avoid tables, JSON, arrays, and technical formatting. Focus on what the results mean in practice."
                )
                
                # Generate a more human-friendly explanation
                simplified_explanation = self.llm.generate_text(
                    prompt=explanation + "\n\n" + simplify_prompt,
                    system_prompt="You are an expert at translating technical explanations into simple language anyone can understand.",
                    temperature=0.7,
                    max_tokens=1500
                )
                
                # Use the simplified explanation
                explanation = simplified_explanation

            # Store the user query and LLM result in the vector store for future retrieval
            if user_query:
                try:
                    doc = {
                        "content": f"User Query: {user_query}\n\nLLM Explanation:\n{explanation}",
                        "metadata": {
                            "type": "qa_pair",
                            "action_type": action_type,
                            "model_type": model_type,
                            "project_name": project_info.get("name") if project_info else None
                        }
                    }
                    documents.append(doc)
                    logger.info("Stored Q&A pair in vector store.")
                except Exception as e:
                    logger.warning(f"Could not store Q&A in vector store: {e}")

            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            return f"I couldn't generate a detailed explanation due to a technical issue: {str(e)}. The model shows that certain features are more important than others in making predictions, but I can't provide specifics at this moment."
        
    def _format_results_for_prompt(self, result_data: Any, action_type: str, model_type: str = None, model=None, project_info=None) -> str:
        """Format technical results, model info, and project info into an LLM-friendly prompt"""
        logger.info(f"Formatting {action_type} results for LLM")
        prompt_parts = []

        try:
            # Add project info if available
            if project_info:
                prompt_parts.append(f"Project: {project_info.get('name', 'Unknown')}")
                prompt_parts.append(f"Description: {project_info.get('description', '')}")

            # Add model info if available
            if model is not None:
                prompt_parts.append(f"Model type: {type(model).__name__}")
                if hasattr(model, "get_params"):
                    # Convert params to JSON-safe format
                    params = self._clean_for_json(model.get_params())
                    try:
                        prompt_parts.append(f"Model parameters: {json.dumps(params, indent=2)}")
                    except TypeError as e:
                        logger.warning(f"Could not serialize model parameters: {e}")
                        prompt_parts.append(f"Model parameters: [Could not serialize due to {e}]")
            
            # Try to get feature names if possible
            if hasattr(model, "feature_names_in_"):
                prompt_parts.append(f"Feature names: {list(model.feature_names_in_)}")
            
            # Add feature importance from the model if available
            try:
                if hasattr(model, "feature_importances_"):
                    # For tree-based models like RandomForest, XGBoost, etc.
                    feature_names = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else [f"feature_{i}" for i in range(len(model.feature_importances_))]
                    model_feature_imp = sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)
                    prompt_parts.append("\nModel's built-in feature importance:")
                    for i, (feat, imp) in enumerate(model_feature_imp[:10], 1):
                        prompt_parts.append(f"{i}. {feat}: {imp:.4f}")
                elif hasattr(model, "coef_"):
                    # For linear models like LinearRegression, LogisticRegression
                    feature_names = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else [f"feature_{i}" for i in range(len(model.coef_[0] if model.coef_.ndim > 1 else model.coef_))]
                    coef = model.coef_[0] if model.coef_.ndim > 1 else model.coef_
                    model_feature_imp = sorted(zip(feature_names, map(abs, coef)), key=lambda x: x[1], reverse=True)
                    prompt_parts.append("\nModel's coefficients (absolute values for importance):")
                    for i, (feat, imp) in enumerate(model_feature_imp[:10], 1):
                        prompt_parts.append(f"{i}. {feat}: {imp:.4f}")
            except Exception as e:
                logger.warning(f"Could not extract model's feature importance: {e}")
        
            # Add feature importance if available in result_data
            if isinstance(result_data, dict):
                # Extract feature importance information from various sources in result_data
                if "feature_importance" in result_data:
                    fi = result_data["feature_importance"]
                    sorted_fi = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)
                    prompt_parts.append("\nFeature importance from analysis (sorted by absolute value):")
                    for i, (feat, val) in enumerate(sorted_fi[:15], 1):
                        direction = "positive" if val > 0 else "negative"
                        prompt_parts.append(f"{i}. {feat}: {val:.4f} ({direction} impact)")
                        
                # Add sorted importance if available (for feature_importance action)
                elif "sorted_importance" in result_data:
                    sorted_imp = result_data["sorted_importance"]
                    prompt_parts.append("\nFeature importance (sorted):")
                    for i, (feat, val) in enumerate(sorted_imp[:15], 1):
                        prompt_parts.append(f"{i}. {feat}: {val:.4f}")

                # Handle all other dictionary data with safe serialization
                # Add data statistics if available for better context
                if "data_stats" in result_data:
                    try:
                        stats = self._clean_for_json(result_data["data_stats"])
                        prompt_parts.append("\nData statistics:")
                        prompt_parts.append(json.dumps(stats, indent=2))
                    except TypeError as e:
                        logger.warning(f"Could not serialize data statistics: {e}")
                        prompt_parts.append("\nData statistics: [Could not serialize]")

                # Add model metadata if available
                if "model_metadata" in result_data:
                    try:
                        meta = self._clean_for_json(result_data["model_metadata"])
                        prompt_parts.append("\nModel metadata:")
                        prompt_parts.append(json.dumps(meta, indent=2))
                    except TypeError as e:
                        logger.warning(f"Could not serialize model metadata: {e}")
                        prompt_parts.append("\nModel metadata: [Could not serialize]")

                # Add sample data if available (for instance explanations)
                if "sample_data" in result_data:
                    try:
                        sample = self._clean_for_json(result_data["sample_data"])
                        prompt_parts.append("\nSample data being explained:")
                        prompt_parts.append(json.dumps(sample, indent=2))
                    except TypeError as e:
                        logger.warning(f"Could not serialize sample data: {e}")
                        prompt_parts.append("\nSample data: [Could not serialize]")

                # Add feature statistics if available
                if "feature_stats" in result_data:
                    try:
                        stats = self._clean_for_json(result_data["feature_stats"])
                        prompt_parts.append("\nFeature statistics:")
                        prompt_parts.append(json.dumps(stats, indent=2))
                    except TypeError as e:
                        logger.warning(f"Could not serialize feature statistics: {e}")
                        prompt_parts.append("\nFeature statistics: [Could not serialize]")
                
                # Add column statistics if available
                if "column_stats" in result_data:
                    try:
                        stats = self._clean_for_json(result_data["column_stats"])
                        prompt_parts.append("\nColumn statistics:")
                        prompt_parts.append(json.dumps(stats, indent=2))
                    except TypeError as e:
                        logger.warning(f"Could not serialize column statistics: {e}")
                        prompt_parts.append("\nColumn statistics: [Could not serialize]")

            elif isinstance(result_data, list):
                # Handle list-type results (for general/feature_importance actions)
                prompt_parts.append("\nList of results:")
                try:
                    # Try to handle the case where the list contains (feature, importance) tuples
                    if result_data and isinstance(result_data[0], tuple) and len(result_data[0]) == 2:
                        for i, (feat, imp) in enumerate(result_data[:15], 1):
                            prompt_parts.append(f"{i}. {feat}: {imp:.4f}")
                    else:
                        for i, item in enumerate(result_data[:15], 1):
                            prompt_parts.append(f"{i}. {str(item)}")
                except Exception as e:
                    logger.warning(f"Could not format list result data: {e}")
                    prompt_parts.append(str(result_data))

            # Add instructions for LLM to provide human-friendly explanation
            prompt_parts.append("\n\nINSTRUCTIONS:")
            prompt_parts.append("1. Provide a clear, concise explanation in plain English")
            prompt_parts.append("2. Start with a brief summary of the most important insights")
            prompt_parts.append("3. Explain the practical meaning of these results for someone who doesn't understand ML")
            prompt_parts.append("4. DO NOT simply repeat the data values back to the user")
            prompt_parts.append("5. Focus on insights rather than technical details")
            
            if action_type.lower() == "shap":
                prompt_parts.append("6. Explain which features most influence the prediction and how they do so")
                prompt_parts.append("7. Clarify whether high/low values of important features increase or decrease the prediction")
            elif action_type.lower() == "lime":
                prompt_parts.append("6. Explain which features drove this specific prediction and why")
                prompt_parts.append("7. Help user understand what changes to these features might change the prediction")
            elif action_type.lower() == "feature_importance":
                prompt_parts.append("6. Explain what the important features tell us about this model and dataset")
                prompt_parts.append("7. Provide context for why these features might matter in this domain")

            # Add the main result data based on action type
            if action_type.lower() == "shap":
                prompt_parts.append(self._format_shap_results(result_data, model_type))
            elif action_type.lower() == "lime":
                prompt_parts.append(self._format_lime_results(result_data, model_type))
            elif action_type.lower() == "feature_importance":
                prompt_parts.append(self._format_feature_importance(result_data, model_type))
            elif action_type.lower() == "prediction":
                prompt_parts.append(self._format_prediction_results(result_data, model_type))
            elif action_type.lower() == "general":
                # Add specific handling for general action type
                prompt_parts.append(self._format_general_results(result_data, model_type))
            else:
                logger.info(f"Using default formatting for action type: {action_type}")
                if isinstance(result_data, dict):
                    # Convert all numpy arrays and other non-serializable objects to strings
                    try:
                        clean_data = self._clean_for_json(result_data)
                        prompt_parts.append(f"Model results ({model_type or 'unknown model'}):\n{json.dumps(clean_data, indent=2)}")
                    except TypeError as e:
                        logger.warning(f"Could not serialize result data: {e}")
                        prompt_parts.append(f"Model results ({model_type or 'unknown model'}): [Could not serialize due to {e}]")
                elif isinstance(result_data, list):
                    try:
                        clean_data = self._clean_for_json(result_data)
                        prompt_parts.append(f"Model results ({model_type or 'unknown model'}):\n{json.dumps(clean_data, indent=2)}")
                    except TypeError as e:
                        logger.warning(f"Could not serialize list result data: {e}")
                        prompt_parts.append(f"Model results ({model_type or 'unknown model'}): {str(result_data)}")
                else:
                    prompt_parts.append(f"Model results ({model_type or 'unknown model'}):\n{str(result_data)}")
        
        except Exception as e:
            logger.error(f"Error formatting results for prompt: {str(e)}")
            # Return a fallback prompt with minimal information
            return f"Please explain the following results for a {model_type or 'machine learning'} model:\n\n" + \
                   f"Action type: {action_type}\n" + \
                   f"Model type: {type(model).__name__ if model else 'Unknown'}\n" + \
                   "Note: Some data could not be included due to formatting errors."

    def _format_general_results(self, general_data: Any, model_type: str = None) -> str:
        """Format general model information results for the LLM prompt"""
        prompt = f"General Information for {model_type or 'ML model'}:\n\n"
        # Handle different data types
        if isinstance(general_data, dict):
            for key, value in general_data.items():
                prompt += f"{key}: {value}\n"
        elif isinstance(general_data, list):
            if general_data and isinstance(general_data[0], tuple) and len(general_data[0]) == 2:
                prompt += "Top features by importance:\n"
                for i, (feature, importance) in enumerate(general_data[:10], 1):
                    prompt += f"{i}. {feature}: {importance:.4f}\n"
            else:
                prompt += "Information points:\n"
                for i, item in enumerate(general_data, 1):
                    prompt += f"{i}. {item}\n"
        else:
            prompt += f"{str(general_data)}\n"
        return prompt

    def _format_shap_results(self, shap_data: Dict[str, Any], model_type: str = None) -> str:
        """Format SHAP results for the LLM prompt"""
        prompt = f"SHAP Explanation for {model_type or 'ML model'}:\n\n"
        if isinstance(shap_data, dict) and "feature_importance" in shap_data and "feature_names" in shap_data:
            importance = shap_data["feature_importance"]
            features = shap_data["feature_names"]
            sorted_features = sorted(zip(features, importance), key=lambda x: abs(x[1]), reverse=True)
            prompt += "Top features by importance:\n"
            for i, (feature, imp) in enumerate(sorted_features[:10], 1):
                direction = "positive" if imp > 0 else "negative"
                prompt += f"{i}. {feature}: {imp:.4f} ({direction} impact)\n"
            
            # Include feature values if this is for a specific instance
            if "feature_values" in shap_data:
                fv = shap_data["feature_values"]
                prompt += "\nFeature values for this instance:\n"
                for i, (feature, imp) in enumerate(sorted_features[:10], 1):
                    if feature in fv:
                        prompt += f"{i}. {feature}: {fv[feature]}\n"
            
            if "base_value" in shap_data:
                prompt += f"\nBase value: {shap_data['base_value']:.4f}\n"
            if "sample_index" in shap_data:
                prompt += f"\nThis explanation is for sample #{shap_data['sample_index']}\n"
        else:
            prompt += f"{str(shap_data)}\n"
        return prompt
        
    def _format_lime_results(self, lime_data: Dict[str, Any], model_type: str = None) -> str:
        """Format LIME results for the LLM prompt"""
        prompt = f"LIME Explanation for {model_type or 'ML model'}:\n\n"
        
        if not lime_data:
            return prompt + "No LIME explanation data available."
            
        if isinstance(lime_data, dict):
            # Safely extract explanation
            if "explanation" in lime_data:
                explanation_text = lime_data.get("explanation")
                if explanation_text:
                    prompt += f"Explanation: {explanation_text}\n\n"
                    
            # Safely extract prediction info
            if "predicted_class" in lime_data:
                prompt += f"Predicted class: {lime_data.get('predicted_class')}\n"
                
            if "predicted_prob" in lime_data:
                prob = lime_data.get("predicted_prob")
                if prob is not None:
                    try:
                        prompt += f"Prediction probability: {float(prob):.4f}\n"
                    except (TypeError, ValueError):
                        prompt += f"Prediction probability: {prob}\n"
            
            # Check for explanation object and format weights
            if "explanation_obj" in lime_data and lime_data["explanation_obj"] is not None:
                try:
                    lime_exp = lime_data["explanation_obj"]
                    if hasattr(lime_exp, "as_list") and callable(getattr(lime_exp, "as_list")):
                        lime_list = lime_exp.as_list()
                        if lime_list:  # Check if list is not empty
                            prompt += "\nFeature weights:\n"
                            for feat, weight in lime_list:
                                prompt += f"- {feat}: {weight:.4f}\n"
                except Exception as e:
                    logger.warning(f"Error formatting LIME explanation object: {e}")
                    
        else:
            prompt += f"{str(lime_data)}\n"
            
        return prompt
            
    def _format_feature_importance(self, importance_data: Any, model_type: str = None) -> str:
        """Format feature importance results for the LLM prompt"""
        prompt = f"Feature Importance for {model_type or 'ML model'}:\n\n"
        if isinstance(importance_data, dict) and "sorted_importance" in importance_data:
            sorted_imp = importance_data["sorted_importance"]
            for i, (feature, imp) in enumerate(sorted_imp[:15], 1):  # Top 15
                prompt += f"{i}. {feature}: {imp:.4f}\n"
        elif isinstance(importance_data, list):
            # Handle list of tuples case directly
            try:
                if importance_data and isinstance(importance_data[0], tuple) and len(importance_data[0]) == 2:
                    for i, (feature, imp) in enumerate(importance_data[:15], 1):
                        prompt += f"{i}. {feature}: {float(imp):.4f}\n"
                else:
                    for i, item in enumerate(importance_data[:15], 1):
                        prompt += f"{i}. {str(item)}\n"
            except Exception as e:
                logger.warning(f"Could not format importance list: {e}")
                prompt += f"{str(importance_data)}\n"
        else:
            prompt += f"{str(importance_data)}\n"
        return prompt
            
    def _format_prediction_results(self, prediction_data: Any, model_type: str = None) -> str:
        """Format prediction results for the LLM prompt"""
        prompt = f"Prediction Results for {model_type or 'ML model'}:\n\n"
        if isinstance(prediction_data, dict) and "prediction" in prediction_data:
            prompt += f"Prediction: {prediction_data['prediction']}\n"
            if "probability" in prediction_data:
                prompt += f"Probability: {prediction_data['probability']:.4f}\n"
            if "input_features" in prediction_data:
                prompt += "\nInput features used for prediction:\n"
                for feature, value in prediction_data["input_features"].items():
                    prompt += f"- {feature}: {value}\n"
        else:
            prompt += f"{str(prediction_data)}\n"
        return prompt
        
    def _get_system_prompt(self, action_type: str) -> str:
        """Get appropriate system prompt based on action type"""
        base_prompt = (
            "You are an expert at explaining machine learning concepts in simple terms. "
            "Explain the following ML analysis results in natural language that anyone can understand. "
            "Be concise but thorough. Focus on what the results mean for the user. "
            "Provide your explanation in clear, conversational English with a brief summary first, "
            "followed by the key insights. DO NOT repeat the raw data back to the user - instead, "
            "interpret the data and explain what it means in real-world terms."
        )
        if action_type.lower() == "shap":
            return base_prompt + (
                "\n\nSHAP values show how much each feature contributes to pushing the prediction away from the "
                "baseline (average) prediction. Positive values push the prediction higher, negative values push it lower. "
                "Explain which features are most important, how they influence the model's decision, and what this means "
                "for the user in practical terms. Use analogies if helpful."
            )
        elif action_type.lower() == "lime":
            return base_prompt + (
                "\n\nLIME explanations show which features were most important for a specific prediction. "
                "It works by creating a simple local model around the prediction. "
                "Focus on explaining which features drove this specific prediction and why. Help the user understand "
                "what changes to these features might change the prediction. Avoid technical jargon."
            )
        elif action_type.lower() == "feature_importance":
            return base_prompt + (
                "\n\nFeature importance shows which features have the biggest overall impact on the model's predictions. "
                "Higher values mean more important features. Explain what the top features mean in the context of this model, "
                "why they might be important, and what this tells us about the underlying patterns in the data. "
                "Provide practical insights rather than just listing features."
            )
        elif action_type.lower() == "prediction":
            return base_prompt + (
                "\n\nExplain the prediction result and what it means in practical terms. "
                "Describe which features contributed most to this prediction and how they influenced it. "
                "Help the user understand why the model made this specific prediction and what it means "
                "in the real world. If there's a probability score, explain its significance."
            )
        else:
            return base_prompt + (
                "\n\nProvide a clear summary of the most important aspects of these results. "
                "Focus on insights that would be valuable to someone who doesn't understand machine learning. "
                "Use plain language and, if possible, relate the findings to real-world implications."
            )
    
    def _clean_for_json(self, data):
        """Convert numpy arrays and other non-JSON-serializable objects to Python types"""
        if data is None:
            return None
        elif isinstance(data, (str, int, float)):
            return data
        elif isinstance(data, bool):
            return data  # Explicit handling for boolean values
        elif isinstance(data, dict):
            return {str(k): self._clean_for_json(v) for k, v in data.items()}
        elif isinstance(data, list) or isinstance(data, tuple):
            return [self._clean_for_json(v) for v in data]
        elif isinstance(data, np.ndarray):
            return self._clean_for_json(data.tolist())
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, pd.DataFrame):
            return self._clean_for_json(data.to_dict(orient='records'))
        elif isinstance(data, pd.Series):
            return self._clean_for_json(data.to_dict())
        elif hasattr(data, '__dict__'):
            return str(data)
        else:
            return str(data)  # Convert any other type to string as a fallback
