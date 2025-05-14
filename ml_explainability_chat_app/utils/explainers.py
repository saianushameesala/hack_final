import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from lime.lime_tabular import LimeTabularExplainer
import shap  # Move shap import to top level
from sklearn.pipeline import Pipeline
import streamlit as st
import os
import logging

logger = logging.getLogger("explainers")


def get_shap_explanation(model, data, instance_index=None, num_features=10):
    """
    Generate SHAP explanation for a model and dataset
    
    Args:
        model: Trained model (sklearn compatible)
        data: DataFrame containing features
        instance_index: Index of instance to explain (None for global explanation)
        num_features: Number of top features to include in explanation
    
    Returns:
        Dict with explanation text and plot image (base64 encoded)
    """
    # Handle pipeline models by extracting the final estimator
    if isinstance(model, Pipeline):
        # Process data through all transformers in the pipeline
        for name, transform in model.steps[:-1]:
            data = transform.transform(data)
        # Get the final estimator
        estimator = model.steps[-1][1]
    else:
        estimator = model
    
    try:
        # Initialize SHAP explainer based on model type
        model_type = type(estimator).__name__
        logger.info(f"Initializing SHAP explainer for model type: {model_type}")
        
        if "XGB" in model_type:
            # Special handling for XGBoost models to avoid dimension errors
            try:
                # Check if feature names in model match dataframe columns
                if hasattr(estimator, 'feature_names_in_'):
                    model_features = list(estimator.feature_names_in_)
                    data_features = list(data.columns)
                    
                    # Log feature information for debugging
                    logger.info(f"Model features: {len(model_features)}, Data features: {len(data_features)}")
                    
                    # If feature sets don't match, realign the data
                    if set(model_features) != set(data_features) or len(model_features) != len(data_features):
                        logger.warning("Feature mismatch detected between model and data. Realigning features.")
                        # Ensure data only contains features the model was trained on and in the right order
                        missing_features = set(model_features) - set(data_features)
                        if missing_features:
                            logger.error(f"Data is missing features that model requires: {missing_features}")
                            return {
                                "text": f"Error: Data is missing features that the model requires: {missing_features}",
                                "image": None,
                                "type": "error"
                            }
                        
                        # Reindex data to match model feature order
                        data = data[model_features]
                
                # Create XGBoost-specific safe explainer
                # Note: No need to import shap here as it's now imported at the top level
                
                # Use a safer approach with XGBoost models
                explainer = shap.TreeExplainer(estimator, data=data.iloc[:50])  # Use sample of training data as background
                
                # For XGBoost, get SHAP values with explicit feature names
                if instance_index is not None:
                    # For single instance explanation
                    shap_values = explainer(data.iloc[[instance_index]])
                else:
                    # For global explanation, use a sample for efficiency
                    sample_size = min(100, len(data))
                    sample_indices = np.random.choice(len(data), sample_size, replace=False)
                    data_sample = data.iloc[sample_indices]
                    shap_values = explainer(data_sample)
                
                # Convert the SHAP values to our simplified format for consistency
                shap_values_obj = shap_values
                
            except Exception as e:
                logger.error(f"XGBoost-specific SHAP error: {e}")
                
                # Fallback to model-agnostic explainer if TreeExplainer fails
                logger.info("Falling back to model-agnostic Explainer for XGBoost")
                explainer = shap.Explainer(estimator.predict, data)
                shap_values_obj = explainer(data)
                
        elif "RandomForest" in model_type or "GradientBoosting" in model_type:
            # Tree-based models can use the faster TreeExplainer
            explainer = shap.TreeExplainer(estimator)
            # For tree models, directly calculate SHAP values
            shap_values = explainer.shap_values(data)
            
            # Handle multi-output case (like RandomForestClassifier with >2 classes)
            if isinstance(shap_values, list):
                # For classifiers, we'll use the values for class 1 (or sum all classes)
                if len(shap_values) == 2:  # Binary classifier
                    shap_values = shap_values[1]  # Use class 1 (positive class)
                    expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
                else:  # Multi-class
                    # Sum SHAP values across all classes or pick a specific class
                    shap_values = np.sum(np.array(shap_values), axis=0)
                    expected_value = np.sum(explainer.expected_value) if isinstance(explainer.expected_value, list) else explainer.expected_value
            else:
                expected_value = explainer.expected_value
                
            # Create a wrapper object similar to Explainer output for consistency
            class SimpleShapValues:
                def __init__(self, values, base_value, data, feature_names):
                    self.values = values
                    self.base_values = base_value
                    self.data = data
                    self.feature_names = feature_names
                
                def __getitem__(self, idx):
                    return SimpleShapValues(
                        self.values[idx], 
                        self.base_values, 
                        self.data.iloc[[idx]] if isinstance(self.data, pd.DataFrame) else self.data[idx],
                        self.feature_names
                    )
            
            # Create a SHAP values wrapper for consistent interface
            shap_values_obj = SimpleShapValues(
                shap_values, 
                expected_value, 
                data, 
                data.columns.tolist()
            )
            
        else:
            # For other models, use the generic Explainer
            explainer = shap.Explainer(estimator)
            shap_values_obj = explainer(data)
        
        if instance_index is not None:
            # Convert string index to integer if needed
            if isinstance(instance_index, str) and instance_index.isdigit():
                instance_index = int(instance_index)
                
            # Ensure index is within bounds
            if instance_index >= 0 and instance_index < len(data):
                # Local explanation for a specific instance
                feature_names = data.columns.tolist()
                feature_values = data.iloc[instance_index].tolist()

                # Get SHAP values for this instance
                if hasattr(shap_values_obj, "values"):
                    instance_shap_values = shap_values_obj.values[instance_index]
                    base_value = shap_values_obj.base_values
                else:
                    instance_shap_values = shap_values_obj[instance_index].values
                    base_value = float(shap_values_obj[instance_index].base_values)

                # Ensure instance_shap_values is 1D
                instance_shap_values = np.array(instance_shap_values).flatten()

                # Create waterfall plot for specific instance
                plt.figure(figsize=(10, 6))
                try:
                    # For tree-based models, create a custom waterfall plot
                    if "RandomForest" in model_type or "GradientBoosting" in model_type or "XGB" in model_type:
                        # Fix: Use np.isscalar to check for scalar, else use x.item() for single-element arrays
                        colors = []
                        for x in instance_shap_values[:num_features]:
                            val = x.item() if hasattr(x, "item") and np.size(x) == 1 else x
                            colors.append('red' if val > 0 else 'blue')
                        plt.barh(
                            range(len(feature_names[:num_features])),
                            [abs(float(x)) for x in instance_shap_values[:num_features]],
                            color=colors
                        )
                        plt.yticks(range(len(feature_names[:num_features])), feature_names[:num_features])
                        plt.title(f"Top {num_features} features for instance #{instance_index}")
                        plt.xlabel("SHAP value magnitude")
                    else:
                        shap.waterfall_plot(shap_values_obj[instance_index], max_display=num_features, show=False)
                except Exception as e:
                    logger.error(f"Error creating waterfall plot: {e}")
                waterfall_img = _get_plot_image()

                # Create force plot for specific instance
                plt.figure(figsize=(10, 3))
                try:
                    # For tree-based models
                    if "RandomForest" in model_type or "GradientBoosting" in model_type or "XGB" in model_type:
                        # Using matplotlib for force plot - handle different versions of shap
                        try:
                            # New SHAP version requires base_value first
                            if isinstance(base_value, np.ndarray):
                                base_val = base_value[0] if len(base_value) > 0 else 0.5
                            else:
                                base_val = base_value
                                
                            shap.force_plot(
                                base_value=base_val,
                                shap_values=instance_shap_values,
                                features=feature_values,
                                feature_names=feature_names,
                                matplotlib=True,
                                show=False
                            )
                        except Exception as e1:
                            logger.warning(f"Force plot with new API failed: {e1}, trying older API")
                            # Try with older SHAP version API
                            shap.force_plot(
                                instance_shap_values,
                                features=feature_values,
                                feature_names=feature_names,
                                matplotlib=True,
                                show=False
                            )
                    else:
                        # Standard force plot
                        shap.plots.force(shap_values_obj[instance_index], matplotlib=True, show=False)
                    force_img = _get_plot_image()
                except Exception as e:
                    logger.error(f"Error creating force plot: {e}")
                    force_img = None
                
                # Sort features by importance for this instance
                feature_importance = {}
                for i, name in enumerate(feature_names):
                    feature_importance[name] = float(instance_shap_values[i])
                
                # Fix: Convert base_value to scalar properly
                if isinstance(base_value, np.ndarray):
                    base_val_scalar = float(base_value[0]) if len(base_value) > 0 else 0.5
                else:
                    base_val_scalar = float(base_value)
                
                # Include instance-specific data
                result = {
                    "text": f"SHAP explanation for instance #{instance_index}",
                    "image": waterfall_img,
                    "force_image": force_img,
                    "type": "shap_instance",
                    "values": instance_shap_values,
                    "feature_names": feature_names,
                    "feature_importance": feature_importance,
                    "feature_values": dict(zip(feature_names, feature_values)),
                    "base_value": base_val_scalar,
                    "sample_index": instance_index
                }

            else:
                return {
                    "text": f"Error: Instance index {instance_index} is out of bounds (data has {len(data)} instances)",
                    "image": None,
                    "type": "error"
                }
        else:
            # Global explanation for Random Forest
            feature_names = data.columns.tolist()
            plt.figure(figsize=(10, 6))

            if "RandomForest" in model_type or "GradientBoosting" in model_type or "XGB" in model_type:
                # Handle different SHAP output formats
                
                # Check if shap_values is an Explanation object (newer SHAP versions)
                if hasattr(shap_values_obj, "values") and hasattr(shap_values_obj, "data"):
                    logger.info("Handling SHAP Explanation object format")
                    # Extract values from the Explanation object
                    if hasattr(shap_values_obj, "base_values"):
                        base_value = shap_values_obj.base_values
                    if hasattr(shap_values_obj, "values"):
                        shap_values = shap_values_obj.values
                        
                # For tree models, use the mean absolute SHAP value for each feature
                if isinstance(shap_values, list):  # Multi-class case
                    # Each element in shap_values is (n_samples, n_features)
                    mean_abs_shap = np.mean([np.abs(np.array(sv)) for sv in shap_values], axis=0)
                    mean_abs_shap = np.mean(mean_abs_shap, axis=0)
                elif hasattr(shap_values, "shape") and len(shap_values.shape) > 1:
                    # Regular 2D array case (samples, features)
                    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                elif hasattr(shap_values_obj, "values") and isinstance(shap_values_obj.values, np.ndarray):
                    # Handle case where shap_values_obj is the Explanation object
                    values_array = shap_values_obj.values
                    if len(values_array.shape) > 1:
                        mean_abs_shap = np.mean(np.abs(values_array), axis=0)
                    else:
                        mean_abs_shap = np.abs(values_array)  # Single instance case
                else:
                    # Not a recognized format, try to convert to numpy array as a last resort
                    logger.warning(f"Unknown SHAP values format: {type(shap_values)}, attempting conversion")
                    try:
                        mean_abs_shap = np.mean(np.abs(np.array(shap_values)), axis=0)
                    except Exception as e:
                        logger.error(f"Could not calculate mean abs SHAP: {e}")
                        mean_abs_shap = np.zeros(len(feature_names))

                # Fix: ensure mean_abs_shap is properly shaped and handle indices
                mean_abs_shap = np.array(mean_abs_shap).flatten()
                if len(mean_abs_shap) > len(feature_names):
                    mean_abs_shap = mean_abs_shap[:len(feature_names)]
                
                # Get valid number of features
                valid_num_features = min(num_features, len(mean_abs_shap), len(feature_names))
                
                # Get indices of top features (sort by absolute value)
                idx_sorted = np.argsort(np.abs(mean_abs_shap))
                if len(idx_sorted) > valid_num_features:
                    indices = idx_sorted[-valid_num_features:]
                else:
                    indices = idx_sorted
                
                # Create bar chart of feature importance
                plt.barh(range(len(indices)), [mean_abs_shap[i] for i in indices])
                plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                plt.title(f"Global Feature Importance (SHAP)")
                plt.xlabel("mean(|SHAP value|)")
            else:
                shap.summary_plot(shap_values_obj, data, plot_type="bar", max_display=num_features, show=False)

            summary_image = _get_plot_image()

            # Create a waterfall plot for the first instance as an example
            plt.figure(figsize=(10, 6))
            if "RandomForest" in model_type or "GradientBoosting" in model_type or "XGB" in model_type:
                # Fix: Handle first instance properly for different types of SHAP values
                if isinstance(shap_values, list) and len(shap_values) > 1:
                    if len(shap_values[1]) > 0:
                        first_instance = np.array(shap_values[1][0]).flatten() 
                    else:
                        first_instance = np.array(shap_values[0][0]).flatten()
                else:
                    first_instance = np.array(shap_values[0]).flatten()
                
                # Limit to feature_names length
                if len(first_instance) > len(feature_names):
                    first_instance = first_instance[:len(feature_names)]
                
                # Get valid number of features for display
                valid_num_features = min(num_features, len(first_instance), len(feature_names))
                sorted_indices = np.argsort(np.abs(first_instance))
                
                if len(sorted_indices) > valid_num_features:
                    sorted_indices = sorted_indices[-valid_num_features:]
                
                plt.barh(
                    range(len(sorted_indices)),
                    [first_instance[i] for i in sorted_indices],
                    color=['red' if first_instance[i] > 0 else 'blue' for i in sorted_indices]
                )
                plt.yticks(range(len(sorted_indices)), [feature_names[i] for i in sorted_indices])
                plt.title(f"Example: SHAP values for first instance")
                plt.xlabel("SHAP value")
            else:
                shap.waterfall_plot(shap_values_obj[0], max_display=num_features, show=False)

            waterfall_image = _get_plot_image()

            # Get global feature importance, safely handling array dimensions
            feature_importance = {}
            if "RandomForest" in model_type or "GradientBoosting" in model_type or "XGB" in model_type:
                # Handle different SHAP output formats including Explanation objects
                if isinstance(shap_values, list):
                    mean_abs_shap = np.mean([np.abs(np.array(sv)) for sv in shap_values], axis=0)
                    mean_abs_shap = np.mean(mean_abs_shap, axis=0)
                elif hasattr(shap_values_obj, "values") and isinstance(shap_values_obj.values, np.ndarray):
                    values_array = shap_values_obj.values
                    if len(values_array.shape) > 1:
                        mean_abs_shap = np.mean(np.abs(values_array), axis=0)
                    else:
                        mean_abs_shap = np.abs(values_array)  # Single instance case
                elif hasattr(shap_values, "shape") and len(shap_values.shape) > 1:
                    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
                else:
                    # Last resort, try to convert
                    try:
                        mean_abs_shap = np.mean(np.abs(np.array(shap_values)), axis=0)
                    except Exception as e:
                        logger.error(f"Could not calculate mean abs SHAP for feature importance: {e}")
                        mean_abs_shap = np.zeros(len(feature_names))
                
                mean_abs_shap = np.array(mean_abs_shap).flatten()
                
                # Ensure mean_abs_shap and feature_names align
                n_features = min(len(mean_abs_shap), len(feature_names))
                for i in range(n_features):
                    feature_importance[feature_names[i]] = float(mean_abs_shap[i])
            else:
                # For newer SHAP Explanation objects
                if hasattr(shap_values_obj, "values") and isinstance(shap_values_obj.values, np.ndarray):
                    mean_abs_shap = np.abs(shap_values_obj.values).mean(axis=0)
                else:
                    # For older API
                    mean_abs_shap = np.abs(shap_values_obj.values).mean(axis=0)
                
                for i, name in enumerate(feature_names):
                    if i < len(mean_abs_shap):
                        feature_importance[name] = float(mean_abs_shap[i])

            result = {
                "text": "Global SHAP feature importance",
                "image": summary_image,
                "waterfall_image": waterfall_image,
                "type": "shap_global",
                "values": shap_values,
                "feature_names": feature_names,
                "feature_importance": feature_importance
            }

        return result

    except Exception as e:
        import traceback
        logger.error(f"SHAP explanation error: {e}\n{traceback.format_exc()}")
        return {
            "text": f"Error generating SHAP explanation: {str(e)}",
            "image": None,
            "type": "error"
        }

def _get_plot_image():
    """Helper function to convert matplotlib plot to base64 image"""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    return img_str

def get_lime_explanation(model, data, instance_index, class_names=None, num_features=10):
    """
    Generate LIME explanation for a specific instance
    
    Args:
        model: Trained model (sklearn compatible)
        data: DataFrame containing features
        instance_index: Index of instance to explain
        class_names: List of class names for classification
        num_features: Number of top features to include
        
    Returns:
        Dict with explanation text and plot image (base64 encoded)
    """
    try:
        # Create LIME explainer
        explainer = LimeTabularExplainer(
            data.values,
            feature_names=data.columns.tolist(),
            class_names=class_names,
            mode='classification' if class_names else 'regression'
        )
        
        # Get the instance to explain
        instance = data.iloc[instance_index].values
        
        # Generate explanation
        if class_names:
            explanation = explainer.explain_instance(
                instance, 
                model.predict_proba, 
                num_features=num_features
            )
        else:
            explanation = explainer.explain_instance(
                instance, 
                model.predict, 
                num_features=num_features
            )
        
        # Create plot
        plt.figure(figsize=(10, 6))
        explanation.as_pyplot_figure()
        
        # Convert plot to base64 image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode()
        plt.close()
        
        # Get text explanation
        explanation_text = f"LIME explanation for instance #{instance_index}"
        
        return {
            "text": explanation_text,
            "image": img_str,
            "type": "lime",
            "explanation_obj": explanation
        }
        
    except Exception as e:
        return {
            "text": f"Error generating LIME explanation: {str(e)}",
            "image": None,
            "type": "error",
            "explanation_obj": None
        }

def display_explanation(explanation):
    """Display an explanation in Streamlit"""
    st.write(explanation["text"])
    
    if explanation["image"]:
        st.image(f"data:image/png;base64,{explanation['image']}")
    
    if explanation["type"] == "error":
        st.error(explanation["text"])
