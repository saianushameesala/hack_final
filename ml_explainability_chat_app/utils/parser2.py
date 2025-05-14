"""
Parser module for the ML Explainability Chat App.
Parses user input to determine the type of ML explanation needed.
"""
import re
import logging

logger = logging.getLogger("parser")

def parse_user_input(input_text: str) -> str:
    """
    Parse user input to determine the action to execute
    
    Args:
        input_text: User input text
        
    Returns:
        Action identifier: 'shap', 'lime', 'feature_importance', 'prediction', or 'general'
    """
    input_lower = input_text.lower()
    
    # Check for SHAP-related queries with specific sample
    shap_sample_pattern = r'shap.*(?:sample|instance|row|data point|observation)\s*#?\s*(\d+)|(?:sample|instance|row|data point|observation)\s*#?\s*(\d+).*shap'
    shap_sample_match = re.search(shap_sample_pattern, input_lower)
    
    if shap_sample_match:
        # Find the first non-None group, which contains the sample index
        sample_index = next((g for g in shap_sample_match.groups() if g is not None), "0")
        logger.info(f"Detected request for SHAP analysis of instance #{sample_index}")
        return f"shap:{sample_index}"
    
    # Check for LIME-related queries with specific sample
    lime_sample_pattern = r'lime.*(?:sample|instance|row|data point|observation)\s*#?\s*(\d+)|(?:sample|instance|row|data point|observation)\s*#?\s*(\d+).*lime'
    lime_sample_match = re.search(lime_sample_pattern, input_lower)
    
    if lime_sample_match:
        # Find the first non-None group, which contains the sample index
        sample_index = next((g for g in lime_sample_match.groups() if g is not None), "0")
        logger.info(f"Detected request for LIME analysis of instance #{sample_index}")
        return f"lime:{sample_index}"
    
    # Check for SHAP related queries
    if any(term in input_lower for term in ["shap", "shapley", "shapely", "contribution", "force plot"]):
        logger.info("Detected request for general SHAP analysis")
        return "shap"
    
    # Check for LIME related queries
    if any(term in input_lower for term in ["lime", "local explanation", "surrogate", "locally"]):
        return "lime"
    
    # Check for feature importance related queries
    if any(term in input_lower for term in ["feature importance", "important feature", "which feature", "significant feature"]):
        return "feature_importance"
    
    # Check for prediction explanation requests with instance id
    prediction_patterns = [
        r"why.+predict.*id\s+(\d+)",
        r"why.*instance\s+(\d+).*predict",
        r"explain.*prediction.*instance\s+(\d+)",
        r"explain.*why.*predict.*sample\s+(\d+)",
        r"why.*model.*predict.*id\s+(\d+)"
    ]
    
    for pattern in prediction_patterns:
        match = re.search(pattern, input_lower)
        if match:
            try:
                instance_id = int(match.group(1))
                return f"prediction:{instance_id}"
            except (IndexError, ValueError):
                pass
    
    # Add prediction explanation to the general checks
    if any(keyword in input_lower for keyword in [
        "why predict", "explain prediction", "explain why", "why does the model predict", 
        "reason for prediction", "prediction explanation"
    ]):
        return "prediction"  # General prediction explanation
    
    # Default to general explanations about the model
    return "general"

# Additional helper function to extract sample details
def extract_sample_details(input_text: str) -> dict:
    """
    Extract sample details from input text
    
    Args:
        input_text: User input text
        
    Returns:
        Dictionary with sample details
    """
    input_lower = input_text.lower()
    details = {}
    
    # Extract sample index if present (support more variations in how users might ask)
    sample_pattern = r'(?:sample|instance|row|data point|observation)\s*#?\s*(\d+)'
    sample_match = re.search(sample_pattern, input_lower)
    
    if sample_match:
        sample_index = sample_match.group(1)
        if sample_index:
            details["index"] = int(sample_index)
            logger.info(f"Extracted sample index: {sample_index}")
    
    return details
