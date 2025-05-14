"""
Parser module for the ML Explainability Chat App.
Parses user input to determine the action to take.
"""
import re
import os
import logging
from openai import OpenAI
from utils.vector_store import get_prompt_by_name

logger = logging.getLogger("parser")

def parse_user_input(text):
    """
    Parse user input to determine the action to take.
    
    Possible actions:
    - "shap": Generate SHAP explanation
    - "shap:X": Generate SHAP for sample X
    - "lime": Generate LIME explanation
    - "lime:X": Generate LIME for sample X
    - "feature_importance": Show feature importance
    - "prediction": Make a prediction
    - "prediction:X": Make a prediction for sample X and explain
    - "general": General model information
    """
    if not text:
        return "general"
        
    text = text.lower().strip()
    
    # Try LLM-based parsing first
    try:
        llm_action = llm_parse_input(text)
        if llm_action:
            logger.info(f"LLM parser returned action: {llm_action}")
            return llm_action
    except Exception as e:
        logger.warning(f"LLM parsing failed: {e}, falling back to regex")
    
    # Fallback to regex-based parsing
    
    # Check for SHAP requests with sample/instance ID
    shap_patterns = [
        r"shap.*(?:sample|instance|row|data\s*point|observation|id)\s*(?:#|number|no\.|\s)\s*(\d+)",
        r"shap.*(?:sample|instance|row|data\s*point|observation|id)\s*(\d+)"
    ]
    
    for pattern in shap_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                instance_id = int(match.group(1))
                return f"shap:{instance_id}"
            except (IndexError, ValueError):
                pass
    
    # Check for LIME requests with sample/instance ID
    lime_patterns = [
        r"lime.*(?:sample|instance|row|data\s*point|observation|id)\s*(?:#|number|no\.|\s)\s*(\d+)",
        r"lime.*(?:sample|instance|row|data\s*point|observation|id)\s*(\d+)"
    ]
    
    for pattern in lime_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                instance_id = int(match.group(1))
                return f"lime:{instance_id}"
            except (IndexError, ValueError):
                pass
    
    # Check for prediction explanations with sample/instance ID
    prediction_patterns = [
        r"(?:why|explain|reason|what\s+led|cause).*prediction.*(?:sample|instance|row|data\s*point|observation|id)\s*(?:#|number|no\.|\s)\s*(\d+)",
        r"(?:why|explain|reason|what\s+led|cause).*prediction.*(?:sample|instance|row|data\s*point|observation|id)\s*(\d+)",
        r"prediction.*(?:sample|instance|row|data\s*point|observation|id)\s*(?:#|number|no\.|\s)\s*(\d+)",
        r"prediction.*(?:sample|instance|row|data\s*point|observation|id)\s*(\d+)"
    ]
    
    for pattern in prediction_patterns:
        match = re.search(pattern, text)
        if match:
            try:
                instance_id = int(match.group(1))
                return f"prediction:{instance_id}"
            except (IndexError, ValueError):
                pass
    
    # Check for comprehensive analysis requests
    comprehensive_patterns = [
        r"explain (this|the) model (completely|comprehensively|in detail|thoroughly)",
        r"(comprehensive|complete|detailed|thorough) (analysis|explanation|breakdown) of (the|this) model",
        r"tell me everything about (this|the) model",
        r"analyze (the|this) model (completely|comprehensively|in detail|thoroughly)",
        r"how does (the|this) model work (overall|in general|as a whole)",
        r"what can you tell me about (this|the) model",
        r"what (do|are) (the|this) (model|data) (show|tell) (us|me)",
        r"what insights can (we|i|you) (get|derive|learn) from (the|this) (model|data)",
        r"what (patterns|trends|factors) (does|do) (this|the) model (show|reveal|identify)"
    ]
    
    for pattern in comprehensive_patterns:
        if re.search(pattern, text):
            # Check if a specific instance is mentioned
            instance_match = re.search(r"(instance|sample|data point|id)[#\s:]*(\d+)", text)
            if instance_match:
                try:
                    instance_id = int(instance_match.group(2))
                    return f"comprehensive:{instance_id}"
                except (IndexError, ValueError):
                    pass
            return "comprehensive"  # General comprehensive explanation
    
    # General SHAP explanation
    if any(keyword in text for keyword in ["shap", "shapley"]):
        return "shap"
    
    # General LIME explanation
    if "lime" in text:
        return "lime"
    
    # Feature importance
    if any(pattern in text for pattern in [
        "feature importance", "important feature", "significant feature",
        "key feature", "top feature", "influential feature",
        "impact of feature", "feature rank", "feature significance"
    ]):
        return "feature_importance"
    
    # General prediction
    if any(keyword in text for keyword in ["predict", "forecast", "estimate"]):
        return "prediction"
    
    # Default to general explanation
    return "general"

def extract_sample_details(text):
    """Extract sample/instance ID and other details from text"""
    # Extract sample/instance ID
    id_patterns = [
        r"(?:sample|instance|row|data\s*point|observation|id)\s*(?:#|number|no\.|\s)\s*(\d+)",
        r"(?:sample|instance|row|data\s*point|observation|id)\s*(\d+)"
    ]
    
    for pattern in id_patterns:
        match = re.search(pattern, text.lower())
        if match:
            try:
                return {"sample_id": int(match.group(1))}
            except (IndexError, ValueError):
                pass
    
    return {}

def llm_parse_input(text):
    """
    Use LLM to parse user input into an action.
    Returns action string or None if parsing fails.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
        
    try:
        client = OpenAI(api_key=api_key)
        
        # Get parser prompt template
        parser_prompt = get_prompt_by_name("parser_prompt.txt")
        if not parser_prompt:
            logger.warning("Parser prompt template not found")
            return None
            
        # Replace {user_input} with actual text
        prompt = parser_prompt.replace("{user_input}", text)
        
        # Sanitize for logging - limit length and replace problematic chars
        safe_log_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
        safe_log_prompt = safe_log_prompt.encode('ascii', errors='replace').decode('ascii')
        
        # Call the LLM
        try:
            logger.debug("Sending prompt to LLM: %s", safe_log_prompt)
            response = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=20,
                temperature=0.1
            )
            
            # Extract the action from the response
            action = response.choices[0].message.content.strip()
            
            # Validate the action
            valid_patterns = [
                r"^shap:\d+$",
                r"^lime:\d+$", 
                r"^prediction:\d+$",
                r"^comprehensive:\d+$",
                r"^shap$",
                r"^lime$", 
                r"^feature_importance$",
                r"^prediction$",
                r"^comprehensive$",
                r"^general$"
            ]
            
            if any(re.match(pattern, action) for pattern in valid_patterns):
                return action
            else:
                logger.warning(f"LLM returned invalid action: {action}")
                return None
        except Exception as e:
            # Handle OpenAI API errors more gracefully
            logger.error(f"OpenAI API error: {str(e)}")
            return None
    
    except Exception as e:
        logger.error(f"Error in LLM parsing: {e}")
        return None
