"""
Module for OpenAI to analyze feature importance from a DataFrame
based on user input (e.g., "Show me top 5 feature importances").
"""

import os
import json
import pandas as pd
from openai import OpenAI

def select_top_feature_importances(df_combined, num_features=10):
    """
    Select the top N features by absolute importance from the DataFrame.
    Args:
        df_combined: DataFrame with columns like 'Feature' and 'Importance'
        num_features: Number of top features to return
    Returns:
        List of dicts with feature and importance
    """
    importance_col = None
    for col in ["Feature", "Importance"]:
        if col in df_combined.columns:
            importance_col = col
            break
    if not importance_col:
        if df_combined.shape[1] > 1:
            importance_col = df_combined.columns[1]
        else:
            return df_combined.head(num_features).to_dict(orient="records")

    df_sorted = df_combined.copy()
    df_sorted = df_sorted.sort_values(by=importance_col, key=abs, ascending=False).head(num_features)
    return df_sorted[["Feature", importance_col]].to_dict(orient="records")

def openai_select_top_feature_importances(user_request, df_combined, default_n=10, max_n=20):
    """
    Use OpenAI to analyze feature importance and respond to user queries.
    Args:
        user_request: User's query string (e.g., "Show me top 5 feature importances")
        df_combined: DataFrame with feature importances
        default_n: Default number of features if not specified (unused in new implementation)
        max_n: Maximum number of features to show (unused in new implementation)
    Returns:
        The model's response text
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    # Get path to feature importance prompt (using the same structure as in feature_calling.py)
    fi_prompt_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts", "feature_importance_prompt.txt")
    
    client = OpenAI(api_key=api_key)

    # Read feature importance prompt instructions if available
    system_prompt = None
    if os.path.exists(fi_prompt_path):
        try:
            with open(fi_prompt_path, "r", encoding="utf-8") as f:
                system_prompt = f.read().strip()
        except Exception as e:
            print(f"Error reading feature importance prompt: {e}")
    
    # Sort the DataFrame by absolute importance for better context
    importance_col = None
    for col in ["Importance", "Weight", "SHAP Value", "SHAP Impact"]:
        if col in df_combined.columns:
            importance_col = col
            break
    
    if importance_col:
        df_sorted = df_combined.copy().sort_values(by=importance_col, key=abs, ascending=False)
        # Format DataFrame as markdown table for the model
        df_table = df_sorted.head(15).to_markdown(index=False)
    else:
        df_table = df_combined.head(15).to_markdown(index=False)
    
    # Prepare prompt with user request and data context
    prompt = f"""
{user_request}

Here are the feature importance values for the top features:

{df_table}
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

def test_openai_select_top_feature_importances():
    """
    Test openai_select_top_feature_importances with fake data and user requests for top 3 to top 7 feature importances.
    """
    import pandas as pd
    import numpy as np

    # Create fake data with required columns
    np.random.seed(42)
    features = [f"feature_{i}" for i in range(1, 21)]
    importances = np.random.randn(20)

    df_combined = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    })

    print("Fake DataFrame:\n", df_combined.head())
    
    # Test multiple requests for different top N features
    test_requests = [
        "Show me the top 3 feature importances",
        "What are the 4 most important features?",
        "I need to see the top 5 feature importances",
        "Display the 6 most influential features",
        "Show me the top 7 feature importances"
    ]
    
    for request in test_requests:
        print("\n" + "="*50)
        print(f"Testing with request: '{request}'")
        
        try:
            response = openai_select_top_feature_importances(request, df_combined)
            print(f"OpenAI function calling response:\n{response}")
        except Exception as e:
            print(f"Error during OpenAI function calling test: {e}")

# Uncomment to run the test directly
# if __name__ == "__main__":
#     test_openai_select_top_feature_importances()
