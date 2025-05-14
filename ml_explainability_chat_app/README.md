# ML Explainability Chat App

A Streamlit application that allows users to explore different types of machine learning models, understand their predictions, and get explanations through an interactive chat interface.

## Features

- Browse different model categories (Standard, Deep Learning, Machine Learning, and LLM Models)
- View project summaries and dataset information
- Interactive chatbot to explain model predictions
- Support for multiple explainability methods:
  - SHAP (SHapley Additive exPlanations)
  - LIME (Local Interpretable Model-agnostic Explanations)
  - Feature importance analysis
- Vector store for relevant documentation and prompt retrieval

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ml-explainability-chat-app.git
cd ml-explainability-chat-app
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Train the models (optional if you already have pre-trained models):
```bash
python models/train_ml_pipeline.py
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the URL displayed in the terminal (typically http://localhost:8501)

3. Select a model category and project from the sidebar

4. Interact with the chatbot to explore model explanations:
   - "Show me SHAP values for this model"
   - "Generate LIME explanation for instance #5"
   - "What are the most important features?"
   - "Predict the outcome for instance #10"

## Project Structure

## Note on Model Files
Model files (.pkl) are not included in the repository due to GitHub's file size limitations. You can generate the models locally by running:

```bash
cd ml_explainability_chat_app/models
python train_ml_pipeline.py
```

For production use, you may want to increase the number of estimators (n_estimators parameter) in the training script for better accuracy.

