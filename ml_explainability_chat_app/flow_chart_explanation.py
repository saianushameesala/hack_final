"""
ML Explainability Chat App - Flow Chart and Code Structure
--------------------------------------------------------

User Interface (app.py)
│
├─── Initialize Session State and UI Components
│    ├─── Set page config, initialize session variables for project selection and chat history
│    └─── Initialize LLM connector (OpenAI or Hugging Face)
│
├─── Sidebar Selection 
│    ├─── Category Selection (Standard/Deep Learning/Machine Learning)
│    └─── Project Selection (Based on selected category)
│
├─── Main Content Display
│    ├─── Project Information Display
│    │    ├─── Description
│    │    ├─── Model Status (Load and display model information)
│    │    └─── Data Status (Load and display dataset information)
│    │
│    └─── Chatbot Interface
│         ├─── Display Chat History
│         ├─── Process User Input
│         │    ├─── Parse User Input (utils/parser.py)
│         │    │    └─── Determine action type (SHAP, LIME, feature importance, prediction)
│         │    │
│         │    ├─── Get Relevant Context (utils/vector_store.py)
│         │    │    └─── Use TF-IDF and cosine similarity to find relevant docs
│         │    │
│         │    ├─── Execute Action (utils/executor.py)
│         │    │    ├─── Generate ML Results (utils/explainers.py)
│         │    │    │    ├─── SHAP Explanations
│         │    │    │    ├─── LIME Explanations
│         │    │    │    ├─── Feature Importance
│         │    │    │    └─── Model Predictions
│         │    │    │
│         │    │    └─── Return Structured Results
│         │    │
│         │    ├─── Generate Natural Language Explanation (utils/llm_explainer.py)
│         │    │    ├─── Format ML Results for LLM
│         │    │    ├─── Send to LLM (OpenAI or Hugging Face)
│         │    │    └─── Process LLM Response
│         │    │
│         │    └─── Return Human-Readable Explanation
│         │
│         └─── Update Chat History

Data and Model Management
│
├─── Project Data Configuration
│    └─── Dictionary mapping categories to projects with model and data paths
│
├─── Model Training (models/train_ml_pipeline.py)
│    ├─── Prepare datasets (breast cancer, diabetes)
│    ├─── Train models (logistic regression, random forest, etc.)
│    └─── Save models and datasets to disk
│
└─── Prompts
     ├─── Simple explanation templates
     ├─── Brief summary templates 
     └─── Detailed analysis templates

Utility Functions
│
├─── Explainers (utils/explainers.py)
│    ├─── SHAP explanation generation
│    └─── LIME explanation generation
│
├─── Vector Store (utils/vector_store.py)
│    ├─── Initialize vector store with project data
│    └─── Retrieve similar documents based on query
│
├─── Parser (utils/parser.py)
│    └─── Parse user input to determine required ML action
│
├─── Executor (utils/executor.py)
│    └─── Execute ML actions and return structured results
│
└─── LLM Connector (utils/llm_connector.py)
     ├─── OpenAI API connector
     ├─── Hugging Face API connector
     └─── Generate natural language explanations of technical ML results

Key Workflow:
1. User asks a question about the model
2. Question is parsed to determine the type of ML explanation needed
3. System executes the appropriate ML technique (SHAP, LIME, etc.)
4. Technical ML results are sent to LLM (OpenAI or Hugging Face)
5. LLM generates a natural language explanation of the results
6. User receives an easy-to-understand explanation
"""
