import os
import sys

# Setup logging early
from utils.logging_setup import setup_logging
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "app.log")
setup_logging(log_level="DEBUG", log_file=log_file)

import logging
logger = logging.getLogger("app")

import streamlit as st
import pandas as pd
import pickle as pkl
import numpy as np
import os
import sys
import traceback
from utils.explainers import get_shap_explanation, get_lime_explanation
from utils.vector_store import get_similar_docs, initialize_vector_store
from utils.parser import parse_user_input
from utils.executor import execute_action
from utils.feature_calling import openai_select_top_features
from utils.feature_calling_for_lime import openai_select_top_lime_features
from utils.feature_calling_for_feature_importances import openai_select_top_feature_importances
from utils.feature_calling_for_prediction import openai_explain_prediction
from utils.comprehensive_explainer import openai_comprehensive_explanation
# Set path to make local modules importable if run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Import LLM connector for generating natural language explanations
try:
    from utils.llm_connector import get_available_llm, OpenAIConnector
    from utils.llm_explainer import LLMExplainer
    LLM_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Error importing LLM modules: {e}")
    LLM_MODULES_AVAILABLE = False

st.set_page_config(
    page_title="Explainabile AI Agent",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'selected_category' not in st.session_state:
    st.session_state.selected_category = None
if 'selected_project' not in st.session_state:
    st.session_state.selected_project = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store_initialized' not in st.session_state:
    st.session_state.vector_store_initialized = False
if 'initialization_error' not in st.session_state:
    st.session_state.initialization_error = None
if 'llm_connector' not in st.session_state:
    st.session_state.llm_connector = None

# Project definitions
project_data = {
    "AI/ML Classification Models": {
        "Credit Risk Model": {
            "model_path": "models/rf_model.pkl",
            "data_path": "data/cs_training_processed.csv",
            "preview_data_path": "data/cs_training_rf.csv",  # Separate preview data
            "description": "The credit card company is focused on predicting the likelihood of a borrower experiencing serious financial distress within two years, using a dataset of 250,000 individuals with 11 financial and demographic features. Data preprocessing involved imputing missing values and removing outliers, followed by applying 2-fold cross-validation to evaluate models such as Logistic Regression, Random Forest"
        },
        "Loan Eligibility Model": {
            "model_path": "models/xgb_model.pkl",
            "data_path": "data/loan_sanction_processed.csv",
            "preview_data_path": "data/loan_sanction_train.csv",  # Separate preview data
            "description": " Dream Housing Finance, a company offering home loans across urban, semi-urban, and rural areas, seeks to automate its loan eligibility process using real-time data from online applications. The dataset includes applicant details such as gender, marital status, education, dependents, income, loan amount, and credit history. The objective is to build a machine learning model to predict loan approval status, thereby streamlining the approval process and targeting eligible customer segments effectively"
        }
    },
    "Financial/Traditional Models": {
        "Trading Cost Model (TCOST)": {
            "model_path": "models/tcost_model.pkl",
            "data_path": "data/tcost_data.csv",
            "preview_data_path": "data/tcost_preview.csv",  # Separate preview data
            "description": "A model to estimate and analyze trading costs in financial markets."
        },
        "SecLending Market Fragmentation Model": {
            "model_path": "models/seclending_model.pkl",
            "data_path": "data/seclending_data.csv",
            "preview_data_path": "data/seclending_preview.csv",  # Separate preview data
            "description": "A model to study market fragmentation in securities lending."
        },
        "Muni Similarity": {
            "model_path": "models/muni_similarity_model.pkl",
            "data_path": "data/muni_similarity_data.csv",
            "preview_data_path": "data/muni_preview.csv",  # Separate preview data
            "description": "A model to assess similarity between municipal bonds for portfolio analysis."
        }
    },
    "AI/ML Regression Models": {
        "Triton": {
            "model_path": "models/gpt_model.pkl",
            "data_path": "data/text_data.csv",
            "preview_data_path": "data/text_preview.csv",  # Separate preview data
            "description": "GPT model fine-tuned for specialized text classification tasks."
        },
        "Market Anamoly Detection": {
            "model_path": "models/bert_model.pkl",
            "data_path": "data/text_embeddings.csv",
            "preview_data_path": "data/text_embeddings_preview.csv",  # Separate preview data
            "description": "BERT model for generating text embeddings for semantic analysis."
        }
    }
}

# Initialize vector store and LLM connector
try:
    if not st.session_state.vector_store_initialized:
        initialize_vector_store(project_data)
        st.session_state.vector_store_initialized = True
        
        # Try to initialize LLM connector and explainer
        if LLM_MODULES_AVAILABLE:
            try:
                # Initialize OpenAI connector with gpt-4.1-nano
                logger.info("Attempting to initialize OpenAI with gpt-4.1-nano model")
                st.session_state.llm_connector = OpenAIConnector(model_name="gpt-4.1-nano")
                
                if not st.session_state.llm_connector.is_initialized:
                    st.session_state.initialization_error = "Failed to initialize OpenAI connector"
                    logger.error("Failed to initialize OpenAI connector")
                else:
                    logger.info(f"Initialized LLM connector: {st.session_state.llm_connector.__class__.__name__}")
                    
                    # Initialize LLM explainer
                    model_metadata = {}
                    for category, projects in project_data.items():
                        for name, info in projects.items():
                            if os.path.exists(info["data_path"]):
                                try:
                                    df = pd.read_csv(info["data_path"])
                                    model_metadata[name] = list(df.columns)
                                except:
                                    pass
                    
                    st.session_state.llm_explainer = LLMExplainer(
                        llm_connector=st.session_state.llm_connector
                    )
                    logger.info("Initialized LLM explainer")
                
            except Exception as e:
                st.session_state.initialization_error = f"Error initializing LLM connector: {str(e)}"
                logger.error(f"Error initializing LLM connector: {str(e)}")
except Exception as e:
    st.session_state.initialization_error = f"Error initializing vector store: {str(e)}"
    logger.error(f"Error initializing vector store: {str(e)}")

# Main Title
st.title("Explainabile AI Agent")
st.write("Explore different ML models, get explanations, and understand predictions through interactive chat.")

# Display initialization errors if any
if st.session_state.initialization_error:
    st.error(f"Initialization Error: {st.session_state.initialization_error}")
    st.warning("The app will continue to function with limited capabilities.")

# Sidebar for category selection
with st.sidebar:
    st.header("Model Categories")
    category_order = [
        "AI/ML Classification Models",
        "Financial/Traditional Models",
        "AI/ML Regression Models",
    ]
    category = st.radio(
        "Select a category:",
        category_order
    )
    st.session_state.selected_category = category

    # Project selection based on category
    if st.session_state.selected_category:
        st.header("Projects")
        projects = list(project_data[st.session_state.selected_category].keys())
        selected_project = st.selectbox("Select a project:", projects)
        st.session_state.selected_project = selected_project
        
        if st.button("Load Project"):
            st.success(f"Loaded {selected_project} project!")

# Main content
if st.session_state.selected_category and st.session_state.selected_project:
    project_info = project_data[st.session_state.selected_category][st.session_state.selected_project]
    
    # Display project information
    st.header(f"{st.session_state.selected_project}")
    st.subheader("Project Description")
    st.write(project_info["description"])
    
    # Check if model and data exist
    model_exists = os.path.exists(project_info["model_path"])
    data_exists = os.path.exists(project_info["data_path"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Status")
        if model_exists:
            st.success("Model is available")
            # Try to load the model
            try:
                with open(project_info["model_path"], 'rb') as f:
                    model = pkl.load(f)
                st.write("Model type:", type(model).__name__)
            except Exception as e:
                st.error(f"Error loading model: {e}")
                model = None
        else:
            st.error("Model file not found. Please train the model first.")
            model = None
    
    with col2:
        st.subheader("Data Status")
        if data_exists:
            st.success("Dataset is available")
            # Try to load the data
            try:
                data = pd.read_csv(project_info["data_path"])
                st.write(f"Dataset shape: {data.shape}")
            except Exception as e:
                st.error(f"Error loading data: {e}")
                data = None
        else:
            st.error("Data file not found.")
            data = None

    # Dataset preview section - now using preview data when available
    if data_exists and data is not None:
        st.subheader("Dataset Preview")
        
        # Check if preview data path exists and try to load it
        preview_data = None
        if "preview_data_path" in project_info:
            preview_path = project_info["preview_data_path"]
            if os.path.exists(preview_path):
                try:
                    preview_data = pd.read_csv(preview_path)
                    st.success(f"Loaded preview data from {preview_path}")
                except Exception as e:
                    logger.warning(f"Could not load preview data from {preview_path}: {e}")
                    st.warning(f"Could not load preview data: {e}. Falling back to main dataset.")
        
        if preview_data is not None:
            # Use the dedicated preview dataset
            sample_data = preview_data.head(5)
        else:
            # Fall back to the main dataset
            if 'is_sample' in data.columns:
                # Get only the sample data for preview (limit to 5 rows)
                sample_data = data[data['is_sample'] == 1].head(5)
                # If no sample data found, use regular data
                if len(sample_data) == 0:
                    sample_data = data.head(5)
            else:
                sample_data = data.head(5)
        
        with st.expander("Preview Dataset"):
            st.dataframe(sample_data, use_container_width=True)  # Added use_container_width=True for full width

    # --- Show SHAP and LIME results as DataFrame and Plot ---
    if model_exists and data_exists and model is not None and data is not None:
        # Remove target column and sample indicator if present
        features_only = data.copy()
        if 'target' in features_only.columns:
            features_only = features_only.drop('target', axis=1)
        if 'is_sample' in features_only.columns:
            features_only = features_only.drop('is_sample', axis=1)
        
        st.subheader("Model Explanations")
        
        # Create tabs for different explanations
        tabs = st.tabs(["SHAP Global", "SHAP Instance", "LIME", "Feature Importance"])
        
        with tabs[0]:
            st.subheader("SHAP Global Explanations")
            with st.spinner("Generating SHAP global explanations..."):
                try:
                    # SHAP global - use a sample of the data for efficiency
                    sample_size = min(100, len(features_only))
                    sample_indices = np.random.choice(len(features_only), sample_size, replace=False)
                    data_sample = features_only.iloc[sample_indices]
                    
                    # Generate SHAP explanation
                    shap_result = get_shap_explanation(model, data_sample)
                    
                    # Display SHAP summary plot
                    if "image" in shap_result and shap_result["image"]:
                        st.markdown("**SHAP Summary Plot:**")
                        st.image(f"data:image/png;base64,{shap_result['image']}")
                    
                    # # Display SHAP waterfall plot for the first instance as example
                    # if "waterfall_image" in shap_result and shap_result["waterfall_image"]:
                    #     st.markdown("**SHAP Waterfall Plot (First Sample):**")
                    #     st.image(f"data:image/png;base64,{shap_result['waterfall_image']}")
                    #     st.caption("This waterfall plot shows how each feature contributes to the prediction for the first sample.")
                except Exception as e:
                    st.error(f"Error generating SHAP global explanation: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        
        with tabs[1]:
            st.subheader("SHAP Instance Explanation")
            # Add instance selector
            max_index = len(features_only) - 1
            selected_instance = st.number_input("Select data instance", min_value=0, max_value=max_index, value=0)
            
            if st.button("Analyze Instance with SHAP"):
                with st.spinner("Generating SHAP instance explanation..."):
                    try:
                        instance_shap = get_shap_explanation(model, features_only, instance_index=selected_instance)
                        
                        # Display instance data
                        st.markdown(f"**Selected instance #{selected_instance} features:**")
                        st.dataframe(features_only.iloc[[selected_instance]].reset_index(drop=True))
                        
                        # Display force plot if available
                        if "force_image" in instance_shap and instance_shap["force_image"]:
                            st.markdown(f"**SHAP Force Plot for Instance #{selected_instance}:**")
                            st.image(f"data:image/png;base64,{instance_shap['force_image']}")
                            st.caption("This force plot shows how each feature pushes the prediction higher (red) or lower (blue)")
                        
                        # Display waterfall plot if available
                        if "image" in instance_shap and instance_shap["image"]:
                            st.markdown(f"**SHAP Waterfall Plot for Instance #{selected_instance}:**")
                            st.image(f"data:image/png;base64,{instance_shap['image']}")
                            st.caption("This waterfall plot shows the magnitude of each feature's contribution")
                        
                        # Show feature values and their impact
                        if "feature_importance" in instance_shap and "feature_values" in instance_shap:
                            # Create a DataFrame combining feature values and their SHAP contributions
                            fi = instance_shap["feature_importance"]
                            fv = instance_shap["feature_values"]
                            df_combined = pd.DataFrame({
                                "Feature": list(fi.keys()),
                                "Value": [fv[f] for f in fi.keys()],
                                "SHAP Impact": list(fi.values())
                            })
                            df_combined = df_combined.sort_values(by="SHAP Impact", key=abs, ascending=False)
                            
                            st.markdown("**Feature Values and Their Impact:**")
                            st.dataframe(df_combined)
                    except Exception as e:
                        st.error(f"Error generating SHAP instance explanation: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        with tabs[2]:
            st.subheader("LIME Explanation")
            # LIME for first sample by default
            lime_instance = st.number_input("Select instance for LIME explanation", min_value=0, max_value=max_index, value=0)
            
            if st.button("Generate LIME Explanation"):
                with st.spinner("Generating LIME explanation..."):
                    lime_result = get_lime_explanation(model, data.iloc[:, :-1], lime_instance)
                    if "explanation_obj" in lime_result and lime_result["explanation_obj"] is not None:
                        try:
                            lime_exp = lime_result["explanation_obj"]
                            if hasattr(lime_exp, "as_list"):
                                lime_df = pd.DataFrame(lime_exp.as_list(), columns=["Feature", "Weight"])
                                st.markdown("**LIME Explanation Weights:**")
                                st.dataframe(lime_df)
                        except Exception as e:
                            st.warning(f"Could not display LIME DataFrame: {e}")
        
        with tabs[3]:
            st.subheader("Feature Importance")
            with st.spinner("Calculating feature importance..."):
                fi_result = execute_action("feature_importance", model, data, project_info)
                if "sorted_importance" in fi_result:
                    sorted_imp = fi_result["sorted_importance"]
                    df_fi = pd.DataFrame(sorted_imp, columns=["Feature", "Importance"])
                    
                    st.markdown("**Model Feature Importance:**")
                    st.bar_chart(df_fi.set_index("Feature").head(10))
                    st.dataframe(df_fi)
    
    # Chatbot section
    st.header("Explainability Chat")
    st.write("Ask questions about the model, request explanations, or explore model predictions.")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Remove the trace_execution decorator and ExecutionTracer usage
    def process_user_input():
        user_input = st.chat_input("Ask something about the model...")
        if not user_input:
            return
        
        logger.info(f"Received user input: {user_input}")
        
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        
        try:
            # Get relevant context documents
            context_docs = get_similar_docs(user_input, top_k=3)
            
            # Parse user input to determine action type
            action = parse_user_input(user_input)
            logger.info(f"Parsed action: {action}")
            
            # Execute action based on parsing
            if model_exists and data_exists:
                # Execute the action and get structured results
                result_data = execute_action(
                    action, 
                    model, 
                    data, 
                    project_info,
                    context_docs
                )
                
                # Handle errors in result_data
                if "error" in result_data and result_data.get("type") == "error":
                    error_msg = result_data.get("error", "Unknown error")
                    st.error(f"Error: {error_msg}")
                    logger.error(f"Error in action execution: {error_msg}")
                    if "traceback" in result_data:
                        logger.error(f"Traceback: {result_data['traceback']}")
                    response = f"I encountered an error while trying to analyze this instance: {error_msg}\n\nPlease try with a different instance or action."
                    
                # Handle specific SHAP instance request
                elif action.startswith("shap:"):
                    try:
                        instance_id = int(action.split(":")[1])
                        logger.info("SHAP analysis without specific instance I am here stage 1")

                        # Display feature values and their SHAP contributions as a table
                        if "feature_importance" in result_data and "feature_values" in result_data:
                            fi = result_data["feature_importance"]
                            fv = result_data["feature_values"]
                            df_combined = pd.DataFrame({
                                "Feature": list(fi.keys()),
                                "Value": [fv[f] for f in fi.keys()],
                                "SHAP Impact": list(fi.values())
                            })
                            df_combined = df_combined.sort_values(by="SHAP Impact", key=abs, ascending=False)
                            
                            # Only show the table for reference, don't show images
                            # st.markdown(f"### SHAP Analysis for Instance #{instance_id}")
                            # st.dataframe(df_combined)
                            response = openai_select_top_features(user_input, df_combined)
                            # st.markdown("**Top Features (via OpenAI function calling):**")
                            # st.write(response)   

                    except Exception as e:
                        logger.error(f"Error displaying instance SHAP analysis: {e}")
                        response = f"Error analyzing instance: {str(e)}"
            
                # For general SHAP analysis without specific instance
                elif action == "shap":
                    try:
                        logger.info("SHAP analysis without specific instance I am here ")
                        if "feature_importance" in result_data:
                            fi = result_data["feature_importance"]
                            sorted_fi = sorted(fi.items(), key=lambda x: abs(x[1]), reverse=True)
                            df_fi = pd.DataFrame(sorted_fi, columns=["Feature", "SHAP Value"])

                            response = openai_select_top_features(user_input, df_fi)
                            # st.markdown("**Top Features (via OpenAI function calling):**")
                            # st.write(response)
                        
                    except Exception as e:
                        logger.error(f"Error displaying SHAP analysis: {e}")
                        response = f"Error analyzing model: {str(e)}"
                
                # Handle feature importance action
                elif action == "feature_importance":
                    try:
                        # Show feature importance as table
                        if "sorted_importance" in result_data:
                            sorted_imp = result_data["sorted_importance"]
                            df_fi = pd.DataFrame(sorted_imp, columns=["Feature", "Importance"])
                            
                            # Call OpenAI function calling for feature importance
                            response = openai_select_top_feature_importances(user_input, df_fi)

                    except Exception as e:
                        logger.error(f"Error displaying feature importance: {e}")
                        response = f"Error analyzing model: {str(e)}"

                # Handle LIME requests similarly - show table instead of image
                elif action.startswith("lime"):
                    try:
                        # instance_id = int(action.split(":")[1])
                        
                        # Display LIME results as table if available
                        if "explanation_obj" in result_data and result_data["explanation_obj"] is not None:
                            lime_exp = result_data["explanation_obj"]
                            if hasattr(lime_exp, "as_list"):
                                lime_df = pd.DataFrame(lime_exp.as_list(), columns=["Feature", "Weight"])
                                # Call OpenAI function calling for LIME
                                response = openai_select_top_lime_features(user_input, lime_df)
                                # st.markdown("**Top Features (via OpenAI function calling):**")
                                # st.write(response)
                    except Exception as e:
                        logger.error(f"Error displaying LIME analysis: {e}")
                        response = f"Error analyzing instance with LIME: {str(e)}"
                
                # Handle specific prediction explanation requests
                elif action.startswith("prediction:"):
                    try:
                        instance_id = int(action.split(":")[1])
                        logger.info(f"Prediction explanation for instance {instance_id}")
                        
                        # Get prediction for this instance
                        prediction_result = execute_action(f"prediction:{instance_id}", model, data, project_info)
                        
                        # Get SHAP explanation for this instance (for reasoning)
                        shap_result = execute_action(f"shap:{instance_id}", model, data, project_info)
                        
                        # Get the instance data
                        if "input_features" in prediction_result:
                            instance_data = prediction_result["input_features"]
                        else:
                            # If not in prediction result, get directly from data
                            instance_data = data.iloc[instance_id].to_dict() if instance_id < len(data) else None
                        
                        # Generate explanation using OpenAI
                        if instance_data:
                            response = openai_explain_prediction(
                                user_input, 
                                instance_data, 
                                prediction_result, 
                                shap_result
                            )
                        else:
                            response = f"Could not explain prediction for instance #{instance_id} due to missing data."
                            
                    except Exception as e:
                        logger.error(f"Error explaining prediction: {e}")
                        response = f"Error explaining prediction: {str(e)}"
                
                # Handle general prediction requests without specific instance
                elif action == "prediction":
                    try:
                        # Default to first instance if none specified
                        prediction_result = execute_action("prediction:0", model, data, project_info)
                        shap_result = execute_action("shap:0", model, data, project_info)
                        
                        # Get the instance data
                        if "input_features" in prediction_result:
                            instance_data = prediction_result["input_features"]
                        else:
                            # If not in prediction result, get directly from data
                            instance_data = data.iloc[0].to_dict()
                        
                        # Generate explanation using OpenAI
                        response = openai_explain_prediction(
                            user_input, 
                            instance_data, 
                            prediction_result, 
                            shap_result
                        )
                            
                    except Exception as e:
                        logger.error(f"Error explaining prediction: {e}")
                        response = f"Error explaining prediction: {str(e)}"

                # Handle comprehensive explanation requests
                elif action.startswith("comprehensive:"):
                    try:
                        instance_id = int(action.split(":")[1])
                        logger.info(f"Comprehensive explanation for instance {instance_id}")
                        
                        # Generate comprehensive explanation
                        response = openai_comprehensive_explanation(
                            user_input, 
                            model, 
                            data, 
                            project_info, 
                            sample_index=instance_id
                        )
                            
                    except Exception as e:
                        logger.error(f"Error generating comprehensive explanation: {e}")
                        response = f"Error generating comprehensive explanation: {str(e)}"

                elif action == "comprehensive":
                    try:
                        # Default to first instance if none specified
                        logger.info("Generating comprehensive model explanation")
                        
                        # Generate comprehensive explanation
                        response = openai_comprehensive_explanation(
                            user_input, 
                            model, 
                            data, 
                            project_info, 
                            sample_index=0
                        )
                            
                    except Exception as e:
                        logger.error(f"Error generating comprehensive explanation: {e}")
                        response = f"Error generating comprehensive explanation: {str(e)}"

                # For other action types, generate natural language explanation using LLM if available
                else:
                    if 'llm_explainer' in st.session_state and st.session_state.llm_explainer:
                        try:
                            response = st.session_state.llm_explainer.explain_results(
                                result_data=result_data,
                                action_type=action.split(":")[0],  # Remove sample index from action
                                model_type=st.session_state.selected_project,
                                model=model,
                                project_info=project_info,
                                user_query=user_input
                            )
                            logger.info("Generated LLM explanation successfully")
                        except Exception as e:
                            logger.error(f"Error generating LLM explanation: {e}")
                            response = str(result_data)  # Fall back to raw result data
                    else:
                        # No LLM available, return raw results
                        response = str(result_data)
            else:
                # Model or data doesn't exist
                response = "I can't execute this request because either the model or data is not available. Please ensure both are properly loaded."
            
            logger.info("Generated response")
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            logger.error(traceback.format_exc())
            response = f"An error occurred: {str(e)}"
        
        # Add response to chat history and display
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response, unsafe_allow_html=True)

    # Call the function to process user input
    process_user_input()
else:
    st.info("Please select a category and project from the sidebar to begin.")

# Add an expander with instructions
with st.expander("How to use this app"):
    st.markdown("""
    1. **Select a model category** from the sidebar
    2. **Choose a specific project** within that category
    3. **Click 'Load Project'** to load model information and dataset
    4. **Use the chat interface** to ask questions about the model. Example queries:
        - "Explain how this model works"
        - "Show me SHAP values for this model"
        - "What features are most important?"
        - "Generate LIME explanation for sample #5"
        - "How does feature X affect predictions?"
        - "Make a prediction for [input values]"
    """)
