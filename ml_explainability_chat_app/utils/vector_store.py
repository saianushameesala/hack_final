import os
import json
import numpy as np
import streamlit as st
import logging

# Configure logging
logger = logging.getLogger("vector_store")

# We'll use a simpler approach instead of sentence_transformers and faiss
# to avoid compatibility issues
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Global variables
vectorizer = None
documents = []
document_vectors = None

def initialize_vector_store(project_data):
    """Initialize the vector store with project data and documentation"""
    global vectorizer, documents, document_vectors
    
    logger.info("Initializing vector store with project data")
    
    # Initialize vectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Prepare documents for vector store
    documents = []
    
    # Add project descriptions
    for category, projects in project_data.items():
        for project_name, project_info in projects.items():
            doc = {
                "content": f"Project: {project_name} (Category: {category})\nDescription: {project_info['description']}",
                "metadata": {
                    "type": "project_info",
                    "category": category,
                    "project_name": project_name
                }
            }
            documents.append(doc)
            logger.debug(f"Added project document: {project_name}")
    
    # Add explanation capabilities
    explanation_docs = [
        {
            "content": "SHAP (SHapley Additive exPlanations) is a method to explain the output of any machine learning model by computing the contribution of each feature to the prediction.",
            "metadata": {"type": "explanation", "method": "shap"}
        },
        {
            "content": "LIME (Local Interpretable Model-agnostic Explanations) explains the predictions of any classifier by approximating it locally with an interpretable model.",
            "metadata": {"type": "explanation", "method": "lime"}
        },
        {
            "content": "Feature importance shows which features have the biggest impact on predictions.",
            "metadata": {"type": "explanation", "method": "feature_importance"}
        },
        {
            "content": "You can ask to explain specific instances by their index number, for example 'explain instance 5'.",
            "metadata": {"type": "usage_help"}
        }
    ]
    documents.extend(explanation_docs)
    
    # Load prompts if available (including subdirectories)
    prompt_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
    if os.path.exists(prompt_dir):
        logger.info(f"Loading prompts from {prompt_dir} (including subdirectories)")
        for root, dirs, files in os.walk(prompt_dir):
            for prompt_file in files:
                if prompt_file.endswith(".txt"):
                    try:
                        prompt_path = os.path.join(root, prompt_file)
                        with open(prompt_path, "r", encoding="utf-8") as f:
                            prompt_content = f.read()
                        # Use relative path from prompts dir as the name
                        rel_path = os.path.relpath(prompt_path, prompt_dir)
                        prompt_doc = {
                            "content": prompt_content,
                            "metadata": {"type": "prompt", "name": rel_path.replace("\\", "/")}
                        }
                        documents.append(prompt_doc)
                        logger.debug(f"Added prompt document: {rel_path}")
                    except Exception as e:
                        logger.error(f"Error loading prompt {prompt_file}: {e}")
    
    # Create document vectors
    document_contents = [doc["content"] for doc in documents]
    document_vectors = vectorizer.fit_transform(document_contents)
    
    logger.info(f"Vector store initialized with {len(documents)} documents")
    return True

def get_similar_docs(query, top_k=3):
    """Retrieve similar documents from the vector store"""
    global vectorizer, documents, document_vectors
    
    # Check if vector store is initialized
    if vectorizer is None or document_vectors is None or not documents:
        logger.warning("Vector store not initialized when trying to retrieve documents")
        return []
    
    logger.info(f"Finding documents similar to query: '{query[:50]}...' (top_k={top_k})")
    
    # Vectorize query
    query_vector = vectorizer.transform([query])
    
    # Compute similarity
    similarities = cosine_similarity(query_vector, document_vectors).flatten()
    
    # Get top k indices
    top_indices = similarities.argsort()[-top_k:][::-1]
    
    # Retrieve and return similar documents
    results = []
    for idx in top_indices:
        if 0 <= idx < len(documents):
            doc = documents[idx]
            score = similarities[idx]
            logger.debug(f"Found relevant doc (score={score:.4f}): {doc['content'][:50]}...")
            results.append(doc)
    
    return results

def get_prompt_by_name(name):
    """Retrieve a prompt by name from the vector store"""
    global documents
    
    for doc in documents:
        if doc.get("metadata", {}).get("type") == "prompt" and doc.get("metadata", {}).get("name") == name:
            return doc["content"]
    
    return None
