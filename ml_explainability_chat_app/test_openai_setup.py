"""
Test script to verify OpenAI API setup and connectivity.
"""
import os
import sys
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO,
                  format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("openai_test")

try:
    import openai
except ImportError:
    logger.error("OpenAI package not installed. Please install with: pip install openai")
    sys.exit(1)

def test_openai_connection(api_key: Optional[str] = None, model: str = "gpt-4.1-nano") -> bool:
    """Test connection to OpenAI API"""
    
    # Get API key
    if api_key is None:
        if 'OPENAI_API_KEY' in os.environ:
            api_key = os.environ['OPENAI_API_KEY']
        else:
            logger.error("No OpenAI API key provided. Set OPENAI_API_KEY environment variable.")
            return False
    
    # Set API key
    openai.api_key = api_key
    
    try:
        logger.info(f"Testing connection to OpenAI API with model: {model}")
        
        # Use correct OpenAI API (>=1.0.0)
        response = openai.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, this is a test."}
            ],
            max_tokens=10
        )
        
        # Extract response content
        content = response.choices[0].message.content.strip()
        logger.info(f"Response received: {content}")
        
        print("\n✅ OpenAI API connection successful!")
        print(f"Model: {model}")
        print(f"Sample response: {content}")
        
        return True
    except Exception as e:
        logger.error(f"Error connecting to OpenAI API: {str(e)}")
        print(f"\n❌ OpenAI API connection failed: {str(e)}")
        return False

def main():
    """Main entry point"""
    print("=" * 60)
    print("OpenAI API Connection Tester")
    print("=" * 60)
    
    # Check environment variable
    if 'OPENAI_API_KEY' in os.environ:
        print("✓ OPENAI_API_KEY environment variable found")
    else:
        print("✗ OPENAI_API_KEY environment variable not found")
        
    # Test connection
    api_key = os.environ.get('OPENAI_API_KEY')
    
    # Try with standard models
    models = ["gpt-4.1-nano"]  # Only using gpt-4.1-nano
    
    success = False
    for model in models:
        print(f"\nTesting connection with model: {model}")
        if test_openai_connection(api_key, model):
            success = True
            break
    
    if not success:
        print("\n❌ All connection tests failed. Please check your API key and access permissions.")
    else:
        print("\n✅ Connection test successful. The OpenAI integration should work in the app.")

if __name__ == "__main__":
    main()
