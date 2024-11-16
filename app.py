import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import os
from datetime import datetime
import json
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📱",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: white;
        padding: 20px;
    }
    .stButton > button {
        width: 200px;  /* Modified button width */
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        border: none;
        margin: 0 auto;
        display: block;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .spam {
        background-color: #000000;
        border: 1px solid #ef5350;
    }
    .not-spam {
        background-color: #000000;
        border: 1px solid #4caf50;
    }
    .title-text {
        text-align: center;
        font-size: 36px;
        font-weight: bold;
        color: #1f1f1f;
        margin-bottom: 20px;
    }
    .subtitle-text {
        text-align: center;
        font-size: 18px;
        color: #666;
        margin-bottom: 30px;
    }
    .footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        text-align: center;  /* Fixed center alignment */
        padding: 10px;
        background-color: white;
        color: #000000;
        border-top: 1px solid #eee;
    }
    .history-item {
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin: 5px 0;
    }
    .sidebar-title {
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .content-container {
        max-width: 800px;  /* Added container width */
        margin: 0 auto;
        padding: 0 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Function to load or create history
def load_history():
    try:
        with open('prediction_history.json', 'r') as f:
            return json.load(f)
    except:
        return []

# Function to save history
def save_history(history):
    with open('prediction_history.json', 'w') as f:
        json.dump(history, f)

# Function to ensure all required NLTK data is downloaded
def download_nltk_data():
    try:
        custom_nltk_path = os.path.join(os.getcwd(), 'nltk_data')
        if not os.path.exists(custom_nltk_path):
            os.makedirs(custom_nltk_path)
        nltk.data.path.insert(0, custom_nltk_path)
        resources = ['punkt', 'stopwords', 'averaged_perceptron_tagger', 'english']
        for resource in resources:
            nltk.download(resource, download_dir=custom_nltk_path, quiet=True)
        return True
    except Exception as e:
        st.error(f"Error in NLTK setup: {str(e)}")
        return False

# Function to preprocess text
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    words = text.split()
    words = [word for word in words if word.isalpha()]
    try:
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
    except:
        pass
    return " ".join(words)

# Load the trained model and vectorizer
@st.cache_resource
def load_models():
    try:
        with open('model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
        with open('vectorizer.pkl', 'rb') as vectorizer_file:
            vectorizer = pickle.load(vectorizer_file)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        return None, None

def main():
    # Sidebar
    with st.sidebar:
        st.markdown('<p class="sidebar-title">Prediction History</p>', unsafe_allow_html=True)
        history = load_history()
        
        if not history:
            st.info("No predictions yet!")
        else:
            for item in history[-10:]:  # Show last 10 predictions
                with st.container():
                    st.markdown(f"""
                    <div class="history-item">
                        <strong>Message:</strong> {item['message'][:50]}...
                        <br><strong>Prediction:</strong> {'Spam' if item['prediction'] == 1 else 'Not Spam'}
                        <br><strong>Time:</strong> {item['timestamp']}
                    </div>
                    """, unsafe_allow_html=True)
        
        # Add a clear history button
        if st.button("Clear History"):
            save_history([])
            st.success("History cleared!")
            st.rerun()

    # Main content
    st.markdown('<p class="title-text">SMS Spam Detection</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Enter your message below to check if it\'s spam or not</p>', unsafe_allow_html=True)

    # Create container for main content
    with st.container():
        # Center container div
        st.markdown('<div class="content-container">', unsafe_allow_html=True)
        
        # Load models
        model, vectorizer = load_models()
        if model is None or vectorizer is None:
            st.error("Failed to load the model files.")
            return

        # User input with reduced height and container width
        input_sms = st.text_area("", placeholder="Type or paste your message here...", height=100)
        
        if st.button("Analyze Message"):
            if not input_sms.strip():
                st.warning("Please enter a message to classify.")
                return

            try:
                # Preprocess and predict
                processed_sms = preprocess_text(input_sms)
                vectorized_sms = vectorizer.transform([processed_sms])
                prediction = model.predict(vectorized_sms)

                # Save prediction to history
                history = load_history()
                history.append({
                    'message': input_sms,
                    'prediction': int(prediction[0]),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                save_history(history)

                # Display prediction with styling
                if prediction[0] == 1:
                    st.markdown("""
                        <div class="prediction-box spam">  
                            <h3>Spam Detected!</h3>
                            <p>This message has been classified as spam. Be cautious!</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class="prediction-box not-spam">
                            <h3>Not Spam</h3>
                            <p>This message appears to be legitimate.</p>
                        </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown(
        '<div class="footer">Created by Seemab Hassan</div>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()