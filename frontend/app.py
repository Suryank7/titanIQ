import streamlit as st
import json
import base64
import os
import sys

# Add project root to sys.path to allow importing from frontend and backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frontend.components.simulator import render_simulator
from backend.agent.llm import chat_with_agent
from backend.tools.survival_predictor_tool import predict_survival

# Use local directory for serving charts
CHART_DIR = "frontend/charts"
os.makedirs(CHART_DIR, exist_ok=True)

st.set_page_config(page_title="Titanic AI Data Analyst", page_icon="🚢", layout="wide")

st.title("🚢 Titanic Autonomous AI Data Analyst")
st.markdown("Ask anything about the Titanic data. The AI will query the dataset, run models, and generate charts.")

# Sidebar for Simulator and controls
with st.sidebar:
    st.header("🧮 Prediction Simulator")
    render_simulator()
    
    st.divider()
    st.header("🛠️ Actions")
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# Display chat history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "visualizations" in message and message["visualizations"]:
            for vis in message["visualizations"]:
                st.image(vis, use_column_width=True)

# Accept user input
if prompt := st.chat_input("Ask a question about the Titanic dataset..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Process AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("*(Agent is thinking and querying data...)*")
        
        try:
            # Call agent locally
            # response = chat_with_agent(prompt, st.session_state.session_id)
            # response_text = response.get("text", "")
            # visualizations = response.get("visualizations", [])
            if "predict" in prompt.lower() or "survival" in prompt.lower():

                # Example default values (replace with parsed values later if needed)
                result = predict_survival(
                    pclass=3,
                    sex="male",
                    age=25,
                    fare=7.25,
                    embarked="S"
                )

                response_text = result
                visualizations = []

            else:
                # Normal agent call
                response = chat_with_agent(prompt, st.session_state.session_id)
                response_text = response.get("text", "")
                visualizations = response.get("visualizations", [])
            
            # Verify paths exist
            valid_viz = []
            for vis_path in visualizations:
                if os.path.exists(vis_path):
                    valid_viz.append(vis_path)
            
            message_placeholder.markdown(response_text)
            
            # Display charts
            if valid_viz:
                for vis in valid_viz:
                    st.image(vis, use_column_width=True)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "visualizations": valid_viz
            })

        except Exception as e:
            message_placeholder.error(f"Failed to execute agent: {e}")
