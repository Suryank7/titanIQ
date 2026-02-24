import streamlit as st

from backend.tools.survival_predictor_tool import predict_survival

def render_simulator():
    """Renders the survival prediction simulator in the Streamlit Sidebar."""
    st.markdown("Test the Machine Learning model by changing passenger profiles.")
    
    with st.form("prediction_form"):
        pclass = st.selectbox("Ticket Class (1st is best)", options=[1, 2, 3], index=2)
        sex = st.radio("Sex", options=["male", "female"], horizontal=True)
        age = st.slider("Age (years)", min_value=0.0, max_value=100.0, value=25.0, step=0.5)
        fare = st.number_input("Fare Paid (£)", min_value=0.0, max_value=600.0, value=15.0, step=1.0)
        embarked = st.selectbox("Port of Embarkation", options=["S", "C", "Q"], format_func=lambda x: "Southampton" if x=="S" else ("Cherbourg" if x=="C" else "Queenstown"))
        
        submitted = st.form_submit_button("Predict Chances")
        
        if submitted:
            payload = {
                "Pclass": pclass,
                "Sex": sex,
                "Age": age,
                "Fare": fare,
                "Embarked": embarked
            }
            try:
                # Direct prediction bypasses the LLM agent via function call
                res_str = predict_survival.invoke({
                    "pclass": pclass,
                    "sex": sex,
                    "age": age,
                    "fare": fare,
                    "embarked": embarked
                })
                
                if "not survive" in res_str.lower() or "error" in res_str.lower():
                    st.error(res_str)
                else:
                    st.success(res_str)
            except Exception as e:
                st.error(f"Failed to execute prediction model: {e}")
