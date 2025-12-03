import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json

# Page config
st.set_page_config(
    page_title="Network Intrusion Detection",
    page_icon="🔒",
    layout="wide"
)

# Load model version info
@st.cache_data
def load_version_info():
    try:
        with open('models/versions.json', 'r') as f:
            return json.load(f)
    except:
        return {"versions": {"v1": {"accuracy": 0.7727, "date": "2025-12-03", "status": "production"}}}

# Load model
@st.cache_resource
def load_model(version="v1"):
    model_path = f"models/{version}/model.pkl"
    scaler_path = f"models/{version}/scaler.pkl"
    encoder_path = f"models/{version}/encoder.pkl"
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    
    return model, scaler, encoder

# Main app
st.title("🔒 Network Intrusion Detection System")
st.markdown("**AI-Powered Network Security Monitoring using XGBoost**")

# Sidebar
st.sidebar.title("⚙️ Settings")
version_info = load_version_info()
available_versions = list(version_info['versions'].keys())

selected_version = st.sidebar.selectbox(
    "Model Version",
    available_versions,
    index=0
)

version_data = version_info['versions'][selected_version]
st.sidebar.metric("Accuracy", f"{version_data['accuracy']:.2%}")
st.sidebar.caption(f"Status: {version_data['status']}")

# Load model
try:
    model, scaler, encoder = load_model(selected_version)
    st.sidebar.success("✅ Model loaded successfully")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {str(e)}")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Single Prediction", "📁 Batch Prediction", "ℹ️ About"])

# Tab 1: Single Prediction
with tab1:
    st.header("Single Connection Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        duration = st.number_input("Duration (seconds)", value=0.0, min_value=0.0)
        protocol_type = st.selectbox("Protocol", [0, 1, 2], 
                                     format_func=lambda x: ["TCP", "UDP", "ICMP"][x])
        service = st.number_input("Service", value=0, min_value=0, max_value=70)
    
    with col2:
        flag = st.number_input("Flag", value=0, min_value=0, max_value=11)
        src_bytes = st.number_input("Source Bytes", value=0, min_value=0)
        dst_bytes = st.number_input("Destination Bytes", value=0, min_value=0)
    
    with col3:
        count = st.number_input("Count", value=1, min_value=0)
        srv_count = st.number_input("Srv Count", value=1, min_value=0)
        same_srv_rate = st.slider("Same Service Rate", 0.0, 1.0, 1.0)
    
    if st.button("🔍 Detect Intrusion", type="primary"):
        # Create feature vector (41 features)
        features = np.array([[
            duration, protocol_type, service, flag, src_bytes, dst_bytes,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            count, srv_count, 0.0, 0.0, 0.0, 0.0,
            same_srv_rate, 0.0, 0.0, count, srv_count,
            same_srv_rate, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]])
        
        # Predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        predicted_label = encoder.inverse_transform([prediction])[0]
        confidence = float(np.max(probabilities))
        
        # Display result
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Detection Result")
            if predicted_label == "Normal":
                st.success(f"✅ **{predicted_label} Traffic**")
            else:
                st.error(f"⚠️ **{predicted_label} Attack Detected!**")
            
            st.metric("Confidence", f"{confidence:.1%}")
            st.progress(confidence)
        
        with col2:
            st.markdown("### 📊 Probabilities")
            prob_data = {encoder.classes_[i]: probabilities[i] for i in range(len(encoder.classes_))}
            prob_df = pd.DataFrame(list(prob_data.items()), columns=['Type', 'Probability'])
            prob_df = prob_df.sort_values('Probability', ascending=False)
            
            for idx, row in prob_df.iterrows():
                st.write(f"**{row['Type']}**: {row['Probability']:.1%}")
                st.progress(row['Probability'])

# Tab 2: Batch Prediction
with tab2:
    st.header("Batch Analysis from CSV")
    
    st.markdown("""
    Upload a CSV file with network connection features.  
    **Format**: 41 features + label + difficulty (43 columns total)
    """)
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, header=None)
            st.info(f"📄 Loaded {len(df)} connections")
            
            if st.button("🚀 Analyze All", type="primary"):
                with st.spinner("Analyzing connections..."):
                    # Extract features (first 41 columns)
                    X = df.iloc[:, :41].values
                    X_scaled = scaler.transform(X)
                    
                    # Predict
                    predictions = model.predict(X_scaled)
                    probabilities = model.predict_proba(X_scaled)
                    
                    predicted_labels = encoder.inverse_transform(predictions)
                    confidences = np.max(probabilities, axis=1)
                    
                    # Results
                    results_df = pd.DataFrame({
                        'ID': range(1, len(predicted_labels) + 1),
                        'Attack_Type': predicted_labels,
                        'Confidence': [f"{c:.1%}" for c in confidences],
                        'Is_Attack': predicted_labels != 'Normal'
                    })
                    
                    # Metrics
                    st.success("✅ Analysis Complete!")
                    
                    col1, col2, col3 = st.columns(3)
                    total_attacks = (predicted_labels != 'Normal').sum()
                    
                    col1.metric("Total", len(predictions))
                    col2.metric("Attacks", total_attacks)
                    col3.metric("Normal", len(predictions) - total_attacks)
                    
                    # Attack distribution
                    st.markdown("### 📊 Attack Distribution")
                    attack_counts = pd.Series(predicted_labels).value_counts()
                    st.bar_chart(attack_counts)
                    
                    # Results table
                    st.markdown("### 📋 Results")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        "📥 Download Results",
                        csv,
                        "detection_results.csv",
                        "text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Tab 3: About
with tab3:
    st.header("About This System")
    
    st.markdown("""
    ## 🔒 Network Intrusion Detection System
    
    This system uses **XGBoost machine learning** to detect 5 types of network attacks:
    
    - **DoS** - Denial of Service attacks
    - **Probe** - Surveillance and probing
    - **R2L** - Remote to Local attacks
    - **U2R** - User to Root attacks
    - **Normal** - Legitimate traffic
    
    ### 🛠️ Technology Stack
    - **ML**: XGBoost, scikit-learn
    - **Frontend**: Streamlit
    - **Deployment**: Streamlit Community Cloud
    - **Version Control**: GitHub
    - **CI/CD**: GitHub Actions
    
    ### 👥 Team
    **Ruben Santosh, Vignesh R Nair, Arko Chakraborty**  
    Dayananda Sagar University, Bangalore, India
    
    ### 📊 Model Info
    - **Accuracy**: {:.2%}
    - **Version**: {}
    - **Status**: {}
    """.format(version_data['accuracy'], selected_version, version_data['status']))

# Footer
st.markdown("---")
st.caption("🔒 Network IDS | Built with Streamlit & XGBoost")
