"""
Credit Risk Assessment Web Application

A Streamlit web interface for predicting loan approval/rejection
with uncertainty-based escalation to human agents.

Usage:
    streamlit run app.py

Author: Credit Risk ML Team
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Page configuration
st.set_page_config(
    page_title="Credit Risk Assessment Portal",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Professional Banking Theme
st.markdown("""
<style>
    /* Import Banking Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .main {
        background-color: #f8f9fa;
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Styling */
    .bank-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem 3rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .bank-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .bank-header p {
        color: #e3f2fd;
        font-size: 1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    /* Form Container */
    .form-container {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }
    
    /* Section Headers */
    .section-header {
        color: #1e3c72;
        font-size: 1.3rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e3f2fd;
    }
    
    /* Input Labels */
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stSlider label {
        color: #2c3e50;
        font-weight: 500;
        font-size: 0.95rem;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Decision Boxes */
    .decision-box {
        padding: 2rem;
        border-radius: 12px;
        margin: 2rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .escalate-box {
        background: linear-gradient(135deg, #fff5f5 0%, #ffe0e0 100%);
        border-left: 6px solid #dc3545;
    }
    
    .escalate-box h2 {
        color: #dc3545;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }
    
    .approve-box {
        background: linear-gradient(135deg, #f0fff4 0%, #d4f4dd 100%);
        border-left: 6px solid #28a745;
    }
    
    .approve-box h2 {
        color: #28a745;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }
    
    .reject-box {
        background: linear-gradient(135deg, #fff8f0 0%, #ffe4cc 100%);
        border-left: 6px solid #fd7e14;
    }
    
    .reject-box h2 {
        color: #fd7e14;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }
    
    .decision-text {
        font-size: 1.1rem;
        color: #2c3e50;
        margin: 0.5rem 0;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Sidebar Styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #2c3e50;
    }
    
    .css-1d391kg .sidebar-content, [data-testid="stSidebar"] .sidebar-content {
        color: white;
    }
    
    /* Info Boxes */
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    /* Status Badge */
    .status-badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.5rem 0;
    }
    
    .badge-success {
        background-color: #d4edda;
        color: #155724;
    }
    
    .badge-warning {
        background-color: #fff3cd;
        color: #856404;
    }
    
    .badge-danger {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    /* Table Styling */
    .dataframe {
        border: none !important;
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #ddd, transparent);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load trained models (cached)"""
    models_dir = Path('results/models')
    
    try:
        # Import the preprocessor class
        from src.data_preprocessing import CreditDataPreprocessor
        
        # Load the raw dataset to fit the preprocessor
        # (The saved preprocessor.pkl is just metadata, not the actual object)
        raw_data_path = Path('data/raw/LC_loans_granting_model_dataset.csv')
        if raw_data_path.exists():
            df_raw = pd.read_csv(raw_data_path)
            # Only keep the first 1000 rows for faster loading
            df_sample = df_raw.head(1000)
            
            # Initialize and fit preprocessor
            preprocessor = CreditDataPreprocessor()
            _ = preprocessor.fit_transform(df_sample, 'Default')
        else:
            st.error("❌ Raw data file not found. Cannot initialize preprocessor.")
            st.stop()
        
        with open(models_dir / 'bootstrap_ensemble.pkl', 'rb') as f:
            ensemble = pickle.load(f)
        
        with open(models_dir / 'escalation_system.pkl', 'rb') as f:
            escalation_system = pickle.load(f)
        
        return preprocessor, ensemble, escalation_system
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.info("Please run the training notebooks first!")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()


def predict_loan(loan_data, preprocessor, ensemble, escalation_system):
    """Make prediction for a loan application"""
    
    # Convert to DataFrame
    df = pd.DataFrame([loan_data])
    
    # Preprocess
    try:
        X_processed = preprocessor.transform(df)
    except Exception as e:
        return {'error': str(e)}
    
    # Get predictions with uncertainty
    proba_mean, uncertainty, all_proba = ensemble.predict_with_uncertainty(X_processed)
    
    # Extract values
    prob_default = proba_mean[0, 1]
    prob_paid = proba_mean[0, 0]
    unc = uncertainty[0]
    confidence = np.max(proba_mean[0])
    
    # Get escalation decision
    should_escalate, escalation_reason = escalation_system.should_escalate(
        uncertainty=unc,
        confidence=confidence,
        probability=prob_default
    )
    
    # Make decision
    if should_escalate:
        action = "ESCALATE"
        decision = "PENDING HUMAN REVIEW"
    else:
        if prob_default >= 0.5:
            action = "AUTOMATED"
            decision = "REJECT"
        else:
            action = "AUTOMATED"
            decision = "APPROVE"
    
    return {
        'action': action,
        'decision': decision,
        'prob_default': prob_default,
        'prob_paid': prob_paid,
        'confidence': confidence,
        'uncertainty': unc,
        'should_escalate': should_escalate,
        'escalation_reason': escalation_reason,
        'all_proba': all_proba
    }


def create_gauge_chart(value, title, thresholds=[0.3, 0.7]):
    """Create a professional gauge chart for probabilities"""
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 18, 'color': '#2c3e50', 'family': 'Inter'}},
        number = {'suffix': "%", 'font': {'size': 36, 'color': '#1e3c72', 'family': 'Inter'}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#1e3c72"},
            'bar': {'color': "#1e3c72", 'thickness': 0.75},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#e3f2fd",
            'steps': [
                {'range': [0, thresholds[0]*100], 'color': '#d4edda'},
                {'range': [thresholds[0]*100, thresholds[1]*100], 'color': '#fff3cd'},
                {'range': [thresholds[1]*100, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "#dc3545", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    
    fig.update_layout(
        height=280,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif")
    )
    return fig


def create_uncertainty_chart(all_proba):
    """Create professional visualization of ensemble predictions"""
    
    # Get probability of default for each model
    probs = all_proba[:, 0, 1]
    
    fig = go.Figure()
    
    # Add histogram with gradient colors
    fig.add_trace(go.Histogram(
        x=probs * 100,
        nbinsx=25,
        name='Model Predictions',
        marker=dict(
            color=probs * 100,
            colorscale='RdYlGn_r',
            line=dict(color='white', width=1)
        ),
        opacity=0.8
    ))
    
    # Add mean line
    mean_prob = np.mean(probs) * 100
    fig.add_vline(
        x=mean_prob,
        line_dash="dash",
        line_color="#1e3c72",
        line_width=3,
        annotation_text=f"Mean: {mean_prob:.1f}%",
        annotation_position="top right",
        annotation_font=dict(size=12, color="#1e3c72", family="Inter")
    )
    
    # Add decision boundary
    fig.add_vline(
        x=50,
        line_dash="dot",
        line_color="#dc3545",
        line_width=2,
        annotation_text="Decision Threshold",
        annotation_position="bottom right",
        annotation_font=dict(size=11, color="#dc3545", family="Inter")
    )
    
    fig.update_layout(
        title=dict(
            text="Ensemble Model Predictions Distribution",
            font=dict(size=16, color="#2c3e50", family="Inter")
        ),
        xaxis_title="Default Probability (%)",
        yaxis_title="Number of Models",
        height=280,
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif"),
        xaxis=dict(gridcolor='#e3f2fd'),
        yaxis=dict(gridcolor='#e3f2fd')
    )
    
    return fig


def main():
    """Main application"""
    
    # Professional Banking Header
    st.markdown('''
    <div class="bank-header">
        <h1>🏦 Credit Risk Assessment Portal</h1>
        <p>Intelligent Loan Decision Making with AI-Powered Risk Analysis</p>
    </div>
    ''', unsafe_allow_html=True)
    
    # Load models
    with st.spinner("🔄 Loading risk assessment models..."):
        preprocessor, ensemble, escalation_system = load_models()
    
    st.success("✅ System Ready - All models loaded successfully")
    
    # Sidebar Configuration
    st.sidebar.markdown("## ⚙️ System Configuration")
    st.sidebar.markdown("---")
    
    input_mode = st.sidebar.radio(
        "**Select Application Mode:**",
        ["📝 Manual Entry", "📤 Batch Upload (CSV)", "📋 Use Sample Data"],
        index=0
    )
    
    # Show system info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Escalation Parameters")
    st.sidebar.metric("Uncertainty Threshold", f"{escalation_system.uncertainty_threshold:.3f}")
    st.sidebar.metric("Confidence Threshold", f"{escalation_system.confidence_threshold:.2%}")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info("""
    **AI-Powered Credit Assessment**
    
    This system uses machine learning to:
    - Predict default probability
    - Quantify prediction uncertainty
    - Automatically escalate high-risk cases
    - Optimize decision accuracy
    """)
    
    # Main content area
    if "📝 Manual Entry" in input_mode:
        render_manual_entry_form(preprocessor, ensemble, escalation_system)
    
    elif "📋 Use Sample Data" in input_mode:
        render_sample_data_form(preprocessor, ensemble, escalation_system)
    
    else:  # Batch Upload
        render_csv_upload_form(preprocessor, ensemble, escalation_system)


def render_manual_entry_form(preprocessor, ensemble, escalation_system):
    """Render the manual entry form"""
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    st.markdown("### 📝 Loan Application Form")
    st.markdown("Please complete all required fields below:")
    
    # Personal & Financial Information
    st.markdown('<p class="section-header">💰 Financial Information</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        loan_amnt = st.number_input(
            "Loan Amount ($)", 
            min_value=500, 
            max_value=50000, 
            value=10000, 
            step=500,
            help="Requested loan amount"
        )
        revenue = st.number_input(
            "Annual Income ($)", 
            min_value=0, 
            max_value=500000, 
            value=50000, 
            step=1000,
            help="Total annual income from all sources"
        )
    
    with col2:
        dti_n = st.slider(
            "Debt-to-Income Ratio (%)", 
            min_value=0.0, 
            max_value=50.0, 
            value=15.5, 
            step=0.1,
            help="Percentage of monthly income that goes to debt payments"
        )
        fico_n = st.slider(
            "FICO Credit Score", 
            min_value=300, 
            max_value=850, 
            value=680, 
            step=5,
            help="Your credit score (300-850)"
        )
    
    with col3:
        experience_c = st.selectbox(
            "Credit Experience Level", 
            options=[0, 1, 2, 3, 4, 5],
            index=3,
            format_func=lambda x: {
                0: "⭐ Minimal (New to Credit)",
                1: "⭐⭐ Limited (1-2 years)",
                2: "⭐⭐⭐ Moderate (3-5 years)",
                3: "⭐⭐⭐⭐ Good (6-10 years)",
                4: "⭐⭐⭐⭐⭐ Very Good (11-15 years)",
                5: "⭐⭐⭐⭐⭐ Excellent (15+ years)"
            }[x],
            help="Length and quality of credit history"
        )
    
    # Employment & Housing Information
    st.markdown('<p class="section-header">🏢 Employment & Housing</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        emp_length = st.selectbox(
            "Employment Length", 
            ["< 1 year", "1 year", "2 years", "3 years", "4 years", 
             "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"],
            index=5,
            help="Length of current employment"
        )
    
    with col2:
        home_ownership_n = st.selectbox(
            "Home Ownership Status", 
            ["RENT", "OWN", "MORTGAGE", "OTHER"],
            index=2,
            help="Current housing situation"
        )
    
    with col3:
        purpose = st.selectbox(
            "Loan Purpose", 
            ["debt_consolidation", "credit_card", "home_improvement", 
             "other", "major_purchase", "small_business", "car", 
             "medical", "moving", "vacation", "house", "wedding", "renewable_energy"],
            format_func=lambda x: x.replace("_", " ").title(),
            help="Primary purpose for the loan"
        )
    
    # Location Information
    st.markdown('<p class="section-header">📍 Location Information</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        addr_state = st.selectbox(
            "State", 
            ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI", 
             "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI", "OTHER"],
            index=0,
            help="State of residence"
        )
    
    with col2:
        zip_code = st.text_input(
            "ZIP Code", 
            value="940xx", 
            max_chars=5,
            help="First 3 digits + 'xx' (e.g., 940xx)"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Create loan data dictionary
    loan_data = {
        'revenue': revenue,
        'dti_n': dti_n,
        'loan_amnt': loan_amnt,
        'fico_n': fico_n,
        'experience_c': experience_c,
        'emp_length': emp_length,
        'purpose': purpose,
        'home_ownership_n': home_ownership_n,
        'addr_state': addr_state,
        'zip_code': zip_code,
    }
    
    # Submission Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Submit Application for Risk Assessment", type="primary", use_container_width=True):
            with st.spinner("⚙️ Analyzing application..."):
                result = predict_loan(loan_data, preprocessor, ensemble, escalation_system)
            display_results(result)


def render_sample_data_form(preprocessor, ensemble, escalation_system):
    """Render the sample data form"""
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    st.markdown("### 📋 Sample Loan Application")
    st.markdown("Review this pre-filled example application:")
    
    # Sample data
    sample_data = {
        'revenue': 50000,
        'dti_n': 15.5,
        'loan_amnt': 10000,
        'fico_n': 680.0,
        'experience_c': 5,
        'emp_length': '5 years',
        'purpose': 'debt_consolidation',
        'home_ownership_n': 'MORTGAGE',
        'addr_state': 'CA',
        'zip_code': '940xx',
    }
    
    # Display sample data in a nice format
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("💵 Loan Amount", f"${sample_data['loan_amnt']:,}")
        st.metric("💰 Annual Income", f"${sample_data['revenue']:,}")
        st.metric("📊 DTI Ratio", f"{sample_data['dti_n']}%")
        st.metric("📈 FICO Score", f"{sample_data['fico_n']:.0f}")
        st.metric("⭐ Credit Experience", sample_data['experience_c'])
    
    with col2:
        st.metric("🏢 Employment Length", sample_data['emp_length'])
        st.metric("🏠 Home Ownership", sample_data['home_ownership_n'])
        st.metric("🎯 Loan Purpose", sample_data['purpose'].replace('_', ' ').title())
        st.metric("📍 State", sample_data['addr_state'])
        st.metric("📮 ZIP Code", sample_data['zip_code'])
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Analyze Sample Application", type="primary", use_container_width=True):
            with st.spinner("⚙️ Analyzing application..."):
                result = predict_loan(sample_data, preprocessor, ensemble, escalation_system)
            display_results(result)


def render_csv_upload_form(preprocessor, ensemble, escalation_system):
    """Render the CSV upload form"""
    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    st.markdown("### 📤 Batch Application Processing")
    st.markdown("Upload a CSV file containing multiple loan applications for batch processing.")
    
    st.info("""
    **Required CSV Format:**
    - Columns: `revenue`, `dti_n`, `loan_amnt`, `fico_n`, `experience_c`, `emp_length`, `purpose`, `home_ownership_n`, `addr_state`, `zip_code`
    - One application per row
    - Use the example file as a template
    """)
    
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload your loan applications file"
    )
    
    if uploaded_file is not None:
        # Read CSV
        loans_df = pd.read_csv(uploaded_file)
        
        st.success(f"✅ Successfully loaded {len(loans_df)} applications")
        
        with st.expander("📄 Preview Data"):
            st.dataframe(loans_df.head(10), use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(f"🔍 Analyze All {len(loans_df)} Applications", type="primary", use_container_width=True):
                with st.spinner(f"⚙️ Processing {len(loans_df)} applications..."):
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, row in loans_df.iterrows():
                        result = predict_loan(row.to_dict(), preprocessor, ensemble, escalation_system)
                        results.append(result)
                        progress_bar.progress((i + 1) / len(loans_df))
                    
                    display_batch_results(results)
    else:
        st.markdown('</div>', unsafe_allow_html=True)


def display_results(result):
    """Display prediction results"""
    
    if 'error' in result:
        st.error(f"❌ Error: {result['error']}")
        return
    
    st.markdown("---")
    st.markdown("## 📊 Risk Assessment Results")
    
    # Decision box with professional styling
    if result['should_escalate']:
        st.markdown(f"""
        <div class="decision-box escalate-box">
            <h2>⚠️ MANUAL REVIEW REQUIRED</h2>
            <p class="decision-text"><strong>Status:</strong> {result['decision']}</p>
            <p class="decision-text"><strong>Reason:</strong> {result['escalation_reason']}</p>
            <div class="status-badge badge-warning">PENDING HUMAN REVIEW</div>
            <p style="margin-top: 1rem; color: #666;">This application requires review by a loan officer due to uncertainty in the automated assessment.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        if result['decision'] == 'APPROVE':
            st.markdown(f"""
            <div class="decision-box approve-box">
                <h2>✅ APPLICATION APPROVED</h2>
                <p class="decision-text"><strong>Status:</strong> Automated {result['decision']}</p>
                <div class="status-badge badge-success">APPROVED - NO REVIEW NEEDED</div>
                <p style="margin-top: 1rem; color: #666;">The AI model is highly confident in this approval decision. The loan can be processed immediately.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="decision-box reject-box">
                <h2>❌ APPLICATION DECLINED</h2>
                <p class="decision-text"><strong>Status:</strong> Automated {result['decision']}</p>
                <div class="status-badge badge-danger">DECLINED - NO REVIEW NEEDED</div>
                <p style="margin-top: 1rem; color: #666;">The AI model is highly confident in this rejection decision based on risk factors.</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Metrics in professional cards
    st.markdown("### 📈 Detailed Risk Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Default Risk</div>
            <div class="metric-value">{result['prob_default']:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%);">
            <div class="metric-label">Payment Probability</div>
            <div class="metric-value">{result['prob_paid']:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="metric-label">Confidence Level</div>
            <div class="metric-value">{result['confidence']:.1%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="metric-label">Uncertainty Score</div>
            <div class="metric-value">{result['uncertainty']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualizations
    st.markdown("### 📊 Visual Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gauge chart
        fig_gauge = create_gauge_chart(result['prob_default'], "Default Risk Probability")
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        # Uncertainty distribution
        fig_uncertainty = create_uncertainty_chart(result['all_proba'])
        st.plotly_chart(fig_uncertainty, use_container_width=True)
    
    # Additional insights
    with st.expander("💡 Risk Assessment Details"):
        st.markdown("""
        **How This Assessment Works:**
        
        - **Default Risk**: Probability that the borrower will default on the loan
        - **Confidence Level**: How certain the AI model is about its prediction
        - **Uncertainty Score**: Measure of disagreement among ensemble models (lower is better)
        - **Manual Review**: Triggered when confidence is low or uncertainty is high
        
        This system uses an ensemble of 100+ machine learning models to provide robust predictions.
        """)
    
    # Risk factors explanation
    if result['prob_default'] > 0.5:
        with st.expander("⚠️ Key Risk Factors"):
            st.warning("""
            High default probability detected. This may be due to:
            - High debt-to-income ratio
            - Lower credit score
            - Limited credit history
            - Recent inquiries or delinquencies
            """)
    elif result['uncertainty'] > escalation_system.uncertainty_threshold:
        with st.expander("🔍 Uncertainty Analysis"):
            st.info("""
            High uncertainty detected in prediction. This typically occurs when:
            - The applicant profile is unusual or rare
            - Multiple risk factors are borderline
            - The application has conflicting indicators
            
            Human review is recommended for these cases.
            """)


def display_batch_results(results):
    """Display batch prediction results"""
    
    st.markdown("---")
    st.markdown("## 📊 Batch Processing Results")
    
    # Summary statistics
    n_total = len(results)
    n_escalated = sum(1 for r in results if r['should_escalate'])
    n_automated = n_total - n_escalated
    n_approve = sum(1 for r in results if not r['should_escalate'] and r['decision'] == 'APPROVE')
    n_reject = sum(1 for r in results if not r['should_escalate'] and r['decision'] == 'REJECT')
    
    # Summary metrics in cards
    st.markdown("### 📈 Processing Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
            <div class="metric-label">Total Applications</div>
            <div class="metric-value">{n_total}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #43cea2 0%, #185a9d 100%);">
            <div class="metric-label">Automated Decisions</div>
            <div class="metric-value">{n_automated}</div>
            <div style="font-size: 0.9rem; margin-top: 0.5rem;">{n_automated/n_total*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <div class="metric-label">Manual Review</div>
            <div class="metric-value">{n_escalated}</div>
            <div style="font-size: 0.9rem; margin-top: 0.5rem;">{n_escalated/n_total*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <div class="metric-label">Automation Rate</div>
            <div class="metric-value">{n_automated/n_total*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Breakdown Chart
    st.markdown("### 📊 Decision Distribution")
    
    breakdown_data = pd.DataFrame({
        'Decision': ['✅ Automated Approve', '❌ Automated Reject', '⚠️ Manual Review'],
        'Count': [n_approve, n_reject, n_escalated],
        'Percentage': [n_approve/n_total*100, n_reject/n_total*100, n_escalated/n_total*100]
    })
    
    fig = px.bar(
        breakdown_data, 
        x='Decision', 
        y='Count', 
        text='Percentage',
        color='Decision',
        color_discrete_map={
            '✅ Automated Approve': '#28a745',
            '❌ Automated Reject': '#fd7e14',
            '⚠️ Manual Review': '#dc3545'
        },
        title="Applications by Decision Type"
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(
        showlegend=False, 
        height=400,
        font=dict(family="Inter, sans-serif"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Results table with better styling
    st.markdown("### 📋 Detailed Application Results")
    
    results_df = pd.DataFrame([{
        'App #': i+1,
        'Decision': r['decision'],
        'Action': r['action'],
        'Default Risk': f"{r['prob_default']:.2%}",
        'Confidence': f"{r['confidence']:.2%}",
        'Uncertainty': f"{r['uncertainty']:.4f}",
        'Escalation Reason': r.get('escalation_reason', 'N/A')
    } for i, r in enumerate(results)])
    
    # Color code the dataframe
    def highlight_decision(row):
        if row['Decision'] == 'APPROVE':
            return ['background-color: #d4edda'] * len(row)
        elif row['Decision'] == 'REJECT':
            return ['background-color: #fff3cd'] * len(row)
        else:
            return ['background-color: #f8d7da'] * len(row)
    
    styled_df = results_df.style.apply(highlight_decision, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=400)
    
    # Download section
    st.markdown("### 💾 Export Results")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        csv = results_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"loan_assessment_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    # Summary insights
    with st.expander("📊 Batch Processing Insights"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Approval Metrics:**")
            st.metric("Approved", n_approve)
            st.metric("Approval Rate", f"{n_approve/n_total*100:.1f}%")
        
        with col2:
            st.markdown("**Risk Metrics:**")
            st.metric("Rejected", n_reject)
            st.metric("Escalation Rate", f"{n_escalated/n_total*100:.1f}%")
        
        avg_default_risk = np.mean([r['prob_default'] for r in results])
        avg_uncertainty = np.mean([r['uncertainty'] for r in results])
        
        st.markdown("---")
        st.markdown("**Portfolio Analysis:**")
        st.metric("Average Default Risk", f"{avg_default_risk:.2%}")
        st.metric("Average Uncertainty", f"{avg_uncertainty:.4f}")
        
        if avg_default_risk > 0.5:
            st.warning("⚠️ High average default risk detected in this batch. Consider additional review.")
        else:
            st.success("✅ Overall portfolio shows acceptable risk levels.")


if __name__ == "__main__":
    main()
