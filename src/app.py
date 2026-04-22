import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import requests
import time
from streamlit_lottie import st_lottie

# Ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_ingestion import load_data
from predict import predict_churn

# Configure the Streamlit page
st.set_page_config(page_title="Telco Churn Analytics", page_icon="📈", layout="wide")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load a free public Lottie animation for "Data Processing / AI"
lottie_ai = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_V9t630.json")

def get_human_readable_driver(driver, input_dict):
    """Converts raw SHAP feature names into human-readable business insights."""
    if driver == 'tenure':
        val = input_dict['tenure'][0]
        if val <= 12:
            return f"Short tenure ({val} months) means the customer isn't fully locked in yet."
        else:
            return f"Tenure of {val} months is a major deciding factor."
    elif driver == 'TotalCharges' or driver == 'MonthlyCharges':
        val = input_dict[driver][0]
        if float(val) > 75:
            return f"High {driver.replace('Charges', ' Charges')} (${val}) increases the risk of leaving."
        else:
            return f"The {driver.replace('Charges', ' Charges')} (${val}) strongly influenced this score."
    elif 'TechSupport' in driver:
        if input_dict['TechSupport'][0] == 'No':
            return "Lack of Tech Support makes them vulnerable to unresolved issues."
        else:
            return "Having Tech Support generally keeps customers happy."
    elif 'Contract' in driver:
        val = input_dict['Contract'][0]
        if val == 'Month-to-month':
            return "Month-to-month contract allows them to leave at any time."
        else:
            return f"A {val} contract provides good stability."
    elif 'InternetService' in driver:
        val = input_dict['InternetService'][0]
        return f"Using {val} internet service is a key historical indicator."
    elif 'PaymentMethod' in driver:
        val = input_dict['PaymentMethod'][0]
        if 'automatic' not in val.lower():
            return "Using a manual payment method makes it easier to cancel."
        else:
            return "Automatic payments usually improve retention."
    elif 'OnlineSecurity' in driver or 'OnlineBackup' in driver:
        service = "Cybersecurity Pack" if 'OnlineSecurity' in driver else "Cloud Backup"
        base_feature = 'OnlineSecurity' if 'OnlineSecurity' in driver else 'OnlineBackup'
        if input_dict[base_feature][0] == 'No':
            return f"Not having the {service} reduces their dependency on us."
        else:
            return f"Having the {service} increases their stickiness."
    else:
        return f"The '{driver}' factor strongly influenced this prediction."

# ==========================================
# CUSTOM CSS FOR ANIMATIONS & DARK MODE
# ==========================================
st.markdown("""
<style>
    /* Card aesthetics with Hover Animation */
    .pbi-card {
        background-color: #262730;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        text-align: center;
        margin-bottom: 20px;
        border-top: 4px solid #3B82F6;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .pbi-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 20px -5px rgba(0, 0, 0, 0.5);
    }
    
    .pbi-card-red {
        border-top: 4px solid #EF4444;
    }
    
    .pbi-card-green {
        border-top: 4px solid #10B981;
    }
    
    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 40px 20px;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    
    .hero-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 10px;
        letter-spacing: 1px;
    }
    
    .hero-subtitle {
        font-size: 18px;
        font-weight: 300;
        opacity: 0.9;
    }
    
    /* Typography */
    .kpi-title {
        color: #9CA3AF;
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 5px;
    }
    
    .kpi-value {
        color: #F9FAFB;
        font-size: 32px;
        font-weight: bold;
        margin: 0;
    }
    
    .kpi-value-red {
        color: #FCA5A5;
    }
    
    .kpi-value-green {
        color: #6EE7B7;
    }
    
    /* Custom Tab Styling */
    button[data-baseweb="tab"] {
        font-size: 16px !important;
        font-weight: 600 !important;
        color: #9CA3AF !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.3s ease !important;
    }
    
    /* Unique Highlight for Active Tab */
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #3B82F6 !important;
        border-bottom: 3px solid #3B82F6 !important;
        background: linear-gradient(0deg, rgba(59, 130, 246, 0.1) 0%, transparent 100%) !important;
    }
    
    button[data-baseweb="tab"]:hover {
        color: #F9FAFB !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class='hero-banner'>
    <div class='hero-title'>🌟 Telecom Customer Retention Hub</div>
    <div class='hero-subtitle'>Smart AI Insights to predict risk and improve customer loyalty</div>
</div>
""", unsafe_allow_html=True)

# Navigation
tab1, tab2 = st.tabs(["Predict Customer Risk", "Overall Performance Dashboard"])

# ==========================================
# TAB 1: CUSTOMER SUCCESS AGENT (ML PREDICTOR)
# ==========================================
with tab1:
    st.markdown("""
        <h2 style='background: linear-gradient(90deg, #F59E0B 0%, #EF4444 100%); 
                   -webkit-background-clip: text; 
                   -webkit-text-fill-color: transparent; 
                   font-weight: 800; 
                   font-size: 36px;
                   margin-bottom: 5px;'>
            Predict Customer Risk
        </h2>
    """, unsafe_allow_html=True)
    st.write("Fill in the customer details below to instantly see if they are at risk of leaving.")
    
    # Placeholder for Results so they appear above the form
    results_placeholder = st.empty()
    
    with st.form("churn_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            tenure = st.slider("Tenure (Months)", 0, 72, 12)
            internet_service = st.selectbox("Internet Service", ["Standard Broadband", "Premium Fiber Optic", "Voice Only (No Data)"])
            online_security = st.selectbox("Cybersecurity Pack", ["No", "Yes"])
            
        with col2:
            monthly_charges = st.number_input("Monthly Charges ($)", value=50.0)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes"])
            
        with col3:
            payment_method_ui = st.selectbox("Payment Method", [
                "Online Bank Debit (eCheck)", 
                "Mailed check", 
                "Auto-Pay (Bank Account)", 
                "Auto-Pay (Credit Card)"
            ])
            online_backup = st.selectbox("Cloud Backup Add-on", ["No", "Yes"])
            senior_citizen_ui = st.selectbox("Senior Citizen", ["No", "Yes"])
            
        total_charges = str(monthly_charges * max(tenure, 1))
        
        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button(label="🔮 Check Risk Score")
        
    if submit_button:
        # Convert Yes/No back to 1/0 for the ML Model
        senior_citizen_val = 1 if senior_citizen_ui == "Yes" else 0
        
        # Map internet service
        internet_mapping = {
            "Standard Broadband": "DSL",
            "Premium Fiber Optic": "Fiber optic",
            "Voice Only (No Data)": "No"
        }
        internet_service_val = internet_mapping[internet_service]
        
        # Map modern payment method names back to original for ML Model
        payment_mapping = {
            "Online Bank Debit (eCheck)": "Electronic check",
            "Mailed check": "Mailed check",
            "Auto-Pay (Bank Account)": "Bank transfer (automatic)",
            "Auto-Pay (Credit Card)": "Credit card (automatic)"
        }
        payment_method_val = payment_mapping[payment_method_ui]
        
        input_data = {
            'SeniorCitizen': [senior_citizen_val],
            'tenure': [tenure],
            'InternetService': [internet_service_val],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'TechSupport': [tech_support],
            'Contract': [contract],
            'PaperlessBilling': ['Yes'],  # Defaulting as we removed it from UI to save space
            'PaymentMethod': [payment_method_val],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        }
        
        df_input = pd.DataFrame(input_data)
        
        with results_placeholder.container():
            st.markdown("---")
            st.markdown("### 📊 Results")
            
            # Create an empty placeholder for the animation
            anim_placeholder = st.empty()
            
            # Show Lottie animation
            if lottie_ai:
                with anim_placeholder:
                    st.markdown("<h4 style='text-align: center; color: #9CA3AF;'>Analyzing customer profile...</h4>", unsafe_allow_html=True)
                    st_lottie(lottie_ai, height=200, key="loading")
                    time.sleep(2.5) # Artificial delay for premium UX feel
                    
            # Clear animation
            anim_placeholder.empty()
            
            with st.spinner('Finalizing risk score...'):
                try:
                    results = predict_churn(
                        df_input, 
                        model_path="xgboost_model.pkl", 
                        scaler_path="scaler.pkl", 
                        feature_cols_path="feature_columns.pkl"
                    )
                    
                    res = results[0]
                    prob = res['Probability'] * 100
                    drivers = res['Top_Churn_Drivers']
                    action = res['Recommended_Action']
                    
                    col_res1, col_res2 = st.columns(2)
                    
                    with col_res1:
                        fig = go.Figure(go.Indicator(
                            mode = "gauge+number",
                            value = prob,
                            title = {'text': "Churn Probability (%)", 'font': {'size': 20, 'color': '#9CA3AF'}},
                            gauge = {
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "#EF4444" if prob > 50 else "#10B981"},
                                'steps': [
                                    {'range': [0, 50], 'color': "rgba(16, 185, 129, 0.1)"},
                                    {'range': [50, 100], 'color': "rgba(239, 68, 68, 0.1)"}
                                ],
                                'threshold': {
                                    'line': {'color': "white", 'width': 4},
                                    'thickness': 0.75,
                                    'value': prob
                                }
                            }
                        ))
                        fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#F9FAFB"})
                        st.plotly_chart(fig, use_container_width=True)
                        
                    with col_res2:
                        st.markdown("""
                        <div class='pbi-card'>
                            <h4 style='color:#3B82F6; margin-bottom:15px;'>🔍 Key Reasons</h4>
                        """, unsafe_allow_html=True)
                        
                        seen_reasons = []
                        for driver in drivers:
                            human_text = get_human_readable_driver(driver, input_data)
                            if human_text not in seen_reasons:
                                seen_reasons.append(human_text)
                                
                        for idx, text in enumerate(seen_reasons):
                            st.markdown(f"**{idx+1}. {text}**")
                            
                        st.markdown("</div>", unsafe_allow_html=True)
                            
                        st.markdown("#### 💡 Suggested Next Steps")
                        if prob > 80:
                            st.error(action)
                        elif prob > 50:
                            st.warning(action)
                        else:
                            st.success(action)
                except Exception as e:
                    st.error(f"Prediction failed. Ensure models are trained. Error: {e}")

# ==========================================
# TAB 2: DATA ANALYST VIEW (POWER BI STYLE)
# ==========================================
with tab2:
    # Fetch Data
    @st.cache_data
    def fetch_data():
        try:
            return load_data()
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return pd.DataFrame()
            
    df = fetch_data()
    
    if not df.empty:
        # Data Preprocessing for BI (ensure Churn is a string for plotting if it was converted)
        if df['Churn'].dtype != 'O':
            df['Churn_Label'] = df['Churn'].map({1: 'Yes', 0: 'No'})
        else:
            df['Churn_Label'] = df['Churn']
            
        # --- SLICERS (FILTERS) ---
        st.markdown("""
            <h2 style='background: linear-gradient(90deg, #10B981 0%, #3B82F6 100%); 
                       -webkit-background-clip: text; 
                       -webkit-text-fill-color: transparent; 
                       font-weight: 800; 
                       font-size: 30px;
                       margin-bottom: 5px;'>
                Filters (Slicers)
            </h2>
        """, unsafe_allow_html=True)
        col_f1, col_f2, col_f3 = st.columns(3)
        
        with col_f1:
            contract_filter = st.multiselect("Contract Type", options=df['Contract'].unique(), default=df['Contract'].unique())
        with col_f2:
            internet_filter = st.multiselect("Internet Service", options=df['InternetService'].unique(), default=df['InternetService'].unique())
        with col_f3:
            payment_filter = st.multiselect("Payment Method", options=df['PaymentMethod'].unique(), default=df['PaymentMethod'].unique())
            
        # Apply Filters
        filtered_df = df[
            (df['Contract'].isin(contract_filter)) &
            (df['InternetService'].isin(internet_filter)) &
            (df['PaymentMethod'].isin(payment_filter))
        ]
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- KPI CARDS ROW ---
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        
        total_customers = len(filtered_df)
        churned_df = filtered_df[filtered_df['Churn_Label'] == 'Yes']
        retained_df = filtered_df[filtered_df['Churn_Label'] == 'No']
        
        churn_rate = (len(churned_df) / total_customers * 100) if total_customers > 0 else 0
        lost_mrr = churned_df['MonthlyCharges'].sum()
        retained_mrr = retained_df['MonthlyCharges'].sum()
        
        with kpi1:
            st.markdown(f"""
                <div class='pbi-card'>
                    <p class='kpi-title'>Total Customers</p>
                    <p class='kpi-value'>{total_customers:,}</p>
                </div>
            """, unsafe_allow_html=True)
            
        with kpi2:
            st.markdown(f"""
                <div class='pbi-card pbi-card-red'>
                    <p class='kpi-title'>Churn Rate</p>
                    <p class='kpi-value kpi-value-red'>{churn_rate:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
        with kpi3:
            st.markdown(f"""
                <div class='pbi-card pbi-card-red'>
                    <p class='kpi-title'>Lost MRR (Monthly)</p>
                    <p class='kpi-value kpi-value-red'>${lost_mrr:,.0f}</p>
                </div>
            """, unsafe_allow_html=True)
            
        with kpi4:
            st.markdown(f"""
                <div class='pbi-card pbi-card-green'>
                    <p class='kpi-title'>Retained MRR</p>
                    <p class='kpi-value kpi-value-green'>${retained_mrr:,.0f}</p>
                </div>
            """, unsafe_allow_html=True)
            
        # --- CHARTS ROW 1 (DONUTS) ---
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            fig_churn = px.pie(filtered_df, names='Churn_Label', title='Customer Churn Breakdown', 
                               hole=0.6, color='Churn_Label', color_discrete_map={'No':'#10B981', 'Yes':'#EF4444'})
            fig_churn.update_traces(textposition='inside', textinfo='percent+label')
            fig_churn.update_layout(margin=dict(t=40, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#F9FAFB'))
            st.plotly_chart(fig_churn, use_container_width=True)
            
        with col_c2:
            fig_contract = px.pie(filtered_df, names='Contract', title='Customers by Contract Type', 
                               hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_contract.update_traces(textposition='inside', textinfo='percent+label')
            fig_contract.update_layout(margin=dict(t=40, b=0, l=0, r=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#F9FAFB'))
            st.plotly_chart(fig_contract, use_container_width=True)
            
        # --- CHARTS ROW 2 (BAR & BOX) ---
        col_c3, col_c4 = st.columns(2)
        
        with col_c3:
            # Create Tenure Groups
            bins = [0, 12, 24, 48, 60, 100]
            labels = ['0-1 Year', '1-2 Years', '2-4 Years', '4-5 Years', '5+ Years']
            filtered_df['Tenure_Group'] = pd.cut(filtered_df['tenure'], bins=bins, labels=labels, right=False)
            
            tenure_churn = filtered_df.groupby(['Tenure_Group', 'Churn_Label'], observed=False).size().reset_index(name='Count')
            fig_tenure = px.bar(tenure_churn, x='Tenure_Group', y='Count', color='Churn_Label', barmode='group',
                                title='Churn by Tenure Group', color_discrete_map={'No':'#10B981', 'Yes':'#EF4444'})
            fig_tenure.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#F9FAFB'))
            st.plotly_chart(fig_tenure, use_container_width=True)
            
        with col_c4:
            fig_box = px.box(filtered_df, x='Churn_Label', y='MonthlyCharges', color='Churn_Label',
                             title='Monthly Charges Impact on Churn', color_discrete_map={'No':'#10B981', 'Yes':'#EF4444'})
            fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#F9FAFB'))
            st.plotly_chart(fig_box, use_container_width=True)

    else:
        st.warning("No data found. Please run setup_database.py first.")
