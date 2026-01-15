import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import norm
from datetime import datetime
import re

# --- Constants: JNP 2024 Parameters ---
NUCLEI_PARAMS = {
    "13C":    {"exp_c": 0.547, "sig_c": 0.253, "exp_i": -0.487, "sig_i": 0.533},
    "1H":     {"exp_c": 0.478, "sig_c": 0.305, "exp_i": -0.786, "sig_i": 0.835},
    "13C+1H": {"exp_c": 0.512, "sig_c": 0.209, "exp_i": -0.637, "sig_i": 0.499}
}

st.set_page_config(page_title="CP3-Bayes Analyzer v3.4.2", layout="wide")
st.title("🧪 Advanced CP3-Bayes Analyzer Ver. 3.4.2")
st.markdown("##### *Abe-lab Official Platform - Triple-Mode Integration*")

# Initialize session state
if 'results' not in st.session_state: st.session_state.results = None
if 'analyzed' not in st.session_state: st.session_state.analyzed = False

# --- Phase 1: Dual-Window Input (Fixed UI) ---
st.subheader("📁 Phase 1: Dual-Window Data Input")
col_exp, col_calc = st.columns(2)

with col_exp:
    st.info("**Window 1: Experimental Data**")
    exp_file = st.file_uploader("Upload Experimental Peak List (CSV)", type="csv", key="exp_up")

with col_calc:
    st.info("**Window 2: Calculated Data**")
    calc_file = st.file_uploader("Upload Calculated Shifts from BRIDGE (CSV)", type="csv", key="calc_up")

# --- Phase 2: Action Section ---
st.divider()
st.subheader("🚀 Phase 2: Analysis")

# Check if both files are uploaded
files_ready = (exp_file is not None) and (calc_file is not None)

def run_analysis_logic(df, n_key):
    p = NUCLEI_PARAMS[n_key]
    # CP3 Core Calculation
    de = (df['Exp_Shift_A'] - df['Exp_Shift_B']).replace(0, 0.0001)
    dc = (df['Calc_A_Scaled'] - df['Calc_B_Scaled']).replace(0, 0.0001)
    
    f3_c = np.where(dc/de > 1, (de**3)/dc, de*dc)
    f3_i = np.where((-dc)/de > 1, (de**3)/(-dc), de*(-dc))
    
    sum_de2 = np.sum(de**2)
    c_cor, c_inc = np.sum(f3_c)/sum_de2, np.sum(f3_i)/sum_de2
    
    # Bayesian Probability Calculation
    p1 = 1 - norm.cdf(c_cor, p['exp_c'], p['sig_c'])
    p2 = 1 - norm.cdf(c_inc, p['exp_i'], p['sig_i'])
    p3 = 1 - norm.cdf(c_cor, p['exp_i'], p['sig_i'])
    p4 = 1 - norm.cdf(c_inc, p['exp_c'], p['sig_c'])
    
    prob = (p1 * p2) / ((p1 * p2) + (p3 * p4)) * 100
    return c_cor, c_inc, prob

if st.button("Run Triple-Mode Analysis", use_container_width=True, disabled=not files_ready):
    try:
        df_exp = pd.read_csv(exp_file)
        df_calc = pd.read_csv(calc_file)
        merged = pd.merge(df_exp, df_calc, on="Atom_Label")
        
        # Nuclei Filtering & Analysis
        c_df = merged[merged['Atom_Label'].str.contains('C', na=False)]
        h_df = merged[merged['Atom_Label'].str.contains('H', na=False)]
        
        c_cor, c_inc, c_p = run_analysis_logic(c_df, "13C") if not c_df.empty else (0,0,0)
        h_cor, h_inc, h_p = run_analysis_logic(h_df, "1H") if not h_df.empty else (0,0,0)
        
        # Mixed Mode Calculation
        m_cor, m_inc = (c_cor + h_cor)/2, (c_inc + h_inc)/2
        pm = NUCLEI_PARAMS["13C+1H"]
        m1 = 1 - norm.cdf(m_cor, pm['exp_c'], pm['sig_c'])
        m2 = 1 - norm.cdf(m_inc, pm['exp_i'], pm['sig_i'])
        m3 = 1 - norm.cdf(m_cor, pm['exp_i'], pm['sig_i'])
        m4 = 1 - norm.cdf(m_inc, pm['exp_c'], pm['sig_c'])
        m_p = (m1 * m2) / ((m1 * m2) + (m3 * m4)) * 100
        
        st.session_state.results = {
            "summary": pd.DataFrame({
                "Mode": ["13C Only", "1H Only", "13C + 1H Mixed"],
                "Probability (A=a)": [f"{c_p:.2f}%", f"{h_p:.2f}%", f"{m_p:.2f}%"],
                "CP3 (Correct)": [f"{c_cor:.4f}", f"{h_cor:.4f}", f"{m_cor:.4f}"],
                "CP3 (Incorrect)": [f"{c_inc:.4f}", f"{h_inc:.4f}", f"{m_inc:.4f}"]
            }),
            "data": merged
        }
        st.session_state.analyzed = True
    except Exception as e:
        st.error(f"Analysis Error: Please check CSV headers. Details: {e}")

# Results and Plotting
if st.session_state.analyzed:
    st.success("Analysis Complete")
    st.table(st.session_state.results["summary"])
    
    # JNP 2024 Correlation Plot
    df_plot = st.session_state.results["data"]
    df_plot['de'] = df_plot['Exp_Shift_A'] - df_plot['Exp_Shift_B']
    df_plot['dc'] = df_plot['Calc_A_Scaled'] - df_plot['Calc_B_Scaled']
    
    fig = px.scatter(df_plot, x="de", y="dc", text="Atom_Label", color="Atom_Label",
                     labels={"de": "Δδ Experimental (A-B)", "dc": "Δδ Calculated (a-b)"},
                     title="Structural Correlation Plot (JNP 2024 Style)")
    fig.add_shape(type="line", x0=min(df_plot['de']), y0=min(df_plot['de']), 
                  x1=max(df_plot['de']), y1=max(df_plot['de']),
                  line=dict(color="Red", dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

# Export Section
st.divider()
if st.session_state.analyzed:
    csv = st.session_state.results["data"].to_csv(index=False).encode('utf-8')
    st.download_button("💾 Export Detailed SI Report (CSV)", data=csv, file_name="CP3_Analysis_Report.csv", use_container_width=True)
else:
    st.button("💾 Export Detailed SI Report (CSV)", disabled=True, use_container_width=True)