import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import norm
from datetime import datetime

# --- Constants: Bayesian Parameters from PDF [cite: 61] ---
NUCLEI_PARAMS = {
    "13C":    {"exp_c": 0.547, "sig_c": 0.253, "exp_i": -0.487, "sig_i": 0.533},
    "1H":     {"exp_c": 0.478, "sig_c": 0.305, "exp_i": -0.786, "sig_i": 0.835},
    "13C+1H": {"exp_c": 0.512, "sig_c": 0.209, "exp_i": -0.637, "sig_i": 0.499}
}

# --- Page Configuration ---
st.set_page_config(page_title="CP3-Bayes Analyzer v3.4.1", layout="wide")
st.title("🧪 Advanced CP3-Bayes Analyzer Ver. 3.4.1")
st.markdown("##### *Abe-lab Official Statistical Platform for Structural Determination*")

# Initialize session state for tracking analysis
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'final_df' not in st.session_state:
    st.session_state.final_df = None
if 'summary_df' not in st.session_state:
    st.session_state.summary_df = None

# --- Abe-lab's 6 Crucial Points (Notes) ---
with st.expander("📝 Abe-lab's 6 Crucial Points (Check before analysis)"):
    st.markdown("""
    1. **Pairing Issues**: Verify ID matching (e.g., conf01) in BRIDGE v1.0.
    2. **Energy Extraction**: Ensure 'Normal termination' and ΔG exist in logs.
    3. **Atom Count**: Match CSV rows with the Gaussian atom numbering.
    4. **Probability Near 50%**: Check for minimal structural difference or input errors.
    5. **Low CP3 Score**: Verify Scaling Factors or conformational search depth.
    6. **Error Bypass**: The tool automatically substitutes $0.0001$ for Δ=0.
    """)

# --- UI: Dual-Window File Uploader ---
st.subheader("📁 Phase 1: Data Input")
col_exp, col_calc = st.columns(2)

with col_exp:
    st.info("**Window 1: Experimental Data**")
    exp_file = st.file_uploader("Upload Experimental Peak List (CSV)", type="csv")

with col_calc:
    st.info("**Window 2: Calculated Data**")
    calc_file = st.file_uploader("Upload Boltzmann-Averaged Shifts (CSV)", type="csv")

# Analysis Readiness
ready_to_run = exp_file is not None and calc_file is not None

# --- Analysis Engine [cite: 12-63] ---
def run_stat_analysis(df, n_key):
    p = NUCLEI_PARAMS[n_key]
    
    # 1. Delta Calculation & Zero-Division Mitigation [cite: 16-17, 24]
    de = (df['Exp_Shift_A'] - df['Exp_Shift_B']).replace(0, 0.0001)
    dc = (df['Calc_A_Scaled'] - df['Calc_B_Scaled']).replace(0, 0.0001)

    # 2. f3 Function Branching Logic 
    f3_c = np.where(dc/de > 1, (de**3)/dc, de*dc)
    f3_i = np.where((-dc)/de > 1, (de**3)/(-dc), de*(-dc))

    # 3. CP3 Score Calculation [cite: 26-27]
    sum_de2 = np.sum(de**2)
    cp3_c, cp3_i = np.sum(f3_c)/sum_de2, np.sum(f3_i)/sum_de2
    
    # 4. Bayesian Transformation (1 - NORM.DIST) [cite: 57-60]
    pr1_ac1 = 1 - norm.cdf(cp3_c, p['exp_c'], p['sig_c'])
    pr2_ac1 = 1 - norm.cdf(cp3_i, p['exp_i'], p['sig_i'])
    pr1_ac2 = 1 - norm.cdf(cp3_c, p['exp_i'], p['sig_i'])
    pr2_ac2 = 1 - norm.cdf(cp3_i, p['exp_c'], p['sig_c'])
    
    # Final Bayesian Probability [cite: 63]
    prob = (pr1_ac1 * pr2_ac1) / ((pr1_ac1 * pr2_ac1) + (pr1_ac2 * pr2_ac2)) * 100
    return cp3_c, cp3_i, prob

# --- Action Section ---
st.divider()
if st.button("🚀 Run Triple-Mode Analysis", use_container_width=True, disabled=not ready_to_run):
    try:
        df_exp = pd.read_csv(exp_file)
        df_calc = pd.read_csv(calc_file)
        
        merged = pd.merge(df_exp, df_calc, on="Atom_Label")
        df_c = merged[merged['Atom_Label'].str.contains('C', na=False)]
        df_h = merged[merged['Atom_Label'].str.contains('H', na=False)]
        
        # 13C Analysis
        c_c, c_i, c_p = run_stat_analysis(df_c, "13C")
        # 1H Analysis
        h_c, h_i, h_p = run_stat_analysis(df_h, "1H")
        
        # 13C+1H Mixed Mode [cite: 25]
        m_c, m_i = (c_c + h_c)/2, (c_i + h_i)/2
        p_m = NUCLEI_PARAMS["13C+1H"]
        pr1_ac1_m = 1-norm.cdf(m_c, p_m['exp_c'], p_m['sig_c'])
        pr2_ac1_m = 1-norm.cdf(m_i, p_m['exp_i'], p_m['sig_i'])
        pr1_ac2_m = 1-norm.cdf(m_c, p_m['exp_i'], p_m['sig_i'])
        pr2_ac2_m = 1-norm.cdf(m_i, p_m['exp_c'], p_m['sig_c'])
        m_p = (pr1_ac1_m * pr2_ac1_m) / ((pr1_ac1_m * pr2_ac1_m) + (pr1_ac2_m * pr2_ac2_m)) * 100

        # Store results
        st.session_state.summary_df = pd.DataFrame({
            "Analysis Mode": ["13C Only", "1H Only", "13C + 1H Mixed"],
            "Bayesian Prob. (A=a)": [f"{c_p:.2f}%", f"{h_p:.2f}%", f"{m_p:.2f}%"],
            "CP3 Correct": [f"{c_c:.4f}", f"{h_c:.4f}", f"{m_c:.4f}"],
            "CP3 Incorrect": [f"{c_i:.4f}", f"{h_i:.4f}", f"{m_i:.4f}"]
        })
        
        merged['de'] = (merged['Exp_Shift_A'] - merged['Exp_Shift_B']).replace(0, 0.0001)
        merged['dc'] = (merged['Calc_A_Scaled'] - merged['Calc_B_Scaled']).replace(0, 0.0001)
        st.session_state.final_df = merged
        st.session_state.analyzed = True

    except Exception as e:
        st.error(f"Error: {str(e)}")

# --- Result Display ---
if st.session_state.analyzed:
    st.subheader("📊 Statistical Analysis Summary")
    st.table(st.session_state.summary_df)

    st.subheader("📈 JNP 2024 Correlation Plot (Δδ Analysis)")
    fig = px.scatter(st.session_state.final_df, x="de", y="dc", text="Atom_Label", color="Atom_Label",
                     labels={"de": "Δδ Experimental (ppm)", "dc": "Δδ Calculated (ppm)"},
                     template="plotly_white")
    fig.add_shape(type="line", x0=-1, y0=-1, x1=1, y1=1, line=dict(color="Red", dash="dash"))
    st.plotly_chart(fig, use_container_width=True)

# --- Download Section ---
st.divider()
if st.session_state.analyzed:
    csv_data = st.session_state.final_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="💾 Export Integrated SI Data (CSV)",
        data=csv_data,
        file_name=f"SI_Report_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        use_container_width=True
    )
else:
    st.button("💾 Export Integrated SI Data (CSV)", disabled=True, use_container_width=True)