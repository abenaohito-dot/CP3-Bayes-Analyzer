import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

# --- Constants: JNP 2024 Parameters ---
NUCLEI_PARAMS = {
    "13C":    {"exp_c": 0.547, "sig_c": 0.253, "exp_i": -0.487, "sig_i": 0.533},
    "1H":     {"exp_c": 0.478, "sig_c": 0.305, "exp_i": -0.786, "sig_i": 0.835},
    "13C+1H": {"exp_c": 0.512, "sig_c": 0.209, "exp_i": -0.637, "sig_i": 0.499}
}

st.set_page_config(page_title="CP3-Bayes Analyzer v3.7", layout="wide")

# --- UI Header ---
st.title("🧪 Advanced CP3-Bayes Analyzer Ver. 3.7")
st.markdown("##### *Abe-lab Official Platform - Triple-Mode & BRIDGE Integration*")

# --- Phase 1: Data Input (4-File System) ---
st.subheader("📁 Phase 1: Data Input")
st.info("Structure A (a) and Structure B (b) files are required for CP3 comparative analysis.")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 📍 **Set A (Structure a / Exp A)**")
    exp_a_file = st.file_uploader("Upload Experimental Peak List A (CSV)", type="csv", key="exp_a")
    calc_a_file = st.file_uploader("Upload BRIDGE Output A (CSV)", type="csv", key="calc_a")

with col_b:
    st.markdown("### 📍 **Set B (Structure b / Exp B)**")
    exp_b_file = st.file_uploader("Upload Experimental Peak List B (CSV)", type="csv", key="exp_b")
    calc_b_file = st.file_uploader("Upload BRIDGE Output B (CSV)", type="csv", key="calc_b")

# --- Phase 2: Scaling Strategy ---
st.divider()
st.subheader("⚙️ Phase 2: Scaling Strategy")

st.markdown("""
> 💡 **スケーリング選択の目安:**
> - **Internal (自動):** 各核種（13C/1H）の原子数が **10点以上** ある場合に推奨。データに最適な補正を行います。
> - **JNP 2024 Fixed (固定):** 原子数が **少ない場合** や、計算レベルが B3LYP/6-31G(d) の場合に安定します。
""")

scaling_mode = st.radio(
    "Choose Scaling Method:",
    ["Internal (Auto-fit to your data)", "JNP 2024 Fixed (B3LYP/6-31G(d))"],
    horizontal=True,
    help="原子数が少ない（特に6個以下）場合は、Fixedを選択することを検討してください。"
)

# --- Phase 3: Analysis Execution ---
def run_analysis_logic(df, n_key):
    """CP3 values and Bayesian probability calculation."""
    p = NUCLEI_PARAMS[n_key]
    
    # Delta-delta calculation
    de = (df['Exp_A'] - df['Exp_B']).replace(0, 0.0001)
    dc = (df['Scaled_a'] - df['Scaled_b']).replace(0, 0.0001)
    
    # CP3 Core Formula
    f3_c = np.where(dc/de > 1, (de**3)/dc, de*dc)
    f3_i = np.where((-dc)/de > 1, (de**3)/(-dc), de*(-dc))
    
    sum_de2 = np.sum(de**2)
    c_cor = np.sum(f3_c) / sum_de2
    c_inc = np.sum(f3_i) / sum_de2
    
    # Bayesian Probability
    p1 = 1 - norm.cdf(c_cor, p['exp_c'], p['sig_c'])
    p2 = 1 - norm.cdf(c_inc, p['exp_i'], p['sig_i'])
    p3 = 1 - norm.cdf(c_cor, p['exp_i'], p['sig_i'])
    p4 = 1 - norm.cdf(c_inc, p['exp_c'], p['sig_c'])
    
    prob = (p1 * p2) / ((p1 * p2) + (p3 * p4) + 1e-15) * 100
    return c_cor, c_inc, prob

if st.button("Run Comprehensive Analysis", use_container_width=True):
    if not (exp_a_file and exp_b_file and calc_a_file and calc_b_file):
        st.error("Please upload all 4 required files.")
    else:
        try:
            # Load Data
            df_exp_a = pd.read_csv(exp_a_file).rename(columns={'Shift': 'Exp_A', 'Chemical_Shift': 'Exp_A'})
            df_exp_b = pd.read_csv(exp_b_file).rename(columns={'Shift': 'Exp_B', 'Chemical_Shift': 'Exp_B'})
            df_calc_a = pd.read_csv(calc_a_file).rename(columns={'Shift': 'Calc_a', 'Isotropic': 'Calc_a'}, errors='ignore')
            df_calc_b = pd.read_csv(calc_b_file).rename(columns={'Shift': 'Calc_b', 'Isotropic': 'Calc_b'}, errors='ignore')
            
            # Merge 4 files by Atom_Label
            m = pd.merge(df_exp_a[['Atom_Label', 'Exp_A']], df_exp_b[['Atom_Label', 'Exp_B']], on="Atom_Label")
            m = pd.merge(m, df_calc_a[['Atom_Label', 'Calc_a']], on="Atom_Label")
            m = pd.merge(m, df_calc_b[['Atom_Label', 'Calc_b']], on="Atom_Label").dropna()

            # Apply Scaling
            for nuclei in ['C', 'H']:
                mask = m['Atom_Label'].str.contains(nuclei, na=False)
                if not mask.any(): continue
                
                if scaling_mode == "Internal (Auto-fit to your data)":
                    # Internal Linear Regression (y = ax + b)
                    x_train = pd.concat([m.loc[mask, 'Calc_a'], m.loc[mask, 'Calc_b']]).values.reshape(-1, 1)
                    y_train = pd.concat([m.loc[mask, 'Exp_A'], m.loc[mask, 'Exp_B']]).values
                    reg = LinearRegression().fit(x_train, y_train)
                    m.loc[mask, 'Scaled_a'] = reg.predict(m.loc[mask, ['Calc_a']].values)
                    m.loc[mask, 'Scaled_b'] = reg.predict(m.loc[mask, ['Calc_b']].values)
                else:
                    # JNP 2024 Fixed Parameters (Slope/Intercept)
                    sl, ic = (-1.053, 181.2) if nuclei == 'C' else (-1.078, 31.8)
                    m.loc[mask, 'Scaled_a'] = m.loc[mask, 'Calc_a'] * sl + ic
                    m.loc[mask, 'Scaled_b'] = m.loc[mask, 'Calc_b'] * sl + ic

            # Analyze each mode
            c_df = m[m['Atom_Label'].str.contains('C', na=False)]
            h_df = m[m['Atom_Label'].str.contains('H', na=False)]
            
            c_cor, c_inc, c_p = run_analysis_logic(c_df, "13C") if not c_df.empty else (0,0,0)
            h_cor, h_inc, h_p = run_analysis_logic(h_df, "1H") if not h_df.empty else (0,0,0)
            
            # Mixed Mode (13C + 1H)
            m_cor, m_inc = (c_cor + h_cor)/2, (c_inc + h_inc)/2
            pm = NUCLEI_PARAMS["13C+1H"]
            m1, m2 = 1-norm.cdf(m_cor, pm['exp_c'], pm['sig_c']), 1-norm.cdf(m_inc, pm['exp_i'], pm['sig_i'])
            m3, m4 = 1-norm.cdf(m_cor, pm['exp_i'], pm['sig_i']), 1-norm.cdf(m_inc, pm['exp_c'], pm['sig_c'])
            m_p = (m1 * m2) / ((m1 * m2) + (m3 * m4) + 1e-15) * 100

            # --- Display Results ---
            st.divider()
            st.balloons()
            
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric("13C Probability", f"{c_p:.2f} %")
            res_col2.metric("1H Probability", f"{h_p:.2f} %")
            res_col3.metric("Mixed Mode (FINAL)", f"{m_p:.2f} %", delta="CP3-Bayes Score")

            # Visualization
            m['de'] = m['Exp_A'] - m['Exp_B']
            m['dc'] = m['Scaled_a'] - m['Scaled_b']
            
            fig = px.scatter(m, x="de", y="dc", color="Atom_Label", text="Atom_Label",
                             labels={"de": "Δδ Experimental (A-B)", "dc": "Δδ Scaled (a-b)"},
                             title="Structural Correlation Plot (JNP 2024 Style)")
            fig.add_shape(type="line", x0=m['de'].min(), y0=m['de'].min(), x1=m['de'].max(), y1=m['de'].max(),
                         line=dict(color="Red", dash="dash"))
            st.plotly_chart(fig, use_container_width=True)

            # Raw Data Table
            with st.expander("See Detailed Merged Data Table"):
                st.dataframe(m)

        except Exception as e:
            st.error(f"Analysis Error: {e}. Please check if Atom_Labels are consistent across all files.")

# --- Footer Export ---
st.divider()
st.markdown("© 2024 Abe-lab. CP3-Bayes Analyzer for Structural Elucidation.")
