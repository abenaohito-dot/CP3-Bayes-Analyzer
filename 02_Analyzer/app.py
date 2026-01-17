import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import norm

# --- Constants: JNP 2024 Parameters ---
NUCLEI_PARAMS = {
    "13C":    {"exp_c": 0.547, "sig_c": 0.253, "exp_i": -0.487, "sig_i": 0.533},
    "1H":     {"exp_c": 0.478, "sig_c": 0.305, "exp_i": -0.786, "sig_i": 0.835},
    "13C+1H": {"exp_c": 0.512, "sig_c": 0.209, "exp_i": -0.637, "sig_i": 0.499}
}

st.set_page_config(page_title="CP3-Bayes Analyzer v3.7.1", layout="wide")

# --- UI Header ---
st.title("🧪 Advanced CP3-Bayes Analyzer Ver. 3.7.1")
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
> - **Internal (自動):** 各核種（13C/1H）の原子数が **10点以上** ある場合に推奨。
> - **JNP 2024 Fixed (固定):** 原子数が少ない場合や、計算レベルが B3LYP/6-31G(d) の場合に安定。
""")

scaling_mode = st.radio(
    "Choose Scaling Method:",
    ["Internal (Auto-fit to your data)", "JNP 2024 Fixed (B3LYP/6-31G(d))"],
    horizontal=True
)

# --- Phase 3: Analysis Execution ---
def run_analysis_logic(df, n_key):
    p = NUCLEI_PARAMS[n_key]
    de = (df['Exp_A'] - df['Exp_B']).replace(0, 0.0001)
    dc = (df['Scaled_a'] - df['Scaled_b']).replace(0, 0.0001)
    
    f3_c = np.where(dc/de > 1, (de**3)/dc, de*dc)
    f3_i = np.where((-dc)/de > 1, (de**3)/(-dc), de*(-dc))
    
    sum_de2 = np.sum(de**2)
    c_cor = np.sum(f3_c) / sum_de2
    c_inc = np.sum(f3_i) / sum_de2
    
    p1, p2 = 1 - norm.cdf(c_cor, p['exp_c'], p['sig_c']), 1 - norm.cdf(c_inc, p['exp_i'], p['sig_i'])
    p3, p4 = 1 - norm.cdf(c_cor, p['exp_i'], p['sig_i']), 1 - norm.cdf(c_inc, p['exp_c'], p['sig_c'])
    
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
            
            # Merge 4 files
            m = pd.merge(df_exp_a[['Atom_Label', 'Exp_A']], df_exp_b[['Atom_Label', 'Exp_B']], on="Atom_Label")
            m = pd.merge(m, df_calc_a[['Atom_Label', 'Calc_a']], on="Atom_Label")
            m = pd.merge(m, df_calc_b[['Atom_Label', 'Calc_b']], on="Atom_Label").dropna()

            # Scaling (Numpy-based logic)
            for nuclei in ['C', 'H']:
                mask = m['Atom_Label'].str.contains(nuclei, na=False)
                if not mask.any(): continue
                
                if scaling_mode == "Internal (Auto-fit to your data)":
                    x = pd.concat([m.loc[mask, 'Calc_a'], m.loc[mask, 'Calc_b']]).values
                    y = pd.concat([m.loc[mask, 'Exp_A'], m.loc[mask, 'Exp_B']]).values
                    sl, ic = np.polyfit(x, y, 1) # numpyだけで線形回帰
                    m.loc[mask, 'Scaled_a'] = m.loc[mask, 'Calc_a'] * sl + ic
                    m.loc[mask, 'Scaled_b'] = m.loc[mask, 'Calc_b'] * sl + ic
                else:
                    sl, ic = (-1.053, 181.2) if nuclei == 'C' else (-1.078, 31.8)
                    m.loc[mask, 'Scaled_a'] = m.loc[mask, 'Calc_a'] * sl + ic
                    m.loc[mask, 'Scaled_b'] = m.loc[mask, 'Calc_b'] * sl + ic

            # Process
            c_df, h_df = m[m['Atom_Label'].str.contains('C')], m[m['Atom_Label'].str.contains('H')]
            c_res = run_analysis_logic(c_df, "13C") if not c_df.empty else (0,0,0)
            h_res = run_analysis_logic(h_df, "1H") if not h_df.empty else (0,0,0)
            
            # Display
            st.divider()
            st.balloons()
            cols = st.columns(3)
            cols[0].metric("13C Prob", f"{c_res[2]:.1f}%")
            cols[1].metric("1H Prob", f"{h_res[2]:.1f}%")
            
            # Mixed Mode
            m_cor, m_inc = (c_res[0]+h_res[0])/2, (c_res[1]+h_res[1])/2
            pm = NUCLEI_PARAMS["13C+1H"]
            m1, m2 = 1-norm.cdf(m_cor, pm['exp_c'], pm['sig_c']), 1-norm.cdf(m_inc, pm['exp_i'], pm['sig_i'])
            m3, m4 = 1-norm.cdf(m_cor, pm['exp_i'], pm['sig_i']), 1-norm.cdf(m_inc, pm['exp_c'], pm['sig_c'])
            m_p = (m1 * m2) / ((m1 * m2) + (m3 * m4) + 1e-15) * 100
            cols[2].metric("Mixed Prob", f"{m_p:.1f}%")

            # Plot
            m['de'], m['dc'] = m['Exp_A'] - m['Exp_B'], m['Scaled_a'] - m['Scaled_b']
            fig = px.scatter(m, x="de", y="dc", color="Atom_Label", text="Atom_Label",
                             labels={"de": "Exp Delta", "dc": "Calc Delta"}, title="Correlation")
            fig.add_shape(type="line", x0=m['de'].min(), y0=m['de'].min(), x1=m['de'].max(), y1=m['de'].max(), line=dict(color="Red", dash="dash"))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(m)

        except Exception as e:
            st.error(f"Error: {e}")

st.divider()
st.caption("CP3-Bayes Ver. 3.7.1 - Dependency Optimized")
