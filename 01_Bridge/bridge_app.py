import streamlit as st
import pandas as pd
import numpy as np
import re
import math
from datetime import datetime

# --- Constants: Boltzmann Parameters ---
KB_KCAL = 1.9872e-3  # kcal/mol·K
HARTREE_TO_KCAL = 627.509
TEMP_DEFAULT = 298.15

# --- Page Configuration ---
st.set_page_config(page_title="NMR DATA BRIDGE v1.3", layout="wide")
st.title("🌉 NMR DATA BRIDGE Ver. 1.3")
st.markdown("##### *Abe-lab Official Platform - Advanced Data Pre-processing*")

# Session State for safe operation
if 'processed_analysis' not in st.session_state: st.session_state.processed_analysis = False
if 'processed_backup' not in st.session_state: st.session_state.processed_backup = False
if 'data_analysis' not in st.session_state: st.session_state.data_analysis = None
if 'data_backup' not in st.session_state: st.session_state.data_backup = None

# --- Helper Functions ---
def parse_gaussian_log(file_content, filename):
    content = file_content.decode("utf-8")
    e_match = re.search(r"SCF Done:.*?=\s+(-?\d+\.\d+)", content)
    if not e_match: return None
    energy = float(e_match.group(1))
    
    nmr_pattern = re.compile(r"(\d+)\s+([A-Za-z]+)\s+Isotropic =\s+(-?\d+\.\d+)")
    atoms = [{"index": int(m[0]), "element": m[1], "sigma": float(m[2])} for m in nmr_pattern.findall(content)]
    
    if not atoms: return None
    return {"filename": filename, "energy": energy, "atoms": atoms}

def auto_pad_label(label, element, no):
    """Formats labels for Excel compatibility (e.g., C1 -> C01)."""
    if label == "": return f"{element}{str(no).zfill(2)}_raw"
    match = re.search(r'(\D+)(\d+)$', label)
    if match:
        prefix, num = match.groups()
        return f"{prefix}{num.zfill(2)}"
    return label

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Configuration")
    temp = st.number_input("Temperature (K):", value=TEMP_DEFAULT)
    st.divider()
    st.caption("Developed for Gifu Pharmaceutical University, Abe-lab.")

# --- Phase 1: Upload (Always Visible) ---
st.subheader("📁 Phase 1: Upload Gaussian Log Files")
uploaded_files = st.file_uploader("Drop .log or .out files here", type=["log", "out"], accept_multiple_files=True, key="up")

# Logic Processing
edited_df = pd.DataFrame()
if uploaded_files:
    raw_results = []
    for f in uploaded_files:
        parsed = parse_gaussian_log(f.getvalue(), f.name)
        if parsed: raw_results.append(parsed)
    
    if raw_results:
        # Boltzmann Weights
        energies = [r['energy'] for r in raw_results]
        min_e = min(energies)
        kb_t = KB_KCAL * temp / HARTREE_TO_KCAL
        weights = [math.exp(-(e - min_e) / kb_t) for e in energies]
        final_w = [w / sum(weights) for w in weights]
        
        # Boltzmann Table
        st.subheader("📊 Phase 2: Boltzmann Distribution")
        dist_df = pd.DataFrame({
            "File Name": [r['filename'] for r in raw_results],
            "Rel. Energy (kcal/mol)": [(e - min_e) * HARTREE_TO_KCAL for e in energies],
            "Weight (%)": [w * 100 for w in final_w]
        })
        st.dataframe(dist_df.style.format(subset=["Rel. Energy (kcal/mol)", "Weight (%)"], formatter="{:.2f}"), use_container_width=True)

        # Atomic Averaging
        atom_count = len(raw_results[0]['atoms'])
        atom_data = []
        for i in range(atom_count):
            avg_s = sum(r['atoms'][i]['sigma'] * final_w[idx] for idx, r in enumerate(raw_results))
            atom_data.append({"Atom_No": raw_results[0]['atoms'][i]['index'], "Element": raw_results[0]['atoms'][i]['element'], "Avg_Shielding": avg_s, "Atom_Label": ""})
        
        # Phase 3: Labeling
        st.subheader("🏷️ Phase 3: Atom Labeling & Grouping")
        edited_df = st.data_editor(pd.DataFrame(atom_data), hide_index=True, use_container_width=True, key="editor")

# --- Phase 4: Export Control (Always Visible) ---
st.divider()
st.subheader("🚀 Phase 4: Data Integration & Export")

ready_to_process = not edited_df.empty
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("##### **[Analysis Mode]**")
    if st.button("Generate Labeled-Only CSV", disabled=not ready_to_process, use_container_width=True):
        # Filter: Only labeled atoms
        df_labeled = edited_df[edited_df['Atom_Label'] != ""].copy()
        if df_labeled.empty:
            st.warning("No labels found. Analysis mode requires labels.")
        else:
            df_labeled['Atom_Label'] = df_labeled['Atom_Label'].apply(lambda x: auto_pad_label(x, "", ""))
            res = df_labeled.groupby('Atom_Label')['Avg_Shielding'].mean().reset_index().rename(columns={'Avg_Shielding': 'Calc_Raw'})
            # Natural sort
            def ns(s): return [int(t) if t.isdigit() else t.lower() for t in re.split('([0-9]+)', s)]
            st.session_state.data_analysis = res.sort_values(by='Atom_Label', key=lambda x: x.map(ns))
            st.session_state.processed_analysis = True

    if st.session_state.processed_analysis:
        st.download_button("💾 Download Analysis CSV", data=st.session_state.data_analysis.to_csv(index=False).encode('utf-8'), file_name="Calc_Data_Analysis.csv", use_container_width=True)
    else:
        st.button("💾 Download Analysis CSV", disabled=True, use_container_width=True)

with col_b:
    st.markdown("##### **[Backup Mode]**")
    if st.button("Generate All-Atoms CSV", disabled=not ready_to_process, use_container_width=True):
        df_all = edited_df.copy()
        df_all['Atom_Label'] = df_all.apply(lambda x: auto_pad_label(x['Atom_Label'], x['Element'], x['Atom_No']), axis=1)
        res_all = df_all.groupby('Atom_Label')['Avg_Shielding'].mean().reset_index().rename(columns={'Avg_Shielding': 'Calc_Raw'})
        # Natural sort
        def ns(s): return [int(t) if t.isdigit() else t.lower() for t in re.split('([0-9]+)', s)]
        st.session_state.data_backup = res_all.sort_values(by='Atom_Label', key=lambda x: x.map(ns))
        st.session_state.processed_backup = True

    if st.session_state.processed_backup:
        st.download_button("💾 Download Backup CSV", data=st.session_state.data_backup.to_csv(index=False).encode('utf-8'), file_name="Calc_Data_Full_Backup.csv", use_container_width=True)
    else:
        st.button("💾 Download Backup CSV", disabled=True, use_container_width=True)

if not uploaded_files:
    st.info("Awaiting Gaussian log files to activate Phase 4 buttons.")