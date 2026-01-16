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
st.set_page_config(page_title="NMR DATA BRIDGE v1.4", layout="wide")
st.title("🌉 NMR DATA BRIDGE Ver. 1.4")
st.markdown("##### *Abe-lab Official Platform - Advanced Data Pre-processing*")

# --- Helper Functions ---

def get_clean_name(filename):
    """Remove common suffixes to match energy and nmr files."""
    name = filename.lower().rsplit('.', 1)[0]
    # 共通の接尾辞を削除してマッチングしやすくする
    suffixes = ['_optfreq', '_opt', '_nmr', '_giao', '_sp']
    for s in suffixes:
        if name.endswith(s):
            name = name[:-len(s)]
    return name

def parse_energy_source(file_content, filename):
    """Extract the best available energy from opt/optfreq files."""
    content = file_content.decode("utf-8")
    
    # 優先順位 1: Gibbs Free Energy (optfreqの場合)
    gibbs_match = re.search(r"Sum of electronic and thermal Free Energies=\s+(-?\d+\.\d+)", content)
    if gibbs_match:
        return {"energy": float(gibbs_match.group(1)), "type": "Gibbs Free Energy (G)", "filename": filename}
    
    # 優先順位 2: Sum of electronic and zero-point Energies (freq計算あり)
    zpve_match = re.search(r"Sum of electronic and zero-point Energies=\s+(-?\d+\.\d+)", content)
    if zpve_match:
        return {"energy": float(zpve_match.group(1)), "type": "Zero-point Energy (E0)", "filename": filename}
    
    # 優先順位 3: SCF Done (optのみ、または一点計算)
    # ファイル内の「最後」のSCF Doneを取得する
    scf_matches = re.findall(r"SCF Done:.*?=\s+(-?\d+\.\d+)", content)
    if scf_matches:
        return {"energy": float(scf_matches[-1]), "type": "Electronic Energy (SCF)", "filename": filename}
    
    return None

def parse_nmr_source(file_content, filename):
    """Extract NMR shielding constants."""
    content = file_content.decode("utf-8")
    nmr_pattern = re.compile(r"(\d+)\s+([A-Za-z]+)\s+Isotropic =\s+(-?\d+\.\d+)")
    atoms = [{"index": int(m[0]), "element": m[1], "sigma": float(m[2])} for m in nmr_pattern.findall(content)]
    if not atoms:
        return None
    return {"atoms": atoms, "filename": filename}

def auto_pad_label(label, element, no):
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

# --- Phase 1: Upload (Divided Windows) ---
st.subheader("📁 Phase 1: Upload Gaussian Log Files")
col_up1, col_up2 = st.columns(2)

with col_up1:
    st.info("**Step A: Energy Source** (opt / optfreq)")
    energy_files = st.file_uploader("Drop OPT files here", type=["log", "out"], accept_multiple_files=True, key="up_energy")

with col_up2:
    st.success("**Step B: NMR Data Source** (GIAO)")
    nmr_files = st.file_uploader("Drop NMR files here", type=["log", "out"], accept_multiple_files=True, key="up_nmr")

# --- Main Logic Processing ---
if energy_files and nmr_files:
    # 1. Parse Energy Files
    energy_data_map = {}
    for f in energy_files:
        parsed = parse_energy_source(f.getvalue(), f.name)
        if parsed:
            clean_name = get_clean_name(f.name)
            energy_data_map[clean_name] = parsed

    # 2. Parse NMR Files and Match
    matched_results = []
    unmatched_nmr = []
    
    for f in nmr_files:
        parsed_nmr = parse_nmr_source(f.getvalue(), f.name)
        clean_name = get_clean_name(f.name)
        
        if parsed_nmr and clean_name in energy_data_map:
            # Combine energy and NMR
            combined = {
                "filename_nmr": f.name,
                "filename_energy": energy_data_map[clean_name]["filename"],
                "energy": energy_data_map[clean_name]["energy"],
                "energy_type": energy_data_map[clean_name]["type"],
                "atoms": parsed_nmr["atoms"]
            }
            matched_results.append(combined)
        else:
            unmatched_nmr.append(f.name)

    # 3. Validation and Boltzmann Calculation
    if matched_results:
        # Check atomic consistency
        base_atom_count = len(matched_results[0]['atoms'])
        consistent = all(len(r['atoms']) == base_atom_count for r in matched_results)
        
        if not consistent:
            st.error("❌ Error: Files have inconsistent atom counts. Check your log files.")
        else:
            # Boltzmann calculation
            energies = [r['energy'] for r in matched_results]
            min_e = min(energies)
            kb_t = KB_KCAL * temp / HARTREE_TO_KCAL
            weights = [math.exp(-(e - min_e) / kb_t) for e in energies]
            total_w = sum(weights)
            final_w = [w / total_w for w in weights]

            # Display Boltzmann and Energy Source Summary
            st.subheader("📊 Phase 2: Boltzmann Distribution & Energy Sources")
            dist_df = pd.DataFrame({
                "NMR File": [r['filename_nmr'] for r in matched_results],
                "Energy File": [r['filename_energy'] for r in matched_results],
                "Energy Type": [r['energy_type'] for r in matched_results],
                "Rel. Energy (kcal/mol)": [(e - min_e) * HARTREE_TO_KCAL for e in energies],
                "Weight (%)": [w * 100 for w in final_w]
            })
            st.dataframe(dist_df.style.format(subset=["Rel. Energy (kcal/mol)", "Weight (%)"], formatter="{:.2f}"), use_container_width=True)
            
            if unmatched_nmr:
                st.warning(f"Unmatched NMR files (no corresponding energy file): {', '.join(unmatched_nmr)}")

            # 4. Atomic Averaging
            atom_data = []
            for i in range(base_atom_count):
                avg_s = sum(r['atoms'][i]['sigma'] * final_w[idx] for idx, r in enumerate(matched_results))
                atom_data.append({
                    "Atom_No": matched_results[0]['atoms'][i]['index'], 
                    "Element": matched_results[0]['atoms'][i]['element'], 
                    "Avg_Shielding": avg_s, 
                    "Atom_Label": ""
                })
            
            # Phase 3: Labeling
            st.subheader("🏷️ Phase 3: Atom Labeling & Grouping")
            edited_df = st.data_editor(pd.DataFrame(atom_data), hide_index=True, use_container_width=True, key="editor")

            # Phase 4: Export (Logic from previous stable version)
            st.divider()
            st.subheader("🚀 Phase 4: Data Integration & Export")
            
            col_a, col_b = st.columns(2)
            ready_to_process = not edited_df.empty

            def natural_sort_key(s):
                return [int(t) if t.isdigit() else t.lower() for t in re.split('([0-9]+)', str(s))]

            with col_a:
                st.markdown("##### **[Analysis Mode]**")
                if st.button("Generate Labeled-Only CSV", disabled=not ready_to_process, use_container_width=True):
                    df_labeled = edited_df[edited_df['Atom_Label'] != ""].copy()
                    if df_labeled.empty:
                        st.warning("No labels found.")
                    else:
                        df_labeled['Atom_Label'] = df_labeled['Atom_Label'].apply(lambda x: auto_pad_label(x, "", ""))
                        res = df_labeled.groupby('Atom_Label')['Avg_Shielding'].mean().reset_index().rename(columns={'Avg_Shielding': 'Calc_Raw'})
                        st.session_state.data_analysis = res.sort_values(by='Atom_Label', key=lambda x: x.map(natural_sort_key))
                        st.session_state.processed_analysis = True

                if st.session_state.get('processed_analysis'):
                    st.download_button("💾 Download Analysis CSV", data=st.session_state.data_analysis.to_csv(index=False).encode('utf-8'), file_name="Calc_Data_Analysis.csv", use_container_width=True)

            with col_b:
                st.markdown("##### **[Backup Mode]**")
                if st.button("Generate All-Atoms CSV", disabled=not ready_to_process, use_container_width=True):
                    df_all = edited_df.copy()
                    df_all['Atom_Label'] = df_all.apply(lambda x: auto_pad_label(x['Atom_Label'], x['Element'], x['Atom_No']), axis=1)
                    res_all = df_all.groupby('Atom_Label')['Avg_Shielding'].mean().reset_index().rename(columns={'Avg_Shielding': 'Calc_Raw'})
                    st.session_state.data_backup = res_all.sort_values(by='Atom_Label', key=lambda x: x.map(natural_sort_key))
                    st.session_state.processed_backup = True

                if st.session_state.get('processed_backup'):
                    st.download_button("💾 Download Backup CSV", data=st.session_state.data_backup.to_csv(index=False).encode('utf-8'), file_name="Calc_Data_Full_Backup.csv", use_container_width=True)

elif (energy_files or nmr_files):
    st.info("Please upload both Energy (opt/freq) and NMR (giao) files to proceed.")