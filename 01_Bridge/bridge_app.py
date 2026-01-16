import streamlit as st
import pandas as pd
import numpy as np
import re
import math

# --- Constants ---
KB_KCAL = 1.9872e-3
HARTREE_TO_KCAL = 627.509
TEMP_DEFAULT = 298.15

# --- Page Configuration ---
st.set_page_config(page_title="NMR DATA BRIDGE v1.5", layout="wide")
st.title("🌉 NMR DATA BRIDGE Ver. 1.5")
st.markdown("##### *Abe-lab Official Platform - Number-based Matching*")

# --- Helper Functions ---

def get_file_number(filename):
    """ファイル名から最初の数字の塊を抽出する (例: conf01_opt -> 01)"""
    match = re.search(r'(\d+)', filename)
    return match.group(1) if match else None

def parse_energy_source(file_content, filename):
    content = file_content.decode("utf-8")
    
    # Gibbs Free Energy
    gibbs_match = re.search(r"Sum of electronic and thermal Free Energies=\s+(-?\d+\.\d+)", content)
    if gibbs_match:
        return {"energy": float(gibbs_match.group(1)), "type": "Gibbs (G)", "filename": filename}
    
    # Zero-point Energy
    zpve_match = re.search(r"Sum of electronic and zero-point Energies=\s+(-?\d+\.\d+)", content)
    if zpve_match:
        return {"energy": float(zpve_match.group(1)), "type": "ZPVE (E0)", "filename": filename}
    
    # SCF Done (Last match in file)
    scf_matches = re.findall(r"SCF Done:.*?=\s+(-?\d+\.\d+)", content)
    if scf_matches:
        return {"energy": float(scf_matches[-1]), "type": "SCF (E)", "filename": filename}
    
    return None

def parse_nmr_source(file_content, filename):
    content = file_content.decode("utf-8")
    nmr_pattern = re.compile(r"(\d+)\s+([A-Za-z]+)\s+Isotropic =\s+(-?\d+\.\d+)")
    atoms = [{"index": int(m[0]), "element": m[1], "sigma": float(m[2])} for m in nmr_pattern.findall(content)]
    return {"atoms": atoms, "filename": filename} if atoms else None

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
    st.caption("Abe-lab, Gifu Pharmaceutical University.")

# --- Phase 1: Upload ---
st.subheader("📁 Phase 1: Upload Gaussian Log Files")
col_up1, col_up2 = st.columns(2)

with col_up1:
    st.info("**Step A: Energy Source** (opt / optfreq)")
    energy_files = st.file_uploader("Drop OPT files", type=["log", "out"], accept_multiple_files=True, key="up_energy")

with col_up2:
    st.success("**Step B: NMR Data Source** (GIAO)")
    nmr_files = st.file_uploader("Drop NMR files", type=["log", "out"], accept_multiple_files=True, key="up_nmr")

# --- Processing Logic ---
if energy_files and nmr_files:
    # 1. 辞書に番号(ID)をキーにして保存
    energy_map = {}
    for f in energy_files:
        file_id = get_file_number(f.name)
        parsed = parse_energy_source(f.getvalue(), f.name)
        if file_id and parsed:
            energy_map[file_id] = parsed

    # 2. NMRファイルを番号でマッチング
    matched_results = []
    unmatched_ids = []

    for f in nmr_files:
        file_id = get_file_number(f.name)
        parsed_nmr = parse_nmr_source(f.getvalue(), f.name)
        
        if parsed_nmr and file_id in energy_map:
            matched_results.append({
                "id": file_id,
                "filename_nmr": f.name,
                "filename_energy": energy_map[file_id]["filename"],
                "energy": energy_map[file_id]["energy"],
                "energy_type": energy_map[file_id]["type"],
                "atoms": parsed_nmr["atoms"]
            })
        else:
            unmatched_ids.append(f.name)

    # 3. Display Results
    if matched_results:
        # Check Atom Consistency
        counts = [len(r['atoms']) for r in matched_results]
        if len(set(counts)) > 1:
            st.error(f"❌ Atom count mismatch! Detected counts: {set(counts)}")
        else:
            # Boltzmann
            energies = [r['energy'] for r in matched_results]
            min_e = min(energies)
            kb_t = KB_KCAL * temp / HARTREE_TO_KCAL
            weights = [math.exp(-(e - min_e) / kb_t) for e in energies]
            total_w = sum(weights)
            final_w = [w / total_w for w in weights]

            st.subheader("📊 Phase 2: Boltzmann Summary (Linked by Number)")
            dist_df = pd.DataFrame({
                "ID": [r['id'] for r in matched_results],
                "Energy Source": [r['filename_energy'] for r in matched_results],
                "NMR Source": [r['filename_nmr'] for r in matched_results],
                "Used Energy": [r['energy_type'] for r in matched_results],
                "Rel. E (kcal/mol)": [(e - min_e) * HARTREE_TO_KCAL for e in energies],
                "Weight (%)": [w * 100 for w in final_w]
            })
            st.dataframe(dist_df.style.format(subset=["Rel. E (kcal/mol)", "Weight (%)"], formatter="{:.2f}"), use_container_width=True)

            if unmatched_ids:
                st.warning(f"⚠️ Files not linked (no matching number): {', '.join(unmatched_ids)}")

            # Atomic Averaging & Labeling (Phase 3 & 4)
            # ... (ここから下のラベル編集・CSV出力ロジックは安定版を継承)
            base_atoms = matched_results[0]['atoms']
            atom_data = []
            for i in range(len(base_atoms)):
                avg_s = sum(r['atoms'][i]['sigma'] * final_w[idx] for idx, r in enumerate(matched_results))
                atom_data.append({"Atom_No": base_atoms[i]['index'], "Element": base_atoms[i]['element'], "Avg_Shielding": avg_s, "Atom_Label": ""})
            
            st.subheader("🏷️ Phase 3: Atom Labeling")
            edited_df = st.data_editor(pd.DataFrame(atom_data), hide_index=True, use_container_width=True, key="editor")

            st.divider()
            st.subheader("🚀 Phase 4: Export")
            col_a, col_b = st.columns(2)
            
            # --- Export Helper ---
            def get_csv(df, mode="labeled"):
                if mode == "labeled":
                    df = df[df['Atom_Label'] != ""].copy()
                    df['Atom_Label'] = df['Atom_Label'].apply(lambda x: auto_pad_label(x, "", ""))
                else:
                    df = df.copy()
                    df['Atom_Label'] = df.apply(lambda x: auto_pad_label(x['Atom_Label'], x['Element'], x['Atom_No']), axis=1)
                
                res = df.groupby('Atom_Label')['Avg_Shielding'].mean().reset_index().rename(columns={'Avg_Shielding': 'Calc_Raw'})
                ns = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('([0-9]+)', str(s))]
                return res.sort_values(by='Atom_Label', key=lambda x: x.map(ns)).to_csv(index=False).encode('utf-8')

            if not edited_df.empty:
                with col_a:
                    st.download_button("💾 Download Analysis CSV", data=get_csv(edited_df, "labeled"), file_name="Calc_Analysis.csv", use_container_width=True)
                with col_b:
                    st.download_button("💾 Download Backup CSV", data=get_csv(edited_df, "all"), file_name="Calc_Full_Backup.csv", use_container_width=True)

elif (energy_files or nmr_files):
    st.info("Awaiting files in both windows to start matching.")