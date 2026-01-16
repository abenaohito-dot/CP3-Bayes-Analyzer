import streamlit as st
import pandas as pd
import numpy as np
import re
import math

# --- Constants ---
KB_KCAL = 1.9872e-3
HARTREE_TO_KCAL = 627.509
TEMP_DEFAULT = 298.15

# --- Helper Functions ---

def get_file_id(filename):
    """ファイル名から拡張子を除いた部分の『最後』の数字を抽出して整数で返す。"""
    name_part = filename.rsplit('.', 1)[0]
    nums = re.findall(r'(\d+)', name_part)
    return int(nums[-1]) if nums else None

def parse_energy_source(file_content, filename):
    content = file_content.decode("utf-8")
    gibbs_match = re.search(r"Sum of electronic and thermal Free Energies=\s+(-?\d+\.\d+)", content)
    if gibbs_match:
        return {"energy": float(gibbs_match.group(1)), "type": "Gibbs (G)", "filename": filename}
    zpve_match = re.search(r"Sum of electronic and zero-point Energies=\s+(-?\d+\.\d+)", content)
    if zpve_match:
        return {"energy": float(zpve_match.group(1)), "type": "ZPVE (E0)", "filename": filename}
    scf_matches = re.findall(r"SCF Done:.*?=\s+(-?\d+\.\d+)", content)
    if scf_matches:
        return {"energy": float(scf_matches[-1]), "type": "SCF (E)", "filename": filename}
    return None

def parse_nmr_source_v18(file_content, filename):
    """v1.8: Isotropic値に加え、テンソル成分(XX, YY, ZZ)も抽出する"""
    content = file_content.decode("utf-8")
    # Isotropic値とそれに続くテンソル行をキャプチャ
    nmr_pattern = re.compile(
        r"(\d+)\s+([A-Za-z]+)\s+Isotropic =\s+(-?\d+\.\d+).*?"
        r"XX=\s+(-?\d+\.\d+).*?"
        r"YY=\s+(-?\d+\.\d+).*?"
        r"ZZ=\s+(-?\d+\.\d+)", 
        re.DOTALL
    )
    
    atoms = []
    for m in nmr_pattern.findall(content):
        atoms.append({
            "index": int(m[0]), 
            "element": m[1], 
            "sigma": float(m[2]),
            "XX": float(m[3]),
            "YY": float(m[4]),
            "ZZ": float(m[5])
        })
    return {"atoms": atoms, "filename": filename} if atoms else None

def auto_pad_label(label, element, no):
    if label == "": return f"{element}{str(no).zfill(2)}_raw"
    match = re.search(r'(\D+)(\d+)$', label)
    if match:
        prefix, num = match.groups()
        return f"{prefix}{num.zfill(2)}"
    return label

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split('([0-9]+)', str(s))]

# --- UI Setup ---
st.set_page_config(page_title="NMR DATA BRIDGE v1.8", layout="wide")
st.title("🌉 NMR DATA BRIDGE Ver. 1.8")
st.markdown("##### *Professional Output Mode with Raw Tensor Verification*")

# Session State
if 'processed_analysis' not in st.session_state: st.session_state.processed_analysis = False
if 'processed_backup' not in st.session_state: st.session_state.processed_backup = False

with st.sidebar:
    st.header("⚙️ Configuration")
    temp = st.number_input("Temperature (K):", value=TEMP_DEFAULT)
    st.divider()
    st.caption("Gifu Pharmaceutical University, Abe-lab.")

# --- Phase 1: Upload ---
st.subheader("📁 Phase 1: Upload Gaussian Log Files")
col_up1, col_up2 = st.columns(2)
with col_up1:
    energy_files = st.file_uploader("Drop OPT files (Energy)", type=["log", "out"], accept_multiple_files=True)
with col_up2:
    nmr_files = st.file_uploader("Drop NMR files (Shielding)", type=["log", "out"], accept_multiple_files=True)

if energy_files and nmr_files:
    # 1. Matching
    energy_map = {get_file_id(f.name): parse_energy_source(f.getvalue(), f.name) for f in energy_files if get_file_id(f.name) is not None}
    matched_results = []
    for f in nmr_files:
        fid = get_file_id(f.name)
        parsed_nmr = parse_nmr_source_v18(f.getvalue(), f.name)
        if parsed_nmr and fid in energy_map and energy_map[fid]:
            matched_results.append({
                "id": fid, "filename_nmr": f.name, "filename_energy": energy_map[fid]["filename"],
                "energy": energy_map[fid]["energy"], "energy_type": energy_map[fid]["type"], "atoms": parsed_nmr["atoms"]
            })

    if matched_results:
        # 2. Boltzmann Table
        energies = [r['energy'] for r in matched_results]
        min_e = min(energies)
        kb_t = KB_KCAL * temp / HARTREE_TO_KCAL
        weights = [math.exp(-(e - min_e) / kb_t) for e in energies]
        final_w = [w / sum(weights) for w in weights]

        st.subheader("📊 Phase 2: Boltzmann Summary")
        dist_df = pd.DataFrame({
            "ID": [r['id'] for r in matched_results],
            "Energy File": [r['filename_energy'] for r in matched_results],
            "NMR File": [r['filename_nmr'] for r in matched_results],
            "Energy Type": [r['energy_type'] for r in matched_results],
            "Rel. E (kcal/mol)": [(e - min_e) * HARTREE_TO_KCAL for e in energies],
            "Weight (%)": [w * 100 for w in final_w]
        }).sort_values("ID")
        st.dataframe(dist_df.style.format(subset=["Rel. E (kcal/mol)", "Weight (%)"], formatter="{:.2f}"), use_container_width=True)

        # --- v1.8 New Section: Raw Data Verification ---
        st.subheader("🔍 Phase 2.5: Raw Data Verification (All Conformers)")
        raw_rows = []
        for r in matched_results:
            for atom in r['atoms']:
                raw_rows.append({
                    "Conf_ID": r['id'],
                    "Atom_No": atom['index'],
                    "Element": atom['element'],
                    "Isotropic": atom['sigma'],
                    "XX": atom['XX'], "YY": atom['YY'], "ZZ": atom['ZZ']
                })
        raw_df = pd.DataFrame(raw_rows)
        st.write("各配座異性体から抽出されたシールド定数およびテンソル成分の生データです：")
        st.dataframe(raw_df, use_container_width=True)
        
        st.download_button(
            "💾 Download Raw Tensors CSV",
            data=raw_df.to_csv(index=False).encode('utf-8'),
            file_name="NMR_Raw_Tensors_Check.csv",
            use_container_width=False
        )

        # 3. Atomic Labeling
        st.subheader("🏷️ Phase 3: Atom Labeling & Averaging")
        base_atoms = matched_results[0]['atoms']
        atom_data = []
        for i in range(len(base_atoms)):
            avg_s = sum(r['atoms'][i]['sigma'] * final_w[idx] for idx, r in enumerate(matched_results))
            atom_data.append({"Atom_No": base_atoms[i]['index'], "Element": base_atoms[i]['element'], "Avg_Shielding": avg_s, "Atom_Label": ""})
        
        edited_df = st.data_editor(pd.DataFrame(atom_data), hide_index=True, use_container_width=True, key="editor")

        # 4. Export
        st.divider()
        st.subheader("🚀 Phase 4: Data Integration & Export")
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("##### **[Analysis Mode]**")
            if st.button("Prepare Analysis Data", use_container_width=True):
                df_labeled = edited_df[edited_df['Atom_Label'] != ""].copy()
                if df_labeled.empty:
                    st.warning("No labels found.")
                else:
                    df_labeled['Atom_Label'] = df_labeled['Atom_Label'].apply(lambda x: auto_pad_label(x, "", ""))
                    res = df_labeled.groupby('Atom_Label')['Avg_Shielding'].mean().reset_index().rename(columns={'Avg_Shielding': 'Calc_Raw'})
                    st.session_state.data_analysis = res.sort_values(by='Atom_Label', key=lambda x: x.map(natural_sort_key))
                    st.session_state.processed_analysis = True

            if st.session_state.processed_analysis:
                st.download_button("💾 Download Analysis CSV", data=st.session_state.data_analysis.to_csv(index=False).encode('utf-8'), file_name="Calc_Data_Analysis.csv", use_container_width=True)

        with col_b:
            st.markdown("##### **[Backup Mode]**")
            if st.button("Prepare Backup Data", use_container_width=True):
                df_all = edited_df.copy()
                df_all['Atom_Label'] = df_all.apply(lambda x: auto_pad_label(x['Atom_Label'], x['Element'], x['Atom_No']), axis=1)
                res_all = df_all.groupby('Atom_Label')['Avg_Shielding'].mean().reset_index().rename(columns={'Avg_Shielding': 'Calc_Raw'})
                st.session_state.data_backup = res_all.sort_values(by='Atom_Label', key=lambda x: x.map(natural_sort_key))
                st.session_state.processed_backup = True

            if st.session_state.processed_backup:
                st.download_button("💾 Download Backup CSV", data=st.session_state.data_backup.to_csv(index=False).encode('utf-8'), file_name="Calc_Data_Full_Backup.csv", use_container_width=True)

elif (energy_files or nmr_files):
    st.info("Awaiting both OPT and NMR files to enable verification and export functions.")
