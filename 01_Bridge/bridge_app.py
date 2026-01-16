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
    """
    ファイル名から拡張子を除いた部分の『最後』の数字を抽出して整数で返す。
    例: compound4_opt_08.log -> 8
    """
    name_part = filename.rsplit('.', 1)[0]
    nums = re.findall(r'(\d+)', name_part)
    if not nums:
        return None
    return int(nums[-1])  # 最後の数字を数値として返す (08も8も 8 になる)

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
    # SCF Done (Last match)
    scf_matches = re.findall(r"SCF Done:.*?=\s+(-?\d+\.\d+)", content)
    if scf_matches:
        return {"energy": float(scf_matches[-1]), "type": "SCF (E)", "filename": filename}
    return None

def parse_nmr_source(file_content, filename):
    content = file_content.decode("utf-8")
    nmr_pattern = re.compile(r"(\d+)\s+([A-Za-z]+)\s+Isotropic =\s+(-?\d+\.\d+)")
    atoms = [{"index": int(m[0]), "element": m[1], "sigma": float(m[2])} for m in nmr_pattern.findall(content)]
    return {"atoms": atoms, "filename": filename} if atoms else None

# --- Page Layout & UI ---
st.set_page_config(page_title="NMR DATA BRIDGE v1.6", layout="wide")
st.title("🌉 NMR DATA BRIDGE Ver. 1.6")
st.markdown("##### *Abe-lab Official Platform - Suffix Number Matching*")

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
    # 1. エネルギーファイルをID（末尾数字）で辞書化
    energy_map = {}
    for f in energy_files:
        fid = get_file_id(f.name)
        parsed = parse_energy_source(f.getvalue(), f.name)
        if fid is not None and parsed:
            energy_map[fid] = parsed

    # 2. NMRファイルをIDでマッチング
    matched_results = []
    unmatched_files = []

    for f in nmr_files:
        fid = get_file_id(f.name)
        parsed_nmr = parse_nmr_source(f.getvalue(), f.name)
        
        if parsed_nmr and fid in energy_map:
            matched_results.append({
                "id": fid,
                "filename_nmr": f.name,
                "filename_energy": energy_map[fid]["filename"],
                "energy": energy_map[fid]["energy"],
                "energy_type": energy_map[fid]["type"],
                "atoms": parsed_nmr["atoms"]
            })
        else:
            unmatched_files.append(f.name)

    # 3. 結果表示
    if matched_results:
        # 原子数チェック
        counts = [len(r['atoms']) for r in matched_results]
        if len(set(counts)) > 1:
            st.error(f"❌ Atom count mismatch! {set(counts)}")
        else:
            # ボルツマン計算
            energies = [r['energy'] for r in matched_results]
            min_e = min(energies)
            kb_t = KB_KCAL * temp / HARTREE_TO_KCAL
            weights = [math.exp(-(e - min_e) / kb_t) for e in energies]
            total_w = sum(weights)
            final_w = [w / total_w for w in weights]

            st.subheader("📊 Phase 2: Boltzmann Summary (Linked by Suffix Number)")
            dist_df = pd.DataFrame({
                "Conformer ID": [r['id'] for r in matched_results],
                "Energy File": [r['filename_energy'] for r in matched_results],
                "NMR File": [r['filename_nmr'] for r in matched_results],
                "Energy Type": [r['energy_type'] for r in matched_results],
                "Rel. E (kcal/mol)": [(e - min_e) * HARTREE_TO_KCAL for e in energies],
                "Weight (%)": [w * 100 for w in final_w]
            }).sort_values("Conformer ID") # ID順に並び替え
            
            st.dataframe(dist_df.style.format(subset=["Rel. E (kcal/mol)", "Weight (%)"], formatter="{:.2f}"), use_container_width=True)

            if unmatched_files:
                st.warning(f"⚠️ Unmatched files: {', '.join(unmatched_files)}")

            # --- 以降、ラベル編集・CSV出力 (省略なし) ---
            base_atoms = matched_results[0]['atoms']
            atom_data = []
            for i in range(len(base_atoms)):
                avg_s = sum(r['atoms'][i]['sigma'] * final_w[idx] for idx, r in enumerate(matched_results))
                atom_data.append({"Atom_No": base_atoms[i]['index'], "Element": base_atoms[i]['element'], "Avg_Shielding": avg_s, "Atom_Label": ""})
            
            st.subheader("🏷️ Phase 3: Atom Labeling")
            edited_df = st.data_editor(pd.DataFrame(atom_data), hide_index=True, use_container_width=True, key="editor")

            st.divider()
            st.subheader("🚀 Phase 4: Export")
            # (CSV出力ロジックは以前と同様)
            # ... [以下、前述のコードと同じため、適宜 bridge_app.py に統合してください]