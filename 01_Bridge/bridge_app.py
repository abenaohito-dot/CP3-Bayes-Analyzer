import streamlit as st
import pandas as pd
import numpy as np
import re
import math

# --- Constants ---
KB_KCAL = 1.9872e-3
HARTREE_TO_KCAL = 627.509
TEMP_DEFAULT = 298.15

# --- Helper Functions (with @st.cache_data) ---

def get_file_id(filename):
    """Extract the last integer from filename before extension."""
    name_part = filename.rsplit('.', 1)[0]
    nums = re.findall(r'(\d+)', name_part)
    return int(nums[-1]) if nums else None

@st.cache_data(show_spinner=False)
def parse_energy_source(file_bytes, filename):
    """Parse energy from Gaussian log file bytes."""
    content = file_bytes.decode("utf-8", errors="replace")
    
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

@st.cache_data(show_spinner=False)
def parse_frequency_source(file_bytes, filename):
    """Read Gaussian frequency blocks and report imaginary vibrational modes."""
    content = file_bytes.decode("utf-8", errors="replace")
    frequencies = []
    for line in content.splitlines():
        if "Frequencies --" in line:
            frequencies.extend(float(value) for value in re.findall(r"-?\d+\.\d+", line))

    normal_termination = "Normal termination of Gaussian" in content
    imaginary = sorted(value for value in frequencies if value < 0)
    if not frequencies:
        status = "⚪ Frequency data not found"
    elif not normal_termination:
        status = "⚪ Calculation not completed"
    elif not imaginary:
        status = "✅ No imaginary frequency"
    elif max(abs(value) for value in imaginary) <= 20.0:
        status = "⚠️ Soft imaginary frequency (≤20 cm⁻¹)"
    else:
        status = "❌ Imaginary frequency detected (>20 cm⁻¹)"

    return {
        "filename": filename,
        "normal_termination": normal_termination,
        "imaginary": imaginary,
        "status": status,
    }

@st.cache_data(show_spinner=False)
def parse_nmr_source_v181(file_bytes, filename):
    """
    Extract Isotropic values and diagonal tensor components (XX, YY, ZZ) safely and quickly.
    Iterates line by line to avoid Catastrophic Backtracking on large logs.
    """
    content = file_bytes.decode("utf-8", errors="replace")
    lines = content.splitlines()
    
    atoms = []
    
    # 正規表現を行単位で適用（爆発的なバックトラックを防ぐ）
    iso_pattern = re.compile(r"^\s*(\d+)\s+([A-Za-z]+)\s+Isotropic\s*=\s*(-?\d+\.\d+)")
    xx_pattern = re.compile(r"XX=\s*(-?\d+\.\d+)")
    yy_pattern = re.compile(r"YY=\s*(-?\d+\.\d+)")
    zz_pattern = re.compile(r"ZZ=\s*(-?\d+\.\d+)")
    
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        iso_m = iso_pattern.search(line)
        if iso_m:
            idx = int(iso_m.group(1))
            element = iso_m.group(2)
            sigma = float(iso_m.group(3))
            
            # XX, YY, ZZ は通常、直後の行（1〜3行以内）に出現する
            xx, yy, zz = None, None, None
            for offset in range(1, 5):
                if i + offset >= n:
                    break
                subline = lines[i + offset]
                if xx is None:
                    xx_m = xx_pattern.search(subline)
                    if xx_m: xx = float(xx_m.group(1))
                if yy is None:
                    yy_m = yy_pattern.search(subline)
                    if yy_m: yy = float(yy_m.group(1))
                if zz is None:
                    zz_m = zz_pattern.search(subline)
                    if zz_m: zz = float(zz_m.group(1))
                
                # 次の原子の行に達した場合は抜ける
                if iso_pattern.search(subline):
                    break
            
            if xx is not None and yy is not None and zz is not None:
                atoms.append({
                    "index": idx,
                    "element": element,
                    "sigma": sigma,
                    "XX": xx,
                    "YY": yy,
                    "ZZ": zz
                })
        i += 1

    return {"atoms": atoms, "filename": filename} if atoms else None

def auto_pad_label(label, element, no):
    if not label: return f"{element}{str(no).zfill(2)}_raw"
    match = re.search(r'(\D+)(\d+)$', str(label))
    if match:
        prefix, num = match.groups()
        return f"{prefix}{num.zfill(2)}"
    return str(label)

def natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'([0-9]+)', str(s))]

# --- UI Setup ---
st.set_page_config(page_title="NMR DATA BRIDGE v1.8.2", layout="wide")
st.title("🌉 NMR DATA BRIDGE Ver. 1.8.2")
st.markdown("##### *Professional Output Mode - Gifu Pharm. Univ. Abe-lab*")

# Session State
if 'processed_analysis' not in st.session_state: st.session_state.processed_analysis = False
if 'processed_backup' not in st.session_state: st.session_state.processed_backup = False

with st.sidebar:
    st.header("⚙️ Configuration")
    temp = st.number_input("Temperature (K):", value=TEMP_DEFAULT)
    st.divider()
    st.caption("Abe-lab, Gifu Pharmaceutical University.")

# --- Phase 1: Upload ---
st.subheader("📁 Phase 1: Upload Gaussian Log Files")
col_up1, col_up2 = st.columns(2)
with col_up1:
    energy_files = st.file_uploader("Drop OPT files (Energy)", type=["log", "out"], accept_multiple_files=True)
with col_up2:
    nmr_files = st.file_uploader("Drop NMR files (Shielding)", type=["log", "out"], accept_multiple_files=True)

if energy_files and nmr_files:
    energy_map = {get_file_id(f.name): parse_energy_source(f.getvalue(), f.name) for f in energy_files if get_file_id(f.name) is not None}
    frequency_map = {get_file_id(f.name): parse_frequency_source(f.getvalue(), f.name) for f in energy_files if get_file_id(f.name) is not None}
    matched_results = []
    
    for f in nmr_files:
        fid = get_file_id(f.name)
        parsed_nmr = parse_nmr_source_v181(f.getvalue(), f.name)
        if parsed_nmr and fid in energy_map and energy_map[fid]:
            matched_results.append({
                "id": fid, "filename_nmr": f.name, "filename_energy": energy_map[fid]["filename"],
                "energy": energy_map[fid]["energy"], "energy_type": energy_map[fid]["type"], "atoms": parsed_nmr["atoms"],
                "frequency": frequency_map[fid]
            })

    if matched_results:
        # 1.5 Frequency / imaginary-mode check
        st.subheader("🫨 Phase 1.5: Frequency Check")
        freq_df = pd.DataFrame({
            "ID": [r["id"] for r in matched_results],
            "Energy File": [r["filename_energy"] for r in matched_results],
            "Normal Termination": ["Yes" if r["frequency"]["normal_termination"] else "No" for r in matched_results],
            "Imaginary Modes": [len(r["frequency"]["imaginary"]) for r in matched_results],
            "Imaginary Frequencies (cm⁻¹)": [", ".join(f"{v:.2f}" for v in r["frequency"]["imaginary"]) or "—" for r in matched_results],
            "Frequency Status": [r["frequency"]["status"] for r in matched_results],
        }).sort_values("ID")
        st.dataframe(freq_df, use_container_width=True)
        st.download_button(
            "💾 Download Frequency Check CSV",
            data=freq_df.to_csv(index=False).encode("utf-8"),
            file_name="Gaussian_Frequency_Check.csv",
            use_container_width=False,
        )

        imaginary_ids = [r["id"] for r in matched_results if r["frequency"]["imaginary"]]
        if imaginary_ids:
            st.warning(f"Imaginary frequency detected in conformer(s): {', '.join(map(str, imaginary_ids))}. Review before interpreting Boltzmann populations.")
        exclude_imaginary = st.checkbox(
            "Exclude conformers with any imaginary frequency from Boltzmann averaging",
            value=False,
            help="Off by default: inspect the table first. Enable only when you decide those structures should not enter the ensemble.",
        )
        if exclude_imaginary:
            matched_results = [r for r in matched_results if not r["frequency"]["imaginary"]]
            if not matched_results:
                st.error("No conformers remain after excluding imaginary-frequency structures.")
                st.stop()

        # 2. Boltzmann Summary
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

        # --- Phase 2.5: Raw Data Verification ---
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
        st.write("Raw shielding constants and tensor components extracted from each conformer:")
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
                if not df_labeled.empty:
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
