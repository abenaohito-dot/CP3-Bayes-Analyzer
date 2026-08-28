import streamlit as st
import pandas as pd
import numpy as np
import re
import math
import zipfile
from io import BytesIO

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

def clean_gaussian_route_string(raw_route):
    """Normalize route string by joining split words caused by Gaussian line-wrapping."""
    lines = [line.strip() for line in raw_route.splitlines() if line.strip()]
    full_str = " ".join(lines)
    full_str = re.sub(r'=\s+', '=', full_str)
    full_str = re.sub(r'\s+=', '=', full_str)
    full_str = re.sub(r'empiricaldispersion=g\s+d3', 'empiricaldispersion=gd3', full_str, flags=re.IGNORECASE)
    full_str = re.sub(r'geom=conn\s+ectivity', 'geom=connectivity', full_str, flags=re.IGNORECASE)
    return full_str

def parse_route_details(original_route):
    """
    Extract method_basis, solvent_model, solvent_name, and dispersion from original Gaussian route.
    """
    if not original_route:
        return {
            "method_basis": "wb97xd/6-311+g(d,p)",
            "solvent_model": "iefpcm",
            "solvent_name": "acetone",
            "dispersion": None
        }
    
    normalized = clean_gaussian_route_string(original_route)
    
    # 1. Method / Basis (例: wb97xd/6-311+g(d,p), B3LYP/6-31G*, m062x/def2tzvp)
    method_basis = None
    mb_match = re.search(r'(?:^|\s)(?:#\w*\s+)?([A-Za-z0-9_\-\+\*]+/[A-Za-z0-9_\-\+\*\(\),]+)', normalized, re.IGNORECASE)
    if mb_match:
        method_basis = mb_match.group(1)
    else:
        tokens = normalized.split()
        method_basis = "wb97xd/6-311+g(d,p)"
        for tok in tokens:
            if "/" in tok and not tok.lower().startswith(("opt", "int", "scrf", "geom", "guess")):
                method_basis = tok
                break
                
    # 2. Solvation (SCRF)
    solvent_model = "gas"
    solvent_name = ""
    s_m = re.search(r'scrf(?:\s*=\s*(?:\(([^\)]+)\)|([^\s]+)))?', normalized, re.IGNORECASE)
    if s_m:
        scrf_content = s_m.group(1) or s_m.group(2) or ""
        scrf_lower = scrf_content.lower()
        if "smd" in scrf_lower:
            solvent_model = "smd"
        elif "iefpcm" in scrf_lower:
            solvent_model = "iefpcm"
        elif "pcm" in scrf_lower:
            solvent_model = "pcm"
        else:
            solvent_model = "iefpcm"
            
        solv_m = re.search(r'solvent\s*=\s*([A-Za-z0-9_\-]+)', scrf_content, re.IGNORECASE)
        if solv_m:
            solvent_name = solv_m.group(1).lower()
        else:
            solvent_name = "acetone"
            
    # 3. Empirical Dispersion
    d_m = re.search(r'(empiricaldispersion=[^\s]+)', normalized, re.IGNORECASE)
    dispersion = d_m.group(1) if d_m else None
    
    return {
        "method_basis": method_basis,
        "solvent_model": solvent_model,
        "solvent_name": solvent_name,
        "dispersion": dispersion
    }

def compose_route(method_basis, solvent_model, solvent_name, dispersion=None):
    """Safely build route section from modular components."""
    parts = ["#p", method_basis.strip()]
    if dispersion and "b3lyp" in method_basis.lower() and "empiricaldispersion" not in method_basis.lower():
        parts.append(dispersion)
        
    if solvent_model and solvent_model.lower() != "gas":
        model_name = solvent_model.lower()
        if solvent_name and solvent_name.strip():
            parts.append(f"scrf=({model_name},solvent={solvent_name.strip().lower()})")
        else:
            parts.append(f"scrf={model_name}")
            
    parts.extend([
        "int=ultrafine",
        "opt=(tight,calcfc,cartesian,maxstep=5,maxcycles=300)",
        "freq"
    ])
    return " ".join(parts)

@st.cache_data(show_spinner=False)
def parse_frequency_source(file_bytes, filename):
    """Read Gaussian frequencies and, when possible, the first imaginary mode and original route."""
    content = file_bytes.decode("utf-8", errors="replace")
    frequencies = []
    frequency_matches = list(re.finditer(r"Frequencies --\s+([^\n]+)", content))
    for match in frequency_matches:
        frequencies.extend(float(value) for value in re.findall(r"-?\d+\.\d+", match.group(1)))

    normal_termination = "Normal termination of Gaussian" in content
    imaginary = sorted(value for value in frequencies if value < 0)
    mode_frequency, mode_vector, geometry = None, None, None

    # 元の Route section を抽出
    route_match = re.search(r"-{5,}\n\s*(#[^\n]+(?:\n\s*[^\n-]+)*)\n\s*-{5,}", content)
    original_route = None
    if route_match:
        original_route = " ".join(line.strip() for line in route_match.group(1).splitlines())

    # Standard orientation または Input orientation から最適化構造を取得
    orient_pattern = re.compile(
        r"(?:Standard|Input)\s+orientation:\s*\n\s*-+\s*\n\s*Center\s+Atomic\s+Atomic\s+Coordinates \(Angstroms\)\s*\n"
        r"\s*Number\s+Number\s+Type\s+X\s+Y\s+Z\s*\n\s*-+\s*\n(.*?)\n\s*-+",
        re.DOTALL,
    )
    orientation_blocks = orient_pattern.findall(content)
    if orientation_blocks:
        parsed_geometry = []
        for row in orientation_blocks[-1].splitlines():
            fields = row.split()
            if len(fields) >= 6 and fields[0].isdigit() and fields[1].isdigit():
                parsed_geometry.append((int(fields[1]), float(fields[3]), float(fields[4]), float(fields[5])))
        geometry = parsed_geometry or None

    # 第一虚振動の変位ベクトルを取得
    if geometry and imaginary:
        for freq_match in frequency_matches:
            block_frequencies = [float(value) for value in re.findall(r"-?\d+\.\d+", freq_match.group(1))]
            for mode_column, value in enumerate(block_frequencies):
                if value >= 0:
                    continue
                following_lines = content[freq_match.end():].splitlines()
                header_index = next((i for i, line in enumerate(following_lines[:40]) if re.search(r"Atom\s+AN\s+X\s+Y\s+Z", line)), None)
                if header_index is None:
                    continue
                vectors = []
                for line in following_lines[header_index + 1:]:
                    fields = line.split()
                    if len(fields) < 2 + (mode_column + 1) * 3 or not fields[0].isdigit() or not fields[1].isdigit():
                        if vectors:
                            break
                        continue
                    try:
                        xyz = [float(number) for number in fields[2 + 3 * mode_column: 5 + 3 * mode_column]]
                    except ValueError:
                        continue
                    vectors.append((int(fields[1]), *xyz))
                if len(vectors) == len(geometry) and [row[0] for row in vectors] == [row[0] for row in geometry]:
                    mode_frequency, mode_vector = value, vectors
                    break
            if mode_vector:
                break

    charge_matches = re.findall(r"Charge\s*=\s*(-?\d+)\s+Multiplicity\s*=\s*(\d+)", content)
    charge, multiplicity = (int(charge_matches[-1][0]), int(charge_matches[-1][1])) if charge_matches else (0, 1)
    
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
        "geometry": geometry,
        "mode_vector": mode_vector,
        "mode_frequency": mode_frequency,
        "charge": charge,
        "multiplicity": multiplicity,
        "original_route": original_route,
    }

def build_displaced_gjf(freq_data, conformer_id, direction, displacement, route, link0):
    """Generate an explicit-coordinate reoptimization input; no old checkpoint is required."""
    element_symbols = {
        1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
        11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
        19: "K", 20: "Ca", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn", 34: "Se", 35: "Br", 53: "I"
    }
    sign = 1.0 if direction == "plus" else -1.0
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(conformer_id))
    mode_label = f"{abs(freq_data['mode_frequency']):.2f}".replace(".", "p")
    chk_name = f"conf_{safe_id}_imag{mode_label}_{direction}.chk"
    
    clean_route = " ".join(route.split())
    if not clean_route.startswith("#"):
        clean_route = "#p " + clean_route
    
    # %chkの重複を除外
    lines = [line.strip() for line in link0.splitlines() if line.strip() and not line.strip().lower().startswith("%chk")]
    lines.append(f"%chk={chk_name}")
    lines.extend([
        clean_route,
        "",
        f"Conformer {conformer_id}: {direction} displacement along {freq_data['mode_frequency']:.4f} cm-1",
        "",
        f"{freq_data['charge']} {freq_data['multiplicity']}"
    ])
    
    for (atomic_no, x, y, z), (_, vx, vy, vz) in zip(freq_data["geometry"], freq_data["mode_vector"]):
        symbol = element_symbols.get(atomic_no, str(atomic_no))
        lines.append(f"{symbol:<2} {x + sign * displacement * vx: .8f} {y + sign * displacement * vy: .8f} {z + sign * displacement * vz: .8f}")
    
    return "\n".join(lines) + "\n\n"

@st.cache_data(show_spinner=False)
def parse_nmr_source_v181(file_bytes, filename):
    """
    Extract Isotropic values and diagonal tensor components (XX, YY, ZZ) safely and quickly.
    Iterates line by line to avoid Catastrophic Backtracking on large logs.
    """
    content = file_bytes.decode("utf-8", errors="replace")
    lines = content.splitlines()
    
    atoms = []
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
st.set_page_config(page_title="NMR DATA BRIDGE v2.0", layout="wide")
st.title("🌉 NMR DATA BRIDGE Ver. 2.0")
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

# OPTファイルがあれば、まず虚振動チェックと再投入GJF生成を有効化
if energy_files:
    energy_map = {get_file_id(f.name): parse_energy_source(f.getvalue(), f.name) for f in energy_files if get_file_id(f.name) is not None}
    frequency_map = {get_file_id(f.name): parse_frequency_source(f.getvalue(), f.name) for f in energy_files if get_file_id(f.name) is not None}
    
    opt_results = []
    for fid, freq in frequency_map.items():
        if fid in energy_map and energy_map[fid]:
            opt_results.append({
                "id": fid,
                "filename_energy": energy_map[fid]["filename"],
                "energy": energy_map[fid]["energy"],
                "energy_type": energy_map[fid]["type"],
                "frequency": freq,
            })

    if opt_results:
        # 1.5 Frequency / imaginary-mode check (OPTファイルのみで動作)
        st.subheader("🫨 Phase 1.5: Frequency Check")
        freq_df = pd.DataFrame({
            "ID": [r["id"] for r in opt_results],
            "Energy File": [r["filename_energy"] for r in opt_results],
            "Normal Termination": ["Yes" if r["frequency"]["normal_termination"] else "No" for r in opt_results],
            "Imaginary Modes": [len(r["frequency"]["imaginary"]) for r in opt_results],
            "Imaginary Frequencies (cm⁻¹)": [", ".join(f"{v:.2f}" for v in r["frequency"]["imaginary"]) or "—" for r in opt_results],
            "Frequency Status": [r["frequency"]["status"] for r in opt_results],
        }).sort_values("ID")
        st.dataframe(freq_df, use_container_width=True)
        st.download_button(
            "💾 Download Frequency Check CSV",
            data=freq_df.to_csv(index=False).encode("utf-8"),
            file_name="Gaussian_Frequency_Check.csv",
            use_container_width=False,
        )

        # 1.6 Generate +/- displaced reoptimization inputs (OPTファイルのみで動作)
        ready_for_restart = [r for r in opt_results if r["frequency"]["mode_vector"]]
        if ready_for_restart:
            st.subheader("🧭 Phase 1.6: Generate ±-Mode Reoptimization GJFs")
            st.caption("Creates explicit-coordinate inputs from the first detected imaginary mode. These files do not need an old checkpoint.")
            
            # 元ログから条件を抽出
            orig_details = parse_route_details(ready_for_restart[0]["frequency"].get("original_route"))
            
            # 3つのセレクターを横並びで配置
            col_m1, col_m2, col_m3 = st.columns([1.3, 1.0, 1.0])
            
            with col_m1:
                method_options = [
                    f"✨ [Inherit] {orig_details['method_basis']}",
                    "🌟 wB97X-D / 6-311+G(d,p)",
                    "M06-2X / def2-TZVP",
                    "B3LYP-D3BJ / 6-311+G(d,p)",
                    "⚠️ B3LYP / 6-31G(d) (Exploration only)",
                    "Custom (自由入力)"
                ]
                sel_method = st.selectbox("1. 汎関数 / 基底関数", options=method_options, index=0)
                if "Inherit" in sel_method:
                    active_method = orig_details['method_basis']
                elif "wB97X-D" in sel_method:
                    active_method = "wb97xd/6-311+g(d,p)"
                elif "M06-2X" in sel_method:
                    active_method = "m062x/def2tzvp"
                elif "B3LYP-D3BJ" in sel_method:
                    active_method = "b3lyp/6-311+g(d,p) empiricaldispersion=gd3bj"
                elif "B3LYP / 6-31G(d)" in sel_method:
                    active_method = "b3lyp/6-31g(d)"
                else:
                    active_method = st.text_input("Method/Basis 手動入力", value=orig_details['method_basis'])

            with col_m2:
                model_options = [
                    f"✨ [Inherit] {orig_details['solvent_model'].upper()}" if orig_details['solvent_model'] != 'gas' else "✨ [Inherit] Gas Phase (気相)",
                    "IEFPCM (scrf=iefpcm)",
                    "SMD (scrf=smd)",
                    "PCM (scrf=pcm)",
                    "None / Gas Phase (気相)"
                ]
                sel_model = st.selectbox("2. 溶媒モデル", options=model_options, index=0)
                if "Inherit" in sel_model:
                    active_model = orig_details['solvent_model']
                elif "IEFPCM" in sel_model:
                    active_model = "iefpcm"
                elif "SMD" in sel_model:
                    active_model = "smd"
                elif "PCM" in sel_model:
                    active_model = "pcm"
                else:
                    active_model = "gas"

            with col_m3:
                if active_model != "gas":
                    preset_solvents = ["acetone", "chloroform", "methanol", "dmso", "water", "thf", "acetonitrile", "dichloromethane", "toluene"]
                    solvent_list = [f"✨ [Inherit] {orig_details['solvent_name']}"] if orig_details['solvent_name'] else []
                    for s_name in preset_solvents:
                        if s_name not in solvent_list:
                            solvent_list.append(s_name)
                    solvent_list.append("Custom (自由入力)")
                    
                    sel_solvent = st.selectbox("3. 溶媒名", options=solvent_list, index=0)
                    if "Inherit" in sel_solvent:
                        active_solvent = orig_details['solvent_name'] or "acetone"
                    elif "Custom" in sel_solvent:
                        active_solvent = st.text_input("溶媒名 手動入力", value="acetone").strip()
                    else:
                        active_solvent = sel_solvent
                else:
                    st.selectbox("3. 溶媒名", options=["— (気相: 溶媒なし)"], disabled=True)
                    active_solvent = ""

            # リアルタイムで Route section を合成
            composed_route = compose_route(active_method, active_model, active_solvent, orig_details.get("dispersion"))

            # プレビュー表示 & Link 0 設定
            col_prev1, col_prev2 = st.columns([1.3, 0.9])
            with col_prev1:
                restart_route = st.text_area(
                    "📝 Reoptimization Route Preview (自動生成・微調整可能)",
                    value=composed_route,
                    help="上記で選択された条件から自動構築された Route section です。手動で追記・微調整も可能です。",
                    height=100
                )
            with col_prev2:
                restart_link0 = st.text_area(
                    "⚙️ Optional Link 0 settings",
                    value="%mem=48GB\n%nprocshared=12",
                    help="One directive per line. %chk is added automatically.",
                    height=100
                )

            # 変位スケールと対象配座の選択カラム
            col_param1, col_param2 = st.columns([1, 2])
            with col_param1:
                displacement = st.number_input(
                    "Mode displacement scale",
                    min_value=0.01,
                    max_value=1.00,
                    value=0.20,
                    step=0.01,
                    help="0.20 is a gentle starting displacement along the Gaussian-printed normal-mode vector."
                )
            with col_param2:
                restart_ids = st.multiselect(
                    "Conformers to generate",
                    options=[r["id"] for r in ready_for_restart],
                    default=[r["id"] for r in ready_for_restart],
                )

            # GJF 生成 & ZIP ダウンロード
            if restart_route.strip() and restart_ids:
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                    for result in ready_for_restart:
                        if result["id"] not in restart_ids:
                            continue
                        mode_label = f"{abs(result['frequency']['mode_frequency']):.2f}".replace(".", "p")
                        for direction in ("plus", "minus"):
                            filename = f"conf_{result['id']}_imag{mode_label}_{direction}.gjf"
                            archive.writestr(
                                filename,
                                build_displaced_gjf(
                                    result["frequency"],
                                    result["id"],
                                    direction,
                                    displacement,
                                    restart_route,
                                    restart_link0
                                )
                            )
                st.download_button(
                    "💾 Download ±-Mode Reoptimization GJFs (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="imaginary_mode_reoptimization_inputs.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            elif not restart_route.strip():
                st.info("Enter or select a route section to enable GJF generation.")

        imaginary_ids = [r["id"] for r in opt_results if r["frequency"]["imaginary"]]
        if imaginary_ids:
            st.warning(f"Imaginary frequency detected in conformer(s): {', '.join(map(str, imaginary_ids))}. Review before interpreting Boltzmann populations.")
        exclude_imaginary = st.checkbox(
            "Exclude conformers with any imaginary frequency from Boltzmann averaging",
            value=False,
            help="Off by default: inspect the table first. Enable only when you decide those structures should not enter the ensemble.",
        )

        # --- Phase 2 以降: NMRファイルもアップロードされた場合に実行 ---
        if nmr_files:
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
        else:
            st.info("💡 Awaiting NMR files (Drop NMR files in Phase 1) to enable Boltzmann averaging and shielding integration.")

elif nmr_files:
    st.info("Awaiting OPT files to extract energies and conformer geometries.")
