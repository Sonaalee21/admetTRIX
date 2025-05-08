import streamlit as st
from datetime import datetime
import base64
import io
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem, Lipinski, Crippen
from rdkit.Chem import rdMolDescriptors # For TPSA, ExactMolWt etc.
import matplotlib.pyplot as plt
from math import pi

# Check if stmol is available, otherwise provide a message
try:
    from stmol import showmol
    STMOL_AVAILABLE = True
except ImportError:
    STMOL_AVAILABLE = False

# --- PAGE CONFIG ---
st.set_page_config(layout="wide", page_title="ADMETrix", page_icon="🧪")

# --- NAVIGATION STATE ---
if "nav_selected" not in st.session_state:
    st.session_state["nav_selected"] = "Home"
if "analysis_triggered" not in st.session_state:
    st.session_state.analysis_triggered = False
if "input_data" not in st.session_state:
    st.session_state.input_data = None
if "input_type" not in st.session_state:
    st.session_state.input_type = None
if "molecule_data_store" not in st.session_state:
    st.session_state.molecule_data_store = []
if "selected_molecule_index" not in st.session_state:
    st.session_state.selected_molecule_index = 0


def nav_button(label, icon=""):
    if st.button(f"{icon} {label}".strip(), key=f"nav_{label}", use_container_width=True):
        st.session_state["nav_selected"] = label
        if label != "Analysis":
            st.session_state.analysis_triggered = False
            st.session_state.molecule_data_store = []
            st.session_state.selected_molecule_index = 0


# --- RDKIT HELPER FUNCTIONS ---

def get_mol_from_sdf_string(sdf_string):
    try:
        suppl = Chem.SDMolSupplier()
        suppl.SetData(sdf_string)
        if len(suppl) > 0 and suppl[0] is not None:
            mol = suppl[0]
            AllChem.Compute2DCoords(mol)
            return mol
        return None
    except Exception:
        return None

def get_mol_from_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            AllChem.Compute2DCoords(mol)
        return mol
    except Exception:
        return None

def generate_2d_image(mol):
    if not mol: return None
    try:
        img = Draw.MolToImage(mol, size=(300,300), kekulize=True)
        bio = io.BytesIO()
        img.save(bio, format='PNG')
        return bio.getvalue()
    except Exception:
        return None

def get_xyz_from_mol(mol):
    if not mol: return None
    try:
        mol_3d = Chem.Mol(mol)
        mol_3d.RemoveAllConformers()
        mol_3d_h = Chem.AddHs(mol_3d)
        AllChem.EmbedMolecule(mol_3d_h, AllChem.ETKDGv3())
        AllChem.UFFOptimizeMolecule(mol_3d_h)
        return Chem.MolToXYZBlock(mol_3d_h)
    except Exception:
        return None

def calculate_physicochemical_properties(mol):
    if not mol: return {"Error": "Invalid molecule object"}
    properties = {}
    try:
        properties["Molecular Weight (MW)"] = round(rdMolDescriptors.CalcExactMolWt(mol), 2)
        properties["LogP (Crippen)"] = round(Crippen.MolLogP(mol), 2)
        properties["Topological Polar Surface Area (TPSA)"] = round(rdMolDescriptors.CalcTPSA(mol), 2)
        properties["Hydrogen Bond Acceptors (HBA)"] = Lipinski.NumHAcceptors(mol)
        properties["Hydrogen Bond Donors (HBD)"] = Lipinski.NumHDonors(mol)
        properties["Rotatable Bonds"] = Lipinski.NumRotatableBonds(mol)
        properties["Number of Rings"] = rdMolDescriptors.CalcNumRings(mol)
        properties["Fraction Csp3"] = round(Lipinski.FractionCSP3(mol), 2)
        properties["Molar Refractivity"] = round(Crippen.MolMR(mol), 2)
        properties["Number of Atoms"] = mol.GetNumAtoms()
        properties["ESOL Solubility (LogS)"] = "calc_pending" # Placeholder, not numeric
    except Exception as e:
        # If any error during calculation, return a dict indicating global error
        # Also include any properties calculated so far, but they might be incomplete
        properties["Error_PropertyCalculation"] = f"Error during property calculation: {e}"
        return properties # Return partially filled + error
    return properties

def check_drug_likeness_rules(props):
    if not props: return {}
    if isinstance(props.get("Error"), str) and len(props) == 1: return {} # Global error from upstream
    if props.get("Error_PropertyCalculation"): return {"Error": props.get("Error_PropertyCalculation")}


    required_numeric_keys = [
        "Molecular Weight (MW)", "LogP (Crippen)", "Topological Polar Surface Area (TPSA)",
        "Hydrogen Bond Acceptors (HBA)", "Hydrogen Bond Donors (HBD)", "Rotatable Bonds",
        "Molar Refractivity", "Number of Atoms", "Number of Rings"
    ]
    for key in required_numeric_keys:
        val = props.get(key)
        if not isinstance(val, (int, float)):
            return {"Error": f"Required property '{key}' is not a valid number for drug-likeness rules."}

    mw = props["Molecular Weight (MW)"]
    logp = props["LogP (Crippen)"]
    hbd = props["Hydrogen Bond Donors (HBD)"]
    hba = props["Hydrogen Bond Acceptors (HBA)"]
    tpsa = props["Topological Polar Surface Area (TPSA)"]
    rb = props["Rotatable Bonds"]
    mr = props["Molar Refractivity"]
    num_atoms = props["Number of Atoms"]
    num_rings = props["Number of Rings"]
    rules = {}

    lipinski_violations = 0
    if mw > 500: lipinski_violations += 1
    if logp > 5: lipinski_violations += 1
    if hbd > 5: lipinski_violations += 1
    if hba > 10: lipinski_violations += 1
    rules["Lipinski's Rule of Five"] = "Pass" if lipinski_violations == 0 else f"Fail ({lipinski_violations} violations)"

    ghose_violations = 0
    if not (160 <= mw <= 480): ghose_violations +=1
    if not (-0.4 <= logp <= 5.6): ghose_violations +=1
    if not (40 <= mr <= 130): ghose_violations +=1
    if not (20 <= num_atoms <= 70): ghose_violations +=1
    rules["Ghose Filter"] = "Pass" if ghose_violations == 0 else f"Fail ({ghose_violations} violations)"

    veber_violations = 0
    if rb > 10: veber_violations += 1
    if tpsa > 140: veber_violations += 1
    rules["Veber Rule"] = "Pass" if veber_violations == 0 else f"Fail ({veber_violations} violations)"

    egan_violations = 0
    if logp > 5.88: egan_violations += 1
    if tpsa > 131.6: egan_violations += 1
    rules["Egan Rule (Bioavailability)"] = "Pass" if egan_violations == 0 else f"Fail ({egan_violations} violations)"

    muegge_violations = 0
    if not (200 <= mw <= 600): muegge_violations +=1
    if not (-2 <= logp <= 5): muegge_violations +=1
    if tpsa > 150: muegge_violations +=1
    if num_rings > 7: muegge_violations +=1
    if rb > 15: muegge_violations +=1
    if hba > 10: muegge_violations +=1
    if hbd > 5: muegge_violations +=1
    rules["Muegge Rule"] = "Pass" if muegge_violations == 0 else f"Fail ({muegge_violations} violations)"
    return rules

def predict_pharmacokinetics(props):
    if not props: return {}
    if isinstance(props.get("Error"), str) and len(props) == 1: return {}
    if props.get("Error_PropertyCalculation"): return {"Error": props.get("Error_PropertyCalculation")}

    required_numeric_keys = [
        "Molecular Weight (MW)", "LogP (Crippen)", "Topological Polar Surface Area (TPSA)",
        "Hydrogen Bond Donors (HBD)", "Hydrogen Bond Acceptors (HBA)"
    ]
    for key in required_numeric_keys:
        val = props.get(key)
        if not isinstance(val, (int, float)):
            return {"Error": f"Required property '{key}' is not a valid number for PK predictions."}

    mw = props["Molecular Weight (MW)"]
    logp = props["LogP (Crippen)"]
    tpsa = props["Topological Polar Surface Area (TPSA)"]
    hbd = props["Hydrogen Bond Donors (HBD)"]
    hba = props["Hydrogen Bond Acceptors (HBA)"]
    pk_preds = {}

    lipinski_violations = 0
    if mw > 500: lipinski_violations += 1
    if logp > 5: lipinski_violations += 1
    if hbd > 5: lipinski_violations += 1
    if hba > 10: lipinski_violations += 1

    if lipinski_violations == 0: pk_preds["GI Absorption"] = "High"
    elif lipinski_violations == 1: pk_preds["GI Absorption"] = "Moderate"
    else: pk_preds["GI Absorption"] = "Low"

    if tpsa < 80 and logp > 0 and logp < 5:
        pk_preds["BBB Permeation"] = "Likely Permeant"
    else:
        pk_preds["BBB Permeation"] = "Likely Non-Permeant"

    pk_preds["P-gp Substrate"] = "Not predicted"
    pk_preds["CYP Inhibition"] = "Not predicted"
    return pk_preds

def get_bioavailability_radar_data(props):
    if not props: return None # Radar specific, returns None if error
    if isinstance(props.get("Error"), str) and len(props) == 1: return None
    if props.get("Error_PropertyCalculation"): return None


    required_numeric_keys = [
        "LogP (Crippen)", "Molecular Weight (MW)", "Topological Polar Surface Area (TPSA)",
        "Fraction Csp3", "Rotatable Bonds"
    ]
    temp_data = {}
    for key in required_numeric_keys:
        val = props.get(key)
        if not isinstance(val, (int, float)):
            # For radar, we can try to proceed with defaults if a value is missing, or return None
            # st.warning(f"Radar data: Required property '{key}' is not a valid number. Using default for radar.")
            # For simplicity now, let's return None if any critical radar param is bad
            return None 
        temp_data[key] = val
    
    data = {
        "LIPO (LogP)": temp_data["LogP (Crippen)"],
        "SIZE (MW)": temp_data["Molecular Weight (MW)"],
        "POLAR (TPSA)": temp_data["Topological Polar Surface Area (TPSA)"],
        "INSOLU (5-LogP)": max(0, 5 - temp_data["LogP (Crippen)"]),
        "INSATU (FracCsp3)": temp_data["Fraction Csp3"],
        "FLEX (9-RotB)": max(0, 9 - temp_data["Rotatable Bonds"])
    }
    return data

def plot_bioavailability_radar(radar_data, mol_name="Molecule"):
    if not radar_data: return None
    
    labels_raw = list(radar_data.keys())
    stats_raw = list(radar_data.values())

    ideal_plot_ranges = {
        "LIPO (LogP)": (-2, 6),
        "SIZE (MW)": (0, 600),
        "POLAR (TPSA)": (0, 160),
        "INSOLU (5-LogP)": (-1, 7),
        "INSATU (FracCsp3)": (0, 1),
        "FLEX (9-RotB)": (-1, 10)
    }
    
    stats_normalized = []
    for i, label in enumerate(labels_raw):
        min_val, max_val = ideal_plot_ranges[label]
        val = stats_raw[i]
        normalized = (val - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else 0
        stats_normalized.append(np.clip(normalized, 0, 1))

    angles = np.linspace(0, 2 * np.pi, len(labels_raw), endpoint=False).tolist()
    stats_plot = stats_normalized + stats_normalized[:1]
    angles_plot = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(4,4), subplot_kw=dict(polar=True))
    ax.plot(angles_plot, stats_plot, color='#D32F2F', linewidth=1.5)
    ax.fill(angles_plot, stats_plot, '#EF9A9A', alpha=0.5)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels_raw, fontsize=7)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2","0.4","0.6","0.8","1.0"], fontsize=6)
    ax.set_ylim(0, 1)

    ideal_zone_normalized = [
        ( (5.0 - ideal_plot_ranges["LIPO (LogP)"][0]) / (ideal_plot_ranges["LIPO (LogP)"][1] - ideal_plot_ranges["LIPO (LogP)"][0]) + \
          (-0.7 - ideal_plot_ranges["LIPO (LogP)"][0]) / (ideal_plot_ranges["LIPO (LogP)"][1] - ideal_plot_ranges["LIPO (LogP)"][0]) ) / 2,
        ( (500 - ideal_plot_ranges["SIZE (MW)"][0]) / (ideal_plot_ranges["SIZE (MW)"][1] - ideal_plot_ranges["SIZE (MW)"][0]) + \
          (150 - ideal_plot_ranges["SIZE (MW)"][0]) / (ideal_plot_ranges["SIZE (MW)"][1] - ideal_plot_ranges["SIZE (MW)"][0]) ) / 2,
        ( (130 - ideal_plot_ranges["POLAR (TPSA)"][0]) / (ideal_plot_ranges["POLAR (TPSA)"][1] - ideal_plot_ranges["POLAR (TPSA)"][0]) + \
          (20 - ideal_plot_ranges["POLAR (TPSA)"][0]) / (ideal_plot_ranges["POLAR (TPSA)"][1] - ideal_plot_ranges["POLAR (TPSA)"][0]) ) / 2,
        ( ( (5 - (-0.7)) - ideal_plot_ranges["INSOLU (5-LogP)"][0]) / (ideal_plot_ranges["INSOLU (5-LogP)"][1] - ideal_plot_ranges["INSOLU (5-LogP)"][0]) + \
          ( (5 - 4.0) - ideal_plot_ranges["INSOLU (5-LogP)"][0]) / (ideal_plot_ranges["INSOLU (5-LogP)"][1] - ideal_plot_ranges["INSOLU (5-LogP)"][0]) ) / 2,
        ( (1.0 - ideal_plot_ranges["INSATU (FracCsp3)"][0]) / (ideal_plot_ranges["INSATU (FracCsp3)"][1] - ideal_plot_ranges["INSATU (FracCsp3)"][0]) + \
          (0.25 - ideal_plot_ranges["INSATU (FracCsp3)"][0]) / (ideal_plot_ranges["INSATU (FracCsp3)"][1] - ideal_plot_ranges["INSATU (FracCsp3)"][0]) ) / 2,
        ( ( (9-0) - ideal_plot_ranges["FLEX (9-RotB)"][0]) / (ideal_plot_ranges["FLEX (9-RotB)"][1] - ideal_plot_ranges["FLEX (9-RotB)"][0]) + \
          ( (9-9) - ideal_plot_ranges["FLEX (9-RotB)"][0]) / (ideal_plot_ranges["FLEX (9-RotB)"][1] - ideal_plot_ranges["FLEX (9-RotB)"][0]) ) / 2,
    ]
    ideal_zone_normalized_clipped = [np.clip(val, 0, 1) for val in ideal_zone_normalized]
    ideal_zone_plot = ideal_zone_normalized_clipped + ideal_zone_normalized_clipped[:1]
    ax.plot(angles_plot, ideal_zone_plot, color='#4CAF50', linewidth=1, linestyle='dashed', label='Approx. Ideal Zone')

    ax.set_title(f"Bioavailability Radar: {mol_name}", size=9, y=1.15)
    
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=100)
    plt.close(fig)
    return buf.getvalue()

# --- STYLING ---
st.markdown("""
    <style>
    body { font-family: 'Arial', sans-serif; color: #333333; }
    .main-title {
        font-size: 3rem; font-weight: bold; color: #00695C; /* Darker Teal */
        text-align: center; letter-spacing: 2px; margin-bottom: 1rem;
        padding-top: 1rem; text-shadow: 1px 1px 2px #B0B0B0;
    }
    .page-title {
        font-size: 2.2rem; color: #00796B; /* Teal */
        border-bottom: 2px solid #004D40; /* Darkest Teal */
        padding-bottom: 0.5rem; margin-top: 1rem; margin-bottom: 1.5rem;
    }
    .page-subtitle {
        font-size: 1.3rem; color: #00897B; /* Lighter Teal */
        margin-bottom: 1rem; font-weight: bold;
    }
    .content-box {
        background: #F8F9FA; border-left: 5px solid #00796B; /* Teal border */
        border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .feature-box {background: #E0F2F1; border-left-color: #00796B; padding: 1.2rem; margin: 1rem 0; border-radius: 8px;} /* Light Teal bg */
    .step-box {background: #E8F5E9; border-left-color: #388E3C; padding: 1.2rem; margin: 1rem 0; border-radius: 8px; font-size: 1.1rem;} /* Light Green bg */
    .purpose-box {background: #FFFDE7; border-left-color: #FBC02D; padding: 1.2rem; margin: 1rem 0; border-radius: 8px;} /* Light Yellow bg */
    .benefit-box {background: #E1F5FE; border-left-color: #0288D1; padding: 1.2rem; margin: 1rem 0; border-radius: 8px;} /* Light Blue bg */
    .acknowledgement-box {background: #F3E5F5; border-left-color: #8E24AA; padding: 1.2rem; margin: 1rem 0; border-radius: 8px;} /* Light Purple bg */

    .circle-img-container { display: flex; justify-content: center; margin-bottom: 1rem; }
    .circle-img {
        border-radius: 50%; width: 150px; height: 150px; object-fit: cover;
        border: 6px solid #00695C; box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }
    .author-name { text-align:center; font-size:1.4rem; font-weight:bold; color: #00695C; margin-top: 0.5rem; }
    .author-title { text-align:center; font-size:1.1rem; color: #555555; margin-bottom: 1.5rem; }
    .linkedin-btn {
        background-color: #0077b5; color: white !important; padding: 0.6em 1.2em;
        border-radius: 20px; text-decoration: none; font-weight: bold;
        display: inline-block; border: none; cursor: pointer; transition: background-color 0.3s ease;
    }
    .linkedin-btn:hover { background-color: #005582; color: white !important; text-decoration: none; }
    ul { padding-left: 20px; }
    li { margin-bottom: 0.5rem; }
    .stDataFrame { width: 100% !important; }
    </style>
""", unsafe_allow_html=True)

# --- HEADER & NAVIGATION BAR ---
st.markdown('<div class="main-title">ADMETrix</div>', unsafe_allow_html=True)

cols_nav = st.columns([1, 1.5, 1.5, 1.5, 1.5, 1])
with cols_nav[1]: nav_button("Home", "🏠")
with cols_nav[2]: nav_button("User Guide", "📖")
with cols_nav[3]: nav_button("Analysis", "🧪")
with cols_nav[4]: nav_button("About", "ℹ️")

nav_selected = st.session_state["nav_selected"]
st.markdown("---")

# --- PAGE CONTENT ---
if nav_selected == "Home":
    st.markdown("<div class='page-title'>ADMETrix : Bioinformatics Drug Nexus</div>", unsafe_allow_html=True)
    st.markdown("""
    Welcome to **ADMETrix**, your integrated bioinformatics web server for essential compound analysis tasks.
    This platform is designed for students, researchers, and educators working in the fields of **structural bioinformatics, drug discovery, and computational biology!**
    """)
    st.markdown("<div class='page-subtitle'>Key Features</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-box">
    <ul>
        <li><b>📊 2D & 3D Visualization</b>: Instantly view 2D chemical structures and interactive 3D molecular models.</li>
        <li><b>🧪 Physicochemical Properties Panel</b>: Access a comprehensive set of descriptors including Molecular Weight, TPSA, LogP, LogS (ESOL placeholder), and more.</li>
        <li><b>💊 Drug-likeness Rules Panel</b>: Evaluate compounds against Lipinski, Ghose, Veber, Egan, and Muegge rules, complete with violation counts.</li>
        <li><b>🔬 Medicinal Chemistry Friendliness</b>: (Planned) Automatically check for PAINS alerts, Brenk structural alerts, lead-likeness, and synthetic accessibility.</li>
        <li><b> pharmacokinetic Predictions</b>: Predict key ADME properties like GI absorption, Blood-Brain Barrier (BBB) permeation. P-gp and CYP status noted as complex.</li>
        <li><b>📈 Bioavailability Radar</b>: Utilize SwissADME-style radar plots for a quick visual assessment of oral bioavailability.</li>
        <li><b>🥚 BOILED-Egg Plot</b>: (Planned) Visualize GI absorption and BBB permeation predictions for multiple molecules simultaneously.</li>
        <li><b>🔗 Drug-Drug Interaction (DDI) Demo</b>: (Planned) Identify potential interactions within your set of compounds.</li>
        <li><b>🔄 Batch Input & Download</b>: Analyze multiple molecules at once (SMILES or SDF) and download all results conveniently.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>Why use ADMETrix?</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="content-box">
    <ul>
        <li>🌐 **No installation needed** — fully browser-based and accessible anywhere.</li>
        <li>👍 **Beginner-friendly**, with clear explanations, expandable sections, and interactive controls.</li>
        <li>🎓 **Educational & Research-Oriented**: Designed for learning, teaching, research, and practical applications in bioinformatics and cheminformatics.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

elif nav_selected == "User Guide":
    st.markdown("<div class='page-title'>User Guide 🚀</div>", unsafe_allow_html=True)
    st.markdown("Follow these simple steps to analyze your compounds using ADMETrix:")
    st.markdown("""
    <div class="step-box">
        <b>Step 1: Navigate to Analysis</b><br>
        Click on the <b>🧪 Analysis</b> tab from the navigation bar at the top.
    </div>
    <div class="step-box">
        <b>Step 2: Select Input Type</b><br>
        Choose your preferred input method:
        <ul>
            <li><b>SDF Files</b>: For uploading one or more <code>.sdf</code> files containing molecular structures.</li>
            <li><b>SMILES Strings</b>: For pasting SMILES strings directly (one per line for multiple molecules).</li>
        </ul>
    </div>
    <div class="step-box">
        <b>Step 3: Upload or Paste Data</b><br>
        <ul>
            <li>If using SDF: Click <b>Browse files</b> to select your <code>.sdf</code> file(s).</li>
            <li>If using SMILES: Paste your SMILES string(s) into the text area. Example SMILES are provided for quick testing.</li>
        </ul>
        Click the <b>🚀 Start Analysis</b> button to process your input.
    </div>
    <div class="step-box">
        <b>Step 4: Explore Results</b><br>
        A summary table of all processed molecules will be shown. If multiple molecules are analyzed, a selector will appear above the detailed results. Choose a molecule to view its specific data in expandable panels:
        <ul>
            <li>2D/3D Structures</li>
            <li>Physicochemical Properties</li>
            <li>Drug-Likeness Rules</li>
            <li>Pharmacokinetics Predictions</li>
            <li>Bioavailability Radar</li>
        </ul>
    </div>
    <div class="step-box">
        <b>Step 5: Download Data</b><br>
        Look for download buttons 📥 associated with individual images, plots, or tables. A button to download all results as a combined CSV is also provided at the end of the results section.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='content-box'><b>💡 Tip:</b> When using the SMILES option, try the example SMILES provided on the Analysis page for a quick start!</div>", unsafe_allow_html=True)

elif nav_selected == "Analysis":
    st.markdown("<div class='page-title'>🧪 ADMETrix Analysis</div>", unsafe_allow_html=True)
    st.markdown("#### Upload your molecules below:")

    cols_analysis_input = st.columns([1, 2, 1])
    with cols_analysis_input[1]:
        file_type = st.radio("Select Input Type:", ["SDF", "SMILES"], horizontal=True, label_visibility="collapsed", key="analysis_file_type_radio")
        uploaded_files_list = None
        smiles_input_str = ""

        if file_type == "SDF":
            uploaded_files_list = st.file_uploader("Browse SDF files", type=["sdf"], accept_multiple_files=True, key="upload_sdf_widget")
            if uploaded_files_list: st.success(f"{len(uploaded_files_list)} SDF file(s) selected.")
        elif file_type == "SMILES":
            st.markdown("##### Example SMILES:")
            st.code("CCOc1ccc(cc1)NC(=O)C\nCC(C)Cc1ccc(cc1)C(C)C(=O)O\nCN1C=NC2=C1C(=O)N(C(=O)N2C)C", language="smiles")
            smiles_input_str = st.text_area("Paste SMILES (one per line):", height=100,
                                        placeholder="CCOc1ccc(cc1)NC(=O)C\n...", key="analysis_smiles_input_area")
            if smiles_input_str: st.success("SMILES input received.")

        if st.button("🚀 Start Analysis", use_container_width=True, type="primary", key="start_analysis_action_button"):
            st.session_state.molecule_data_store = [] 
            st.session_state.analysis_triggered = False 
            st.session_state.selected_molecule_index = 0
            mols_to_process, mol_names, error_messages = [], [], []

            if file_type == "SDF" and uploaded_files_list:
                st.session_state.input_type = "SDF"
                for i, uploaded_file in enumerate(uploaded_files_list):
                    try:
                        sdf_string = uploaded_file.getvalue().decode("utf-8")
                        mol = get_mol_from_sdf_string(sdf_string)
                        if mol:
                            mols_to_process.append(mol)
                            name_from_sdf = mol.GetProp("_Name") if mol.HasProp("_Name") else uploaded_file.name.split('.')[0]
                            mol_names.append(name_from_sdf + f"_sdf_{i+1}")
                        else: error_messages.append(f"Could not parse molecule from SDF: {uploaded_file.name}")
                    except Exception as e: error_messages.append(f"Error processing SDF {uploaded_file.name}: {e}")
            elif file_type == "SMILES" and smiles_input_str.strip():
                st.session_state.input_type = "SMILES"
                smiles_list = [s.strip() for s in smiles_input_str.strip().split('\n') if s.strip()]
                for i, smi in enumerate(smiles_list):
                    mol = get_mol_from_smiles(smi)
                    if mol:
                        mols_to_process.append(mol)
                        mol_names.append(f"SMILES_{i+1}" + (f" ({smi[:15]}...)" if len(smi)>15 else f" ({smi})") )
                    else: error_messages.append(f"Could not parse SMILES: {smi}")
            
            if error_messages:
                for err in error_messages: st.warning(err)

            if mols_to_process:
                with st.spinner("Analyzing molecules..."):
                    for i, mol_obj in enumerate(mols_to_process):
                        mol_data = {"name": mol_names[i], "mol_obj_rdkit": mol_obj}
                        mol_data["physchem"] = calculate_physicochemical_properties(mol_obj)
                        mol_data["drug_likeness"] = check_drug_likeness_rules(mol_data["physchem"])
                        mol_data["pharmacokinetics"] = predict_pharmacokinetics(mol_data["physchem"])
                        mol_data["radar_params"] = get_bioavailability_radar_data(mol_data["physchem"])
                        mol_data["2d_image_bytes"] = generate_2d_image(mol_obj)
                        mol_data["xyz_str"] = get_xyz_from_mol(mol_obj)
                        st.session_state.molecule_data_store.append(mol_data)
                st.session_state.analysis_triggered = True
                st.success(f"Analysis complete for {len(st.session_state.molecule_data_store)} molecule(s).")
            elif not error_messages:
                st.warning("No valid input provided. Please upload/paste data and click 'Start Analysis'.")

    st.markdown("---")

    if st.session_state.get("analysis_triggered", False) and st.session_state.molecule_data_store:
        st.markdown("<div class='page-subtitle'>📊 Analysis Results Overview</div>", unsafe_allow_html=True)
        summary_data_list = []
        for idx, data in enumerate(st.session_state.molecule_data_store):
            lipinski_status = data["drug_likeness"].get("Lipinski's Rule of Five", "N/A")
            lip_viol = "N/A"
            if "Fail" in lipinski_status:
                try: lip_viol = lipinski_status.split('(')[1].split(' ')[0]
                except: lip_viol = "err"
            elif "Pass" in lipinski_status: lip_viol = "0"
            summary_data_list.append({
                "ID": idx + 1, "Name": data["name"],
                "MW": data["physchem"].get("Molecular Weight (MW)", "N/A"),
                "LogP": data["physchem"].get("LogP (Crippen)", "N/A"),
                "TPSA": data["physchem"].get("Topological Polar Surface Area (TPSA)", "N/A"),
                "Lipinski Viol.": lip_viol
            })
        st.dataframe(pd.DataFrame(summary_data_list), use_container_width=True, hide_index=True)

        mol_display_names = [f"{idx+1}. {data['name']}" for idx, data in enumerate(st.session_state.molecule_data_store)]
        if len(mol_display_names) > 1 :
             selected_mol_display_name = st.selectbox("Select molecule for detailed results:", mol_display_names,
                index=st.session_state.selected_molecule_index, key="molecule_selector_dropdown")
             st.session_state.selected_molecule_index = mol_display_names.index(selected_mol_display_name)
        
        current_mol_data = st.session_state.molecule_data_store[st.session_state.selected_molecule_index]
        mol_name_display = current_mol_data["name"]
        st.markdown(f"#### Detailed Results for: **{mol_name_display}**")

        with st.expander("🖼️ Molecular Structure (2D/3D)", expanded=True):
            col1_struct, col2_struct = st.columns(2)
            with col1_struct:
                st.subheader("2D Structure")
                if current_mol_data["2d_image_bytes"]:
                    st.image(current_mol_data["2d_image_bytes"])
                    st.download_button(label="Download 2D Image", data=current_mol_data["2d_image_bytes"],
                        file_name=f"{mol_name_display.replace(' ','_').replace('/','_')}_2D.png", mime="image/png",
                        key=f"download_2d_btn_{st.session_state.selected_molecule_index}")
                else: st.error("Could not generate 2D image.")
            with col2_struct:
                st.subheader("3D Structure")
                if STMOL_AVAILABLE and current_mol_data["xyz_str"]:
                    showmol(current_mol_data["xyz_str"], style='stick', height=350, width=350)
                elif not STMOL_AVAILABLE: st.warning("`stmol` library not installed. 3D view unavailable.")
                else: st.error("Could not generate 3D structure data (XYZ).")
        
        def display_and_download_df(title, data_dict, df_key_suffix, error_message="Data could not be processed."):
            if data_dict and not data_dict.get("Error"): # Check for specific error key or empty
                df = pd.DataFrame(list(data_dict.items()), columns=['Property', 'Value'])
                st.dataframe(df, use_container_width=True, hide_index=True)
                st.download_button(
                    label=f"Download {title} CSV", data=df.to_csv(index=False).encode('utf-8'),
                    file_name=f'{mol_name_display.replace(" ","_").replace("/","_")}_{df_key_suffix}.csv', mime='text/csv',
                    key=f"download_{df_key_suffix}_btn_{st.session_state.selected_molecule_index}"
                )
                if title == "Physicochemical Properties":
                     st.markdown("<small><b>Note:</b> ESOL Solubility (LogS) is a placeholder.</small>", unsafe_allow_html=True)
            else:
                st.error(data_dict.get("Error", error_message)) # Display specific error if available

        with st.expander("🧪 Physicochemical Properties", expanded=False):
            display_and_download_df("Properties", current_mol_data["physchem"], "physchem", "Physicochemical properties could not be calculated.")
        with st.expander("💊 Drug-likeness Rules", expanded=False):
            display_and_download_df("Drug-likeness", current_mol_data["drug_likeness"], "druglikeness", "Drug-likeness rules could not be evaluated.")
        with st.expander("💉 Pharmacokinetics Predictions", expanded=False):
            display_and_download_df("PK Predictions", current_mol_data["pharmacokinetics"], "pk", "Pharmacokinetic predictions could not be made.")
        
        with st.expander("📈 Bioavailability Radar", expanded=False):
            st.write("Visualizes key parameters for a quick assessment of oral bioavailability potential.")
            # ... (markdown description remains the same)
            st.markdown("""
            <small>The radar plot shows normalized values for:
            <ul>
                <li><b>LIPO (LogP):</b> Target -0.7 to +5.0</li>
                <li><b>SIZE (MW):</b> Target 150 to 500 g/mol</li>
                <li><b>POLAR (TPSA):</b> Target 20 to 130 Å²</li>
                <li><b>INSOLU (5-LogP):</b> Higher is better (LogP < 5, ideally < 4).</li>
                <li><b>INSATU (Fraction Csp3):</b> Target ≥ 0.25</li>
                <li><b>FLEX (9-RotB):</b> Higher is better (Rot.Bonds ≤ 9).</li>
            </ul> An approximate ideal zone (green dashed line) is shown based on typical drug-like ranges. Points closer to the periphery (value 1.0) are generally more favorable for each axis on the normalized plot.</small>
            """, unsafe_allow_html=True)
            if current_mol_data["radar_params"]:
                radar_plot_bytes = plot_bioavailability_radar(current_mol_data["radar_params"], mol_name_display)
                if radar_plot_bytes:
                    st.image(radar_plot_bytes)
                    st.download_button(label="Download Radar Plot", data=radar_plot_bytes,
                       file_name=f"{mol_name_display.replace(' ','_').replace('/','_')}_radar.png", mime="image/png",
                       key=f"download_radar_btn_{st.session_state.selected_molecule_index}")
                else: st.error("Could not generate Bioavailability Radar plot.")
            else: st.warning("Data for Bioavailability Radar not available or incomplete.")

        st.markdown("---")
        all_results_data_list = []
        for data_item_all in st.session_state.molecule_data_store:
            flat_data_item = {"Molecule_Name": data_item_all["name"]}
            flat_data_item.update({f"PhysChem_{k.replace(' ','_')}": v for k,v in data_item_all["physchem"].items()})
            flat_data_item.update({f"DrugLike_{k.replace(' ','_')}": v for k,v in data_item_all["drug_likeness"].items()})
            flat_data_item.update({f"PK_{k.replace(' ','_')}": v for k,v in data_item_all["pharmacokinetics"].items()})
            if data_item_all["radar_params"]:
                 flat_data_item.update({f"Radar_{k.replace(' ','_')}": v for k,v in data_item_all["radar_params"].items()})
            all_results_data_list.append(flat_data_item)
        
        if all_results_data_list:
            df_all_results = pd.DataFrame(all_results_data_list)
            st.download_button(label="📥 Download All Results as CSV", data=df_all_results.to_csv(index=False).encode('utf-8'),
                file_name="ADMETrix_all_results.csv", mime="text/csv",
                key="download_all_results_action_button", use_container_width=True)

    elif nav_selected == "Analysis":
            st.info("☝️ Please upload your molecular data and click 'Start Analysis' to view results.")

elif nav_selected == "About":
    st.markdown("<div class='page-title'>ℹ️ ADMETrix and The Idea Behind It!</div>", unsafe_allow_html=True)
    st.markdown("""<div class="circle-img-container"><img src="https://raw.githubusercontent.com/SonaliLakhamade/ADMETrix/main/Sonali_Lakhamade.jpg" alt="Sonali Lakhamade" class="circle-img"/></div>""", unsafe_allow_html=True)
    st.markdown('<div class="author-name">Sonali Lakhamade</div>', unsafe_allow_html=True)
    st.markdown('<div class="author-title">Author & Developer</div>', unsafe_allow_html=True)
    st.markdown("""<div class="content-box"><b>ADMETrix</b> was developed as part of my Master's degree in Bioinformatics at Fergusson College (Autonomous), affiliated with Savitribai Phule Pune University... </div>""", unsafe_allow_html=True) # Content truncated for brevity
    # ... (Rest of the About page content remains the same as previous correct version) ...
    st.markdown("<div class='page-subtitle'>🎯 Purpose</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="purpose-box">
    <ul>
        <li>To enable rapid, browser-based compound analysis, focusing on ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties.</li>
        <li>To support students and researchers in drug discovery, medicinal chemistry, and computational toxicology.</li>
        <li>To provide an educational tool for understanding key concepts in cheminformatics and ADME profiling.</li>
        <li>To allow batch analysis of multiple compounds, interactive visualization of results, and easy data export for reporting and further analysis.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>✨ Key Highlights</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-box">
    <ul>
        <li>Comprehensive ADMET profiling: 2D/3D visualization, physicochemical properties, drug-likeness rules, medicinal chemistry alerts (planned), and pharmacokinetics predictions.</li>
        <li>User-friendly interface suitable for both beginners and experienced users.</li>
        <li>Batch processing capability for efficient analysis of multiple molecules.</li>
        <li>Downloadable outputs: Images (PNG) and results tables (CSV) for offline use, presentations, and publications.</li>
        <li>Interactive plots like Bioavailability Radar and BOILED-Egg plot (planned) for intuitive data interpretation.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>🛠️ Tools and Technologies Used</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="content-box">
    ADMETrix is built using a combination of powerful open-source technologies:
    <ul>
        <li><b>Programming Language:</b> Python</li>
        <li><b>Web Framework:</b> Streamlit</li>
        <li><b>Cheminformatics:</b> RDKit</li>
        <li><b>Data Handling:</b> pandas, numpy</li>
        <li><b>Visualization:</b> matplotlib, stmol (for 3D structures)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>🌟 Benefits</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="benefit-box">
    <ul>
        <li><b>Accessibility:</b> No installation required; accessible from any modern web browser on any device.</li>
        <li><b>Ease of Use:</b> Intuitive design makes complex analyses straightforward.</li>
        <li><b>Educational Value:</b> Helps in understanding ADMET concepts and cheminformatics principles.</li>
        <li><b>Research Support:</b> Facilitates preliminary screening and hypothesis generation in drug discovery projects.</li>
        <li><b>Time-Saving:</b> Automates many routine calculations and report generation for ADMET properties.</li>
        <li><b>Cost-Effective:</b> Leverages open-source tools, making it a free resource for the community.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>🔮 Planned Future Enhancements</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-box">
    <ul>
        <li>Integration of Medicinal Chemistry filters (e.g., PAINS, Brenk).</li>
        <li>Implementation of the BOILED-Egg plot.</li>
        <li>More robust ESOL calculation for solubility (LogS).</li>
        <li>Advanced predictions (e.g., CYP isoform inhibition, P-gp substrate likelihood using ML models if feasible).</li>
        <li>User accounts for saving results and customizing analysis workflows.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>🙏 Mentorship & Acknowledgement</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="acknowledgement-box">
    I extend my sincere gratitude to <b>Dr. Kushagra Kashyap</b>, Assistant Professor in Bioinformatics, School of Science and Mathematics, DES Pune University, for his invaluable guidance and academic support throughout the conceptualization and development phases of this project. His expertise and mentorship played a pivotal role in shaping the project's scientific direction and scope.<br>
    <br>
    His encouragement and expert insights were instrumental in refining the technical implementation strategy and ensuring the project's alignment with current research needs. I am deeply grateful for his significant contributions to this endeavor.<br>
    <br>
    <a href="https://www.linkedin.com/in/dr-kushagra-kashyap-b230a3bb?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank" class="linkedin-btn">Connect with Dr. Kashyap on LinkedIn</a>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>📧 Feedback & Contact</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="content-box">
    Your feedback is highly appreciated and will help improve ADMETrix!
    <ul>
        <li><b>Email:</b> <a href="mailto:sonaalee21@gmail.com">sonaalee21@gmail.com</a></li>
        <li><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/sonali-lakhamade-b6a83230a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank">Sonali Lakhamade</a></li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(f"<div style='font-size:0.9rem;color:#777777;text-align:center;margin-top:2rem;'><i>ADMETrix | Last major update: November 2023 | Current date: {datetime.now().strftime('%A, %B %d, %Y, %I:%M %p')}</i></div>", unsafe_allow_html=True)