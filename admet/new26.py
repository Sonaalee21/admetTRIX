# admetrix_app.py

import streamlit as st
from datetime import datetime
import base64
import io
import pandas as pd
import py3Dmol
from stmol import showmol
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches # For BOILED-Egg ellipse
from math import pi
import requests # For PubChem
import re # For cleaning molecule names

# --- RDKit Import ---
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw, AllChem, Lipinski, Crippen
    from rdkit.Chem import rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    # Error will be displayed prominently on the Analysis page if needed
    RDKIT_AVAILABLE = False

# --- stmol Import ---
try:
    from stmol import showmol
    STMOL_AVAILABLE = True
except ImportError:
    STMOL_AVAILABLE = False

# --- PAGE CONFIG ---
st.set_page_config(
    layout="wide",
    page_title="ADMETriX",
    page_icon="🧬" # Changed Icon
)

# --- NAVIGATION STATE & OTHER SESSION STATES ---
if "nav_selected" not in st.session_state:
    st.session_state["nav_selected"] = "Home"
if "analysis_triggered" not in st.session_state:
    st.session_state.analysis_triggered = False
if "molecule_data_store" not in st.session_state:
    st.session_state.molecule_data_store = []
if "selected_molecule_index" not in st.session_state:
    st.session_state.selected_molecule_index = 0
if "current_pubchem_search_results" not in st.session_state:
    st.session_state.current_pubchem_search_results = None

# --- Navigation Function ---
def update_nav(label):
    """Callback function to update navigation state."""
    st.session_state["nav_selected"] = label
    if label != "Analysis":
        # Reset analysis state if navigating away
        st.session_state.analysis_triggered = False
        st.session_state.molecule_data_store = []
        st.session_state.selected_molecule_index = 0
        st.session_state.current_pubchem_search_results = None

def show_3d_structure(xyz_str):
    xyzview = py3Dmol.view(width=800, height=500)
    xyzview.addModel(xyz_str, "xyz")
    xyzview.setStyle({'stick': {}})
    xyzview.zoomTo()
    showmol(xyzview, height=500, width=800)

# --- RDKIT/HELPER FUNCTIONS (Conditional on RDKit availability) ---
if RDKIT_AVAILABLE:
    def get_mol_from_sdf_string(sdf_string):
        try:
            suppl = Chem.SDMolSupplier()
            suppl.SetData(sdf_string)
            if len(suppl) > 0 and suppl[0] is not None:
                mol = suppl[0]; AllChem.Compute2DCoords(mol); return mol
            return None
        except Exception: return None

    def get_mol_from_smiles(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol: AllChem.Compute2DCoords(mol)
            return mol
        except Exception: return None

    def generate_2d_image(mol, size=(300, 300)):
        if not mol: return None
        try:
            mol_copy = Chem.Mol(mol); Chem.Kekulize(mol_copy, clearAromaticFlags=True)
            img = Draw.MolToImage(mol_copy, size=size, kekulize=False)
        except Exception:
             try: img = Draw.MolToImage(mol, size=size)
             except Exception: return None
        bio = io.BytesIO(); img.save(bio, format='PNG')
        return bio.getvalue()

if STMOL_AVAILABLE:
if current_mol_data["xyz_str"]:
        show_3d_structure(current_mol_data["xyz_str"])
    else:
        st.warning("Could not generate 3D structure.")

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
            properties["ESOL Solubility (LogS)"] = "calc_pending"
        except Exception as e: properties["Error_PropertyCalculation"] = f"Error calculating properties: {e}"
        return properties

    def check_drug_likeness_rules(props):
        if not props or props.get("Error_PropertyCalculation"): return {"Error": props.get("Error_PropertyCalculation", "Missing property data")}
        required_keys = ["Molecular Weight (MW)", "LogP (Crippen)", "Topological Polar Surface Area (TPSA)", "Hydrogen Bond Acceptors (HBA)", "Hydrogen Bond Donors (HBD)", "Rotatable Bonds", "Molar Refractivity", "Number of Atoms", "Number of Rings"]
        for key in required_keys:
            if not isinstance(props.get(key), (int, float)): return {"Error": f"Invalid property for rules: {key}"}
        mw, logp, hbd, hba, tpsa, rb, mr, num_atoms, num_rings = (props[k] for k in required_keys)
        rules = {}
        lip_v = (mw > 500) + (logp > 5) + (hbd > 5) + (hba > 10)
        rules["Lipinski's Rule of Five"] = f"Passed ✅" if lip_v == 0 else f"Failed ({lip_v} violations) ❌"
        ghose_v = (not 160 <= mw <= 480) + (not -0.4 <= logp <= 5.6) + (not 40 <= mr <= 130) + (not 20 <= num_atoms <= 70)
        rules["Ghose Filter"] = f"Passed ✅" if ghose_v == 0 else f"Failed ({ghose_v} violations) ❌"
        veber_v = (rb > 10) + (tpsa > 140)
        rules["Veber Rule"] = f"Passed ✅" if veber_v == 0 else f"Failed ({veber_v} violations) ❌"
        egan_v = (logp > 5.88) + (tpsa > 131.6)
        rules["Egan Rule (Bioavailability)"] = f"Passed ✅" if egan_v == 0 else f"Failed ({egan_v} violations) ❌"
        muegge_v = (not 200 <= mw <= 600) + (not -2 <= logp <= 5) + (tpsa > 150) + (num_rings > 7) + (rb > 15) + (hba > 10) + (hbd > 5)
        rules["Muegge Rule"] = f"Passed ✅" if muegge_v == 0 else f"Failed ({muegge_v} violations) ❌"
        return rules

    def predict_pharmacokinetics(props):
        if not props or props.get("Error_PropertyCalculation"): return {"Error": props.get("Error_PropertyCalculation", "Missing property data")}
        required_keys = ["Molecular Weight (MW)", "LogP (Crippen)", "Topological Polar Surface Area (TPSA)", "Hydrogen Bond Donors (HBD)", "Hydrogen Bond Acceptors (HBA)"]
        for key in required_keys:
            if not isinstance(props.get(key), (int, float)): return {"Error": f"Invalid property for PK preds: {key}"}
        mw, logp, tpsa, hbd, hba = (props[k] for k in required_keys)
        pk_preds = {}
        lip_v = (mw > 500) + (logp > 5) + (hbd > 5) + (hba > 10)
        pk_preds["GI Absorption"] = "High" if lip_v == 0 else ("Moderate" if lip_v == 1 else "Low")
        pk_preds["BBB Permeation"] = "Likely Permeant" if tpsa < 80 and 0 < logp < 5 else "Likely Non-Permeant"
        pk_preds["P-gp Substrate"] = "Not predicted"
        pk_preds["CYP Inhibition"] = "Not predicted"
        return pk_preds

    def get_bioavailability_radar_data(props):
        if not props or props.get("Error_PropertyCalculation"): return None
        required_keys = ["LogP (Crippen)", "Molecular Weight (MW)", "Topological Polar Surface Area (TPSA)", "Fraction Csp3", "Rotatable Bonds"]
        temp_data = {}
        for key in required_keys:
            if not isinstance(props.get(key), (int, float)): return None
            temp_data[key] = props[key]
        data = {"LIPO (LogP)": temp_data["LogP (Crippen)"], "SIZE (MW)": temp_data["Molecular Weight (MW)"], "POLAR (TPSA)": temp_data["Topological Polar Surface Area (TPSA)"], "INSOLU (5-LogP)": max(0, 5 - temp_data["LogP (Crippen)"]), "INSATU (FracCsp3)": temp_data["Fraction Csp3"], "FLEX (9-RotB)": max(0, 9 - temp_data["Rotatable Bonds"])}
        return data

    def calculate_boiled_egg_data(all_molecule_data):
        boiled_data = []
        for data in all_molecule_data:
            props = data.get("physchem")
            if props and not props.get("Error_PropertyCalculation"):
                 logp = props.get("LogP (Crippen)")
                 tpsa = props.get("Topological Polar Surface Area (TPSA)")
                 name = data.get("name", "Unknown")
                 if isinstance(logp, (int, float)) and isinstance(tpsa, (int, float)):
                     boiled_data.append({"name": name, "WLOGP": logp, "TPSA": tpsa})
        return boiled_data

# --- PLOTTING FUNCTIONS ---
def plot_bioavailability_radar(radar_data, mol_name="Molecule"):
    if not radar_data: return None
    labels_raw, stats_raw = list(radar_data.keys()), list(radar_data.values())
    ideal_plot_ranges = {"LIPO (LogP)": (-2, 6), "SIZE (MW)": (0, 600), "POLAR (TPSA)": (0, 160), "INSOLU (5-LogP)": (-1, 7), "INSATU (FracCsp3)": (0, 1), "FLEX (9-RotB)": (-1, 10)}
    stats_normalized = []
    for i, label in enumerate(labels_raw):
        min_val, max_val = ideal_plot_ranges[label]
        normalized = np.clip(((stats_raw[i] - min_val) / (max_val - min_val)) if (max_val - min_val) != 0 else 0, 0, 1)
        stats_normalized.append(normalized)
    angles = np.linspace(0, 2 * pi, len(labels_raw), endpoint=False).tolist()
    stats_plot, angles_plot = stats_normalized + stats_normalized[:1], angles + angles[:1]
    plt.style.use('seaborn-v0_8-pastel')
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles_plot, stats_plot, color='#1E88E5', linewidth=2, linestyle='solid', marker='o', markersize=4)
    ax.fill(angles_plot, stats_plot, '#90CAF9', alpha=0.4)
    ideal_mids_norm = [np.clip(mid, 0, 1) for mid in [0.66, 0.54, 0.47, 0.71, 0.63, 0.5]]
    ideal_zone_plot = ideal_mids_norm + ideal_mids_norm[:1]
    ax.plot(angles_plot, ideal_zone_plot, color='#FBC02D', linewidth=1.5, linestyle='dashed', label='Approx. Ideal Zone')
    ax.set_xticks(angles); ax.set_xticklabels(labels_raw, fontsize=8, weight='bold')
    ax.set_yticks(np.arange(0.2, 1.1, 0.2)); ax.set_yticklabels([f"{t:.1f}" for t in np.arange(0.2, 1.1, 0.2)], fontsize=7)
    ax.set_ylim(0, 1); ax.grid(color='grey', linestyle=':', linewidth=0.5)
    ax.set_title(f"Bioavailability Radar: {mol_name}", size=10, weight='bold', y=1.15)
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches='tight', dpi=120); plt.close(fig)
    return buf.getvalue()

def plot_boiled_egg(boiled_data):
    if not boiled_data: return None
    wlogp = [d['WLOGP'] for d in boiled_data]
    tpsa = [d['TPSA'] for d in boiled_data]
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(6, 5))
    yolk_ellipse = patches.Ellipse((2.9, 75), width=5.0, height=110, angle=0, fill=True, color='#FFECB3', alpha=0.6, zorder=0)
    white_ellipse = patches.Ellipse((2.9, 55), width=7.0, height=140, angle=0, fill=True, color='#E3F2FD', alpha=0.4, zorder=0)
    ax.add_patch(white_ellipse); ax.add_patch(yolk_ellipse)
    ax.scatter(wlogp, tpsa, c='#D32F2F', s=30, alpha=0.8, zorder=1, label='Compounds')
    ax.set_xlabel("WLOGP (LogP as proxy)", fontsize=10, weight='bold')
    ax.set_ylabel("TPSA (Å²)", fontsize=10, weight='bold')
    ax.set_title("BOILED-Egg Plot (GI Absorption & BBB Permeation)", fontsize=12, weight='bold')
    ax.set_xlim(-4, 8); ax.set_ylim(0, 200)
    ax.text(6.5, 185, 'GI Abs (Yolk)', fontsize=8, color='#B8860B', ha='center', weight='bold')
    ax.text(6.5, 170, 'BBB Perm (White)', fontsize=8, color='#1E88E5', ha='center', weight='bold')
    ax.text(6.5, 155, 'Both', fontsize=8, color='#7B1FA2', ha='center', weight='bold')
    ax.grid(True, linestyle=':', linewidth=0.5)
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches='tight', dpi=120); plt.close(fig)
    return buf.getvalue()

# --- PUBCHEM HELPER FUNCTIONS ---
def pubchem_get_cid_from_name(name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        cids = response.json().get('IdentifierList', {}).get('CID', [])
        return cids[0] if cids else None
    except requests.exceptions.RequestException as e: st.error(f"PubChem API error (name->CID): {e}"); return None
    except ValueError: st.error("PubChem API error (name->CID): Could not decode response."); return None

def pubchem_get_similar_compounds(cid, threshold=90, max_results=5):
    if not cid: return []
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/similarity/cid/{cid}/cids/JSON?Threshold={threshold}&MaxRecords={max_results}"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json().get('IdentifierList', {}).get('CID', [])
    except requests.exceptions.RequestException as e: st.error(f"PubChem API error (similarity): {e}"); return []
    except ValueError: st.error("PubChem API error (similarity): Could not decode response."); return []

def pubchem_get_compound_info(cid):
    if not cid: return {}
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName,CanonicalSMILES,MolecularFormula,MolecularWeight/JSON"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'PropertyTable' in data and 'Properties' in data['PropertyTable'] and data['PropertyTable']['Properties']:
            return data['PropertyTable']['Properties'][0]
        return {}
    except requests.exceptions.RequestException as e: st.error(f"PubChem API error (info CID {cid}): {e}"); return {}
    except ValueError: st.error(f"PubChem API error (info CID {cid}): Could not decode response."); return {}

def clean_molecule_name_for_pubchem(name):
    if not name: return ""
    name = re.sub(r'_sdf_\d+$', '', name); name = re.sub(r' \(.+\)$', '', name)
    name = name.replace("_", " ").strip()
    return name

# --- STYLING ---
st.markdown("""
<style>
    /* Base Font */
    body, .stApp {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #333;
        background-color: #f0f8ff; /* Light AliceBlue background */
    }

    /* Main Title */
    .main-title-container {
        padding: 1rem 0;
        margin-bottom: 1rem;
        border-radius: 10px;
        background: linear-gradient(90deg, #00796B, #00ACC1); /* Teal/Cyan gradient */
        box-shadow: 0 4px 15px rgba(0, 121, 107, 0.3);
        text-align: center;
    }
    .main-title {
        font-size: 3.5rem;
        font-weight: 700; /* Bolder */
        color: white;
        letter-spacing: 3px;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        margin: 0; /* Remove default margins */
    }

    /* Navigation Buttons - Target standard Streamlit buttons */
     .stButton>button {
        border-radius: 20px !important;
        border: 1px solid #00796B !important; /* Teal border */
        background: linear-gradient(145deg, #ffffff, #e0f2f1) !important; /* White to light teal */
        color: #00695C !important; /* Dark Teal text */
        padding: 0.5rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1) !important;
        width: 100% !important; /* Make buttons fill columns */
        margin-bottom: 0.5rem; /* Add space below nav buttons */
    }
    .stButton>button:hover {
        background: linear-gradient(145deg, #00796B, #00ACC1) !important; /* Teal/Cyan gradient on hover */
        color: white !important;
        border-color: #004D40 !important;
        box-shadow: 3px 3px 8px rgba(0, 0, 0, 0.2) !important;
        transform: translateY(-2px) !important;
    }
    /* Style specific buttons differently if needed */
     .stDownloadButton>button { /* Standard download button style */
        background: #FFB300 !important; /* Amber */
        color: #424242 !important;
        border-color: #FFA000 !important;
     }
     .stDownloadButton>button:hover {
        background: #FFA000 !important;
        border-color: #FF6F00 !important;
        color: white !important;
     }
     /* Primary Action Button (e.g., Start Analysis) */
      button[kind="primary"] {
         background: linear-gradient(145deg, #00897B, #00695C) !important; /* Darker Teal gradient */
         color: white !important;
         border-color: #004D40 !important;
      }
       button[kind="primary"]:hover {
         background: linear-gradient(145deg, #00695C, #004D40) !important;
         border-color: #00332C !important;
       }


    /* Page Titles */
    .page-title {
        font-size: 2.5rem; color: #004D40; /* Darkest Teal */
        border-bottom: 3px solid #00ACC1; /* Cyan accent */
        padding-bottom: 0.6rem; margin-top: 1.5rem; margin-bottom: 2rem;
        font-weight: 600;
    }
    .page-subtitle {
        font-size: 1.5rem; color: #00796B; /* Teal */
        margin-bottom: 1.2rem; font-weight: 600;
        border-left: 4px solid #00ACC1; padding-left: 0.8rem;
    }

    /* Content Boxes - General */
    .content-box, .feature-box, .step-box, .purpose-box, .benefit-box, .acknowledgement-box {
        background: #ffffff; /* White background for content */
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.07);
        border-left: 5px solid; /* Keep border-left */
    }
    /* Specific Box Colors */
    .feature-box { border-left-color: #00ACC1; background-color: #E0F7FA; } /* Cyan Accent, Light Cyan BG */
    .step-box { border-left-color: #4CAF50; background-color: #E8F5E9; } /* Green Accent, Light Green BG */
    .purpose-box { border-left-color: #FFB300; background-color: #FFFDE7; } /* Amber Accent, Light Yellow BG */
    .benefit-box { border-left-color: #1E88E5; background-color: #E3F2FD; } /* Blue Accent, Light Blue BG */
    .acknowledgement-box { border-left-color: #8E24AA; background-color: #F3E5F5; } /* Purple Accent, Light Purple BG */
    .content-box { border-left-color: #00796B; background-color: #f8f9fa; } /* Teal Accent, Off-white BG */

    /* About Page Specific */
    .circle-img-container { display: flex; justify-content: center; margin-bottom: 1rem; }
    .circle-img {
        border-radius: 50%;
        width: 150px; /* Moderate size */
        height: 150px;
        object-fit: cover;
        border: 7px solid #00796B; /* Teal border */
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
     }
    .author-name { text-align:center; font-size:1.6rem; font-weight:600; color: #004D40; margin-top: 0.5rem; }
    .author-title { text-align:center; font-size:1.2rem; color: #555; margin-bottom: 1rem; font-style: italic;} /* Reduced bottom margin */
    .linkedin-btn { /* Style for specific LinkedIn buttons */
        display: inline-block;
        padding: 0.7em 1.5em;
        border: none;
        border-radius: 25px;
        background: linear-gradient(145deg, #0077b5, #005582); /* LinkedIn Blue Gradient */
        color: white !important;
        font-weight: bold;
        text-decoration: none;
        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        text-align: center; /* Center text inside button */
    }
    .linkedin-btn:hover {
        background: linear-gradient(145deg, #005582, #003350);
        box-shadow: 4px 4px 12px rgba(0, 0, 0, 0.3);
        transform: translateY(-2px);
        color: white !important; text-decoration: none;
    }

    /* Footer */
    .footer-text { font-size: 0.9rem; color: #6c757d; text-align: center; margin-top: 3rem; padding-bottom: 1rem; font-style: italic; }

    /* Dataframe styling */
    .stDataFrame { border: 1px solid #dee2e6; border-radius: 5px; }
    /* Enhance dataframe header */
     .stDataFrame thead th { background-color: #E0F2F1; color: #00695C; font-weight: bold; }
    /* Style Drug-Likeness table generated by markdown */
    table.dataframe { width: 100%; border-collapse: collapse; }
    table.dataframe th { background-color: #E0F2F1; color: #00695C; font-weight: bold; padding: 8px; text-align: left; border-bottom: 2px solid #00796B;}
    table.dataframe td { padding: 8px; border-bottom: 1px solid #dee2e6; }
    table.dataframe tr:nth-child(even) { background-color: #f8f9fa; }


    /* Expander styling */
    .st-expander { border: 1px solid #B2DFDB; border-radius: 8px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05); margin-bottom: 1rem;}
    .st-expander header { background-color: #E0F2F1; border-radius: 8px 8px 0 0; font-weight: 600; color: #00695C; }
    .st-expander header:hover { background-color: #B2DFDB; }

    /* Ensure text wraps in st.text */
    .stText { white-space: pre-wrap !important; }

</style>
""", unsafe_allow_html=True)

# --- HEADER & NAVIGATION BAR ---
st.markdown('<div class="main-title-container"><div class="main-title">🧬 ADMETriX 🧪</div></div>', unsafe_allow_html=True)

cols_nav = st.columns(4)
with cols_nav[0]:
    if st.button("🏠 Home", key="nav_Home_main", use_container_width=True): update_nav("Home")
with cols_nav[1]:
    if st.button("📖 User Guide", key="nav_User_Guide_main", use_container_width=True): update_nav("User Guide")
with cols_nav[2]:
    if st.button("🔬 Analysis", key="nav_Analysis_main", use_container_width=True): update_nav("Analysis")
with cols_nav[3]:
    if st.button("ℹ️ About", key="nav_About_main", use_container_width=True): update_nav("About")

nav_selected = st.session_state.get("nav_selected", "Home")
st.divider()

# --- PAGE CONTENT ---
if nav_selected == "Home":
    st.markdown("<div class='page-title'>Welcome to ADMETriX: Your Bioinformatics Drug Nexus</div>", unsafe_allow_html=True)
    st.markdown("""
    ADMETriX is your integrated, web-based platform for essential compound analysis.
    Designed for students, researchers, and educators in **structural bioinformatics, drug discovery, and computational biology**.
    Streamline your ADMET analysis without installations!
    """)
    st.markdown("<div class='page-subtitle'>🚀 Key Features</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-box">
    <ul>
        <li><b>📊 2D & 3D Visualization</b>: View 2D structures and interactive 3D models instantly.</li>
        <li><b>🧪 Physicochemical Properties</b>: Comprehensive descriptors (MW, TPSA, LogP, etc.).</li>
        <li><b>💊 Drug-likeness Rules</b>: Evaluate against Lipinski, Ghose, Veber, Egan, Muegge rules with icons ✅/❌.</li>
        <li><b>🔗 Similar Compounds Search</b>: Find structurally similar compounds via PubChem with 2D structures.</li>
        <li><b> pharmacokinetic Predictions</b>: Predict GI absorption & BBB permeation likelihood.</li>
        <li><b>📈 Bioavailability Radar</b>: SwissADME-style radar plots for rapid assessment.</li>
        <li><b>🥚 BOILED-Egg Plot</b>: Visualize GI absorption & BBB permeation for multiple compounds.</li>
        <li><b>🔄 Batch Input & Download</b>: Analyze multiple molecules (SMILES/SDF) & download results.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>💡 Why Use ADMETriX?</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="content-box">
    <ul>
        <li>🌐 No installation needed — fully browser-based and accessible anywhere.</li>
        <li>👍 Beginner-friendly, with clear explanations and interactive controls.</li>
        <li>🎓 Educational & Research-Oriented: Designed for learning, teaching, and research.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

elif nav_selected == "User Guide":
    st.markdown("<div class='page-title'>📖 User Guide</div>", unsafe_allow_html=True)
    st.markdown("Follow these simple steps to analyze your compounds using ADMETriX:")
    st.markdown("""
    <div class="step-box"><b>Step 1: Navigate to Analysis</b><br>Click on the <b>🔬 Analysis</b> tab above.</div>
    <div class="step-box"><b>Step 2: Select Input Type</b><br>Choose 'SDF' to upload <code>.sdf</code> files or 'SMILES' to paste text or upload <code>.smi</code>/<code>.txt</code> files.</div>
    <div class="step-box"><b>Step 3: Provide Data</b><br>Use the 'Browse files' button or paste SMILES into the text area (one per line).</div>
    <div class="step-box"><b>Step 4: Start Analysis</b><br>Click the <b>🚀 Start Analysis</b> button.</div>
    <div class="step-box"><b>Step 5: Explore Results</b><br>View the summary table. Select a molecule (if multiple) to see detailed results in expandable sections (including 2D/3D structures, PubChem search, properties, rules, PK predictions, Radar plot, and BOILED-Egg plot).</div>
    <div class="step-box"><b>Step 6: Download Data</b><br>Use the download buttons 📥 for individual results or the 'Download All Results' button at the bottom.</div>
    """, unsafe_allow_html=True)

elif nav_selected == "Analysis":
    st.markdown("<div class='page-title'>🔬 ADMETriX Analysis</div>", unsafe_allow_html=True)

    if not RDKIT_AVAILABLE:
        st.error("🚫 RDKit installation not found. The Analysis tab requires RDKit. Please install it (`conda install -c conda-forge rdkit` or `pip install rdkit-pypi`) and restart the app.")
    else:
        st.markdown("#### Upload Your Molecules")
        cols_analysis_input = st.columns([1, 2, 1])
        with cols_analysis_input[1]:
            file_type = st.radio("Select Input Type:", ["SDF", "SMILES"], horizontal=True, label_visibility="collapsed", key="analysis_file_type_radio")
            uploaded_sdf_files = None
            uploaded_smiles_file = None
            smiles_input_str = ""

            if file_type == "SDF":
                uploaded_sdf_files = st.file_uploader("Browse SDF files (.sdf)", type=["sdf"], accept_multiple_files=True, key="upload_sdf_widget")
                if uploaded_sdf_files: st.success(f"{len(uploaded_sdf_files)} SDF file(s) selected.")
            elif file_type == "SMILES":
                uploaded_smiles_file = st.file_uploader("Browse SMILES file (.smi, .txt)", type=["smi", "txt"], accept_multiple_files=False, key="upload_smiles_file_widget")
                st.markdown("<p style='text-align: center; margin: 0.5rem 0;'>OR</p>", unsafe_allow_html=True) # Centered OR
                st.markdown("###### Paste SMILES (one per line):")
                smiles_input_str = st.text_area("Paste SMILES here...", height=100, placeholder="CCOc1ccc(cc1)NC(=O)C\n...", key="analysis_smiles_input_area", label_visibility="collapsed")

                if uploaded_smiles_file: st.success(f"SMILES file '{uploaded_smiles_file.name}' selected.")
                elif smiles_input_str: st.success("SMILES pasted in text area.")

            if st.button("🚀 Start Analysis", use_container_width=True, type="primary", key="start_analysis_action_button"):
                # Reset state variables
                st.session_state.molecule_data_store, st.session_state.analysis_triggered, st.session_state.selected_molecule_index, st.session_state.current_pubchem_search_results = [], False, 0, None
                mols_to_process, mol_names, error_messages = [], [], []
                input_provided = False

                # --- Input Processing Logic ---
                if file_type == "SDF" and uploaded_sdf_files:
                    st.session_state.input_type = "SDF"; input_provided = True
                    for i, uploaded_file in enumerate(uploaded_sdf_files):
                        try: mol = get_mol_from_sdf_string(uploaded_file.getvalue().decode("utf-8", errors='ignore'));
                        except Exception as e: error_messages.append(f"Error processing SDF {uploaded_file.name}: {e}"); mol=None
                        if mol: mols_to_process.append(mol); name_from_sdf = mol.GetProp("_Name") if mol.HasProp("_Name") else uploaded_file.name.split('.')[0]; mol_names.append(name_from_sdf + f"_sdf_{i+1}")
                        elif uploaded_file: error_messages.append(f"Could not parse molecule from SDF: {uploaded_file.name}")
                elif file_type == "SMILES":
                    smiles_list = []
                    if uploaded_smiles_file:
                        st.session_state.input_type = "SMILES_File"; input_provided = True
                        try: smiles_content = uploaded_smiles_file.getvalue().decode("utf-8", errors='ignore')
                        except Exception as e: error_messages.append(f"Error reading SMILES file {uploaded_smiles_file.name}: {e}"); smiles_content = ""
                        smiles_list = [s.strip() for s in smiles_content.strip().split('\n') if s.strip()]
                        if not smiles_list and not error_messages: error_messages.append(f"SMILES file '{uploaded_smiles_file.name}' appears empty or invalid.")
                    elif smiles_input_str.strip():
                        st.session_state.input_type = "SMILES_Text"; input_provided = True
                        smiles_list = [s.strip() for s in smiles_input_str.strip().split('\n') if s.strip()]
                    else: input_provided = False
                    for i, smi in enumerate(smiles_list):
                        mol = get_mol_from_smiles(smi)
                        if mol: mols_to_process.append(mol); mol_names.append(f"SMILES_{i+1}" + (f" ({smi[:15]}...)" if len(smi)>15 else f" ({smi})") )
                        else: error_messages.append(f"Could not parse SMILES: {smi}")
                if error_messages: st.warning('\n'.join(error_messages))
                if mols_to_process:
                    with st.spinner("⚙️ Analyzing molecules..."):
                        # --- Analysis Loop ---
                        for i, mol_obj in enumerate(mols_to_process):
                            mol_data = {"name": mol_names[i], "mol_obj_rdkit": mol_obj};
                            mol_data["physchem"] = calculate_physicochemical_properties(mol_obj);
                            mol_data["drug_likeness"] = check_drug_likeness_rules(mol_data["physchem"]);
                            mol_data["pharmacokinetics"] = predict_pharmacokinetics(mol_data["physchem"]);
                            mol_data["radar_params"] = get_bioavailability_radar_data(mol_data["physchem"]);
                            mol_data["2d_image_bytes"] = generate_2d_image(mol_obj);
                            mol_data["xyz_str"] = get_xyz_from_mol(mol_obj);
                            st.session_state.molecule_data_store.append(mol_data)
                    st.session_state.analysis_triggered = True; st.success(f"✅ Analysis complete for {len(st.session_state.molecule_data_store)} molecule(s).")
                elif not input_provided and not error_messages: st.warning("⚠️ No valid input provided. Please upload/paste data.")
                elif not mols_to_process and input_provided: st.error("❌ No valid molecules could be processed from the input.")
        st.divider()

        # --- Results Display Area ---
        if st.session_state.get("analysis_triggered", False) and st.session_state.molecule_data_store:
            st.markdown("<div class='page-subtitle'>📊 Analysis Results Overview</div>", unsafe_allow_html=True)
            summary_data_list = []
            for idx, data in enumerate(st.session_state.molecule_data_store):
                lipinski_status = data["drug_likeness"].get("Lipinski's Rule of Five", "N/A")
                lip_viol = "N/A"
                if isinstance(lipinski_status, str) and "Fail" in lipinski_status:
                    try: lip_viol = lipinski_status.split('(')[1].split(' ')[0]
                    except: lip_viol = "err"
                elif isinstance(lipinski_status, str) and "Passed" in lipinski_status: lip_viol = "0"
                elif data["drug_likeness"].get("Error"): lip_viol = "Err"
                summary_data_list.append({ "ID": idx + 1, "Name": data["name"], "MW": data["physchem"].get("Molecular Weight (MW)", "N/A"), "LogP": data["physchem"].get("LogP (Crippen)", "N/A"), "TPSA": data["physchem"].get("Topological Polar Surface Area (TPSA)", "N/A"), "Lipinski Viol.": lip_viol })
            st.dataframe(pd.DataFrame(summary_data_list), use_container_width=True, hide_index=True)

            mol_display_names = [f"{idx+1}. {data['name']}" for idx, data in enumerate(st.session_state.molecule_data_store)]
            if len(mol_display_names) > 1 :
                 selected_mol_display_name = st.selectbox("Select molecule for detailed results:", mol_display_names, index=st.session_state.selected_molecule_index, key="molecule_selector_dropdown", on_change=lambda: setattr(st.session_state, 'current_pubchem_search_results', None))
                 st.session_state.selected_molecule_index = mol_display_names.index(selected_mol_display_name)

            current_mol_data = st.session_state.molecule_data_store[st.session_state.selected_molecule_index]
            mol_name_display = current_mol_data["name"]
            st.markdown(f"#### Detailed Results for: **{mol_name_display}**")

            # --- Result Expanders ---
            with st.expander("🖼️ Molecular Structure (2D/3D)", expanded=True):
                col1_struct, col2_struct = st.columns(2)
                with col1_struct:
                    st.subheader("2D Structure")
                    if current_mol_data["2d_image_bytes"]:
                        st.image(current_mol_data["2d_image_bytes"])
                        st.download_button(label="Download 2D Image", data=current_mol_data["2d_image_bytes"], file_name=f"{mol_name_display.replace(' ','_').replace('/','_')}_2D.png", mime="image/png", key=f"download_2d_btn_{st.session_state.selected_molecule_index}")
                    else: st.error("Could not generate 2D image.")
                with col2_struct:
                    st.subheader("3D Structure")
            
                
                
                st.warning("Could not generate 3D structure.")

  

            with st.expander("🔗 Find Similar Compounds via PubChem", expanded=False):
                st.write(f"Search PubChem for compounds similar to: **{mol_name_display}**")
                default_search_name = clean_molecule_name_for_pubchem(current_mol_data["name"])
                pubchem_search_name_key = f"pubchem_search_name_{st.session_state.selected_molecule_index}"
                if pubchem_search_name_key not in st.session_state or st.session_state.get(pubchem_search_name_key, '') != default_search_name:
                    st.session_state[pubchem_search_name_key] = default_search_name
                user_search_name = st.text_input("Compound name for PubChem search:", value=st.session_state.get(pubchem_search_name_key, default_search_name), key=pubchem_search_name_key, help="Edit name if needed.")

                # Use standard button style for search
                if st.button("Search PubChem 🔎", key=f"pubchem_search_btn_{st.session_state.selected_molecule_index}"):
                    st.session_state.current_pubchem_search_results = None
                    if user_search_name:
                        with st.spinner(f"Searching PubChem for '{user_search_name}'..."):
                            cid = pubchem_get_cid_from_name(user_search_name)
                            if cid:
                                st.success(f"Found CID {cid}. Fetching similar compounds...", icon="✅")
                                similar_cids = pubchem_get_similar_compounds(cid, threshold=90, max_results=5)
                                if similar_cids:
                                    similar_compound_data = []
                                    with st.spinner("Fetching details for similar compounds..."):
                                        for scid in similar_cids:
                                            info = pubchem_get_compound_info(scid)
                                            img_bytes = None
                                            if RDKIT_AVAILABLE and info.get("CanonicalSMILES"):
                                                mol_sim = get_mol_from_smiles(info["CanonicalSMILES"])
                                                img_bytes = generate_2d_image(mol_sim, size=(200, 200))
                                            similar_compound_data.append({"cid": scid, "info": info, "image": img_bytes})
                                    st.session_state.current_pubchem_search_results = similar_compound_data
                                else: st.session_state.current_pubchem_search_results = "no_similar"
                            else: st.session_state.current_pubchem_search_results = "not_found"
                    else: st.warning("Please enter a compound name for PubChem search.")

                current_search_name_for_display = st.session_state.get(pubchem_search_name_key, default_search_name)
                if st.session_state.current_pubchem_search_results == "not_found": st.error(f"Compound '{current_search_name_for_display}' not found in PubChem.")
                elif st.session_state.current_pubchem_search_results == "no_similar": st.warning(f"No similar compounds found with >90% similarity for '{current_search_name_for_display}'.")
                elif isinstance(st.session_state.current_pubchem_search_results, list):
                    st.markdown(f"##### Top Similar Compounds (>90% similarity to '{current_search_name_for_display}')")
                    results_list = st.session_state.current_pubchem_search_results
                    for i, item in enumerate(results_list):
                        info, img_bytes = item["info"], item["image"]
                        col_info, col_img = st.columns([3,2])
                        with col_info:
                            st.markdown(f"**{i+1}. Compound CID: {item['cid']}**")
                            st.markdown(f"   *Name:* {info.get('IUPACName', 'N/A')}")
                            st.markdown(f"   *Formula:* {info.get('MolecularFormula', 'N/A')}")
                            st.markdown(f"   *MW:* {info.get('MolecularWeight', 'N/A')}")
                            st.markdown(f"   *SMILES:* `{info.get('CanonicalSMILES', 'N/A')}`")
                        with col_img:
                             if img_bytes: st.image(img_bytes, width=200)
                             else: st.caption("No 2D Image")
                        if i < len(results_list) - 1: st.markdown("---")

            # Helper function for displaying tables with icons
            def display_results_table(title, data_dict, df_key_suffix, error_message="Data could not be processed."):
                st.subheader(title)
                if data_dict and not data_dict.get("Error") and not data_dict.get("Error_PropertyCalculation"):
                    is_drug_likeness = title == "Drug-likeness Rules"
                    df_data = []
                    for prop, value in data_dict.items():
                         if "Error" not in prop:
                            df_data.append({"Property": prop, "Value": value})
                    if df_data:
                        df = pd.DataFrame(df_data)
                        if is_drug_likeness:
                            # Use st.markdown to render HTML table for icons
                            html_table = df.to_html(escape=False, index=False, classes=["dataframe"], border=0)
                            st.markdown(html_table, unsafe_allow_html=True)
                            # Prepare CSV without icons
                            df_download = df.copy()
                            df_download['Value'] = df_download['Value'].astype(str).str.replace(' ✅','', regex=False).str.replace(' ❌','', regex=False)
                            csv_data = df_download.to_csv(index=False).encode('utf-8')
                        else:
                            st.dataframe(df, use_container_width=True, hide_index=True)
                            csv_data = df.to_csv(index=False).encode('utf-8')
                        st.download_button(label=f"Download {title} CSV", data=csv_data, file_name=f'{mol_name_display.replace(" ","_").replace("/","_")}_{df_key_suffix}.csv', mime='text/csv', key=f"download_{df_key_suffix}_btn_{st.session_state.selected_molecule_index}")
                        if title == "Properties": st.markdown("<small><b>Note:</b> ESOL Solubility (LogS) is a placeholder.</small>", unsafe_allow_html=True)
                    else: st.warning("No valid data to display.")
                elif data_dict and data_dict.get("Error"): st.error(data_dict.get("Error"))
                elif data_dict and data_dict.get("Error_PropertyCalculation"): st.error(data_dict.get("Error_PropertyCalculation"))
                else : st.error(error_message)

            with st.expander("🧪 Physicochemical Properties", expanded=False): display_results_table("Properties", current_mol_data["physchem"], "physchem", "Physicochemical properties failed or pending.")
            with st.expander("💊 Drug-likeness Rules", expanded=False): display_results_table("Drug-likeness Rules", current_mol_data["drug_likeness"], "druglikeness", "Drug-likeness rules evaluation failed or pending.")
            with st.expander("💉 Pharmacokinetics Predictions", expanded=False): display_results_table("PK Predictions", current_mol_data["pharmacokinetics"], "pk", "Pharmacokinetic predictions failed or pending.")

            with st.expander("📈 Bioavailability Radar", expanded=False):
                st.subheader("Bioavailability Radar Plot")
                radar_plot_bytes = None
                if current_mol_data["radar_params"]: radar_plot_bytes = plot_bioavailability_radar(current_mol_data["radar_params"], mol_name_display)
                if radar_plot_bytes:
                     st.image(radar_plot_bytes);
                     st.download_button(label="Download Radar Plot", data=radar_plot_bytes, file_name=f"{mol_name_display.replace(' ','_').replace('/','_')}_radar.png", mime="image/png", key=f"download_radar_btn_{st.session_state.selected_molecule_index}")
                     st.info("""**Interpretation:** The Bioavailability Radar visualizes drug-likeness based on six properties. The blue area is the molecule's profile; the yellow dashed line approximates the ideal space (LogP: -0.7 to +5.0, MW: 150-500, TPSA: 20-130, etc., normalized). Points near the edge (1.0) are generally better. Deviations suggest potential bioavailability issues.""")
                else: st.error("Could not generate Bioavailability Radar plot (check data).")

            with st.expander("🥚 BOILED-Egg Plot", expanded=False):
                 st.subheader("BOILED-Egg Plot (All Molecules)")
                 boiled_egg_data = calculate_boiled_egg_data(st.session_state.molecule_data_store)
                 if boiled_egg_data:
                     boiled_egg_plot_bytes = plot_boiled_egg(boiled_egg_data)
                     if boiled_egg_plot_bytes:
                         st.image(boiled_egg_plot_bytes)
                         st.download_button(label="Download BOILED-Egg Plot", data=boiled_egg_plot_bytes, file_name="ADMETriX_BOILED_Egg.png", mime="image/png", key="download_boiled_egg_plot")
                         st.info("""**Interpretation:** Predicts GI absorption (HIA) and BBB permeation. **Yellow Yolk:** High HIA probability. **Light Blue White:** High BBB permeation probability. **Violet Overlap:** High probability for both. **Grey:** Low probability for both. Each red dot is a molecule. (WLOGP approximated using LogP).""")
                     else: st.error("Could not generate BOILED-Egg plot.")
                 else: st.warning("No valid data available for BOILED-Egg plot (requires multiple molecules with valid LogP and TPSA).")

            st.divider()
            all_results_data_list = []
            for data_item_all in st.session_state.molecule_data_store:
                flat_data_item = {"Molecule_Name": data_item_all["name"]}
                if data_item_all.get("physchem"): flat_data_item.update({f"PhysChem_{k.replace(' ','_')}": v for k,v in data_item_all["physchem"].items() if "Error" not in k})
                if data_item_all.get("drug_likeness"): flat_data_item.update({f"DrugLike_{k.replace(' ','_')}": v for k,v in data_item_all["drug_likeness"].items() if "Error" not in k})
                if data_item_all.get("pharmacokinetics"): flat_data_item.update({f"PK_{k.replace(' ','_')}": v for k,v in data_item_all["pharmacokinetics"].items() if "Error" not in k})
                if data_item_all.get("radar_params"): flat_data_item.update({f"Radar_{k.replace(' ','_')}": v for k,v in data_item_all["radar_params"].items() if "Error" not in k})
                all_results_data_list.append(flat_data_item)

            if all_results_data_list:
                 try:
                     df_all_results = pd.DataFrame(all_results_data_list)
                     st.download_button(label="📥 Download All Results as CSV", data=df_all_results.to_csv(index=False).encode('utf-8'), file_name="ADMETrix_all_results.csv", mime="text/csv", key="download_all_results_action_button", use_container_width=True, type="primary")
                 except Exception as e:
                     st.error(f"Error preparing 'Download All Results' CSV: {e}")


        elif nav_selected == "Analysis": st.info("☝️ Please upload your molecular data and click 'Start Analysis' to view results.")


elif nav_selected == "About":
    st.markdown("<div class='page-title'>ℹ️ ADMETriX and The Idea Behind It!</div>", unsafe_allow_html=True);

    # Display image and author info centrally
    st.markdown("""<div class="circle-img-container"><img src="https://media.licdn.com/dms/image/v2/D4D03AQG40jMywf_Vrg/profile-displayphoto-shrink_800_800/B4DZW9E1mDH4Ac-/0/1742633921321?e=1752105600&v=beta&t=KtKUnuLZf_1CfHp3y2YVY0x9UwKplrbanEiq5MFbncU " alt="Sonali Lakhamade" class="circle-img"/></div>""", unsafe_allow_html=True) # Corrected image URL if needed
    st.markdown('<div class="author-name">Sonali Lakhamade</div>', unsafe_allow_html=True)
    st.markdown('<div class="author-title">Author & Developer</div>', unsafe_allow_html=True)
    # Centered LinkedIn button below author info
    st.markdown('<div style="text-align: center; margin-bottom: 2rem;"><a href="https://linkedin.com/in/sonali-lakhamade-b6a83230a" target="_blank" class="linkedin-btn">Connect with Sonali on LinkedIn</a></div>', unsafe_allow_html=True)

    # Main description text
    st.markdown("""
    <div class="content-box">
    ADMETriX was developed as part of my Master's degree in Bioinformatics at DES Pune University, with the goal of providing a user-friendly platform for the bioinformatics community.<br><br>
    The author holds a strong interest in the areas of structural bioinformatics, genomic data analysis, and computational biology tools development.<br><br>
    This project marks an initial step toward developing more advanced, feature-rich bioinformatics applications in the future.<br><br>
    The web server will continue to be enhanced with additional tools, improved UI, and expanded compound analysis features based on user feedback and technological trends.
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Rest of the About sections
    st.markdown("<div class='page-subtitle'>🎯 Purpose</div>", unsafe_allow_html=True); st.markdown("""<div class="purpose-box"><ul><li>Enable rapid, browser-based compound analysis, focusing on ADMET properties.</li><li>Support students and researchers in drug discovery, medicinal chemistry, and computational toxicology.</li><li>Provide an educational tool for understanding key concepts in cheminformatics and ADME profiling.</li><li>Allow batch analysis, interactive visualization, and easy data export.</li></ul></div>""", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>✨ Key Highlights</div>", unsafe_allow_html=True); st.markdown("""<div class="feature-box"><ul><li>Comprehensive ADMET profiling: 2D/3D visualization, properties, rules, PK predictions.</li><li>User-friendly interface suitable for all levels.</li><li>Batch processing capability.</li><li>Downloadable outputs (PNG, CSV).</li><li>Interactive plots (Radar, BOILED-Egg) and PubChem Similarity Search.</li></ul></div>""", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>🛠️ Tools and Technologies</div>", unsafe_allow_html=True); st.markdown("""<div class="content-box">Built with: Python, Streamlit, RDKit, Pandas, NumPy, Matplotlib, Requests, stmol.</div>""", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>🌟 Benefits</div>", unsafe_allow_html=True); st.markdown("""<div class="benefit-box"><ul><li>Accessibility: No installation needed.</li><li>Ease of Use: Intuitive design.</li><li>Educational Value: Helps understand ADMET/cheminformatics.</li><li>Research Support: Facilitates screening & hypothesis generation.</li><li>Time-Saving: Automates routine calculations.</li><li>Cost-Effective: Free, open-source based.</li></ul></div>""", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>🔮 Planned Future Enhancements</div>", unsafe_allow_html=True); st.markdown("""<div class="feature-box"><ul><li>Medicinal Chemistry filters (PAINS, Brenk).</li><li>Robust ESOL calculation.</li><li>Advanced ML-based predictions.</li><li>Enhanced PubChem integration.</li><li>User accounts & workflow customization.</li></ul></div>""", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>🙏 Mentorship & Acknowledgement</div>", unsafe_allow_html=True); st.markdown("""<div class="acknowledgement-box">I extend my sincere gratitude to <b>Dr. Kushagra Kashyap</b>, Assistant Professor in Bioinformatics, School of Science and Mathematics, DES Pune University for his invaluable guidance and academic support throughout this project. His expertise and mentorship played a pivotal role in shaping the project's scientific direction.<br><br>His encouragement and expert insights were instrumental in refining the technical implementation and ensuring the project's successful completion. I'm grateful for his contributions to this project.<br><br><a href="https://www.linkedin.com/in/dr-kushagra-kashyap-b230a3bb?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank" class="linkedin-btn">Connect with Dr. Kashyap on LinkedIn</a></div>""", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>📧 Feedback & Contact</div>", unsafe_allow_html=True); st.markdown("""<div class="content-box">Your feedback is highly appreciated!...<ul><li><b>Email:</b> <a href="mailto:sonaalee21@gmail.com">sonaalee21@gmail.com</a></li><li><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/sonali-lakhamade-b6a83230a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app" target="_blank">Sonali Lakhamade</a></li></ul></div>""", unsafe_allow_html=True) # Corrected this line

# --- FOOTER ---
st.divider()
st.markdown("<div class='footer-text'>Made with passion by Sonali Lakhamade, an enthusiastic bioinformatics student | © 2025</div>", unsafe_allow_html=True)
