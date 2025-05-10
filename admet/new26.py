import streamlit as st
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from math import pi
import requests
import re
from PIL import Image # For RDKit 2D image

# --- PAGE CONFIG (MUST BE THE FIRST STREAMLIT COMMAND) ---

st.set_page_config(
layout="wide",
page_title="ADMETriX",
page_icon="🧬"
)

# --- RDKit Availability & Imports ---
# Initialize flags first

RDKIT_AVAILABLE = True
STMOL_AVAILABLE = True
RDKIT_IMPORT_ERROR_MESSAGE = None # To store error message if import fails

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw, AllChem
    import py3Dmol # For stmol, we need py3Dmol as well for direct view creation
    from stmol import showmol
except ImportError as e:
    RDKIT_AVAILABLE = False
    STMOL_AVAILABLE = False
    RDKIT_IMPORT_ERROR_MESSAGE = f"CRITICAL ERROR: RDKit, stmol, or py3Dmol is not installed or could not be imported: {e}. Please ensure these are correctly installed in your environment. App will have limited or no analysis functionality."

# Now, if there was an import error, display it using st.error() AFTER set_page_config
if RDKIT_IMPORT_ERROR_MESSAGE:
    st.error(RDKIT_IMPORT_ERROR_MESSAGE)

# --- NAVIGATION STATE & OTHER SESSION STATES ---

if "nav_selected" not in st.session_state:
    st.session_state["nav_selected"] = "Home"
if "analysis_triggered" not in st.session_state:
    st.session_state.analysis_triggered = False
if "molecule_data_store" not in st.session_state:
    st.session_state.molecule_data_store = None
if "current_smiles" not in st.session_state:
    st.session_state.current_smiles = ""
if "current_mol_name" not in st.session_state:
    st.session_state.current_mol_name = "Molecule"
if "current_pubchem_search_results" not in st.session_state:
    st.session_state.current_pubchem_search_results = None
if "current_pubchem_main_cid" not in st.session_state:
    st.session_state.current_pubchem_main_cid = None
if "current_pubchem_search_name" not in st.session_state:
    st.session_state.current_pubchem_search_name = ""
if "analysis_triggered_by_pubchem_or_sdf" not in st.session_state:
    st.session_state.analysis_triggered_by_pubchem_or_sdf = False
if "current_mol_name_sdf_upload_attempt" not in st.session_state:
    st.session_state.current_mol_name_sdf_upload_attempt = "Molecule from SDF"


# --- Navigation Function ---

def update_nav(label):
    st.session_state["nav_selected"] = label
    if label != "Analysis":
        st.session_state.analysis_triggered = False
        # st.session_state.molecule_data_store = None # Decided against clearing here to persist results across nav


# --- RDKIT HELPER FUNCTIONS ---

if RDKIT_AVAILABLE:
    def get_mol_from_smiles(smiles_string):
        try:
            mol = Chem.MolFromSmiles(smiles_string)
            if mol: AllChem.SanitizeMol(mol) 
            return mol
        except Exception as e:
            print(f"Error parsing SMILES '{smiles_string}': {e}") 
            return None

    def get_mol_from_sdf_block(sdf_block):
        try:
            mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
            if mol: AllChem.SanitizeMol(mol) 
            return mol
        except Exception as e:
            print(f"Error parsing SDF block: {e}") 
            return None

    def generate_3d_coordinates(mol_input):
        if not mol_input: return None
        mol = Chem.Mol(mol_input) 
        mol_with_hs = Chem.AddHs(mol)
        embed_result = AllChem.EmbedMolecule(mol_with_hs, AllChem.ETKDGv3())
        if embed_result == 0: 
            try: AllChem.UFFOptimizeMolecule(mol_with_hs)
            except Exception: pass 
            return mol_with_hs
        elif embed_result == -1: 
            print("Initial 3D embedding failed. Trying with random coordinates...") 
            params = AllChem.EmbedParameters()
            params.useRandomCoords = True
            params.randomSeed = 42 
            embed_result_random = AllChem.EmbedMolecule(mol_with_hs, params)
            if embed_result_random == 0:
                try: AllChem.UFFOptimizeMolecule(mol_with_hs)
                except Exception: pass
                return mol_with_hs
            else:
                print("Could not generate 3D coordinates even with random coords.") 
                return None 
        return None 

    def mol_to_xyz(mol_3d, mol_name="Molecule"):
        if not mol_3d or mol_3d.GetNumConformers() == 0: return ""
        conformer = mol_3d.GetConformer()
        xyz_lines = [f"{mol_3d.GetNumAtoms()}", mol_name]
        for atom in mol_3d.GetAtoms():
            pos = conformer.GetAtomPosition(atom.GetIdx())
            xyz_lines.append(f"{atom.GetSymbol()} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f}")
        return "\n".join(xyz_lines)

    def generate_2d_image(mol, size=(300, 300)): 
        if mol:
            has_2d_conformer = False
            if mol.GetNumConformers() > 0:
                conf = mol.GetConformer()
                if not conf.Is3D():
                    sum_sq_coords = sum(conf.GetAtomPosition(i).x**2 + conf.GetAtomPosition(i).y**2 for i in range(mol.GetNumAtoms()))
                    if sum_sq_coords > 1e-4: 
                        has_2d_conformer = True
            
            if not has_2d_conformer:
                AllChem.Compute2DCoords(mol)
                if mol.GetNumConformers() == 0 or (mol.GetNumConformers() > 0 and mol.GetConformer().Is3D()):
                    print("Warning: Compute2DCoords did not result in a usable 2D conformer.")

            try:
                pil_image = Draw.MolToImage(mol, size=size, kekulize=True)
                if pil_image:
                    img_byte_arr = io.BytesIO()
                    pil_image.save(img_byte_arr, format='PNG')
                    return img_byte_arr.getvalue()
                else:
                    print("Draw.MolToImage returned None.") 
                    return None
            except Exception as e:
                print(f"Error generating 2D image with RDKit: {e}") 
                return None
        return None

    def calculate_physicochemical_properties(mol):
        if not mol: return {}
        return {
            "Molecular Weight (MW)": Descriptors.MolWt(mol),
            "LogP (Octanol-Water Partition Coefficient)": Descriptors.MolLogP(mol),
            "Topological Polar Surface Area (TPSA)": Descriptors.TPSA(mol),
            "Number of Hydrogen Bond Donors (HBD)": Descriptors.NumHDonors(mol),
            "Number of Hydrogen Bond Acceptors (HBA)": Descriptors.NumHAcceptors(mol),
            "Number of Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
            "Number of Rings": Descriptors.RingCount(mol),
            "Number of Heavy Atoms": mol.GetNumHeavyAtoms(),
            "Number of Aromatic Rings": Descriptors.NumAromaticRings(mol),
            "Fraction of C sp3 Atoms (FracCsp3)": Descriptors.FractionCSP3(mol),
            "Molar Refractivity": Descriptors.MolMR(mol),
        }

    def check_drug_likeness_rules(props, mol_obj): 
        rules_violations = {}
        lipinski_violations = []
        if props.get("LogP (Octanol-Water Partition Coefficient)", 0) > 5: lipinski_violations.append("LogP > 5")
        if props.get("Molecular Weight (MW)", 0) > 500: lipinski_violations.append("MW > 500")
        if props.get("Number of Hydrogen Bond Donors (HBD)", 0) > 5: lipinski_violations.append("HBD > 5")
        if props.get("Number of Hydrogen Bond Acceptors (HBA)", 0) > 10: lipinski_violations.append("HBA > 10")
        rules_violations["Lipinski's Rule of Five"] = f"{len(lipinski_violations)} violations: {', '.join(lipinski_violations) if lipinski_violations else 'Pass'}"
        
        total_atoms = mol_obj.GetNumAtoms() if mol_obj else 0 
        ghose_pass = (160 <= props.get("Molecular Weight (MW)", 0) <= 480 and
                      -0.4 <= props.get("LogP (Octanol-Water Partition Coefficient)", 0) <= 5.6 and
                      40 <= props.get("Molar Refractivity", 0) <= 130 and
                      20 <= total_atoms <= 70) 
        rules_violations["Ghose Filter"] = "Pass" if ghose_pass else "Fail"
        
        veber_pass = (props.get("Number of Rotatable Bonds", 0) <= 10 and
                      props.get("Topological Polar Surface Area (TPSA)", 0) <= 140)
        rules_violations["Veber Rule"] = "Pass" if veber_pass else "Fail"
        
        egan_pass = (props.get("LogP (Octanol-Water Partition Coefficient)", 0) <= 5.88 and
                     props.get("Topological Polar Surface Area (TPSA)", 0) <= 131.6)
        rules_violations["Egan Rule"] = "Pass" if egan_pass else "Fail"
        
        muegge_violations = []
        if not (200 <= props.get("Molecular Weight (MW)", 0) <= 600): muegge_violations.append("MW not in [200,600]")
        if props.get("LogP (Octanol-Water Partition Coefficient)", 0) > 5 : muegge_violations.append("LogP > 5")
        if props.get("Topological Polar Surface Area (TPSA)", 0) > 150 : muegge_violations.append("TPSA > 150")
        if props.get("Number of Rotatable Bonds", 0) > 15: muegge_violations.append("RotBonds > 15")
        if props.get("Number of Hydrogen Bond Donors (HBD)", 0) > 5: muegge_violations.append("HBD > 5")
        if props.get("Number of Hydrogen Bond Acceptors (HBA)", 0) > 10: muegge_violations.append("HBA > 10")
        if props.get("Number of Rings", 0) > 4 : muegge_violations.append("Rings > 4") 
        rules_violations["Muegge (Pfizer) Rule"] = f"{len(muegge_violations)} violations: {', '.join(muegge_violations) if muegge_violations else 'Pass'}"
        
        return rules_violations

# --- PLOTTING FUNCTIONS ---

def plot_bioavailability_radar_core(radar_data_values, mol_name="Molecule"):
    if not radar_data_values: return None
    labels_raw, stats_raw = list(radar_data_values.keys()), list(radar_data_values.values())
    
    ideal_plot_ranges = {
        "LIPO (LogP)": (-2, 6), "SIZE (MW)": (100, 500), "POLAR (TPSA)": (20, 140), 
        "INSOLU (LogS)": (-6, 0), "INSATU (FracCsp3)": (0.25, 1), "FLEX (RotB)": (0, 10)
    }

    stats_normalized = []
    for i, label in enumerate(labels_raw):
        raw_val = stats_raw[i]
        min_ideal, max_ideal = ideal_plot_ranges.get(label, (raw_val, raw_val))

        if label == "POLAR (TPSA)" or label == "FLEX (RotB)":
            if raw_val <= min_ideal: normalized_val = 1.0
            elif raw_val >= max_ideal: normalized_val = 0.0
            else: normalized_val = 1.0 - (raw_val - min_ideal) / (max_ideal - min_ideal) if (max_ideal - min_ideal) != 0 else 0.5
        
        elif label == "LIPO (LogP)" or label == "SIZE (MW)":
            normalized_val = np.clip((raw_val - min_ideal) / (max_ideal - min_ideal), 0, 1) if (max_ideal - min_ideal) != 0 else 0.5
        
        else: 
            normalized_val = np.clip((raw_val - min_ideal) / (max_ideal - min_ideal), 0, 1) if (max_ideal - min_ideal) != 0 else 0.5
        stats_normalized.append(normalized_val)

    angles = np.linspace(0, 2 * pi, len(labels_raw), endpoint=False).tolist()
    stats_plot = stats_normalized + stats_normalized[:1] 
    angles_plot = angles + angles[:1] 

    plt.style.use('seaborn-v0_8-pastel') 
    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles_plot, stats_plot, color='#1E88E5', linewidth=2, linestyle='solid', marker='o', markersize=4)
    ax.fill(angles_plot, stats_plot, '#90CAF9', alpha=0.4)
    ax.set_xticks(angles); ax.set_xticklabels(labels_raw, fontsize=7, weight='bold')
    ax.set_yticks(np.arange(0, 1.1, 0.2)); ax.set_yticklabels([f"{t:.1f}" for t in np.arange(0, 1.1, 0.2)], fontsize=7)
    ax.set_ylim(0, 1); ax.grid(color='grey', linestyle=':', linewidth=0.5)
    ax.set_title(f"Bioavailability Radar: {mol_name}", size=10, weight='bold', y=1.15)
    
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches='tight', dpi=120); plt.close(fig)
    return buf.getvalue()


def plot_bioavailability_radar(phys_props, mol_name="Molecule"):
    if not phys_props or not RDKIT_AVAILABLE: return None
    radar_data_for_core = {
        "LIPO (LogP)": phys_props.get("LogP (Octanol-Water Partition Coefficient)", 0),
        "SIZE (MW)": phys_props.get("Molecular Weight (MW)", 0),
        "POLAR (TPSA)": phys_props.get("Topological Polar Surface Area (TPSA)", 0),
        "INSOLU (LogS)": 5 - phys_props.get("LogP (Octanol-Water Partition Coefficient)", 5), 
        "INSATU (FracCsp3)": phys_props.get("Fraction of C sp3 Atoms (FracCsp3)", 0),
        "FLEX (RotB)": phys_props.get("Number of Rotatable Bonds", 0),
    }
    return plot_bioavailability_radar_core(radar_data_for_core, mol_name)

def plot_boiled_egg(phys_props_list): 
    if not phys_props_list or not RDKIT_AVAILABLE or not isinstance(phys_props_list, list): return None
    
    wlogp_values = []
    tpsa_values = []
    for props in phys_props_list:
        wlogp_values.append(props.get("LogP (Octanol-Water Partition Coefficient)", np.nan))
        tpsa_values.append(props.get("Topological Polar Surface Area (TPSA)", np.nan))

    valid_indices = [i for i, (w, t) in enumerate(zip(wlogp_values, tpsa_values)) if pd.notna(w) and pd.notna(t)]
    if not valid_indices:
        print("No valid data points for BOILED-Egg plot.")
        return None

    wlogp = [wlogp_values[i] for i in valid_indices]
    tpsa = [tpsa_values[i] for i in valid_indices]
    
    if not wlogp or not tpsa: return None

    plt.style.use('seaborn-v0_8-whitegrid') 
    fig, ax = plt.subplots(figsize=(6, 5))
    yolk_ellipse = patches.Ellipse(xy=(2.9, 75), width=5.0, height=110, angle=0, fill=True, color='#FFECB3', alpha=0.6, zorder=0) 
    white_ellipse = patches.Ellipse(xy=(2.9, 55), width=7.0, height=140, angle=0, fill=True, color='#E3F2FD', alpha=0.4, zorder=0) 
    ax.add_patch(white_ellipse); ax.add_patch(yolk_ellipse)
    ax.scatter(wlogp, tpsa, c='#D32F2F', s=40, alpha=0.8, zorder=1, label='Compounds')
    ax.set_xlabel("WLOGP (RDKit MolLogP as proxy)", fontsize=10, weight='bold')
    ax.set_ylabel("TPSA (Å²)", fontsize=10, weight='bold')
    ax.set_title("BOILED-Egg Plot", fontsize=12, weight='bold')
    ax.set_xlim(-4, 8); ax.set_ylim(0, 200)
    ax.text(0.5, 185, 'GI Absorption (Yolk for BBB+)', fontsize=8, color='#4A4A4A', ha='center', weight='bold', style='italic') 
    ax.text(0.5, 170, 'BBB Permeation (White for GI only)', fontsize=8, color='#4A4A4A', ha='center', weight='bold', style='italic') 
    ax.grid(True, linestyle=':', linewidth=0.5); ax.legend(fontsize=8, loc='upper right')
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches='tight', dpi=120); plt.close(fig) 
    return buf.getvalue()


# --- PUBCHEM HELPER FUNCTIONS ---

def pubchem_get_cid_from_name(name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON"
    try:
        response = requests.get(url, timeout=10); response.raise_for_status() 
        cids = response.json().get('IdentifierList', {}).get('CID', [])
        return cids[0] if cids else None
    except requests.exceptions.RequestException as e: print(f"PubChem API error (name->CID for '{name}'): {e}"); return None
    except ValueError: print(f"PubChem API JSON decoding error (name->CID for '{name}')."); return None

def pubchem_get_similar_compounds(cid, threshold=90, max_results=5):
    if not cid: return []
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/similarity/cid/{cid}/cids/JSON?Threshold={threshold}&MaxRecords={max_results}"
    try:
        response = requests.get(url, timeout=15); response.raise_for_status()
        return response.json().get('IdentifierList', {}).get('CID', [])
    except requests.exceptions.RequestException as e: print(f"PubChem API error (similarity for CID {cid}): {e}"); return []
    except ValueError: print(f"PubChem API JSON decoding error (similarity for CID {cid})."); return []

def pubchem_get_compound_info(cid):
    if not cid: return {}
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IUPACName,CanonicalSMILES,MolecularFormula,MolecularWeight/JSON"
    try:
        response = requests.get(url, timeout=10); response.raise_for_status()
        data = response.json()
        if 'PropertyTable' in data and 'Properties' in data['PropertyTable'] and data['PropertyTable']['Properties']:
            return data['PropertyTable']['Properties'][0] 
        return {}
    except requests.exceptions.RequestException as e: print(f"PubChem API error (info for CID {cid}): {e}"); return {}
    except ValueError: print(f"PubChem API JSON decoding error (info for CID {cid})."); return {}

def clean_molecule_name_for_pubchem(name):
    if not name: return ""
    name = re.sub(r'sdf\d*[_ ]*', '', name, flags=re.IGNORECASE)
    return name.replace("_", " ").strip()

# --- STYLING ---

st.markdown("""<style> 
body, .stApp { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #333; background-color: #f0f8ff; }
.main-title-container { padding: 1rem 0; margin-bottom: 1rem; border-radius: 10px; background: linear-gradient(90deg, #00796B, #00ACC1); box-shadow: 0 4px 15px rgba(0, 121, 107, 0.3); text-align: center; }
.main-title { font-size: 3.5rem; font-weight: 700; color: white; letter-spacing: 3px; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3); margin: 0; }
.stButton>button { border-radius: 20px !important; border: 1px solid #00796B !important; background: linear-gradient(145deg, #ffffff, #e0f2f1) !important; color: #00695C !important; padding: 0.5rem 1.5rem !important; font-weight: 600 !important; transition: all 0.3s ease !important; box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1) !important; width: 100% !important; margin-bottom: 0.5rem; }
.stButton>button:hover { background: linear-gradient(145deg, #00796B, #00ACC1) !important; color: white !important; border-color: #004D40 !important; box-shadow: 3px 3px 8px rgba(0, 0, 0, 0.2) !important; transform: translateY(-2px) !important; }
.stDownloadButton>button { border-radius: 20px !important; border: 1px solid #FF8F00 !important; background: linear-gradient(145deg, #fff3e0, #ffe0b2) !important; color: #E65100 !important; padding: 0.5rem 1.5rem !important; font-weight: 600 !important; transition: all 0.3s ease !important; box-shadow: 2px 2px 5px rgba(0,0,0,0.1) !important; width: auto !important; margin-top: 0.5rem; margin-bottom: 0.5rem;}
.stDownloadButton>button:hover { background: linear-gradient(145deg, #FFB74D, #FF9800) !important; color: white !important; border-color: #E65100 !important; box-shadow: 3px 3px 8px rgba(0,0,0,0.2) !important; transform: translateY(-2px) !important; }
.page-title { font-size: 2.5rem; color: #004D40; border-bottom: 3px solid #00ACC1; padding-bottom: 0.6rem; margin-top: 1.5rem; margin-bottom: 2rem; font-weight: 600; }
.page-subtitle { font-size: 1.5rem; color: #00796B; margin-bottom: 1.2rem; font-weight: 600; border-left: 4px solid #00ACC1; padding-left: 0.8rem; }
.content-box, .feature-box, .step-box, .purpose-box, .benefit-box, .acknowledgement-box { background: #ffffff; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 3px 10px rgba(0, 0, 0, 0.07); border-left: 5px solid; }
.feature-box { border-left-color: #00ACC1; background-color: #E0F7FA; } .step-box { border-left-color: #4CAF50; background-color: #E8F5E9; }
.purpose-box { border-left-color: #FFB300; background-color: #FFFDE7; } .benefit-box { border-left-color: #1E88E5; background-color: #E3F2FD; }
.acknowledgement-box { border-left-color: #8E24AA; background-color: #F3E5F5; } .content-box { border-left-color: #00796B; background-color: #f8f9fa; }
.circle-img-container { display: flex; justify-content: center; margin-bottom: 1rem; }
.circle-img { border-radius: 50%; width: 150px; height: 150px; object-fit: cover; border: 7px solid #00796B; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2); }
.author-name { text-align:center; font-size:1.6rem; font-weight:600; color: #004D40; margin-top: 0.5rem; }
.author-title { text-align:center; font-size:1.2rem; color: #555; margin-bottom: 1rem; font-style: italic;}
.linkedin-btn { display: inline-block; padding: 0.7em 1.5em; border: none; border-radius: 25px; background: linear-gradient(145deg, #0077b5, #005582); color: white !important; font-weight: bold; text-decoration: none; box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2); transition: all 0.3s ease; text-align: center; }
.linkedin-btn:hover { background: linear-gradient(145deg, #005582, #003350); box-shadow: 4px 4px 12px rgba(0, 0, 0, 0.3); transform: translateY(-2px); color: white !important; text-decoration: none; }
.footer-text { font-size: 0.9rem; color: #6c757d; text-align: center; margin-top: 3rem; padding-bottom: 1rem; font-style: italic; }
.stDataFrame { border: 1px solid #dee2e6; border-radius: 5px; } .stDataFrame thead th { background-color: #E0F2F1; color: #00695C; font-weight: bold; }
table.dataframe { width: 100%; border-collapse: collapse; } table.dataframe th { background-color: #E0F2F1; color: #00695C; font-weight: bold; padding: 8px; text-align: left; border-bottom: 2px solid #00796B;}
table.dataframe td { padding: 8px; border-bottom: 1px solid #dee2e6; } table.dataframe tr:nth-child(even) { background-color: #f8f9fa; }
.st-expander { border: 1px solid #B2DFDB; border-radius: 8px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05); margin-bottom: 1rem;}
.st-expander header { background-color: #E0F2F1; border-radius: 8px 8px 0 0; font-weight: 600; color: #00695C; }
.st-expander header:hover { background-color: #B2DFDB; } .stText { white-space: pre-wrap !important; }
</style>""", unsafe_allow_html=True)

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
    if st.button("ℹ About", key="nav_About_main", use_container_width=True): update_nav("About")
nav_selected = st.session_state.get("nav_selected", "Home")
st.divider()

# --- PAGE CONTENT ---

if nav_selected == "Home":
    st.markdown("<div class='page-title'>Welcome to ADMETriX: Your Bioinformatics Drug Nexus</div>", unsafe_allow_html=True)
    st.markdown("""
    ADMETriX is an integrated, web-based platform designed for comprehensive compound analysis.
    Explore ADMET properties, visualize structures, and evaluate drug-likeness with ease.
    """)
    st.markdown("<div class='page-subtitle'>🚀 Key Features</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="feature-box">
    <ul>
    <li><b>📝 SMILES/SDF Input</b>: Analyze molecules by providing SMILES strings or uploading SDF files.</li>
    <li><b>🔗 PubChem Compound Search</b>: Find compounds by name and use their SMILES for analysis.</li>
    <li><b>📊 2D & 3D Visualization</b>: View interactive 2D and 3D structures of your molecules.</li>
    <li><b>🧪 Physicochemical Properties Calculation</b>: Calculate key properties like LogP, MW, TPSA, etc.</li>
    <li><b>💊 Drug-likeness Rules Evaluation</b>: Assess compounds against common rules (Lipinski, Ghose, etc.).</li>
    <li><b>📈 Bioavailability Radar & BOILED-Egg Plots</b>: Visualize ADMET profiles with image download options.</li>
    <li><b>📁 Data Download</b>: Download 2D images, plots, and 3D coordinate (XYZ) files.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

elif nav_selected == "User Guide":
    st.markdown("<div class='page-title'>📖 User Guide</div>", unsafe_allow_html=True)
    st.markdown("Follow these steps to use ADMETriX:")
    st.markdown("""
    <div class="step-box"><b>Step 1: Navigate to Analysis</b><br>Click on the <b>🔬 Analysis</b> tab.</div>
    <div class="step-box"><b>Step 2: Input Molecule</b>
    <ul>
    <li><b>By SMILES</b>: Enter a SMILES string in the 'Input SMILES' tab and click "Analyze SMILES".</li>
    <li><b>By SDF File</b>: Upload an SDF file in the 'Upload SDF' tab and click "Analyze SDF".</li>
    <li><b>By PubChem Search</b>: Use the 'PubChem Search' tab. Search for a compound. Click "Use this SMILES for Analysis" on a result.</li>
    </ul>
    </div>
    <div class="step-box"><b>Step 3: Explore Results</b><br>View the 2D/3D structures, properties, drug-likeness, and plots. Download options are available for generated 2D images, plots, and 3D coordinate (XYZ) files.</div>
    <div class="step-box"><b>Important Note on RDKit:</b><br>This application requires RDKit, stmol, and py3Dmol. If you see an error at the top about these libraries not being installed, please ensure they are correctly set up in your Python environment. Plot generation also requires a working Matplotlib setup.</div>
    """, unsafe_allow_html=True)

elif nav_selected == "Analysis":
    st.markdown("<div class='page-title'>🔬 ADMETriX Analysis</div>", unsafe_allow_html=True)

    if not RDKIT_AVAILABLE: 
        st.warning("Core analysis functionality is disabled because required libraries (RDKit/stmol/py3Dmol) could not be loaded. Please check the error message at the top of the page and ensure correct installation.")
    else: 
        input_tabs = st.tabs(["📝 Input SMILES", "📁 Upload SDF", "🔎 PubChem Search"])

        with input_tabs[0]: 
            current_smiles_val = st.session_state.get("current_smiles", "")
            current_name_val = st.session_state.get("current_mol_name", "Molecule from SMILES")
            if st.session_state.get("analysis_triggered_by_pubchem_or_sdf"):
                 current_smiles_val = st.session_state.current_smiles
                 current_name_val = st.session_state.current_mol_name
                 st.session_state.analysis_triggered_by_pubchem_or_sdf = False

            user_smiles = st.text_input(
                "Enter SMILES string:", value=current_smiles_val, key="smiles_input_val_tab", 
                placeholder="e.g., CC(=O)Oc1ccccc1C(=O)OH for Aspirin"
            )
            user_mol_name = st.text_input(
                "Molecule Name (Optional):", value=current_name_val, key="mol_name_smiles_val_tab" 
            )
            if st.button("Analyze SMILES", key="analyze_smiles_btn_main_tab"):
                if user_smiles:
                    st.session_state.current_smiles = user_smiles
                    st.session_state.current_mol_name = user_mol_name if user_mol_name else "Molecule from SMILES"
                    st.session_state.analysis_triggered = True
                    st.session_state.molecule_data_store = None 
                    st.rerun() 
                else:
                    st.warning("Please enter a SMILES string."); st.session_state.analysis_triggered = False

        with input_tabs[1]: 
            sdf_file = st.file_uploader("Upload SDF file (single molecule)", type=["sdf", "mol"], key="sdf_uploader_main_tab")
            
            default_sdf_name = "Molecule from SDF"
            if st.session_state.get("current_mol_name_sdf_upload_attempt"):
                default_sdf_name = st.session_state.current_mol_name_sdf_upload_attempt
            elif sdf_file and not st.session_state.get("current_mol_name_sdf_upload_attempt"):
                 default_sdf_name = sdf_file.name.split('.')[0] if sdf_file.name else "Molecule from SDF"

            sdf_user_mol_name = st.text_input(
                "Molecule Name (Optional, from SDF):", value=default_sdf_name, key="mol_name_sdf_main_tab" 
            )
            st.session_state.current_mol_name_sdf_upload_attempt = sdf_user_mol_name

            if sdf_file is not None:
                if st.button("Analyze SDF", key="analyze_sdf_btn_main_tab"):
                    try:
                        sdf_block = sdf_file.read().decode("utf-8")
                        mol_from_sdf_obj = get_mol_from_sdf_block(sdf_block)
                        if mol_from_sdf_obj:
                            name_to_set = sdf_user_mol_name 
                            if mol_from_sdf_obj.HasProp('_Name') and mol_from_sdf_obj.GetProp('_Name').strip():
                                name_from_sdf_prop = mol_from_sdf_obj.GetProp('_Name').strip()
                                if name_to_set == "Molecule from SDF" or not name_to_set.strip() or name_to_set == (sdf_file.name.split('.')[0] if sdf_file.name else "Molecule from SDF"):
                                    name_to_set = name_from_sdf_prop
                            elif not name_to_set or name_to_set == "Molecule from SDF": 
                                name_to_set = sdf_file.name.split('.')[0] if sdf_file.name else "Molecule from SDF"
                            
                            st.session_state.current_mol_name = name_to_set
                            smiles_from_sdf = Chem.MolToSmiles(mol_from_sdf_obj, isomericSmiles=True)
                            if smiles_from_sdf:
                                st.session_state.current_smiles = smiles_from_sdf
                                st.session_state.analysis_triggered = True
                                st.session_state.analysis_triggered_by_pubchem_or_sdf = True 
                                st.session_state.molecule_data_store = None 
                                st.rerun()
                            else: st.error("Could not convert SDF to SMILES."); st.session_state.analysis_triggered = False
                        else: st.error("Could not parse molecule from the uploaded SDF file. Please ensure it's a valid single-molecule SDF."); st.session_state.analysis_triggered = False
                    except Exception as e: st.error(f"Error processing SDF file: {e}"); st.session_state.analysis_triggered = False
            else: 
                 if not st.session_state.get("analysis_triggered"):
                    st.session_state.current_mol_name_sdf_upload_attempt = "Molecule from SDF"

        with input_tabs[2]: 
            st.markdown("##### Find Compounds via PubChem")
            if "pubchem_search_term_tab" not in st.session_state: st.session_state.pubchem_search_term_tab = ""

            user_search_name_input_pubchem = st.text_input(
                "Compound name for PubChem search:", value=st.session_state.pubchem_search_term_tab,
                key="pubchem_search_text_input_tab", help="Enter a compound name (e.g., Aspirin, Ibuprofen)."
            )
            if st.button("Search PubChem 🔎", key="pubchem_search_btn_action_tab"):
                st.session_state.pubchem_search_term_tab = user_search_name_input_pubchem 
                st.session_state.current_pubchem_search_results = None; st.session_state.current_pubchem_main_cid = None
                
                if user_search_name_input_pubchem:
                    cleaned_name = clean_molecule_name_for_pubchem(user_search_name_input_pubchem)
                    if not cleaned_name: st.warning("Search input was empty after cleaning."); st.session_state.current_pubchem_search_results = "not_found_query_empty"
                    else:
                        with st.spinner(f"Searching PubChem for '{cleaned_name}'..."):
                            cid = pubchem_get_cid_from_name(cleaned_name)
                            if cid:
                                st.session_state.current_pubchem_main_cid = cid
                                main_info = pubchem_get_compound_info(cid)
                                if main_info and main_info.get("CanonicalSMILES"):
                                    st.success(f"Found '{main_info.get('IUPACName', user_search_name_input_pubchem)}' (CID: {cid}). Displaying results below.", icon="✅")
                                    results_list = [{"cid": cid, "info": main_info, "is_main": True}]
                                    similar_cids = pubchem_get_similar_compounds(cid, threshold=90, max_results=3) 
                                    if similar_cids:
                                        for scid_item in similar_cids:
                                            if scid_item != cid: 
                                                s_info = pubchem_get_compound_info(scid_item)
                                                if s_info: results_list.append({"cid": scid_item, "info": s_info, "is_main": False})
                                    st.session_state.current_pubchem_search_results = results_list
                                else: st.error(f"Compound '{user_search_name_input_pubchem}' found (CID: {cid}), but critical info (like SMILES) not available from PubChem."); st.session_state.current_pubchem_search_results = "not_found_smiles"
                            else: st.error(f"Compound '{user_search_name_input_pubchem}' not found in PubChem or API error occurred."); st.session_state.current_pubchem_search_results = "not_found"
                else: st.warning("Please enter a compound name for PubChem search.")
            
            pubchem_results_to_show = st.session_state.get("current_pubchem_search_results")
            pubchem_searched_term = st.session_state.get("pubchem_search_term_tab", "")

            if pubchem_results_to_show:
                if pubchem_results_to_show == "not_found_query_empty": pass 
                elif pubchem_results_to_show == "not_found": pass
                elif pubchem_results_to_show == "not_found_smiles": pass
                elif isinstance(pubchem_results_to_show, list):
                    st.markdown("---"); st.markdown(f"##### Search Results for '{pubchem_searched_term}':")
                    for i, item in enumerate(pubchem_results_to_show):
                        info = item["info"]; smiles = info.get('CanonicalSMILES', 'N/A')
                        name_display = info.get('IUPACName', f"CID: {item['cid']}")
                        header = f"**Main Match: {name_display}**" if item["is_main"] else f"**Similar Compound: {name_display}**"
                        with st.container(): 
                            st.markdown(header)
                            st.markdown(f"    Formula: `{info.get('MolecularFormula', 'N/A')}`, MW: `{info.get('MolecularWeight', 'N/A')}`")
                            st.markdown(f"    SMILES: `{smiles}`")
                            if smiles != 'N/A':
                                if st.button(f"Use this SMILES for Analysis", key=f"use_smiles_pubchem_{item['cid']}_{i}"):
                                    st.session_state.current_smiles = smiles
                                    st.session_state.current_mol_name = name_display if name_display != f"CID: {item['cid']}" else "Molecule from PubChem"
                                    st.session_state.analysis_triggered = True
                                    st.session_state.analysis_triggered_by_pubchem_or_sdf = True 
                                    st.session_state.molecule_data_store = None 
                                    st.rerun()
                            st.markdown("---")
        st.divider()

        if st.session_state.get("analysis_triggered") and st.session_state.get("current_smiles"):
            active_smiles = st.session_state.current_smiles
            active_mol_name = st.session_state.current_mol_name
            mol_object_for_analysis = get_mol_from_smiles(active_smiles)

            if mol_object_for_analysis:
                sanitized_mol_name = re.sub(r'[^a-zA-Z0-9_.\-]', '_', active_mol_name)
                if not sanitized_mol_name: sanitized_mol_name = "molecule_analysis" 
                sanitized_mol_name = (sanitized_mol_name[:60]) if len(sanitized_mol_name) > 60 else sanitized_mol_name

                with st.spinner(f"Analyzing {active_mol_name}... Please wait."):
                    st.session_state.molecule_data_store = {
                        "name": active_mol_name, "smiles": active_smiles, "mol_obj_2d": mol_object_for_analysis, 
                        "img_2d_bytes": None, "mol_3d_obj": None, "xyz_str": "",
                        "phys_props": {}, "drug_likeness": {},
                        "radar_plot_bytes": None, "boiled_egg_bytes": None
                    }
                    
                    img_2d_bytes_data = generate_2d_image(mol_object_for_analysis)
                    if img_2d_bytes_data: st.session_state.molecule_data_store["img_2d_bytes"] = img_2d_bytes_data
                    
                    mol_3d_rdkit_obj = generate_3d_coordinates(mol_object_for_analysis) 
                    if mol_3d_rdkit_obj:
                        st.session_state.molecule_data_store["mol_3d_obj"] = mol_3d_rdkit_obj
                        xyz_str_data = mol_to_xyz(mol_3d_rdkit_obj, active_mol_name)
                        if xyz_str_data: st.session_state.molecule_data_store["xyz_str"] = xyz_str_data

                    phys_props_data = calculate_physicochemical_properties(mol_object_for_analysis)
                    st.session_state.molecule_data_store["phys_props"] = phys_props_data
                    
                    drug_likeness_data = check_drug_likeness_rules(phys_props_data, mol_object_for_analysis)
                    st.session_state.molecule_data_store["drug_likeness"] = drug_likeness_data

                    if phys_props_data: # Only generate plots if properties were calculated
                        radar_plot_bytes_data = plot_bioavailability_radar(phys_props_data, active_mol_name)
                        if radar_plot_bytes_data: st.session_state.molecule_data_store["radar_plot_bytes"] = radar_plot_bytes_data
                        
                        boiled_egg_bytes_data = plot_boiled_egg([phys_props_data]) 
                        if boiled_egg_bytes_data: st.session_state.molecule_data_store["boiled_egg_bytes"] = boiled_egg_bytes_data

                display_data = st.session_state.molecule_data_store
                
                st.subheader(f"Analysis for: {display_data['name']} (`{display_data['smiles']}`)")
                col_struct, col_plots = st.columns(2)

                with col_struct:
                    st.markdown("##### 2D Structure")
                    if display_data.get("img_2d_bytes"):
                        st.image(display_data["img_2d_bytes"], use_column_width=True)
                        st.download_button(
                            label="Download 2D Structure (PNG)", data=display_data["img_2d_bytes"],
                            file_name=f"{sanitized_mol_name}_2D_structure.png", mime="image/png",
                            key="download_2d_structure_btn"
                        )
                    else: st.warning("Could not generate 2D image.")
                    
                    st.markdown("##### 3D Structure (Interactive)")
                    if STMOL_AVAILABLE and display_data.get("xyz_str"):
                        view = py3Dmol.view(width=450, height=400) 
                        view.addModel(display_data["xyz_str"], 'xyz'); view.setStyle({'stick':{}}) 
                        view.setBackgroundColor('0xeeeeee'); view.zoomTo()
                        showmol(view, height=400, width=450)
                        # ADDED DOWNLOAD BUTTON FOR XYZ DATA
                        st.download_button(
                            label="Download 3D Coords (XYZ)",
                            data=display_data["xyz_str"],
                            file_name=f"{sanitized_mol_name}_3D_coordinates.xyz",
                            mime="text/plain", # or chemical/x-xyz
                            key="download_3d_xyz_btn"
                        )
                    elif not display_data.get("xyz_str") and display_data.get("mol_3d_obj") is None and RDKIT_AVAILABLE:
                         st.warning("3D coordinate generation failed. Cannot display 3D structure or provide XYZ data.")
                    elif not STMOL_AVAILABLE: st.warning("stmol/py3Dmol is not available for 3D visualization.")
                    else: st.warning("Could not generate data for 3D structure display or XYZ download.")
                
                with col_plots:
                    st.markdown("##### Bioavailability Radar")
                    if display_data.get("radar_plot_bytes"):
                        st.image(display_data["radar_plot_bytes"], use_column_width=True)
                        st.download_button(
                            label="Download Radar Plot (PNG)", data=display_data["radar_plot_bytes"],
                            file_name=f"{sanitized_mol_name}_bioavailability_radar.png", mime="image/png",
                            key="download_radar_plot_btn"
                        )
                    else: st.warning("Could not generate Bioavailability Radar Plot.")

                    st.markdown("##### BOILED-Egg Plot")
                    if display_data.get("boiled_egg_bytes"):
                        st.image(display_data["boiled_egg_bytes"], use_column_width=True)
                        st.download_button(
                            label="Download BOILED-Egg Plot (PNG)", data=display_data["boiled_egg_bytes"],
                            file_name=f"{sanitized_mol_name}_boiled_egg_plot.png", mime="image/png",
                            key="download_boiled_egg_btn"
                        )
                    else: st.warning("Could not generate BOILED-Egg Plot.")

                st.markdown("---"); st.markdown("##### Physicochemical Properties")
                if display_data.get("phys_props"):
                    props_df = pd.DataFrame(list(display_data["phys_props"].items()), columns=['Property', 'Value'])
                    for col in props_df.columns: 
                        if col == 'Value' and pd.api.types.is_numeric_dtype(props_df[col]):
                            props_df[col] = props_df[col].apply(lambda x: f'{x:.2f}' if pd.notnull(x) and isinstance(x, (int, float)) else x)
                    st.dataframe(props_df, use_container_width=True, hide_index=True)
                else: st.info("No physicochemical properties calculated.")

                st.markdown("---"); st.markdown("##### Drug-Likeness Evaluation")
                if display_data.get("drug_likeness"):
                    likeness_df = pd.DataFrame(list(display_data["drug_likeness"].items()), columns=['Rule', 'Evaluation'])
                    st.dataframe(likeness_df, use_container_width=True, hide_index=True)
                else: st.info("No drug-likeness evaluation performed.")
            
            else: st.error(f"The provided SMILES string '{active_smiles}' is invalid or could not be processed by RDKit. Please check the SMILES.")
            st.session_state.analysis_triggered = False 

        elif st.session_state.get("analysis_triggered") and not st.session_state.get("current_smiles"):
            st.info("No SMILES available for analysis. Please input a molecule using one of the methods above.")
            st.session_state.analysis_triggered = False 

elif nav_selected == "About":
    st.markdown("<div class='page-title'>ℹ ADMETriX and The Idea Behind It!</div>", unsafe_allow_html=True);
    st.markdown("""<div class="circle-img-container"><img src="https://media.licdn.com/dms/image/v2/D4D03AQG40jMywf_Vrg/profile-displayphoto-shrink_800_800/B4DZW9E1mDH4Ac-/0/1742633921321?e=1752105600&v=beta&t=KtKUnuLZf_1CfHp3y2YVY0x9UwKplrbanEiq5MFbncU" alt="Sonali Lakhamade" class="circle-img"/></div>""", unsafe_allow_html=True)
    st.markdown('<div class="author-name">Sonali Lakhamade</div>', unsafe_allow_html=True)
    st.markdown('<div class="author-title">Author & Developer</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align: center; margin-bottom: 2rem;"><a href="https://linkedin.com/in/sonali-lakhamade-b6a83230a" target="_blank" class="linkedin-btn">Connect with Sonali on LinkedIn</a></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="content-box">
    ADMETriX was developed as part of my Master's degree in Bioinformatics at DES Pune University, with the goal of providing a user-friendly platform for the bioinformatics community.
    The author holds a strong interest in structural bioinformatics, genomic data analysis, and computational biology tools development.
    This project marks an initial step toward developing more advanced, feature-rich bioinformatics applications.
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("<div class='page-subtitle'>🛠 Tools and Technologies</div>", unsafe_allow_html=True); st.markdown("""<div class="content-box">Built with: Python, Streamlit, RDKit, Pandas, NumPy, Matplotlib, Requests, stmol, py3Dmol.</div>""", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>🙏 Mentorship & Acknowledgement</div>", unsafe_allow_html=True); st.markdown("""<div class="acknowledgement-box">I extend my sincere gratitude to <b>Dr. Kushagra Kashyap</b>, Assistant Professor in Bioinformatics, School of Science and Mathematics, DES Pune University for his invaluable guidance and academic support throughout this project. His expertise and mentorship played a pivotal role in shaping the project's scientific direction. His encouragement and expert insights were instrumental in refining the technical implementation and ensuring the project's successful completion.</div>""", unsafe_allow_html=True)
    st.markdown("<div class='page-subtitle'>📧 Feedback & Contact</div>", unsafe_allow_html=True); st.markdown("""<div class="content-box">Your feedback is highly appreciated!...<ul><li><b>Email:</b> <a href="mailto:sonaalee21@gmail.com">sonaalee21@gmail.com</a></li><li><b>LinkedIn:</b> <a href="https://www.linkedin.com/in/sonali-lakhamade-b6a83230a" target="_blank">Sonali Lakhamade</a></li></ul></div>""", unsafe_allow_html=True)

# --- FOOTER ---
st.divider()
st.markdown("<div class='footer-text'>Made with passion by Sonali Lakhamade, an enthusiastic bioinformatics student | © 2025</div>", unsafe_allow_html=True)
