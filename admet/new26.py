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

# --- Library Availability & Imports ---
RDKIT_AVAILABLE = False
STMOL_AVAILABLE = False
STREAMLIT_KETCHER_AVAILABLE = False
SEABORN_AVAILABLE = False # Default to False
_IMPORT_ERROR_MESSAGES = []

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Draw, AllChem
    RDKIT_AVAILABLE = True
except ImportError as e:
    _IMPORT_ERROR_MESSAGES.append(f"ERROR: RDKit could not be imported: {e}. Essential analysis features will be unavailable. Please install RDKit.")

try:
    import py3Dmol
    from stmol import showmol
    STMOL_AVAILABLE = True
except ImportError as e:
    _IMPORT_ERROR_MESSAGES.append(f"WARNING: stmol or py3Dmol could not be imported: {e}. 3D molecule visualization will be affected.")

try:
    from streamlit_ketcher import st_ketcher
    STREAMLIT_KETCHER_AVAILABLE = True
except ImportError as e:
    _IMPORT_ERROR_MESSAGES.append(f"WARNING: streamlit-ketcher could not be imported: {e}. The molecule sketcher feature will be unavailable.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError as e:
    _IMPORT_ERROR_MESSAGES.append(f"INFO: Seaborn library not found: {e}. Statistical plots (Distributions, Heatmap) will be unavailable. To enable them, please install seaborn: `pip install seaborn`")


# Display import errors/info, if any, after page config
if _IMPORT_ERROR_MESSAGES:
    for msg in _IMPORT_ERROR_MESSAGES:
        if "ERROR:" in msg:
            st.error(msg)
        elif "WARNING:" in msg:
            st.warning(msg)
        elif "INFO:" in msg:
            st.info(msg)

    if not RDKIT_AVAILABLE:
        st.error("Core application functionality is severely limited due to RDKit not being available.")


# --- NAVIGATION STATE & OTHER SESSION STATES ---
default_session_states = {
    "nav_selected": "Home",
    "analysis_triggered": False,
    "molecule_data_store": None,
    "current_smiles": "",
    "current_mol_name": "Molecule",
    "current_pubchem_search_results": None,
    "current_pubchem_main_compound_info": None,
    "current_pubchem_search_name": "",
    "analysis_triggered_by_pubchem_or_sdf": False,
    "current_mol_name_sdf_upload_attempt": "Molecule from SDF",
    "ketcher_display_smiles": "",
    "smiles_from_sketch_for_analysis": "",
    "current_mol_name_sketcher_attempt": "Molecule from Sketcher",
    "analysis_triggered_by_sketcher": False,
    "extended_pubchem_props_for_current_mol": None,
    "smiles_from_file_buffer": ""
}
for key, value in default_session_states.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Navigation Function ---
def update_nav(label):
    st.session_state["nav_selected"] = label
    if label != "Analysis":
        st.session_state.analysis_triggered = False

# --- RDKIT HELPER FUNCTIONS --- (Keep as is)
if RDKIT_AVAILABLE:
    def get_mol_from_smiles(smiles_string):
        try:
            mol = Chem.MolFromSmiles(smiles_string)
            if mol: AllChem.SanitizeMol(mol)
            return mol
        except Exception as e:
            st.warning(f"RDKit parsing error for SMILES '{smiles_string}': {e}")
            return None

    def get_mol_from_sdf_block(sdf_block):
        try:
            mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
            if mol: AllChem.SanitizeMol(mol)
            return mol
        except Exception as e:
            st.warning(f"RDKit parsing error for SDF block: {e}")
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
            params = AllChem.EmbedParameters()
            params.useRandomCoords = True
            params.randomSeed = 42
            embed_result_random = AllChem.EmbedMolecule(mol_with_hs, params)
            if embed_result_random == 0:
                try: AllChem.UFFOptimizeMolecule(mol_with_hs)
                except Exception: pass
                return mol_with_hs
            else:
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

    def generate_2d_image(mol, size=(280, 280)):
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
                return None
            except Exception as e:
                print(f"Error generating 2D image with RDKit: {e}")
                return None
        return None

    def calculate_rdkit_physicochemical_properties(mol):
        if not mol: return {}
        props = {
            "Molecular Weight (MW)": Descriptors.MolWt(mol),
            "LogP (Octanol-Water Partition Coefficient)": Descriptors.MolLogP(mol),
            "Topological Polar Surface Area (TPSA)": Descriptors.TPSA(mol),
            "Hydrogen Bond Donors (HBD)": Descriptors.NumHDonors(mol),
            "Hydrogen Bond Acceptors (HBA)": Descriptors.NumHAcceptors(mol),
            "Rotatable Bonds": Descriptors.NumRotatableBonds(mol),
            "Number of Rings": Descriptors.RingCount(mol),
            "Number of Heavy Atoms": mol.GetNumHeavyAtoms(),
            "Number of Aromatic Rings": Descriptors.NumAromaticRings(mol),
            "Fraction C sp3 Atoms (FracCsp3)": Descriptors.FractionCSP3(mol),
            "Molar Refractivity": Descriptors.MolMR(mol),
        }
        return props

    def check_drug_likeness_rules(props, mol_obj):
        rules_violations = {}
        lipinski_violations = []
        if props.get("LogP (Octanol-Water Partition Coefficient)", 0) > 5: lipinski_violations.append("LogP > 5")
        if props.get("Molecular Weight (MW)", 0) > 500: lipinski_violations.append("MW > 500 Da")
        if props.get("Hydrogen Bond Donors (HBD)", 0) > 5: lipinski_violations.append("HBD > 5")
        if props.get("Hydrogen Bond Acceptors (HBA)", 0) > 10: lipinski_violations.append("HBA > 10")
        rules_violations["Lipinski's Rule of Five"] = f"{len(lipinski_violations)} violations: {', '.join(lipinski_violations) if lipinski_violations else 'Pass'}"

        total_atoms = mol_obj.GetNumAtoms() if mol_obj else 0
        ghose_pass = (160 <= props.get("Molecular Weight (MW)", 0) <= 480 and
                      -0.4 <= props.get("LogP (Octanol-Water Partition Coefficient)", 0) <= 5.6 and
                      40 <= props.get("Molar Refractivity", 0) <= 130 and
                      20 <= total_atoms <= 70)
        rules_violations["Ghose Filter"] = "Pass" if ghose_pass else "Fail"

        veber_pass = (props.get("Rotatable Bonds", 0) <= 10 and
                      props.get("Topological Polar Surface Area (TPSA)", 0) <= 140)
        rules_violations["Veber Rule"] = "Pass" if veber_pass else "Fail"

        egan_pass = (props.get("LogP (Octanol-Water Partition Coefficient)", 0) <= 5.88 and
                     props.get("Topological Polar Surface Area (TPSA)", 0) <= 131.6)
        rules_violations["Egan Rule"] = "Pass" if egan_pass else "Fail"

        muegge_violations = []
        if not (200 <= props.get("Molecular Weight (MW)", 0) <= 600): muegge_violations.append("MW not in [200,600]")
        if props.get("LogP (Octanol-Water Partition Coefficient)", -99) > 5 : muegge_violations.append("LogP > 5")
        if props.get("Topological Polar Surface Area (TPSA)", 999) > 150 : muegge_violations.append("TPSA > 150")
        if props.get("Rotatable Bonds", 99) > 15: muegge_violations.append("RotBonds > 15")
        if props.get("Hydrogen Bond Donors (HBD)", 99) > 5: muegge_violations.append("HBD > 5")
        if props.get("Hydrogen Bond Acceptors (HBA)", 99) > 10: muegge_violations.append("HBA > 10")
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
        if (max_ideal - min_ideal) == 0: normalized_val = 0.5
        elif label == "POLAR (TPSA)" or label == "FLEX (RotB)":
            if raw_val <= min_ideal: normalized_val = 1.0
            elif raw_val >= max_ideal: normalized_val = 0.0
            else: normalized_val = 1.0 - (raw_val - min_ideal) / (max_ideal - min_ideal)
        elif label == "LIPO (LogP)" or label == "SIZE (MW)":
             normalized_val = np.clip((raw_val - min_ideal) / (max_ideal - min_ideal), 0, 1)
        else:
            normalized_val = np.clip((raw_val - min_ideal) / (max_ideal - min_ideal), 0, 1)
        stats_normalized.append(normalized_val)

    angles = np.linspace(0, 2 * pi, len(labels_raw), endpoint=False).tolist()
    stats_plot = stats_normalized + stats_normalized[:1]; angles_plot = angles + angles[:1]

    plt.style.use('seaborn-v0_8-pastel')
    fig, ax = plt.subplots(figsize=(2.2, 2.2), subplot_kw=dict(polar=True))
    ax.plot(angles_plot, stats_plot, color='#1E88E5', linewidth=1, linestyle='solid', marker='o', markersize=2)
    ax.fill(angles_plot, stats_plot, '#90CAF9', alpha=0.4)
    ax.set_xticks(angles); ax.set_xticklabels(labels_raw, fontsize=3.5, weight='bold')
    ax.set_yticks(np.arange(0, 1.1, 0.25)); ax.set_yticklabels([f"{t:.1f}" for t in np.arange(0, 1.1, 0.25)], fontsize=3.5)
    ax.set_ylim(0, 1); ax.grid(color='grey', linestyle=':', linewidth=0.25)
    ax.set_title(f"Bioavailability: {mol_name}", size=5.5, weight='bold', y=1.28)
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches='tight', dpi=75); plt.close(fig)
    return buf.getvalue()

def plot_bioavailability_radar(phys_props, mol_name="Molecule"):
    if not phys_props or not RDKIT_AVAILABLE: return None
    log_p_val = phys_props.get("LogP (Octanol-Water Partition Coefficient)", phys_props.get("PubChem XLogP3", 2.5))
    log_s_approx = 0.8 - (0.7 * log_p_val)
    radar_data_for_core = {
        "LIPO (LogP)": log_p_val,
        "SIZE (MW)": phys_props.get("Molecular Weight (MW)", phys_props.get("PubChem MolecularWeight", 0)),
        "POLAR (TPSA)": phys_props.get("Topological Polar Surface Area (TPSA)", 0),
        "INSOLU (LogS)": log_s_approx,
        "INSATU (FracCsp3)": phys_props.get("Fraction C sp3 Atoms (FracCsp3)", 0),
        "FLEX (RotB)": phys_props.get("Rotatable Bonds", phys_props.get("PubChem RotatableBondCount",0)),
    }
    return plot_bioavailability_radar_core(radar_data_for_core, mol_name)

def plot_boiled_egg(phys_props_list):
    if not RDKIT_AVAILABLE or not phys_props_list or not isinstance(phys_props_list, list): return None
    wlogp_values = []; tpsa_values = []
    for props in phys_props_list:
        logp = props.get("LogP (Octanol-Water Partition Coefficient)", props.get("PubChem XLogP3", np.nan))
        tpsa = props.get("Topological Polar Surface Area (TPSA)", np.nan)
        wlogp_values.append(logp)
        tpsa_values.append(tpsa)
    valid_indices = [i for i, (w, t) in enumerate(zip(wlogp_values, tpsa_values)) if pd.notna(w) and pd.notna(t)]
    if not valid_indices: return None
    wlogp = [wlogp_values[i] for i in valid_indices]; tpsa = [tpsa_values[i] for i in valid_indices]
    if not wlogp or not tpsa: return None

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(3.8, 3.0), subplot_kw=None)
    yolk_ellipse = patches.Ellipse(xy=(2.9, 67.5), width=6.0, height=135, angle=0, fill=True, color='#FFECB3', alpha=0.6, zorder=0)
    white_ellipse = patches.Ellipse(xy=(2.9, 75), width=8.6, height=155, angle=0, fill=True, color='#E3F2FD', alpha=0.4, zorder=0)
    ax.add_patch(white_ellipse); ax.add_patch(yolk_ellipse)
    ax.scatter(wlogp, tpsa, c='#D32F2F', s=18, alpha=0.8, zorder=1, label='Cmpd(s)')
    ax.set_xlabel("WLOGP (LogP)", fontsize=6, weight='bold'); ax.set_ylabel("TPSA (Å²)", fontsize=6, weight='bold')
    ax.set_title("BOILED-Egg Plot", fontsize=7, weight='bold')
    ax.set_xlim(-4, 8); ax.set_ylim(0, 200)
    ax.tick_params(axis='both', which='major', labelsize=5)
    ax.text(2.9, 60, 'BBB+', fontsize=5, ha='center', va='center', weight='bold', color='#AD8C00')
    ax.text(2.9, 145, 'GI Abs.', fontsize=5, ha='center', va='center', weight='bold', color='#1E88E5')
    ax.grid(True, linestyle=':', linewidth=0.3); ax.legend(fontsize=4.5, loc='upper right')
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches='tight', dpi=75); plt.close(fig)
    return buf.getvalue()

if SEABORN_AVAILABLE:
    def plot_property_distributions(properties_list, mol_name="Dataset"):
        if not properties_list: return None
        df = pd.DataFrame(properties_list)

        props_to_plot = {
            "Molecular Weight (MW)": "MW",
            "LogP (Octanol-Water Partition Coefficient)": "LogP",
            "Topological Polar Surface Area (TPSA)": "TPSA",
            "Hydrogen Bond Donors (HBD)": "HBD",
            "Hydrogen Bond Acceptors (HBA)": "HBA",
            "Rotatable Bonds": "RotB"
        }

        plot_df_cols = [col for col in props_to_plot.keys() if col in df.columns]
        if not plot_df_cols: return None

        plot_df = df[plot_df_cols].copy()

        num_plots = len(plot_df.columns)
        if num_plots == 0: return None

        cols = 2
        rows = (num_plots + cols - 1) // cols

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(rows, cols, figsize=(6, 2.6 * rows + 0.4))
        if num_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()

        for i, col_name in enumerate(plot_df.columns):
            data_series = pd.to_numeric(plot_df[col_name], errors='coerce').dropna()
            ax_current = axes[i]

            if data_series.empty:
                ax_current.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax_current.transAxes)
                ax_current.set_title(props_to_plot.get(col_name, col_name), fontsize=6)
                ax_current.set_xticks([])
                ax_current.set_yticks([])
                continue

            sns.histplot(data_series, kde=True, ax=ax_current, color="#00796B", bins=10 if len(data_series) > 1 else 1, stat="density")
            ax_current.set_title(props_to_plot.get(col_name, col_name), fontsize=6)
            ax_current.set_xlabel("")
            ax_current.tick_params(axis='x', labelsize=4.5)
            ax_current.tick_params(axis='y', labelsize=4.5)
            if len(data_series) == 1:
                 ax_current.axvline(data_series.iloc[0], color='red', linestyle='--', lw=1)

            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            ax_box = inset_axes(ax_current, width="100%", height="25%", loc='lower center',
                                bbox_to_anchor=(0, -0.35, 1, 1),
                                bbox_transform=ax_current.transAxes, borderpad=0)
            sns.boxplot(x=data_series, ax=ax_box, color="#4DB6AC", fliersize=1.2, linewidth=0.5, width=0.35)
            ax_box.set_yticks([])
            ax_box.set_xlabel("")
            ax_box.tick_params(axis='x', labelsize=4.5)


        for j in range(num_plots, len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f"Property Distributions: {mol_name}", fontsize=8, weight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=75)
        plt.close(fig)
        return buf.getvalue()


    def plot_correlation_heatmap(properties_list, mol_name="Dataset"):
        if not properties_list: return None
        df = pd.DataFrame(properties_list)

        props_to_plot_keys = [
            "Molecular Weight (MW)", "LogP (Octanol-Water Partition Coefficient)",
            "Topological Polar Surface Area (TPSA)", "Hydrogen Bond Donors (HBD)",
            "Hydrogen Bond Acceptors (HBA)", "Rotatable Bonds"
        ]

        plot_df_cols = [col for col in props_to_plot_keys if col in df.columns and pd.to_numeric(df[col], errors='coerce').notna().any()]
        if len(plot_df_cols) < 2 :
            return None

        plot_df = df[plot_df_cols].apply(pd.to_numeric, errors='coerce').copy()

        short_names_map = {
            "Molecular Weight (MW)": "MW",
            "LogP (Octanol-Water Partition Coefficient)": "LogP",
            "Topological Polar Surface Area (TPSA)": "TPSA",
            "Hydrogen Bond Donors (HBD)": "HBD",
            "Hydrogen Bond Acceptors (HBA)": "HBA",
            "Rotatable Bonds": "RotB"
        }
        plot_df.rename(columns=short_names_map, inplace=True)

        if len(plot_df.dropna(how='all', subset=plot_df.columns)) < 2:
            return None

        corr_matrix = plot_df.corr()

        if corr_matrix.isnull().all().all() or corr_matrix.shape[0] < 2:
             return None

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(4.2, 3.2)) # Reduced figsize
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.4, ax=ax, annot_kws={"size": 5}, vmin=-1, vmax=1)
        ax.set_title(f"Property Correlation: {mol_name}", fontsize=7, weight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=5)
        plt.yticks(rotation=0, fontsize=5)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=75)
        plt.close(fig)
        return buf.getvalue()

# --- PUBCHEM HELPER FUNCTIONS ---
PUG_PROPS = "IUPACName,CanonicalSMILES,MolecularFormula,MolecularWeight,ExactMass,Charge,Complexity,XLogP3,HBondDonorCount,HBondAcceptorCount,RotatableBondCount,HeavyAtomCount"

def pubchem_get_cid_from_name_or_smiles(identifier, id_type="name"):
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound"
    url = f"{base_url}/{id_type}/{identifier}/cids/JSON"
    try:
        if id_type == "smiles":
            response = requests.post(url, data={'smiles': identifier}, timeout=10)
        else:
            response = requests.get(url, timeout=10)
        response.raise_for_status()
        cids = response.json().get('IdentifierList', {}).get('CID', [])
        return cids[0] if cids else None
    except requests.exceptions.RequestException as e: print(f"PubChem API error ({id_type}->CID for '{identifier}'): {e}"); return None
    except ValueError: print(f"PubChem API JSON decoding error ({id_type}->CID for '{identifier}')."); return None


def pubchem_get_compound_info(cid, props=PUG_PROPS): # Allow specifying properties
    if not cid: return {}
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/{props}/JSON"
    try:
        response = requests.get(url, timeout=10); response.raise_for_status()
        data = response.json()
        if 'PropertyTable' in data and 'Properties' in data['PropertyTable'] and data['PropertyTable']['Properties']:
            return data['PropertyTable']['Properties'][0]
        return {}
    except requests.exceptions.RequestException as e: print(f"PubChem API error (info for CID {cid}): {e}"); return {}
    except ValueError: print(f"PubChem API JSON decoding error (info for CID {cid})."); return {}

def pubchem_get_compound_png_url(cid, width=150, height=150):
    return f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG?width={width}&height={height}"


def pubchem_get_similar_cids(cid, threshold=90, max_results=5):
    if not cid: return []
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/similarity/cid/{cid}/cids/JSON?Threshold={threshold}&MaxRecords={max_results}"
    try:
        response = requests.get(url, timeout=15); response.raise_for_status()
        return response.json().get('IdentifierList', {}).get('CID', [])
    except requests.exceptions.RequestException as e: print(f"PubChem API error (similarity for CID {cid}): {e}"); return []
    except ValueError: print(f"PubChem API JSON decoding error (similarity for CID {cid})."); return []

def clean_molecule_name_for_pubchem(name):
    if not name: return ""
    name = re.sub(r'sdf\d*[_ ]*', '', name, flags=re.IGNORECASE).strip()
    return name.replace("_", " ").strip()

# --- STYLING ---
st.markdown("""<style>
body, .stApp {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    color: #263238; /* Darker grey for text */
    background: linear-gradient(135deg, #e0f2f1 0%, #b2dfdb 50%, #80cbc4 100%); /* Teal gradient background */
}
.main-title-container {
    padding: 1.5rem 0; /* Vertical padding for title */
    margin-top: 10px; /* Shift down slightly RELATIVE TO TOP OF ITS COLUMN */
    margin-bottom: 1.5rem;
    border-radius: 12px;
    background: linear-gradient(90deg, #00695C, #00897B); /* Deeper teal for title */
    box-shadow: 0 6px 20px rgba(0, 77, 64, 0.3);
    text-align: center;
    display: flex;
    align-items: center; /* Vertically center .main-title text within this container */
    justify-content: center; /* Horizontally center .main-title text */
    /* height: 120px; -- Removed, let padding and content define height */
}
.main-title {
    font-size: 3.8rem; /* Larger title */
    font-weight: 700;
    color: white;
    letter-spacing: 2.5px;
    text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.4);
    margin: 0 auto; /* Center title text within its column */
}
/* .gif-image class is not directly used for the header GIFs, they are styled inline */
.stButton>button {
    border-radius: 25px !important; /* More rounded buttons */
    border: 1px solid #00796B !important;
    background: linear-gradient(145deg, #ffffff, #e0f7fa) !important; /* Lighter background for buttons */
    color: #004D40 !important; /* Darker teal text */
    padding: 0.6rem 1.8rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 3px 3px 8px rgba(0, 0, 0, 0.1) !important;
    width: 100% !important; margin-bottom: 0.6rem;
}
.stButton>button:hover {
    background: linear-gradient(145deg, #00695C, #00897B) !important; /* Darker teal gradient on hover */
    color: white !important;
    border-color: #004D40 !important;
    box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.25) !important;
    transform: translateY(-2px) !important;
}
.stDownloadButton>button {
    border-radius: 25px !important;
    border: 1px solid #FB8C00 !important;
    background: linear-gradient(145deg, #fff3e0, #ffe0b2) !important;
    color: #E65100 !important;
    padding: 0.6rem 1.8rem !important;
    font-weight: 600 !important;
    box-shadow: 3px 3px 8px rgba(0,0,0,0.1) !important;
}
.stDownloadButton>button:hover {
    background: linear-gradient(145deg, #FFA726, #FB8C00) !important;
    color: white !important;
    border-color: #E65100 !important;
    box-shadow: 4px 4px 10px rgba(0,0,0,0.25) !important;
}
.page-title { font-size: 2.8rem; color: #004D40; border-bottom: 4px solid #00897B; padding-bottom: 0.8rem; margin-top: 2rem; margin-bottom: 2.5rem; font-weight: 600; }
.page-subtitle { font-size: 1.6rem; color: #006064; margin-bottom: 1.4rem; font-weight: 600; border-left: 5px solid #00897B; padding-left: 1rem; }
.content-box, .feature-box, .step-box, .purpose-box, .benefit-box, .acknowledgement-box, .future-box {
    background: #f5f5f5; /* Lighter grey for content boxes */
    border-radius: 10px;
    padding: 1.8rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    border-left: 6px solid;
}
.feature-box { border-left-color: #26A69A; background-color: #E0F2F1; }
.step-box { border-left-color: #66BB6A; background-color: #E8F5E9; }
.purpose-box { border-left-color: #FFCA28; background-color: #FFFDE7; }
.benefit-box { border-left-color: #42A5F5; background-color: #E3F2FD; }
.acknowledgement-box { border-left-color: #AB47BC; background-color: #F3E5F5; }
.content-box { border-left-color: #00897B; background-color: #E0F2F1; }
.future-box { border-left-color: #EC407A; background-color: #FCE4EC; }
.circle-img-container { display: flex; justify-content: center; margin-bottom: 1rem; }
.circle-img { border-radius: 50%; width: 190px; height: 190px; object-fit: cover; border: 7px solid #00796B; box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2); }
.author-name { text-align:center; font-size:1.6rem; font-weight:600; color: #004D40; margin-top: 0.5rem; }
.author-title { text-align:center; font-size:1.2rem; color: #546E7A; margin-bottom: 1rem; font-style: italic;}
.linkedin-btn { display: inline-block; padding: 0.7em 1.5em; border: none; border-radius: 25px; background: linear-gradient(145deg, #0077b5, #005582); color: white !important; font-weight: bold; text-decoration: none; box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2); transition: all 0.3s ease; text-align: center; }
.linkedin-btn:hover { background: linear-gradient(145deg, #005582, #003350); box-shadow: 4px 4px 12px rgba(0, 0, 0, 0.3); transform: translateY(-2px); color: white !important; text-decoration: none; }
.footer-text { font-size: 0.9rem; color: #455A64; text-align: center; margin-top: 3rem; padding-bottom: 1rem; font-style: italic; }
iframe[title="streamlit_ketcher.st_ketcher"] { width: 100% !important; min-height: 400px !important; border: 1px solid #80CBC4 !important; border-radius: 10px !important; }

/* Custom table styling */
.styled-table { width: 100%; border-collapse: collapse; margin: 1em 0; font-size: 0.9em; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.07); }
.styled-table thead tr { background-color: #00695C; color: #ffffff; text-align: left; font-weight: bold; }
.styled-table th, .styled-table td { padding: 12px 15px; }
.styled-table tbody tr { border-bottom: 1px solid #CFD8DC; }
.styled-table tbody tr:nth-of-type(even) { background-color: #ECEFF1; }
.styled-table tbody tr:last-of-type { border-bottom: 3px solid #00695C; }
.styled-table tbody tr:hover { background-color: #B2DFDB; }
.styled-table td:first-child { font-weight: bold; color: #004D40; }
.pass-eval { color: #4CAF50; font-weight: bold; }
.fail-eval { color: #F44336; font-weight: bold; }
/* Expander styling in Main Page (not sidebar) */
.main .stExpander {
    border: 1px solid #80CBC4 !important;
    border-radius: 10px !important;
    box-shadow: 0 3px 8px rgba(0,0,0,0.08) !important;
    margin-bottom: 1.2rem !important;
}
.main .stExpander header {
    background-color: #B2DFDB !important; /* Lighter teal for expander header */
    border-radius: 9px 9px 0 0 !important;
    font-weight: 600 !important;
    color: #004D40 !important; /* Darker teal text */
    padding: 0.7rem 1.2rem !important;
}
.main .stExpander header:hover {
    background-color: #80CBC4 !important;
}
.main .stExpander p {
    font-size: 0.95rem !important;
}
.main .stImage img {
    max-width: 100% !important;
    height: auto !important;
    margin: 0 auto !important;
    display: block !important;
}
.main .stImage img[width="280"] { /* Specific rule for the 2D molecule image if width is set to 280 */
    max-width: 280px !important;
}
/* Tab Label Styling */
div[data-baseweb="tab-list"] button > div > p {
    font-size: 1.1em !important;
    font-weight: 700 !important;
}
</style>""", unsafe_allow_html=True)

# --- HEADER & NAVIGATION BAR ---
gif_url = "https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExcmg4Nzl0Z3Q4eW9idmQ1dWVvbm0xZ3VxaHpmaXJkYWV1eDFja3QxNyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/5Xfn2En9gZybm/giphy.gif"

# Adjusted column ratios: GIFs get more space
title_cols = st.columns([1.5, 2.5, 1.5]) 

with title_cols[0]:
    st.markdown(f"""
        <div style="display: flex; justify-content: center; align-items: center; height: 100%; margin-top: 10px;">
            <img src="{gif_url}" style="height: 110px; width: auto; max-width: 100%;">
        </div>
    """, unsafe_allow_html=True)

with title_cols[1]:
    st.markdown(
        '<div class="main-title-container">'
        '  <div class="main-title">🧬 ADMETriX 🧪</div>'
        '</div>',
        unsafe_allow_html=True
    )

with title_cols[2]:
    st.markdown(f"""
        <div style="display: flex; justify-content: center; align-items: center; height: 100%; margin-top: 10px;">
            <img src="{gif_url}" style="height: 110px; width: auto; max-width: 100%;">
        </div>
    """, unsafe_allow_html=True)


nav_cols = st.columns(4)
with nav_cols[0]:
    if st.button("🏠 Home", key="nav_Home_main", use_container_width=True): update_nav("Home")
with nav_cols[1]:
    if st.button("📖 User Guide", key="nav_User_Guide_main", use_container_width=True): update_nav("User Guide")
with nav_cols[2]:
    if st.button("🔬 Analysis", key="nav_Analysis_main", use_container_width=True): update_nav("Analysis")
with nav_cols[3]:
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
    <li><b>✏️ Molecule Sketcher</b>: Draw molecules directly in the browser, get SMILES, and analyze.</li>
    <li><b>📝 SMILES/SDF Input</b>: Analyze molecules by providing SMILES strings (typed or from file) or uploading SDF/MOL files.</li>
    <li><b>🔗 PubChem Compound Search</b>: Find compounds by name, view similar structures, and use their SMILES for analysis, enriched with PubChem data.</li>
    <li><b>📊 2D & 3D Visualization</b>: View interactive 2D and 3D structures of your molecules.</li>
    <li><b>🧪 Physicochemical Properties</b>: Calculate key properties (RDKit) and fetch additional data from PubChem.</li>
    <li><b>💊 Drug-likeness Rules Evaluation</b>: Assess compounds against common rules (Lipinski, Ghose, etc.) with clear indicators.</li>
    <li><b>📈 ADMET Plots</b>: Visualize Bioavailability Radar, BOILED-Egg, Property Distributions (Histograms & Boxplots), and Correlation Heatmap with interpretation guides and download options.</li>
    <li><b>📁 Data Download</b>: Download 2D images, plots, and 3D coordinate (XYZ) files.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

elif nav_selected == "User Guide":
    st.markdown("<div class='page-title'>📖 User Guide</div>", unsafe_allow_html=True)
    st.markdown("""
    This guide will walk you through using ADMETriX for your compound analysis needs.

    <div class="step-box">
    <h5>Step 1: Navigate to the Analysis Page</h5>
    From the main navigation bar at the top, click on the <b>🔬 Analysis</b> button. This will take you to the main analysis interface.
    </div>

    <div class="step-box">
    <h5>Step 2: Input Your Molecule</h5>
    The Analysis page provides several tabs for molecule input. Choose **one** method:

    <ol>
        <li><b>✏️ Draw Molecule Tab:</b>
            <ul>
                <li><b>Draw:</b> Use the provided Ketcher molecular sketcher to draw your compound.</li>
                <li><b>Get SMILES:</b> Click the "Get SMILES from Current Sketch" button. This converts your drawing to a SMILES string and displays it. Ensure it's correct.</li>
                <li><b>Name (Optional):</b> Enter a name for your molecule in the text field.</li>
                <li><b>Analyze:</b> Click "Analyze Sketched Molecule" to start the analysis.</li>
            </ul>
        </li>
        <br>
        <li><b>📝 Input SMILES Tab:</b>
            <ul>
                <li><b>Type/Paste:</b> Directly enter or paste a valid SMILES string into the "Enter SMILES string" text box.</li>
                <li><b>Upload File (Optional):</b> Click "Browse files" to upload a <code>.smi</code> or <code>.txt</code> file. The first SMILES string from the first line of this file will be loaded into the text box.</li>
                <li><b>Name (Optional):</b> Enter a name for your molecule.</li>
                <li><b>Analyze:</b> Click "Analyze SMILES". Examples of SMILES strings are provided for reference.</li>
            </ul>
        </li>
        <br>
        <li><b>📁 Upload SDF/MOL Tab:</b>
            <ul>
                <li><b>Upload:</b> Click "Browse files" to upload a single-molecule <code>.sdf</code> or <code>.mol</code> file from your computer.</li>
                <li><b>Name (Optional):</b> The name might be auto-filled from the file or filename. You can edit it.</li>
                <li><b>Analyze:</b> Click "Analyze File".</li>
            </ul>
        </li>
        <br>
        <li><b>🔎 PubChem Search Tab:</b>
            <ul>
                <li><b>Search:</b> Enter a compound name (e.g., "Aspirin", "Caffeine") into the search box.</li>
                <li><b>Click Search:</b> Press the "Search PubChem 🔎" button.</li>
                <li><b>Review Results:</b> Information for the primary match and a table of similar compounds (with 2D structures) will be displayed.</li>
                <li><b>Analyze Primary Match:</b> To analyze the main compound found, click the "Analyze This Compound (Primary Match)" button.</li>
            </ul>
        </li>
    </ol>
    </div>

    <div class="step-box">
    <h5>Step 3: Explore Analysis Results</h5>
    Once you trigger an analysis (and it completes successfully), the results will appear in <b>expandable sections directly on the analysis page</b>, below the input tabs.
    <ul>
        <li>Results are organized into these sections. Click on a section header (e.g., "2D Molecular Structure", "Physicochemical Properties") to open it and view the details.</li>
        <li><b>Available Results:</b>
            <ul>
                <li><b>2D Molecular Structure:</b> A static 2D image of the molecule.</li>
                <li><b>3D Molecular Structure:</b> An interactive 3D view (if coordinates could be generated).</li>
                <li><b>Bioavailability Radar Plot:</b> A radar chart summarizing key ADMET properties.</li>
                <li><b>BOILED-Egg Plot:</b> Predicts GI absorption and BBB permeability.</li>
                <li><b>Physicochemical Properties:</b> A table of calculated and (if available) PubChem-sourced properties.</li>
                <li><b>Drug-Likeness Evaluation:</b> Assessment against common drug-likeness rules (e.g., Lipinski's Rule of Five).</li>
                <li><b>Molecular Property Distributions:</b> Histograms and boxplots showing the distribution of key properties.</li>
                <li><b>Property Relationships (Correlation Heatmap):</b> Shows correlations between properties.</li>
            </ul>
        </li>
        <li><b>Interpretation:</b> For plots, click the "Interpretation Guide" button (if available) or refer to standard interpretations.</li>
        <li><b>Downloads:</b> Most visual results (images, plots) and 3D coordinates have download buttons.</li>
    </ul>
    </div>

    <div class="step-box">
    <h5>Step 4: Clear Analysis (Optional)</h5>
    If you wish to clear the current analysis results from the display to start fresh (without navigating away), click the "Clear Analysis Results" button at the bottom of the Analysis page. This will also reset the input fields.
    </div>

    <div class="step-box">
    <h5>Important Notes:</h5>
    <ul>
        <li><b>Required Libraries:</b> This application relies on RDKit for its core chemical informatics functionalities.</li>
        <li><b>Optional Libraries:</b>
            <ul>
                <li><code>stmol</code> & <code>py3Dmol</code> are needed for 3D molecular visualization.</li>
                <li><code>streamlit-ketcher</code> is required for the molecule drawing feature.</li>
                <li><code>seaborn</code> is needed for generating statistical plots like property distributions and correlation heatmaps.</li>
            </ul>
            If any of these are missing, corresponding features might be unavailable or limited. The application will usually display an informational message if a library is not found.
        </li>
        <li><b>Single Molecule Analysis:</b> Currently, ADMETriX is optimized for analyzing one molecule at a time. Some statistical plots (like correlation heatmaps) are more meaningful with multiple data points.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

elif nav_selected == "Analysis":
    st.markdown("<div class='page-title'>🔬 ADMETriX Analysis</div>", unsafe_allow_html=True)

    if not RDKIT_AVAILABLE:
        st.error("CRITICAL: RDKit is not available. Core analysis functionalities are disabled. Please install RDKit and restart the application.")
    else:
        tab_titles = ["✏️ Draw Molecule", "📝 Input SMILES", "📁 Upload SDF/MOL", "🔎 PubChem Search"]
        input_tabs = st.tabs(tab_titles)

        # Tab 0: Draw Molecule (Ketcher)
        with input_tabs[0]:
            if STREAMLIT_KETCHER_AVAILABLE:
                st.markdown("##### Step 1: Draw Molecule")
                current_smiles_on_canvas = st_ketcher(
                    value=st.session_state.ketcher_display_smiles,
                    key="molecule_sketcher_main_component"
                )

                st.markdown("##### Step 2: Get SMILES from Sketch")
                if st.button("Get SMILES from Current Sketch", key="get_smiles_from_sketch_button", use_container_width=False):
                    if current_smiles_on_canvas and current_smiles_on_canvas.strip():
                        st.session_state.smiles_from_sketch_for_analysis = current_smiles_on_canvas
                        st.session_state.ketcher_display_smiles = current_smiles_on_canvas
                        st.success(f"SMILES captured from sketch: `{st.session_state.smiles_from_sketch_for_analysis}`")
                    else:
                        st.session_state.smiles_from_sketch_for_analysis = ""
                        st.warning("Sketcher is empty or the drawn molecule is invalid.")

                if st.session_state.smiles_from_sketch_for_analysis:
                    st.info(f"**SMILES ready for analysis:** `{st.session_state.smiles_from_sketch_for_analysis}`")

                st.markdown("##### Step 3: Provide Name & Analyze")
                sketcher_mol_name_input = st.text_input(
                    "Molecule Name (Optional):",
                    value=st.session_state.current_mol_name_sketcher_attempt,
                    key="mol_name_sketcher_input_field"
                )
                if sketcher_mol_name_input != st.session_state.current_mol_name_sketcher_attempt:
                     st.session_state.current_mol_name_sketcher_attempt = sketcher_mol_name_input


                if st.button("Analyze Sketched Molecule", key="analyze_button_for_sketcher_mol", use_container_width=False):
                    smiles_to_use = st.session_state.smiles_from_sketch_for_analysis
                    if smiles_to_use and smiles_to_use.strip():
                        st.session_state.current_smiles = smiles_to_use
                        st.session_state.current_mol_name = st.session_state.current_mol_name_sketcher_attempt.strip() \
                                                            if st.session_state.current_mol_name_sketcher_attempt.strip() \
                                                            else "Molecule from Sketcher"
                        st.session_state.analysis_triggered = True
                        st.session_state.molecule_data_store = None
                        st.session_state.extended_pubchem_props_for_current_mol = None
                        st.session_state.analysis_triggered_by_sketcher = True
                        st.session_state.analysis_triggered_by_pubchem_or_sdf = False
                        st.session_state.ketcher_display_smiles = ""
                        st.session_state.smiles_from_sketch_for_analysis = ""
                        st.session_state.current_mol_name_sketcher_attempt = "Molecule from Sketcher"
                        st.rerun()
                    else:
                        st.warning("No SMILES captured from sketch. Please draw/confirm using 'Get SMILES' first.")
                        st.session_state.analysis_triggered = False
            else:
                st.warning("Molecule Sketcher (streamlit-ketcher) is not available.")

        # Tab 1: Input SMILES
        with input_tabs[1]:
            st.markdown("##### Input SMILES String for Analysis")

            uploaded_smiles_file = st.file_uploader(
                "Or, upload a SMILES file (.smi, .txt - first line will be used)",
                type=["smi", "txt"],
                key="smiles_file_uploader"
            )
            if uploaded_smiles_file is not None:
                try:
                    smiles_content = uploaded_smiles_file.read().decode("utf-8").splitlines()
                    if smiles_content:
                        first_smiles = smiles_content[0].strip()
                        if first_smiles:
                            st.session_state.smiles_from_file_buffer = first_smiles
                            st.info(f"SMILES from file loaded: `{first_smiles}`. It has been placed in the text box below. Modify if needed and click 'Analyze SMILES'.")
                        else:
                            st.warning("Uploaded file seems empty or first line is blank.")
                            st.session_state.smiles_from_file_buffer = ""
                    else:
                        st.warning("Uploaded file is empty.")
                        st.session_state.smiles_from_file_buffer = ""
                except Exception as e:
                    st.error(f"Error reading SMILES file: {e}")
                    st.session_state.smiles_from_file_buffer = ""

            st.markdown("""**Examples:**
            *   Aspirin: `CC(=O)Oc1ccccc1C(=O)OH`
            *   Caffeine: `CN1C=NC2=C1C(=O)N(C(=O)N2C)C`
            *   Ibuprofen: `CC(C)CC1=CC=C(C=C1)C(C)C(=O)O`""")

            _initial_smiles = st.session_state.smiles_from_file_buffer if st.session_state.smiles_from_file_buffer else st.session_state.current_smiles
            _initial_name = st.session_state.current_mol_name

            if st.session_state.analysis_triggered_by_pubchem_or_sdf or st.session_state.analysis_triggered_by_sketcher:
                 if not st.session_state.smiles_from_file_buffer:
                    _initial_smiles = st.session_state.current_smiles
                    _initial_name = st.session_state.current_mol_name

            user_smiles = st.text_input("Enter SMILES string (or use uploaded file):", value=_initial_smiles, key="smiles_input_val_tab_field", placeholder="e.g., CC(=O)Oc1ccccc1C(=O)OH")
            user_mol_name = st.text_input("Molecule Name (Optional):", value=_initial_name, key="mol_name_smiles_val_tab_field")

            if st.button("Analyze SMILES", key="analyze_smiles_btn_main_tab", use_container_width=False):
                if user_smiles:
                    st.session_state.current_smiles = user_smiles
                    st.session_state.current_mol_name = user_mol_name.strip() if user_mol_name.strip() else "Molecule from SMILES"
                    st.session_state.analysis_triggered = True
                    st.session_state.molecule_data_store = None
                    st.session_state.extended_pubchem_props_for_current_mol = None
                    st.session_state.analysis_triggered_by_sketcher = False
                    st.session_state.analysis_triggered_by_pubchem_or_sdf = False
                    st.session_state.smiles_from_file_buffer = ""
                    st.rerun()
                else:
                    st.warning("Please enter a SMILES string."); st.session_state.analysis_triggered = False

        # Tab 2: Upload SDF/MOL
        with input_tabs[2]:
            st.markdown("##### Upload SDF/MOL File for Analysis")
            sdf_file = st.file_uploader("Upload SDF or MOL file (single molecule)", type=["sdf", "mol"], key="sdf_uploader_main_tab")

            default_sdf_name = st.session_state.current_mol_name_sdf_upload_attempt
            if sdf_file and default_sdf_name == "Molecule from SDF":
                 default_sdf_name = sdf_file.name.split('.')[0] if sdf_file.name else "Molecule from SDF"

            sdf_user_mol_name = st.text_input("Molecule Name (Optional, from file):", value=default_sdf_name, key="mol_name_sdf_main_tab_field")
            if sdf_user_mol_name != st.session_state.current_mol_name_sdf_upload_attempt:
                 st.session_state.current_mol_name_sdf_upload_attempt = sdf_user_mol_name

            if sdf_file is not None:
                if st.button("Analyze File", key="analyze_sdf_btn_main_tab", use_container_width=False):
                    try:
                        sdf_block = sdf_file.read().decode("utf-8")
                        mol_from_file = get_mol_from_sdf_block(sdf_block)
                        if mol_from_file:
                            name_to_set = st.session_state.current_mol_name_sdf_upload_attempt
                            if mol_from_file.HasProp('_Name') and mol_from_file.GetProp('_Name').strip():
                                name_from_file_prop = mol_from_file.GetProp('_Name').strip()
                                if not name_to_set.strip() or name_to_set == "Molecule from SDF" or \
                                   name_to_set == (sdf_file.name.split('.')[0] if sdf_file.name else "Molecule from SDF"):
                                    name_to_set = name_from_file_prop
                            if not name_to_set.strip() or name_to_set == "Molecule from SDF":
                                 name_to_set = sdf_file.name.split('.')[0] if sdf_file.name else "Molecule from File"

                            st.session_state.current_mol_name = name_to_set
                            smiles_from_file = Chem.MolToSmiles(mol_from_file, isomericSmiles=True)
                            if smiles_from_file:
                                st.session_state.current_smiles = smiles_from_file
                                st.session_state.analysis_triggered = True
                                st.session_state.molecule_data_store = None
                                st.session_state.extended_pubchem_props_for_current_mol = None
                                st.session_state.analysis_triggered_by_pubchem_or_sdf = True
                                st.session_state.analysis_triggered_by_sketcher = False
                                st.rerun()
                            else: st.error("Could not convert file to SMILES."); st.session_state.analysis_triggered = False
                        else: st.error("Could not parse molecule. Ensure valid single-molecule SDF/MOL."); st.session_state.analysis_triggered = False
                    except Exception as e: st.error(f"Error processing file: {e}"); st.session_state.analysis_triggered = False
            else:
                 if not st.session_state.get("analysis_triggered"):
                    st.session_state.current_mol_name_sdf_upload_attempt = "Molecule from SDF"

        # Tab 3: PubChem Search
        with input_tabs[3]:
            st.markdown("##### Find Compounds via PubChem")
            user_search_name = st.text_input("Compound name for PubChem search:", value=st.session_state.current_pubchem_search_name, key="pubchem_search_text_input_tab_field", help="e.g., Aspirin")
            if user_search_name != st.session_state.current_pubchem_search_name:
                st.session_state.current_pubchem_search_name = user_search_name

            if st.button("Search PubChem 🔎", key="pubchem_search_btn_action_tab", use_container_width=False):
                st.session_state.current_pubchem_search_results = None
                st.session_state.current_pubchem_main_compound_info = None
                if user_search_name:
                    cleaned_name = clean_molecule_name_for_pubchem(user_search_name)
                    if not cleaned_name: st.warning("Search input empty after cleaning.")
                    else:
                        with st.spinner(f"Searching PubChem for '{cleaned_name}'..."):
                            cid = pubchem_get_cid_from_name_or_smiles(cleaned_name)
                            if cid:
                                main_info = pubchem_get_compound_info(cid)
                                if main_info and main_info.get("CanonicalSMILES"):
                                    st.session_state.current_pubchem_main_compound_info = main_info

                                    similar_cids_list = pubchem_get_similar_cids(cid, threshold=90, max_results=5)
                                    similar_compounds_data = []
                                    for scid in similar_cids_list:
                                        if scid != cid:
                                            s_info = pubchem_get_compound_info(scid, props="IUPACName,CanonicalSMILES") # Fetch only needed info
                                            if s_info and s_info.get("CanonicalSMILES"):
                                                similar_compounds_data.append({
                                                    "cid": scid,
                                                    "name": s_info.get("IUPACName", f"CID: {scid}"),
                                                    "smiles": s_info.get("CanonicalSMILES"),
                                                    "image_url": pubchem_get_compound_png_url(scid, width=100, height=100)
                                                })
                                    st.session_state.current_pubchem_search_results = similar_compounds_data
                                else:
                                    st.error(f"Found CID {cid} for '{cleaned_name}', but could not retrieve its detailed information from PubChem.")
                                    st.session_state.current_pubchem_main_compound_info = None
                            else:
                                st.error(f"Compound '{cleaned_name}' not found in PubChem or API error.")
                else: st.warning("Please enter a compound name for PubChem search.")

            # Display Primary Match Info
            if st.session_state.current_pubchem_main_compound_info:
                main_info = st.session_state.current_pubchem_main_compound_info
                main_name_disp = main_info.get('IUPACName', st.session_state.current_pubchem_search_name)
                st.markdown(f"--- \n**Primary Match: {main_name_disp}** (CID: {main_info.get('CID', 'N/A')})")
                st.markdown(f"  Formula: `{main_info.get('MolecularFormula', 'N/A')}`, MW: `{main_info.get('MolecularWeight', 'N/A')}`")
                main_smiles = main_info.get('CanonicalSMILES', 'N/A')
                st.markdown(f"  SMILES: `{main_smiles}`")
                if main_smiles != 'N/A':
                    if st.button(f"Analyze This Compound (Primary Match)", key=f"analyze_primary_pubchem_{main_info.get('CID')}", use_container_width=False):
                        st.session_state.current_smiles = main_smiles
                        st.session_state.current_mol_name = main_name_disp
                        st.session_state.analysis_triggered = True
                        st.session_state.molecule_data_store = None
                        st.session_state.extended_pubchem_props_for_current_mol = main_info # Store all fetched props
                        st.session_state.analysis_triggered_by_pubchem_or_sdf = True
                        st.session_state.analysis_triggered_by_sketcher = False
                        st.rerun()
                st.markdown("---")


            # Display Similar Compounds Table
            if st.session_state.current_pubchem_search_results:
                st.markdown("##### Similar Compounds (Up to 5, >90% Similarity):")

                table_html = "<table class='styled-table'><thead><tr><th>Sr.No.</th><th>Name</th><th>SMILES</th><th>2D Structure</th></tr></thead><tbody>"
                for idx, item in enumerate(st.session_state.current_pubchem_search_results):
                    table_html += f"<tr><td>{idx+1}</td><td>{item['name']}</td><td><code>{item['smiles']}</code></td><td><img src='{item['image_url']}' alt='2D structure' width='100'></td></tr>"
                table_html += "</tbody></table>"
                st.markdown(table_html, unsafe_allow_html=True)
                st.markdown("---")


        st.divider()

        # --- ANALYSIS LOGIC & RESULTS DISPLAY (ON MAIN PAGE) ---
        if st.session_state.get("analysis_triggered") and st.session_state.get("current_smiles"):
            active_smiles = st.session_state.current_smiles
            active_mol_name = st.session_state.current_mol_name
            mol_object_for_analysis = get_mol_from_smiles(active_smiles)

            if mol_object_for_analysis:
                with st.spinner(f"Analyzing {active_mol_name}... Please wait."):
                    st.session_state.molecule_data_store = {
                        "name": active_mol_name, "smiles": active_smiles,
                        "mol_obj_2d": mol_object_for_analysis,
                        "img_2d_bytes": None, "mol_3d_obj": None, "xyz_str": "",
                        "phys_props": {}, "drug_likeness": {},
                        "radar_plot_bytes": None, "boiled_egg_bytes": None,
                        "dist_plot_bytes": None, "heatmap_bytes": None
                    }

                    st.session_state.molecule_data_store["img_2d_bytes"] = generate_2d_image(mol_object_for_analysis)
                    mol_3d_rdkit_obj = generate_3d_coordinates(mol_object_for_analysis)
                    if mol_3d_rdkit_obj:
                        st.session_state.molecule_data_store["mol_3d_obj"] = mol_3d_rdkit_obj
                        st.session_state.molecule_data_store["xyz_str"] = mol_to_xyz(mol_3d_rdkit_obj, active_mol_name)

                    rdkit_props = calculate_rdkit_physicochemical_properties(mol_object_for_analysis)
                    final_phys_props = rdkit_props.copy()

                    current_pubchem_data = st.session_state.get("extended_pubchem_props_for_current_mol")
                    if not current_pubchem_data:
                        cid_from_smiles = pubchem_get_cid_from_name_or_smiles(active_smiles, id_type="smiles")
                        if not cid_from_smiles:
                             cid_from_smiles = pubchem_get_cid_from_name_or_smiles(active_mol_name)
                        if cid_from_smiles:
                            current_pubchem_data = pubchem_get_compound_info(cid_from_smiles)
                            st.session_state.extended_pubchem_props_for_current_mol = current_pubchem_data

                    if current_pubchem_data:
                        pubchem_prop_map = {
                            "MolecularWeight": "PubChem MolecularWeight", "ExactMass": "PubChem ExactMass",
                            "Charge": "PubChem Charge", "Complexity": "PubChem Complexity",
                            "XLogP3": "PubChem XLogP3", "HBondDonorCount": "PubChem HBD",
                            "HBondAcceptorCount": "PubChem HBA", "RotatableBondCount": "PubChem RotatableBonds",
                            "HeavyAtomCount": "PubChem HeavyAtoms"
                        }
                        for pc_key, display_key in pubchem_prop_map.items():
                            if pc_key in current_pubchem_data and current_pubchem_data[pc_key] is not None:
                                if display_key not in final_phys_props or "PubChem" in display_key:
                                    final_phys_props[display_key] = current_pubchem_data[pc_key]

                    st.session_state.molecule_data_store["phys_props"] = final_phys_props
                    st.session_state.molecule_data_store["drug_likeness"] = check_drug_likeness_rules(final_phys_props, mol_object_for_analysis)

                    if final_phys_props:
                        st.session_state.molecule_data_store["radar_plot_bytes"] = plot_bioavailability_radar(final_phys_props, active_mol_name)
                        st.session_state.molecule_data_store["boiled_egg_bytes"] = plot_boiled_egg([final_phys_props])
                        if SEABORN_AVAILABLE:
                            st.session_state.molecule_data_store["dist_plot_bytes"] = plot_property_distributions([final_phys_props], active_mol_name)
                            st.session_state.molecule_data_store["heatmap_bytes"] = plot_correlation_heatmap([final_phys_props], active_mol_name)

                # --- Display results using expanders on the main page ---
                display_data = st.session_state.molecule_data_store
                st.subheader(f"Analysis Results for: {display_data['name']} (`{display_data['smiles']}`)")
                sanitized_mol_name = re.sub(r'[^a-zA-Z0-9_.\-]', '_', display_data['name'])
                if not sanitized_mol_name: sanitized_mol_name = "molecule_analysis"
                sanitized_mol_name = (sanitized_mol_name[:60]) if len(sanitized_mol_name) > 60 else sanitized_mol_name


                with st.expander("2D Molecular Structure", expanded=True):
                    if display_data.get("img_2d_bytes"):
                        st.image(display_data["img_2d_bytes"], width=280)
                        st.download_button("Download 2D (PNG)", display_data["img_2d_bytes"], f"{sanitized_mol_name}_2D.png", "image/png", key="dl_2d_img_main", use_container_width=False)
                    else: st.warning("Could not generate 2D image.")

                with st.expander("3D Molecular Structure (Interactive)", expanded=False):
                    if STMOL_AVAILABLE and display_data.get("xyz_str"):
                        view = py3Dmol.view(width=600, height=450)
                        view.addModel(display_data["xyz_str"], 'xyz'); view.setStyle({'stick':{}}); view.setBackgroundColor('0xeeeeee'); view.zoomTo()
                        showmol(view, height=450, width=600)
                        st.download_button("Download 3D Coords (XYZ)", display_data["xyz_str"], f"{sanitized_mol_name}_3D.xyz", "text/plain", key="dl_3d_xyz_main", use_container_width=False)
                    elif not display_data.get("xyz_str") and RDKIT_AVAILABLE:
                         st.warning("3D coordinate generation failed or XYZ data not available.")
                    elif not STMOL_AVAILABLE: st.warning("stmol/py3Dmol is not available for 3D visualization.")
                    else: st.warning("Could not generate data for 3D structure display.")

                with st.expander("Bioavailability Radar Plot", expanded=False):
                    if display_data.get("radar_plot_bytes"):
                        st.image(display_data["radar_plot_bytes"], use_container_width=True)
                        st.download_button("Download Radar Plot (PNG)", display_data["radar_plot_bytes"], f"{sanitized_mol_name}_radar.png", "image/png", key="dl_radar_main", use_container_width=False)
                        st.info("""
                        **Interpreting the Bioavailability Radar:**
                        This plot provides a quick visual assessment of key ADMET-related properties relative to typical "drug-like" ranges. Each axis represents a property, normalized so that the **outer edge (1.0) is generally optimal** within the defined ideal range for that property.
                        - **LIPO (LogP):** Lipophilicity. The ideal range (e.g., 1-3 for many drugs) is mapped. Values outside this become less optimal.
                        - **SIZE (MW):** Molecular Weight. Values within the desirable range (e.g., 100-500 Da) score higher.
                        - **POLAR (TPSA):** Topological Polar Surface Area. Lower TPSA (e.g., <140 Å²) is often better for permeability; hence, lower raw values map to higher normalized scores.
                        - **INSOLU (LogS):** Aqueous solubility (approximated). Higher LogS (less negative, more soluble) scores higher.
                        - **INSATU (FracCsp3):** Fraction of sp3 hybridized carbons (3D character). Higher values are often preferred.
                        - **FLEX (RotB):** Flexibility (Rotatable Bonds). Fewer bonds (e.g., <10) can be better for bioavailability, so lower raw values map to higher scores.

                        **A larger, more regular, and well-filled hexagon suggests a more balanced and potentially favorable ADMET profile** according to these general guidelines. Deviations highlight properties that might need optimization.
                        """)
                    else: st.warning("Could not generate Bioavailability Radar Plot.")

                with st.expander("BOILED-Egg Plot", expanded=False):
                    if display_data.get("boiled_egg_bytes"):
                        st.image(display_data["boiled_egg_bytes"], use_container_width=True)
                        st.download_button("Download BOILED-Egg Plot (PNG)", display_data["boiled_egg_bytes"], f"{sanitized_mol_name}_boiled_egg.png", "image/png", key="dl_boiled_egg_main", use_container_width=False)
                        st.info("""
                        **Interpreting the BOILED-Egg Plot:**
                        This plot, based on the work by Daina and Zoete, predicts passive gastrointestinal (GI) absorption and brain penetration (BBB) using WLOGP (a measure of lipophilicity, x-axis) and TPSA (Topological Polar Surface Area, a measure of polarity, y-axis).
                        - **White Region (Egg White):** Compounds located here are predicted to have good passive GI absorption. These are likely to be absorbed into the bloodstream after oral administration.
                        - **Yellow Region (Yolk):** Compounds in this central area are predicted to have good passive GI absorption AND are likely to permeate the Blood-Brain Barrier (BBB+). This is often desirable for drugs targeting the central nervous system (CNS).
                        - **Outside Regions:** Compounds falling outside both ellipses are generally predicted to have poor passive GI absorption and/or poor BBB permeability. They might require active transport mechanisms or prodrug strategies if oral bioavailability or CNS action is desired.

                        The plot provides a quick visual filter for drug candidates based on these two crucial permeability aspects.
                        """)
                    else: st.warning("Could not generate BOILED-Egg Plot.")

                with st.expander("Physicochemical Properties", expanded=False):
                    if display_data.get("phys_props"):
                        props_html = "<table class='styled-table'><thead><tr><th>Property</th><th>Value</th></tr></thead><tbody>"
                        sorted_props = dict(sorted(display_data["phys_props"].items()))
                        for key, value in sorted_props.items():
                            val_str = f"{value:.2f}" if isinstance(value, (float, np.floating)) else str(value)
                            props_html += f"<tr><td>{key}</td><td>{val_str}</td></tr>"
                        props_html += "</tbody></table>"
                        st.markdown(props_html, unsafe_allow_html=True)
                    else: st.info("No physicochemical properties calculated.")

                with st.expander("Drug-Likeness Evaluation", expanded=False):
                    if display_data.get("drug_likeness"):
                        likeness_html = "<table class='styled-table'><thead><tr><th>Rule</th><th>Evaluation</th></tr></thead><tbody>"
                        for rule, evaluation in display_data["drug_likeness"].items():
                            if "Pass" in evaluation:
                                eval_display = f"<span class='pass-eval'>✅ {evaluation}</span>"
                            elif "Fail" in evaluation or "violation" in evaluation:
                                eval_display = f"<span class='fail-eval'>❌ {evaluation}</span>"
                            else:
                                eval_display = evaluation
                            likeness_html += f"<tr><td>{rule}</td><td>{eval_display}</td></tr>"
                        likeness_html += "</tbody></table>"
                        st.markdown(likeness_html, unsafe_allow_html=True)
                    else: st.info("No drug-likeness evaluation performed.")

                if SEABORN_AVAILABLE:
                    with st.expander("Molecular Property Distributions (Histograms & Boxplots)", expanded=False):
                        if display_data.get("dist_plot_bytes"):
                            st.image(display_data["dist_plot_bytes"], use_container_width=True)
                            st.download_button("Download Distributions (PNG)", display_data["dist_plot_bytes"], f"{sanitized_mol_name}_distributions.png", "image/png", key="dl_dist_plot_main", use_container_width=False)
                            st.info("""
                            **Interpreting Property Distributions:**
                            These plots show the distribution of key physicochemical properties. For a single molecule, they indicate its specific value.
                            - **Histogram (Top Part of Each Subplot):**
                                - Shows the frequency of values. For a single molecule, it's a single bar at the molecule's value.
                                - The red dashed line marks the exact value of the property for the current molecule.
                                - The curve (KDE - Kernel Density Estimate) smoothly estimates the probability distribution.
                            - **Boxplot (Bottom Inset of Each Subplot):**
                                - Visualizes data spread. For a single molecule, this appears as a single line at the property's value.
                                - **Median (Central Line):** The middle value (50th percentile).
                                - **Box (IQR):** Spans from the 1st quartile (Q1, 25th percentile) to the 3rd quartile (Q3, 75th percentile). The length of the box (IQR = Q3 - Q1) indicates variability.
                                - **Whiskers:** Extend from the box to show the range of the data, typically up to 1.5 * IQR from Q1 and Q3. Points beyond whiskers are potential outliers.
                            By observing where the red line (molecule's value) falls relative to typical drug-like ranges (e.g., those implied by Lipinski's rules), you can assess its profile for that property.
                            """)
                        else:
                            st.info("Distribution plots could not be generated.")

                    with st.expander("Property Relationships (Correlation Heatmap)", expanded=False):
                        if display_data.get("heatmap_bytes"):
                            st.image(display_data["heatmap_bytes"], use_container_width=True)
                            st.download_button("Download Heatmap (PNG)", display_data["heatmap_bytes"], f"{sanitized_mol_name}_heatmap.png", "image/png", key="dl_heatmap_main", use_container_width=False)
                            st.info("""
                            **Interpreting the Correlation Heatmap:**
                            This heatmap shows Pearson correlation coefficients between pairs of selected physicochemical properties.
                            - Values range from **-1 to +1**.
                            - **+1 (Warm color, e.g., red):** Perfect positive correlation. As one property increases, the other tends to increase.
                            - **-1 (Cool color, e.g., blue):** Perfect negative correlation. As one property increases, the other tends to decrease.
                            - **0 (Neutral color, e.g., white/light gray):** No linear correlation.
                            - The intensity of the color indicates the strength of the correlation.
                            This plot is most informative when analyzing a dataset of multiple molecules to identify trends. For a single molecule, correlations cannot be computed, and this plot will typically be empty or show a message.
                            """)
                        else:
                            st.info("Correlation heatmap could not be generated.")
                else:
                    with st.expander("Molecular Property Distributions", expanded=False):
                        st.warning("Seaborn library not found. Distribution plots are unavailable. Please install it (`pip install seaborn`) and restart the app.")
                    with st.expander("Property Relationships (Correlation Heatmap)", expanded=False):
                        st.warning("Seaborn library not found. Heatmap plots are unavailable. Please install it (`pip install seaborn`) and restart the app.")

                # Clear analysis trigger flags for the next run
                st.session_state.analysis_triggered = False
                st.session_state.analysis_triggered_by_sketcher = False
                st.session_state.analysis_triggered_by_pubchem_or_sdf = False
                # Do not clear molecule_data_store here so it remains visible until next analysis

            else:
                st.error(f"The provided SMILES '{active_smiles}' is invalid or could not be processed by RDKit.")
                st.session_state.analysis_triggered = False
                st.session_state.molecule_data_store = None # Clear if error

        elif st.session_state.get("analysis_triggered") and not st.session_state.get("current_smiles"):
            st.info("No SMILES available for analysis. Please input a molecule.")
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

    st.markdown("<div class='page-subtitle'>🛠 Tools and Technologies</div>", unsafe_allow_html=True)
    st.markdown("""<div class="content-box">Built with: Python, Streamlit, RDKit, Pandas, NumPy, Matplotlib, Seaborn, Requests, stmol, py3Dmol, streamlit-ketcher.</div>""", unsafe_allow_html=True)

    st.markdown("<div class='page-subtitle'>💡 Future Enhancements</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="future-box">
    The vision for ADMETriX extends beyond its current capabilities. Future enhancements planned include:
    <ul>
        <li>🎯 <b>Target Prediction:</b> Integration with databases like BindingDB to predict potential biological targets for input compounds.</li>
        <li>🔗 <b>Database Connectivity:</b> Direct links to ChEMBL and other relevant databases to fetch information on drug approval status, bioactivity data, and clinical trial information.</li>
        <li>🔬 <b>Advanced Visualization:</b> More sophisticated visualization tools, possibly including pharmacophore modeling overlays or interactive 3D docking previews (if feasible).</li>
        <li>📊 <b>Batch Analysis & Statistical Tools:</b> Capability to analyze multiple compounds simultaneously and provide more in-depth statistical comparisons and outlier detection.</li>
        <li>📈 <b>Predictive Modeling:</b> Incorporation of pre-trained machine learning models for predicting additional ADMET properties (e.g., solubility, permeability, toxicity endpoints).</li>
        <li>⚙️ <b>Customizable Workflows:</b> Allowing users to select specific analyses or rule sets.</li>
        <li>🌐 <b>API Access:</b> Potential for programmatic access to ADMETriX functionalities.</li>
    </ul>
    These enhancements aim to make ADMETriX an even more powerful and comprehensive tool for early-stage drug discovery.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='page-subtitle'>🙏 Mentorship & Acknowledgement</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="acknowledgement-box">
    I extend my sincere gratitude to <b>Dr. Kushagra Kashyap</b>, Assistant Professor in Bioinformatics, School of Science and Mathematics, DES Pune University, for his invaluable guidance and academic support throughout this project. His expertise in computational drug discovery and bioinformatics was instrumental in shaping the project's scientific direction and ensuring its rigor. Dr. Kashyap's mentorship provided critical insights into data interpretation, tool selection, and the practical aspects of ADMET analysis. His encouragement and constructive feedback were pivotal in refining the technical implementation and achieving the project's objectives. I am deeply appreciative of his time, patience, and commitment to fostering student research.
    <br><br>
    <div style="text-align: center;">
        <a href="https://www.linkedin.com/in/dr-kushagra-kashyap-b230a3bb" target="_blank" class="linkedin-btn" style="display: inline-block; padding: 0.5em 1em; margin-top:0.5em;">Connect with Dr. Kushagra Kashyap</a>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='page-subtitle'>📧 Feedback & Contact</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class="content-box">
    Your feedback is highly appreciated! If you have suggestions, encounter issues, or wish to collaborate, please feel free to reach out:
    <ul>
        <li>📧 <b>Email:</b> <a href="mailto:sonaalee21@gmail.com">sonaalee21@gmail.com</a></li>
        <li><img src="https://www.vectorlogo.zone/logos/linkedin/linkedin-icon.svg" alt="LinkedIn" width="16" height="16" style="vertical-align:middle; margin-right:5px;"> <b>LinkedIn:</b> <a href="https://www.linkedin.com/in/sonali-lakhamade-b6a83230a" target="_blank">Sonali Lakhamade</a></li>
        <li><img src="https://www.vectorlogo.zone/logos/github/github-icon.svg" alt="GitHub" width="16" height="16" style="vertical-align:middle; margin-right:5px;"> <b>GitHub:</b> <a href="https://github.com/Sonaalee21/admetTRIX/" target="_blank">ADMETriX Project Repository</a></li>

    </ul>
    </div>
    """, unsafe_allow_html=True)


# --- FOOTER ---
st.divider()
st.markdown("<div class='footer-text'>Made with passion by Sonali Lakhamade, an enthusiastic bioinformatics student | © 2025</div>", unsafe_allow_html=True)
