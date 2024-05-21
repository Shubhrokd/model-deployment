from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

model = joblib.load("rfmodel2.pkl")
app = Flask(__name__)
features_to_exclude = [
    "NumRadicalElectrons",
    "SMR_VSA8",
    "SlogP_VSA7",
    "SlogP_VSA9",
    "EState_VSA11",
    "NumAliphaticCarbocycles",
    "NumSaturatedCarbocycles",
    "fr_Al_OH",
    "fr_Al_OH_noTert",
    "fr_ArN",
    "fr_Ar_COO",
    "fr_Ar_NH",
    "fr_Ar_OH",
    "fr_C_S",
    "fr_HOCCN",
    "fr_N_O",
    "fr_Ndealkylation1",
    "fr_Nhpyrrole",
    "fr_SH",
    "fr_aldehyde",
    "fr_alkyl_carbamate",
    "fr_allylic_oxid",
    "fr_amide",
    "fr_amidine",
    "fr_aniline",
    "fr_azide",
    "fr_azo",
    "fr_barbitur",
    "fr_benzodiazepine",
    "fr_bicyclic",
    "fr_diazo",
    "fr_dihydropyridine",
    "fr_epoxide",
    "fr_ester",
    "fr_ether",
    "fr_furan",
    "fr_guanido",
    "fr_hdrzine",
    "fr_hdrzone",
    "fr_imide",
    "fr_isocyan",
    "fr_isothiocyan",
    "fr_ketone",
    "fr_ketone_Topliss",
    "fr_lactam",
    "fr_lactone",
    "fr_methoxy",
    "fr_morpholine",
    "fr_nitro",
    "fr_nitro_arom",
    "fr_nitro_arom_nonortho",
    "fr_nitroso",
    "fr_oxazole",
    "fr_oxime",
    "fr_para_hydroxylation",
    "fr_phenol",
    "fr_phenol_noOrthoHbond",
    "fr_piperzine",
    "fr_priamide",
    "fr_prisulfonamd",
    "fr_pyridine",
    "fr_sulfide",
    "fr_sulfone",
    "fr_term_acetylene",
    "fr_tetrazole",
    "fr_thiazole",
    "fr_thiophene",
    "fr_urea",
]


def process_molecules(smiles_1):
    mol_1 = Chem.MolFromSmiles(smiles_1)

    descrs = [Descriptors.CalcMolDescriptors(mol_1)]
    df = pd.DataFrame(descrs)
    df["Molecule"] = ["Molecule 1"]
    selected_features = df.drop(features_to_exclude, axis=1)
    return selected_features


@app.route("/")
def index():
    return "Hello world"


@app.route("/predict", methods=["POST"])
def predict():
    smiles_1 = request.form.get("mol1")
    smiles_2 = request.form.get("mol2")
    temp = request.form.get("temp")
    cmol1 = request.form.get("cmol1")
    cmol2 = request.form.get("cmol2")
    feature_1 = process_molecules(smiles_1)
    feature_2 = process_molecules(smiles_2)
    feature_1 *= cmol1
    feature_2 *= cmol2
    feature_sum = feature_1 + feature_2
    vector = feature_sum.to_numpy().flatten()
    vector.append(temp)
    prediction = model.predict(vector)

    return jsonify({"Conductivity": int(prediction)})


if __name__ == "__main__":
    app.run(debug=True)
