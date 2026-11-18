import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parent / "openpom"))
from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import deepchem as dc
from rdkit import Chem
from rdkit.Chem import Draw

from openpom.feat.graph_featurizer import GraphFeaturizer, GraphConvConstants
from openpom.models.mpnn_pom import MPNNPOMModel

ROOT = Path(__file__).resolve().parent
ENSEMBLE_DIR = ROOT / "openpom" / "models" / "ensemble_models"
N_MODELS = 10

TASKS = [
    'alcoholic', 'aldehydic', 'alliaceous', 'almond', 'amber', 'animal',
    'anisic', 'apple', 'apricot', 'aromatic', 'balsamic', 'banana', 'beefy',
    'bergamot', 'berry', 'bitter', 'black currant', 'brandy', 'burnt',
    'buttery', 'cabbage', 'camphoreous', 'caramellic', 'cedar', 'celery',
    'chamomile', 'cheesy', 'cherry', 'chocolate', 'cinnamon', 'citrus', 'clean',
    'clove', 'cocoa', 'coconut', 'coffee', 'cognac', 'cooked', 'cooling',
    'cortex', 'coumarinic', 'creamy', 'cucumber', 'dairy', 'dry', 'earthy',
    'ethereal', 'fatty', 'fermented', 'fishy', 'floral', 'fresh', 'fruit skin',
    'fruity', 'garlic', 'gassy', 'geranium', 'grape', 'grapefruit', 'grassy',
    'green', 'hawthorn', 'hay', 'hazelnut', 'herbal', 'honey', 'hyacinth',
    'jasmin', 'juicy', 'ketonic', 'lactonic', 'lavender', 'leafy', 'leathery',
    'lemon', 'lily', 'malty', 'meaty', 'medicinal', 'melon', 'metallic',
    'milky', 'mint', 'muguet', 'mushroom', 'musk', 'musty', 'natural', 'nutty',
    'odorless', 'oily', 'onion', 'orange', 'orangeflower', 'orris', 'ozone',
    'peach', 'pear', 'phenolic', 'pine', 'pineapple', 'plum', 'popcorn',
    'potato', 'powdery', 'pungent', 'radish', 'raspberry', 'ripe', 'roasted',
    'rose', 'rummy', 'sandalwood', 'savory', 'sharp', 'smoky', 'soapy',
    'solvent', 'sour', 'spicy', 'strawberry', 'sulfurous', 'sweaty', 'sweet',
    'tea', 'terpenic', 'tobacco', 'tomato', 'tropical', 'vanilla', 'vegetable',
    'vetiver', 'violet', 'warm', 'waxy', 'weedy', 'winey', 'woody'
]

EXAMPLES = {
    "Ethanol": "CCO",
    "Ethyl acetate": "CC(=O)OCC",
    "Vanillin": "COc1cc(C=O)ccc1O",
    "Limonene": "CC(=C)C1CCC(C)=CC1",
    "Menthol": "CC(C)C1CCC(C)CC1O",
}


@st.cache_resource(show_spinner="Loading the 10 pretrained models (first time only)...")
def load_models():
    models = []
    for i in range(N_MODELS):
        exp_dir = ENSEMBLE_DIR / f"experiments_{i + 1}"
        model = MPNNPOMModel(
            n_tasks=len(TASKS),
            batch_size=128,
            class_imbalance_ratio=None,
            loss_aggr_type='sum',
            node_out_feats=100,
            edge_hidden_feats=75,
            edge_out_feats=100,
            num_step_message_passing=5,
            mpnn_residual=True,
            message_aggregator_type='sum',
            mode='classification',
            number_atom_features=GraphConvConstants.ATOM_FDIM,
            number_bond_features=GraphConvConstants.BOND_FDIM,
            n_classes=1,
            readout_type='set2set',
            num_step_set2set=3,
            num_layer_set2set=2,
            ffn_hidden_list=[392, 392],
            ffn_embeddings=256,
            ffn_activation='relu',
            ffn_dropout_p=0.12,
            ffn_dropout_at_input_no_act=False,
            weight_decay=1e-5,
            self_loop=False,
            optimizer_name='adam',
            log_frequency=32,
            model_dir=str(exp_dir),
            device_name='cpu',
        )
        model.restore(str(exp_dir / "checkpoint2.pt"))
        models.append(model)
    return models


def predict_odors(models, smiles):
    featurizer = GraphFeaturizer()
    featurized = featurizer.featurize([smiles])
    preds = [m.predict(dc.data.NumpyDataset(featurized)) for m in models]
    return np.mean(np.asarray(preds), axis=0).squeeze()


def to_dataframe(scores):
    df = pd.DataFrame({"odor": TASKS, "score": scores})
    return df.sort_values("score", ascending=False).reset_index(drop=True)



st.set_page_config(page_title="AI Nose", page_icon="👃", layout="wide")
st.title("👃 AI Nose")
st.caption("Predict how a molecule smells, using the pretrained OpenPOM model (10-model ensemble).")

if "smiles" not in st.session_state:
    st.session_state.smiles = "CCO"


def set_example(value):
    st.session_state.smiles = value


st.write("**Try an example:**")
cols = st.columns(len(EXAMPLES))
for col, (name, smi) in zip(cols, EXAMPLES.items()):
    col.button(name, on_click=set_example, args=(smi,), use_container_width=True)

left, right = st.columns([3, 1])
with left:
    smiles = st.text_input("SMILES string (one molecule)", key="smiles")
with right:
    top_n = st.slider("Top N odors", 3, 30, 10)

smiles = smiles.strip()
if smiles:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("That is not a valid SMILES string. Check it and try again.")
    else:
        models = load_models()
        with st.spinner("Predicting..."):
            scores = predict_odors(models, smiles)
        df = to_dataframe(scores)
        top = df.head(top_n)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.subheader("Structure")
            st.image(Draw.MolToImage(mol, size=(350, 350)))
            st.metric("Top descriptor", top.iloc[0]["odor"], f"{top.iloc[0]['score']:.3f}")

        with c2:
            st.subheader(f"Top {top_n} odor descriptors")
            fig, ax = plt.subplots(figsize=(7, 0.4 * top_n + 1))
            ax.barh(top["odor"], top["score"], color="#4C8BF5")
            ax.invert_yaxis()
            ax.set_xlim(0, 1)
            ax.set_xlabel("Predicted score")
            for y, v in enumerate(top["score"]):
                ax.text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=9)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            st.pyplot(fig)

        st.subheader("Results table")
        st.dataframe(top, use_container_width=True, hide_index=True)

        d1, d2, _ = st.columns([1, 1, 3])
        d1.download_button("Download CSV", top.to_csv(index=False),
                           file_name="odor_prediction.csv", mime="text/csv")
        payload = {"smiles": smiles,
                   "predictions": top.to_dict(orient="records")}
        d2.download_button("Download JSON", json.dumps(payload, indent=2),
                           file_name="odor_prediction.json", mime="application/json")
else:
    st.info("Enter a SMILES string or click an example above.")