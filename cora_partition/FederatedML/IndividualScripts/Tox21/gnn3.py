# Imaging imports to get pictures in the notebook
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D


import numpy as np
import deepchem as dc
from deepchem.molnet import load_tox21

# Only for debug!
np.random.seed(123)

# Load Tox21 dataset
n_features = 1024
tox21_tasks, tox21_datasets, transformers = load_tox21()
train_dataset, valid_dataset, test_dataset = tox21_datasets

# Fit models
metric = dc.metrics.Metric(
    dc.metrics.roc_auc_score, np.mean, mode="classification")

nb_epoch = 10
model = dc.models.tensorgraph.fcnet.MultitaskClassifier(
    len(tox21_tasks),
    train_dataset.get_data_shape()[0])

# Fit trained model
model.fit(train_dataset, nb_epoch=nb_epoch)
#model.save('test.h5')

train_scores = model.evaluate(train_dataset, [metric], transformers)
valid_scores = model.evaluate(valid_dataset, [metric], transformers)

from lime import lime_tabular
feature_names = ["fp_%s"  % x for x in range(1024)]
explainer = lime_tabular.LimeTabularExplainer(train_dataset.X,
                                              feature_names=feature_names,
                                              categorical_features=feature_names,
                                              class_names=['not toxic', 'toxic'],
                                              discretize_continuous=True)

# We need a function which takes a 2d numpy array (samples, features) and returns predictions (samples,)
def eval_model(my_model, transformers):
    def eval_closure(x):
        ds = dc.data.NumpyDataset(x, None, None, None)
        # The 0th task is NR-AR
        predictions = model.predict(ds)[:,0]
        return predictions
    return eval_closure
model_fn = eval_model(model, transformers)

# We want to investigate a toxic compound
active_id = np.where(test_dataset.y[:,0]==1)[0][0]

# this returns an Lime Explainer class
# The explainer contains details for why the model behaved the way it did
exp = explainer.explain_instance(test_dataset.X[active_id], model_fn, num_features=5, top_labels=0)