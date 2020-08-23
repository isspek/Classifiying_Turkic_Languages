"""
Error analysis of predictions that are performed by the best model
"""

import json
from pathlib import Path

import numpy as np

from dataset import read_splits, ID2STR
from models import MODELS

MODEL_CONFIG_DIR = "model_configs"
MODELS_DIR = "models"
RANDOM_SEED = 42
FNAME = "baseline_word.json"
RESULTS = "results"


def save_predicts():
    _, _, _, _, X_test, y_test = read_splits(RANDOM_SEED)

    model_config_dir = Path(MODEL_CONFIG_DIR)

    assert model_config_dir.exists()

    model_config_fname = model_config_dir / FNAME

    with open(model_config_fname) as f:
        _config = json.load(f)

    model = MODELS[_config["model"]](_config)

    model.model_dir = Path(MODELS_DIR) / str(RANDOM_SEED)
    model.model_path = model.model_dir / model.model_path

    preds_test = model.predict(X_test)

    preds_test = [ID2STR[pred] for pred in preds_test]
    y_test = [ID2STR[pred[0]] for pred in y_test]

    np.savez_compressed(Path(RESULTS) / "preds.npz", preds=np.asarray(preds_test)
                        , labels=np.asarray(y_test))


if __name__ == '__main__':
    save_predicts()
