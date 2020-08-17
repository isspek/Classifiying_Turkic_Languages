import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from dataset import read_splits, ID2STR
from logger import logger
from cli import args
from models import MODELS

MODEL_CONFIG_DIR = "model_configs"
MODELS_DIR = "models"
random_splits = [42, 1, 15]


def experiment(fname: str):
    reports_dev = []
    reports_test = []
    for random_split in random_splits:
        logger.info("Experimenting split of random seed {}".format(random_split))
        X_train, y_train, X_dev, y_dev, X_test, y_test = read_splits(random_split)

        model_config_dir = Path(MODEL_CONFIG_DIR)

        assert model_config_dir.exists()

        model_config_fname = model_config_dir / fname

        with open(model_config_fname) as f:
            _config = json.load(f)

        model = MODELS[_config["model"]](_config)

        model.model_dir = Path(MODELS_DIR) / str(random_split)
        model.model_dir.mkdir(parents=True, exist_ok=True)

        # TODO check the below
        model.model_path = model.model_dir / model.model_path

        if not model.model_path.exists():
            model.fit(X_train, y_train)

        logger.info("Prediction mode")
        preds_dev = model.predict(X_dev)

        preds_dev = [ID2STR[pred] for pred in preds_dev]
        y_dev = [ID2STR[pred[0]] for pred in y_dev]

        logger.info("===========DEV RESULTS===========")
        logger.info(classification_report(y_true=y_dev, y_pred=preds_dev, digits=4))
        report = classification_report(y_true=y_dev, y_pred=preds_dev, digits=4, output_dict=True)

        data = {}
        for label in set(y_dev):
            data['{key}_f1'.format(key=label)] = [round(report[label]['f1-score'] * 100, 2)]
        data['macro_f1'] = [round(report['macro avg']['f1-score'] * 100, 2)]

        reports_dev.append(pd.DataFrame.from_dict(data))

        preds_test = model.predict(X_test)
        preds_test = [ID2STR[pred] for pred in preds_test]
        y_test = [ID2STR[pred[0]] for pred in y_test]

        logger.info("===========TEST RESULTS===========")
        logger.info(classification_report(y_true=y_test, y_pred=preds_test, digits=4))

        # if we need an average
        report = classification_report(y_true=y_test, y_pred=preds_test, digits=4, output_dict=True)
        data = {}
        for label in set(y_test):
            data['{key}_f1'.format(key=label)] = [round(report[label]['f1-score'] * 100, 2)]
        data['macro_f1'] = [round(report['macro avg']['f1-score'] * 100, 2)]

        reports_test.append(pd.DataFrame.from_dict(data))

    reports_dev = pd.concat(reports_dev)
    reports_test = pd.concat(reports_test)
    for column in reports_dev.columns:
        logger.info(column)
        logger.info('Mean-Dev {}'.format(np.mean(reports_dev[column].values)))
        logger.info('Std-Dev {}'.format(np.std(reports_dev[column].values)))
        logger.info('Mean-Test {}'.format(np.mean(reports_test[column].values)))
        logger.info('Std-Test {}'.format(np.std(reports_test[column].values)))


if __name__ == '__main__':
    experiment(args.fname)
