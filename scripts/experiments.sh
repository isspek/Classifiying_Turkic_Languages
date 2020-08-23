#!/usr/bin/env bash

python -m experiment --fname baseline_char.json
python -m experiment --fname baseline_word.json
python -m experiment --fname feat_based.json
python -m experiment --fname all.json
