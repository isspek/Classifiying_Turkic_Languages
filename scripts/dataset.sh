#!/usr/bin/env bash
python -m dataset --mode split --random_seed 42
python -m dataset --mode split --random_seed 1
python -m dataset --mode split --random_seed 15