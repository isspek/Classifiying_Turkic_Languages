#!/usr/bin/env bash
if [ ! -d pyzemberek ]
then
    git clone https://github.com/kodiks/pyzemberek.git
fi
cd pyzemberek

python setup.py install