#!/usr/bin/env bash

# if venv does not exist, create it
if ! [ -d "venv" ]; then
  #python3 -m venv .venv
  python3 -m venv venv
fi

# activate venv
source venv/bin/activate

# install requirements into venv
pip3 install -r requirements.txt

# launch jupyter
jupyter notebook

deactivate