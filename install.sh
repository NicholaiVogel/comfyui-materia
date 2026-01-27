#!/bin/bash

requirements_txt="$(dirname "$0")/requirements.txt"
python_exec="../../../python_embeded/python.exe"

echo "Installing Materia requirements..."

if [ -f "$python_exec" ]; then
    echo "Installing with ComfyUI Portable"
    "$python_exec" -s -m pip install -r "$requirements_txt"
else
    echo "Installing with system Python"
    pip install -r "$requirements_txt"
fi
