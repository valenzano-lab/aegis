deactivate

. .venv_build/bin/activate

# Make sure the version is up to date

# For verification, use the ~/.pypirc from WSL on Kaki (Martin Bagic)

python3 -m pip install --upgrade twine
python3 -m twine upload --repository testpypi dist/*

deactivate
