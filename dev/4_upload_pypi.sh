deactivate

. .venv_build/bin/activate

# Make sure the version is up to date

# Requires correct .pypirc __token__ authentication

python3 -m pip install --upgrade twine
python3 -m twine upload dist/*

deactivate
