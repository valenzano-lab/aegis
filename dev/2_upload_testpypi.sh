deactivate

. .venv_build/bin/activate

# Make sure the version is up to date

python3 -m pip install --upgrade twine
python3 -m twine upload --repository testpypi dist/*

deactivate
