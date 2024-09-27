# install aegis-sim for development

# uninstall previous installation
. .venv/bin/activate

python3 -m pip uninstall aegis-sim -y
rm -rf .venv

deactivate

# install afresh

# make new venv
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install --upgrade pip

# make new build
python3 -m pip install build pytest
python3 -m build

# install build
python3 -m pip install -e .[dev]
python3 -m pytest tests/ --log-cli-level=WARNING
