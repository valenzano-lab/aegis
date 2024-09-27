deactivate

rm -rf .venv_pypi
python3 -m venv .venv_pypi
. .venv_pypi/bin/activate

python3 -m pip install --upgrade pip pytest
python3 -m pip install --no-cache-dir aegis-sim

pytest

deactivate
