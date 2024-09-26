deactivate

rm -rf .venv_build
python3 -m venv .venv_build

. .venv_build/bin/activate

python3 -m pip install --upgrade pip pytest
python3 -m pip install build

rm -rf build dist src/aegis_sim.egg-info
python3 setup.py sdist
python3 -m pip install dist/*.tar.gz

pytest

deactivate
