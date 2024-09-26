deactivate

rm -rf .venv_testpypi
python3 -m venv .venv_testpypi
. .venv_testpypi/bin/activate

python3 -m pip install --upgrade pip pytest
python3 -m pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple aegis-sim

pytest

deactivate
