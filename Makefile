.ONESHELL:

# ============
# HELPER FUNCS
# ============

# Install AEGIS for development
install_dev:
	# Create, activate and update virtual environment
	# Make fresh local build of AEGIS
	# Install the developer version of the local build of AEGIS
	# Test

	rm -rf .venv ; \
	python3 -m venv .venv ; \
	. .venv/bin/activate ; \
	python3 -m pip install --upgrade pip; \
	rm -rf dist/* ; \
	python3 -m pip install build ; \
	python3 -m build ; \
	python3 -m pip install -e .[dev] ; \
	python3 -m pytest tests/ --log-cli-level=DEBUG

# Uninstall local AEGIS installation
uninstall_dev:
	. .venv/bin/activate ; \
	python3 -m pip uninstall aegis-sim ; \
	deactivate ; \
	rm -rf .venv

# Remove old builds and make a new build
build:
	rm -rf dist/*
	python3 -m build

# Test aegis installed in .venv
test:
	. .venv/bin/activate ; \
	python3 -m pytest tests/ --log-cli-level=DEBUG

# Create and visualize performance profile
# make profile yml=_
# _ is the basic configuration
profile:
	. .venv/bin/activate ; \
	python3 -m cProfile -o profiler/$(yml).prof src/aegis/__main__.py profiler/$(yml).yml
	rm -r profiler/$(yml) ; \
	snakeviz profiler/$(yml).prof


# ========================================
# TESTPYPI
# https://test.pypi.org/project/aegis-sim/
# ========================================

# Upload build to testpypi
upload_testpypi:
	python3 -m pip install --upgrade twine
	python3 -m twine upload --repository testpypi dist/*

# Install build from testpypi
install_testpypi:
	deactivate ; \
	rm -rf temp/venv ; \
	python3 -m venv temp/venv ; \
	. temp/venv/bin/activate ; \
	python3 -m pip install --upgrade pip pytest ; \
	python3 -m pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple aegis-sim ; \
	python3 -m pytest tests/ --log-cli-level=DEBUG


# ===================================
# (REAL) PYPI
# https://pypi.org/project/aegis-sim/
# ===================================

# Upload build to pypi
upload_pypi:
	python3 -m pip install --upgrade twine
	python3 -m twine upload dist/*

# Install build from pypi 
install_pypi:
	rm -rf temp/venv ; \
	python3 -m venv temp/venv ; \
	. temp/venv/bin/activate ; \
	python3 -m pip install --upgrade pip pytest ; \
	python3 -m pip install --no-cache-dir aegis-sim ; \
	python3 -m pytest tests/ --log-cli-level=DEBUG


# =============
# MISCELLANEOUS
# =============

manifest:
	python3 -m pip install check-manifest
	check-manifest --create