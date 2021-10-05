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


# ========================================
# TESTPYPI
# https://test.pypi.org/project/aegis-sim/
# ========================================

# Upload build to testpypi
upload_testpypi:
	twine upload --repository testpypi dist/*

# Install build from testpypi
install_testpypi:
	deactivate
	rm -rf temp/venv
	python3 -m venv temp/venv
	. temp/venv/bin/activate
	python3 -m pip install --index-url https://test.pypi.org/simple --extra-index-url https://pypi.org/simple aegis-sim[dev]


# ===================================
# (REAL) PYPI
# https://pypi.org/project/aegis-sim/
# ===================================

# Upload build to pypi
upload_pypi:
	twine upload dist/*

# Install build from pypi 
install_pypi:
	deactivate
	rm -rf temp/venv
	python3 -m venv temp/venv
	. temp/venv/bin/activate
	python3 -m pip install aegis-sim[dev]


# =============
# MISCELLANEOUS
# =============

manifest:
	python3 -m pip install check-manifest
	check-manifest --create