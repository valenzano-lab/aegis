.ONESHELL:

# NOTES:
# 	recipe_testpypi: build upload_testpypi install_testpypi test_venv
# 	recipe_pypi: build upload_pypi install_pypi test_venv


# ============
# HELPER FUNCS
# ============

# Remove old builds and make new build
build:
	rm dist/*
	python3 -m build

# Test aegis installed in test/venv
test_venv:
	. temp/venv/bin/activate
	aegis misc/misc.yml
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

performance_profile:
	python3 profiling/profiler.py

editable_install:
	python3 -m pip install -e .

editable_uninstall:
	python3 -m pip uninstall aging-of-evolving-genomes