import setuptools
import pathlib

__version__ = "2.3.0.2"

# Latest pypi version is 2.3.0.2
# Latest test_pypi version is 2.3.0


setuptools.setup(
    name="aegis-sim",
    version=__version__,
    description="Numerical model for life history evolution of age-structured populations",
    long_description=(pathlib.Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    keywords=["evolution", "aging", "life history", "agent-based model", "simulation"],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    author="Martin Bagic, Dario Valenzano",
    author_email="martin.bagic@outlook.com, Dario.Valenzano@leibniz-fli.de",
    project_urls={
        "Source": "https://github.com/valenzano-lab/aegis",
        # "Documentation": "",
    },
    url="https://github.com/valenzano-lab/aegis",
    # packages=setuptools.find_packages(include=["aegis", "gui", "documentation", "aegis.*"]),
    packages=setuptools.find_namespace_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.6",
    entry_points={
        "console_scripts": [
            "aegis = aegis.__main__:main",
        ]
    },
    install_requires=[
        "numpy",
        "pandas",
        "PyYAML",
        "pyarrow",
        "platformdirs",
        "dash>=2.17.1",  # GUI
        "dash-bootstrap-components",  # GUI
        "urllib3==1.26.19",
        "psutil",
        "kaleido",  # for image export for dash figures TODO use a simpler method
    ],
    extras_require={
        "dev": [
            "pytest==6.2.4",
            "flake8",
            "black",
            "tabulate",
            "pdoc",
        ]
    },
)
