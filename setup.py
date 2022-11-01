import setuptools
import pathlib

setuptools.setup(
    name="aegis-sim",
    version="2.2",
    description="Numerical model for life history evolution of age-structured populations",
    long_description=(pathlib.Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    author="Martin Bagic, Dario Valenzano",
    author_email="martin.bagic@outlook.com, Dario.Valenzano@leibniz-fli.de",
    url="https://github.com/valenzano-lab/aegis",
    package_dir={"": "src"},
    packages=[
        "aegis",
        "aegis.modules",
        "aegis.parameters",
        "aegis.help",
    ],
    package_data={
        "aegis": ["parameters/default.yml", "help/visor.ipynb"],
    },
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
        "jupyter",
        "matplotlib",
    ],
    extras_require={
        "dev": [
            "pytest==6.2.4",
            "flake8",
            "black",
        ]
    },
)
