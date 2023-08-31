import setuptools
import pathlib

setuptools.setup(
    name="aegis-sim",
    version="2.3.0",
    description="Numerical model for life history evolution of age-structured populations",
    long_description=(pathlib.Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    author="Martin Bagic, Dario Valenzano",
    author_email="martin.bagic@outlook.com, Dario.Valenzano@leibniz-fli.de",
    url="https://github.com/valenzano-lab/aegis",
    packages=["aegis", "aegis.help", "aegis.modules", "aegis.parameters", "aegis.visor"],
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
        "dash",
    ],
    extras_require={
        "dev": [
            "pytest==6.2.4",
            "flake8",
            "black",
        ]
    },
)
