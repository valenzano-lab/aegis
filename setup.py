from setuptools import setup, find_packages
from setuptools.extension import Extension

with open("README-PyPI.md", "r") as fh:
    long_description = fh.read()

setup(name='mpi-age-aegis',
        # Version
        version='1.0',
        # Package data
        packages=find_packages(),
        #packages=['aegis'],
        zip_safe=False, #?
        setup_requires=['pytest-runner'],
        tests_require=['pytest'],
        install_requires=[
            'numpy>=1.15.4',
            'scipy>=1.1.0',
            'python-dateutil>=2.7.5',
            'pandas>=0.23.4',
            'matplotlib==2.1.0',
            'seaborn>=0.9.0',
            'pytest>=4.3.0'
            ],
        scripts=['bin/aegis'],
        # Medadata for upload to PyPI
        author='Will Bradshaw, Arian Sajina and Dario Valenzano',
        author_email='wbradshaw@age.mpg.de, asajina@age.mpg.de, Dario.Valenzano@age.mpg.de',
        description='AEGIS - Ageing of Evolving Genomes In Silico',
        long_description=long_description,
        long_description_content_type="text/markdown",
        license='MIT',
        url="https://github.com/valenzano-lab/aegis"
        )
