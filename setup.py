from setuptools import setup, find_packages
from setuptools.extension import Extension

setup(name='aegis',
        # Version
        version='0.5',
        # Package data
        packages=find_packages(),
        #packages=['aegis'],
        zip_safe=False, #?
        setup_requires=['pytest-runner'],
        tests_require=['pytest'],
        install_requires=[
            'numpy>=1.13.3',
            'scipy>=0.19.1',
            'python-dateutil>=2.6.1',
            'pandas>=0.22.0',
            'matplotlib==2.1.0',
            'ggplot>=0.11.5'
            # pytest?
            ],
        scripts=['bin/aegis'],
        # Medadata for upload to PyPI
        author='Will Bradshaw, Dario Valenzano and Arian Sajina',
        author_email='wbradshaw@age.mpg.de, asajina@age.mpg.de',
        description='AEGIS - Ageing of Evolving Genomes In Silico',
        license='MIT',
        url="tbc"
        )
