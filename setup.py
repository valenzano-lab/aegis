from setuptools import setup, find_packages
from setuptools.extension import Extension

setup(name='aegis',
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
        author='Will Bradshaw, Dario Valenzano and Arian Sajina',
        author_email='wbradshaw@age.mpg.de, asajina@age.mpg.de',
        description='AEGIS - Ageing of Evolving Genomes In Silico',
        license='MIT',
        url="tbc"
        )
