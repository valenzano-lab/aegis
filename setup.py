from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
        Extension(
            'aegis.Core.Population',
            ['aegis/Core/Population.pyx'],
            include_dirs=[numpy.get_include()]
            )
        ]

setup(name='aegis',
        # Version
        version='0.5',
        # Package data
        packages=find_packages(),
        #packages=['aegis'],
        ext_modules = cythonize(extensions),
        zip_safe=False, #?
        setup_requires=['pytest-runner'],
        tests_require=['pytest'],
        install_requires=[
            'numpy>=1.13.3',
            'scipy>=0.19.1',
            'python-dateutil>=2.6.1'
            # add plotting package
            ],
        scripts=['bin/aegis'],
        # Medadata for upload to PyPI
        author='Will Bradshaw & Dario Valenzano',
        author_email='wbradshaw@age.mpg.de',
        description='AEGIS - Ageing of Evolving Genomes In Silico',
        license='MIT',
        url="tbc"
        )
