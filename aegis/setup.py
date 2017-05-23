from setuptools import setup, find_packages

setup(name='aegis',
        # Version
        version='0.1',
        # Package data
        packages=find_packages(),
        #packages=['aegis'],
        zip_safe=False, #?
        setup_requires=['pytest-runner'],
        tests_require=['pytest'],
        install_requires=[], #! Set dependencies
        # Medadata for upload to PyPI
        author='Will Bradshaw & Dario Valenzano',
        author_email='wbradshaw@age.mpg.de',
        description='AEGIS - Ageing of Evolving Genomes In Silico',
        license='MIT',
        url="tbc"
        )
