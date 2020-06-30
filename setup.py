from setuptools import setup, find_packages

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='deepGTA',
    version='0.1.1',
    description='analyze time-resolved spectra using global-target analysis and deep learning',
    long_description=readme,
    author='Philipp Kollenz',
    author_email='p.kollenz@stud.uni-heidelberg.de',
    license=license,
    packages=find_packages(exclude=('tests', 'docs', 'examples'))
)
