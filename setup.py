from setuptools import find_packages, setup


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('requirements.txt') as f:
    requirements = f.read()


setup(
    name='deimos_gui',
    version='1.0.1',
    description='GUI for Data Extraction for Integrated Multidimensional Spectrometry',
    long_description=readme,
    author='Marjolein Oostrom',
    author_email='marjolein.oostrom@pnnl.gov',
    url='https://github.com/pnnl/deimos_gui',
    install_requires=requirements,
    license=license
)
