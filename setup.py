from setuptools import setup, find_packages

setup(
    name='scCRAFT',
    version='1.0.0',
    author='Chuan He',
    author_email='ch2343@yale.edu',
    packages=find_packages(),
    install_requires=[
        'torch',
        'scanpy',
        'numpy',
        'umap-learn',
        'scipy',
        'pandas',
        'sklearn',
        'jax',
        'matplotlib'  # Assume dependencies based on your initial code
    ],
    url='http://pypi.python.org/pypi/scCRAFT/',
    license='LICENSE.txt',
    description='An package for single-cell data integration using deep learning.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)