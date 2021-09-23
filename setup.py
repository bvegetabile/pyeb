from setuptools import setup, find_packages

setup(
    author = 'Brian G. Vegetabile',
    description = 'Python package for entropy balancing',
    name = 'pyeb',
    version = '0.1.0',
    packages = find_packages(include = ['pyeb', 'pyeb.*']),
    install_requires = [
        'numpy',
        'scipy',
        'pandas',
        'matplotlib'
    ]
)