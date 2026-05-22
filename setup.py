from setuptools import setup, find_packages

setup(
    name='sleep_monitor',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'matplotlib',
        'ipykernel',
        'pyarrow',
    ],
    python_requires='>=3.9',
)
