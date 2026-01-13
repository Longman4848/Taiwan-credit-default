#@"
from setuptools import setup, find_packages

setup(
    name='taiwan_credit_default',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'pytest'
    ]
)
#@ | Out-File -FilePath "setup.py" -Encoding UTF8