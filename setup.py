from setuptools import setup

setup(
    name='tsar',
    version='0.0.3',
    description='Time series auto-regressor.',
    author='Enzo Busseti',
    packages=['tsar'],
    install_requires=['numpy', 'pandas', 'numba']
)
