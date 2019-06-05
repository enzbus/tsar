from setuptools import setup

setup(
    name='tsar',
    version='0.0.6',
    description='Time series auto-regressor.',
    author='Enzo Busseti',
    license='GPLv3+',
    packages=['tsar'],
    install_requires=['numpy', 'pandas', 'numba']
)
