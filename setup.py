from setuptools import setup

setup(
    name='tsar',
    version='0.3',
    description='Time series auto-regressor.',
    author='Enzo Busseti',
    license='GPLv3+',
    packages=['tsar'],
    install_requires=['numpy', 'scipy', 'pandas', 'numba']
)
