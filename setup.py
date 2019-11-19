from setuptools import setup

setup(
    name='tsar',
    version='0.7.2',
    description='Time series auto-regressor.',
    author='Enzo Busseti',
    license='GPLv3+',
    packages=['tsar'],
    tests_requires=['nose >= 1.3.7'],
    install_requires=['scipy>=1.3.1', 'numpy>=1.17.3',
                      'numba>=0.46.0', 'pandas>=0.25.3']
)
