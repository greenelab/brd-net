from setuptools import setup

setup(
   name='brdnet',
   version='1.0',
   description='Transfer Learning on Gene Expression Data',
   author='Ben Heil',
   author_email='foomail@foo.com',
   packages=['brdnet'],
   install_requires=['requests'],
   tests_require=['pytest'],
)
