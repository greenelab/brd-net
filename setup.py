from setuptools import setup

test_dependencies = ['pytest']

setup(
   name='brdnet',
   version='1.0',
   description='Transfer Learning on Gene Expression Data',
   author='Ben Heil',
   author_email='ben.jer.heil@gmail.com',
   packages=['brdnet'],
   install_requires=['requests'],
   tests_require=test_dependencies,

   extras_require={
       'test': test_dependencies,
   },
)
