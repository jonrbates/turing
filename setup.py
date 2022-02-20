from setuptools import setup, find_packages

setup(name='turing',
      version='0.1',
      description='Explore neural networks as turing machines', 
      license='Apache License 2.0',
      url='https://jonrbates.ai',
      package_dir={"": "src"},
      packages=find_packages("src"),
)