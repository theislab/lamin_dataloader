from setuptools import setup, find_packages

setup(
    name='transformer_io',
    version='0.1.0',
    url="https://github.com/theislab/transformer_io",
    license='MIT',
    description='data io for large models',
    author='Theislab',
    packages=find_packages(),
    install_requires=[
        'lamindb==0.75.0',
        # List your other project dependencies here
    ],
)
