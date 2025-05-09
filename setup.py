from setuptools import setup, find_packages

setup(
    name='lamin_dataloader',
    version='0.1.0',
    url="https://github.com/theislab/lamin_dataloader",
    license='MIT',
    description='data loader and pre-processing for large-scale models based on lamindb',
    author='Theislab',
    packages=find_packages(),
    install_requires=[
        'lamindb==0.75.0',
        # List your other project dependencies here
    ],
)
