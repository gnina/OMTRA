from setuptools import setup, find_packages

setup(
    name='omtra',  # Replace with the desired package name
    version='0.0.0',
    packages=find_packages(),
    include_package_data=True,
    description='a multitask model for small-molecule drug discovery',
    author='da koes group',
    license='Apache 2.0',  # Or any other license you prefer
    entry_points={
        'console_scripts': [
            'omtra=omtra.cli:main',
        ],
    },
)