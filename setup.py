import pathlib
from io import open
from os import path
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

with open(path.join(HERE, 'requirements.txt'), encoding='utf-8') as file:
    all_reqs = file.read().split('\n')

install_requires = [package.strip() for package in all_reqs
                    if ('git+' not in package) and
                    not package.startswith("#") and
                    (not package.startswith('-'))]
dependency_links = [package.strip().replace('git+', '') for package in all_reqs if 'git+' not in package]

setup(
    name='Virtual drawing board',
    description='Drawing on a virtual canvas using hand pose estimation',
    version='1.0.0',
    packages=find_packages(),  
    install_requires=install_requires,
    python_requires='>=3.7',
    license='MIT',  
    author="gursi26",
    url='https://github.com/gursi26/virtual_drawing_board',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ]
)