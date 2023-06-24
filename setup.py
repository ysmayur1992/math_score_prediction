from typing import List
from setuptools import setup, find_packages

hyphen_e_dot = "-e."

def get_packages(file_path:str)->List[str]:
    requirement = []
    with open(file_path) as file_obj:
        requirement = file_obj.readlines()
        requirement = [words.replace("\n","") for words in requirement]
        if hyphen_e_dot in requirement:
            requirement.remove(hyphen_e_dot)

    return requirement

setup(
    name="Math Score Prediction",
    version='0.0.3',
    author='Yash Mayur',
    author_email='ysmayur1992@gmail.com',
    packages=find_packages(),
    install_requires=get_packages("requirements.txt")
)