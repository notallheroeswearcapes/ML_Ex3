import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def read_requirements():
    with open('requirements.txt') as req:
        content = req.read()
        requirements = content.split('\n')
    return requirements


setuptools.setup(
    name="deeplearn",
    version="1.0",
    author="Matthias Eder",
    author_email="e1624856@student.tuwien.ac.at",
    description="The third exercise of group 09 for the Machine Learning course of TU Wien in the "
                "winter semester of 2020/2021.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/notallheroeswearcapes/ML_Ex3",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8.0',
    include_package_data=True,
    install_requires=read_requirements(),
    entry_points='''
        [console_scripts]
        deeplearn=deeplearn.cli:cli
    '''
)
