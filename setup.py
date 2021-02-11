import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="exercise3-group09-ml-tu-ws20",
    version="0.0.1",
    author="Matthias Eder",
    author_email="e1624856@student.tuwien.ac.at",
    description="The third exercise of group 09 for the Machine Learning course of TU Wien in the winter semester of "
                "2020/2021.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/notallheroeswearcapes/ML_Ex3",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9.0',
    py_modules=['main'],
    install_requires=[
        'Click',
    ],
    entry_points='''
        [console_scripts]
        test=main:cli
    '''
)
