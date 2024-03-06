from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os

from setuptools import setup, find_packages

install_requires = [
    "transformers>=4.26.1",
    "numpy>=1.23.4",
    "pandas>=1.5.2",
    "scikit_learn>=1.1.3",
    "tensorboard>=2.10.0",
    "tqdm>=4.64.1",
    "jieba>=0.42.1",
    "gensim>=4.2.0",
    "pillow>=9.3.0",
    "nltk>=3.7",
    "sphinx-markdown-tables>=0.0.17"
]

setup_requires = []

classifiers = ["License :: OSI Approved :: MIT License"]

# Readthedocs requires Sphinx extensions to be specified as part of
# install_requires in order to build properly.
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if on_rtd:
    install_requires.extend(setup_requires)

filepath = os.path.join(os.path.dirname(__file__), "README.md")

setup(
    name="faknow",
    version="0.0.3",  # edit faknow/__init__.py in response
    description="A unified library for fake news detection.",
    long_description=open(filepath, encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/NPURG/FaKnow",
    author="NPURG",
    author_email="faknow@outlook.com",
    python_requires=">=3.8.0",
    packages=[package for package in find_packages() if package.startswith("faknow")],
    include_package_data=True,  # include files in MANIFEST.in
    install_requires=install_requires,
    setup_requires=setup_requires,
    zip_safe=False,
    classifiers=classifiers,
)
