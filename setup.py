from setuptools import setup, find_packages

setup(
    name="text_miner",
    version='0.2',
    author="Jose Alberto Varona Labrada",
    author_email="jovalab92@gmail.com",
    description="Full text mining utility for extract and pre-process text from documents, incluiding language detection",
    python_requires=">=3.6",
    url="https://github.com/JoseVL92/text-miner",
    download_url="https://github.com/JoseVL92/text-miner/archive/v_01.tar.gz",
    packages=find_packages(),
    data_files=[
        ("", ["LICENSE.txt", "README.md"])
    ],
    # install_requires=[]
    # before install pdftotext, poppler should be installed
    # sudo apt install build-essential libpoppler-cpp-dev pkg-config python3-dev
    # install_requires=['chardet==3.*', 'numpy', 'pdftotext', 'pycld2',
    #                   'python-magic', 'scikit-learn', 'spacy', 'textract==1.5.0'],
    extras_require={
        'vsm': ['numpy', 'scipy'],
        'extractor': ['chardet==3.*', 'pdftotext', 'python-magic', 'textract==1.5.0'],
        'nlp': ['numpy', 'pycld2', 'scikit-learn', 'spacy'],
        'full': ['chardet==3.*', 'numpy', 'pdftotext', 'pycld2',
                 'python-magic', 'scikit-learn', 'spacy', 'textract==1.5.0']
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    license='MIT',
    keywords=['text mining', 'vectorization', 'text extraction']
)
