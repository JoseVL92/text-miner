from setuptools import setup, find_packages

setup(
    name="text_miner",
    version=0.1,
    author="Jose Alberto Varona Labrada",
    author_email="jovalab92@gmail.com",
    description="Full text mining utility for extract and pre-process text from documents, incluiding language detection",
    python_requires=">=3.5",
    url="https://github.com/JoseVL92/text-miner",
    download_url="https://github.com/JoseVL92/text-miner/archive/v_01.tar.gz",
    packages=find_packages(),
    data_files=[
        ("", ["LICENSE.txt", "README.md"])
    ],
    install_requires=['python-magic', 'textract', 'chardet', 'pycld2', 'spacy', 'numpy', 'sklearn', 'nltk'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    license='MIT',
    keywords=['text mining', 'tf-idf']
)
