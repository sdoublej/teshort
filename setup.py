from setuptools import setup, find_packages

setup(
    name='teshort',
    version='0.0.2',
    description='Transformer-based item reduction tool',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='humansdoublej',
    author_email='humansdoublej@naver.com',
    url='https://github.com/sdoublej/teshort/master', 
    license='MIT',  # 또는 다른 라이선스
    packages=find_packages(),
    install_requires=[
        'sentence-transformers',
        'scikit-learn',
        'matplotlib',
        'pandas',
        'numpy',
        'umap-learn'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.7',
)