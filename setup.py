from setuptools import setup, find_packages

setup(
    name='learnviz',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'torch',  # Assuming we'll need PyTorch for neural network integration
    ],
    author='Edan Meyer',
    description='A tool for visualizing neural network training metrics and weights',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/edanmeyer/learnviz',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
) 