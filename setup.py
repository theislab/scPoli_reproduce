from pathlib import Path

from setuptools import setup, find_packages

long_description = Path('README.rst').read_text('utf-8')

try:
    from lataq_reproduce import __author__, __email__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = ''

setup(name='lataq_reproduce',
      version='0.0.1',
      description='Semi-supervised model for architecture surgery on Single-cell data',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/theislab/lataq_reproduce',
      author=__author__,
      author_email=__email__,
      license='MIT',
      packages=find_packages(),
      zip_safe=False,
      install_requires=[
          l.strip() for l in Path('requirements.txt').read_text('utf-8').splitlines()
      ],
      classifiers=[
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          'Environment :: Console',
          'Framework :: Jupyter',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          "License :: OSI Approved :: MIT License",
          'Natural Language :: English',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
      ],
      doc=[
          'sphinx',
          'sphinx_rtd_theme',
          'sphinx_autodoc_typehints',
          'typing_extensions; python_version < "3.8"',
      ],
      )