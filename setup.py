from setuptools import setup, find_packages

setup(name='aiml',
      version='1.0',
      description='ML Automation',
      author='MSDS ML',
      author_email='edavid@aim.edu',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'pandas',
          'matplotlib',
          'scikit-learn',
          'tqdm'
      ],
      )
