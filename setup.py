from setuptools import setup, find_packages

setup(
    name='guitar-transcription-inhibition',
    url='https://github.com/cwitkowitz/guitar-transcription-with-inhibition',
    author='Frank Cwitkowitz',
    author_email='fcwitkow@ur.rochester.edu',
    # TODO - why does 'pip install -e' consider generated/training/inference as packages?
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=['amt_tools', 'pyguitarpro'],
    version='0.0.9',
    license='MIT',
    description='Code for guitar transcription with inhibition',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)
