# coding=utf-8

print('''
python setup.py install
python setup.py bdist_wheel
''')

from setuptools import setup, find_packages
import os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wycv import __version__

appname = "wycv"
version = __version__

packages = find_packages()
readme = appname + ' ' + version
install_requires = []

setup(
    name=appname,
    version=version,
    description=(
        '''%s''' % appname
    ),

    author='yewei.song',
    author_email='yewei.song@micro-i.com.cn',
    maintainer='yewei.song',
    maintainer_email='yewei.song@micro-i.com.cn',
    packages=packages,
    platforms='any',
    url='',
    classifiers=[
        'Programming Language :: Python',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries'
    ],
    install_requires=install_requires,
    include_package_data=True,

    long_description=readme,
    long_description_content_type='text/markdown'
)
