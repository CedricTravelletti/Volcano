from setuptools import setup
from setuptools import find_packages
from distutils.cmd import Command
from distutils.extension import Extension
import os
import sys
import io
import subprocess
import platform
import numpy as np
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

if "--line_trace" in sys.argv:
    line_trace = True
    print("Build with line trace enabled ...")
    sys.argv.remove("--line_trace")
else:
    line_trace = False

PACKAGE = "Volcapy"
NAME = "Volcapy"
VERSION = "0.0.2"
DESCRIPTION = "Volcapy " + VERSION
AUTHOR = "Cedric Travelletti"
AUTHOR_EMAIL = "cedrictravelletti@gmail.com"
URL = 'https://github.com/CedricTravelletti/Volcapy'


requirements = "requirements.txt"

ext_modules = [
    # "volcapy/kernels/squared_exponential.pyx",
    # "volcapy/kernels/exponential.pyx",
    "volcapy/niklas/banerjee.pyx",
    "volcapy/grid/covariance_tools.pyx",
]


def generate_extensions(ext_modules, line_trace=False):

    extensions = []

    if line_trace:
        print("define cython trace to True ...")
        define_macros = [('CYTHON_TRACE', 1), ('CYTHON_TRACE_NOGIL', 1)]
    else:
        define_macros = []

    for pyxfile in ext_modules:
        ext = Extension(name='.'.join(pyxfile.split('/'))[:-4],
                        sources=[pyxfile],
                        define_macros=define_macros)
        extensions.append(ext)
    return extensions


n_cpu = 4
ext_modules_settings = cythonize(generate_extensions(ext_modules, line_trace), 
                                 compiler_directives={'embedsignature': True, 'linetrace': line_trace}, 
                                 nthreads=n_cpu)


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    packages=find_packages(),
    include_package_data=False,
    install_requires=io.open(requirements, encoding='utf8').read(),
    classifiers=[],
    ext_modules=ext_modules_settings,
    include_dirs=[np.get_include()],
)
