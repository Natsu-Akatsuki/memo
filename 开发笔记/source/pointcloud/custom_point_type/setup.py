import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion
from pathlib import Path


cur_path = Path("./build").resolve()
cur_path.mkdir(parents=True, exist_ok=True)
subprocess.check_call(['cmake', '..'], cwd="./build")
subprocess.check_call(['make'], cwd="./build")

