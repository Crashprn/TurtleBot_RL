import os

from setuptools import find_namespace_packages, find_packages, setup

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + "/requirements.txt"

install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setup(name="TurtleBot_RL",
      version="0.1",
      description="Turtle bot reinforcement learning package",
      author="Cody Grogan, Jarom Burr",
      install_requires=install_requires,
      packages=find_namespace_packages(include=["TurtleBot_RL.*"]),
      zip_safe=False)