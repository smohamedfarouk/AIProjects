from pathlib import Path
from setuptools import setup, find_packages

BASE_DIR = Path(__file__).resolve().parent

requirements = [
    line.strip()
    for line in (BASE_DIR / "requirement.txt").read_text().splitlines()
    if line.strip() and not line.strip().startswith("#")
]

setup(
    name="LLMOps",
    version="0.1.0",
    packages=find_packages(),
    description="A package for LLM operations",
    author="Mohamed Farouk Sherefudheen",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/LLMOps",
    install_requires=requirements,
    python_requires=">=3.10",
)