from setuptools import find_packages, setup

setup(
    name="mem0rylol",
    version="0.2.1",  # Incremented version number
    packages=find_packages(exclude=["tests*"]),
    install_requires=[
        "langchain>=0.2.16",
        "langchain-cerebras>=0.1.0",
        "langchain-google-genai>=1.0.10",
        "lancedb>=0.13.0",
        "langchain-community>=0.2.16",
        "pydantic>=2.9.1",
        "langsmith>=0.1.117",
    ],
    extras_require={
        "dev": [
            "black",
            "isort",
            "pytest",
            "pytest-asyncio",
            "mypy",
            "pytest-cov",
            "bandit",
            "Sphinx",
            "sphinx-autobuild",
            "pre-commit",
            "twine",
            "tox",
        ],
    },
    author="toeknee",
    description="A sophisticated AI memory layer.",
    long_description=open("README.rst").read(),
    long_description_content_type="text/x-rst",
    license="GNU",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
)
