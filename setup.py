"""
Raga AI - Yaman Error Detector

A system for analyzing Indian classical music performances to detect errors in the Yaman raga.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="raga-ai",
    version="1.0.0",
    author="Raga AI Development Team",
    author_email="info@raga-ai.example.com",
    description="An advanced system for analyzing Indian classical music performances, specifically designed to detect errors in the Yaman raga.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/raga-ai/yaman-error-detector",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "raga-ai=main:main",
        ],
    },
    keywords="indian-classical-music, raga-analysis, audio-processing, music-information-retrieval, pitch-estimation",
    project_urls={
        "Bug Reports": "https://github.com/raga-ai/yaman-error-detector/issues",
        "Source": "https://github.com/raga-ai/yaman-error-detector",
        "Documentation": "https://github.com/raga-ai/yaman-error-detector/tree/main/docs",
    },
)