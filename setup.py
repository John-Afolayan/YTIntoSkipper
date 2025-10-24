from setuptools import setup, find_packages

setup(
    name="youtube-intro-skipper",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "yt-dlp>=2023.0.0",
    ],
    extras_require={
        "speech": ["webrtcvad>=2.0.10"],
    },
    entry_points={
        "console_scripts": [
            "intro-skipper=youtube_intro_skipper.cli:main",
            "intro-skipper-cloud=youtube_intro_skipper.cloud_cli:main",
        ],
    },
    python_requires=">=3.8",
)