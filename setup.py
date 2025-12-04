from setuptools import setup, find_packages

setup(
    name="youtube-intro-skipper",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "requests>=2.31.0",
        "yt-dlp>=2023.0.0",
        "librosa>=0.10.0",  # NEW: For robust audio analysis
        "scipy>=1.10.0",    # NEW: For signal correlation
        "soundfile>=0.12.0" # NEW: For fast audio loading
    ],
    extras_require={
        "speech": ["webrtcvad>=2.0.10"],
    },
    entry_points={
        "console_scripts": [
            "intro-skipper=main:main",
        ],
    },
    python_requires=">=3.8",
)