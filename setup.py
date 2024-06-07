from setuptools import setup, find_packages

setup(
    name="TemporalOdyssey",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "gym==0.17.3",
        "numpy==1.18.5",
        "tensorflow==2.3.0",
        "matplotlib==3.2.2"
    ],
    entry_points={
        "console_scripts": [
            "temporal_odyssey=temporal_odyssey.main:main",
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="An immersive reinforcement learning project inspired by H.G. Wells' 'The Time Machine'.",
    url="https://github.com/YourUsername/Temporal-Odyssey",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)

