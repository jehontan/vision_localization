import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vision_localization",
    version="1.0.3",
    author="Je Hon Tan",
    author_email="jehontan@gmail.com",
    description="Vision-based localization using ArUco markers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jehontan/vision_localization",
    project_urls={
        "Bug Tracker": "https://github.com/jehontan/vision_localization/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        # 'numpy >= 1.19', # Debian Bullseye Stable python3-numpy
        # 'scipy >= 1.6', # Debian Bullseye Stable python3-scipy
        # 'opencv-contrib-python >= 4.5.1', # Debian Bullseye Stable python3-opencv
        'flask >= 2.0.2',
        'pyyaml >= 5.3.1',
        'pyzmq >= 22.1.0',
    ],
    entry_points={
        'console_scripts': [
            'localization_node=vision_localization.node:main',
            'localization_server=vision_localization.server:main',
            'camera_calibrate=vision_localization.calibrate:main',
            'offline_calibrate=vision_localization.offline_calibrate:main'
        ],
    }
)