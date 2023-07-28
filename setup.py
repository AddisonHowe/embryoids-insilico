from setuptools import setup

setup(
    name="insilemb",
    version="0.1.0",
    packages=["insilemb"],
    entry_points={
        "console_scripts": [
            "insilemb = insilemb.main:main",
            "voronoi = insilemb.voronoi:main",
        ]
    },
)
