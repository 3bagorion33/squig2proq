from setuptools import setup, find_packages

setup(
    name='squig2proq',
    version='1.0',
    packages=find_packages(),
    package_data={
        'squig2proq.data': ['Clean.ffp'],
    },
    include_package_data=True,
    install_requires=[
        'tkinterdnd2',
    ],
    entry_points={
        'console_scripts': [
            'squig2proq=squig2proq:main',
        ],
    },
)
