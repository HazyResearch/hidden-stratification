from setuptools import find_packages, setup

setup(name='stratification', version='1.0', packages=find_packages())
package_data = ({"stratification": ["py.typed"], "shared": ["py.typed"]},)
