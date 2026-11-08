from setuptools import setup, find_packages
from glob import glob
import os
setup(name='localization', version="0.1.0", packages=find_packages(),
 data_files=[("share/ament_index/resource_index/packages", ["resource/localization"]),
 ("share/localization", ["package.xml"])] + [("share/localization/"+d, glob(d+"/*")) for d in ("config", "launch", "rviz")],
 install_requires=["setuptools"], zip_safe=True, maintainer="CHEESECC maintainers", maintainer_email="maintainers@example.invalid", description="Simulation-first RC autonomy", license="Apache-2.0",
 entry_points={"console_scripts": ['phone_ingestion = localization.phone_node:main']})
