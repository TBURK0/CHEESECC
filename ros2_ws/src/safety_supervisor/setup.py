from setuptools import setup, find_packages
from glob import glob
import os
setup(name='safety_supervisor', version="0.1.0", packages=find_packages(),
 data_files=[("share/ament_index/resource_index/packages", ["resource/safety_supervisor"]),
 ("share/safety_supervisor", ["package.xml"])] + [("share/safety_supervisor/"+d, glob(d+"/*")) for d in ("config", "launch", "rviz")],
 install_requires=["setuptools"], zip_safe=True, maintainer="CHEESECC maintainers", maintainer_email="maintainers@example.invalid", description="Simulation-first RC autonomy", license="Apache-2.0",
 entry_points={"console_scripts": ['safety_node = safety_supervisor.node:main']})
