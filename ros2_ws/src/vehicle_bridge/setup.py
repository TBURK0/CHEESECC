from glob import glob

from setuptools import setup

package_name = 'vehicle_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='CHEESECC',
    maintainer_email='dev@example.com',
    description='Owns the Jetson <-> UNO Q serial link; actuator command bridge.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={'console_scripts': ['vehicle_bridge_node = vehicle_bridge.node:main']},
)
