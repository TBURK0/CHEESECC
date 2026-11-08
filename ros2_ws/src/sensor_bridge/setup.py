from glob import glob

from setuptools import setup

package_name = 'sensor_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='CHEESECC',
    maintainer_email='dev@example.com',
    description='UNO Q ultrasonic and IMU frames -> sensor_msgs Range/Imu with calibration and diagnostics.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={'console_scripts': ['sensor_bridge_node = sensor_bridge.node:main']},
)
