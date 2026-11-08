import os
from glob import glob
from setuptools import setup

package_name = 'autonomy'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='team',
    maintainer_email='dev@localhost',
    description='Experimental local depth mapping, scene analysis and command safety.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'perception = autonomy.perception:main',
            'fusion = autonomy.fusion:main',
            'planning = autonomy.planning:main',
            'autonomy_pipeline = autonomy.pipeline:main',
            'autonomy_sim = autonomy.simulation:main',
            'semantic_perception = autonomy.semantic_node:main',
            'wheel_imu_odometry = autonomy.odometry_node:main',
            'autonomy_physics_sim = autonomy.physics_sim:main',
        ],
    },
)
