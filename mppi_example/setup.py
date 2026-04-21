import os
import sys

from setuptools import find_packages, setup

package_name = 'mppi_example'

# Older setuptools builds on the Jetson can choke on the legacy editable flag
# that colcon passes for --symlink-install. Strip it so develop/install still
# work across both environments.
while '--editable' in sys.argv:
    sys.argv.remove('--editable')

data_files = [
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),
    ('share/' + package_name, [
        'mppi_example/config.yaml',
        'mppi_example/README.md',
        'mppi_example/LICENSE',
    ]),
]

for root, _, files in os.walk('mppi_example/waypoints'):
    if files:
        destination = os.path.join('share', package_name, root.replace('mppi_example/', ''))
        data_files.append((destination, [os.path.join(root, name) for name in files]))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cedric',
    maintainer_email='cedrich@seas.upenn.edu',
    description='JAX MPPI controller for F1TENTH/Roboracer.',
    license='BSD-3-Clause',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "mppi_node = mppi_example.mppi_node:main",
        ],
    },
)
