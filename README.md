# Vision-based localization using ArUco markers

This package implements a localization systtem using ArUco markers.

Markers may be fixed or roaming. Cameras view the markers and calculate
roaming marker poses relative to fixed markers. Using fixed marker poses
relative to world the roaming markers world pose is known.

Each camera must see at least 1 fixed marker.

Multiple cameras can be used, and pose information is published over ZMQ
by each camera node. A server subscribes to all the pose information and
makes it available over a HTTP API.

## Installation

The same package can be installed on camera nodes and server.

```sh
pip install -e .
```

## Usage

A configuration file needs to be provided to camera nodes and server.
An example is provided in `/config/config.yaml`.

Once installed the package provides 2 entrypoints, one for camera node
and one for server.

To run on camera node:

```sh
localization_node --log <log-level> <config_file> <camera_name>
```

To run on localization server:

```sh
localization_server --log <log_level> <config_file>
```