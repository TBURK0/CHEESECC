"""ROS Image-shaped depth buffer decoding without importing ROS."""
import struct


def decode_depth(msg):
    formats = {'16UC1': ('H', 2, 0.001), '32FC1': ('f', 4, 1.0)}
    if msg.encoding not in formats:
        raise ValueError('expected 16UC1 or 32FC1 depth')
    fmt, size, scale = formats[msg.encoding]
    if msg.width <= 0 or msg.height <= 0 or msg.step < msg.width * size:
        raise ValueError('invalid depth dimensions')
    if len(msg.data) < msg.step * msg.height:
        raise ValueError('truncated depth buffer')
    fmt = ('>' if msg.is_bigendian else '<') + str(msg.width) + fmt
    return [struct.unpack_from(fmt, msg.data, v * msg.step) for v in range(msg.height)], scale
