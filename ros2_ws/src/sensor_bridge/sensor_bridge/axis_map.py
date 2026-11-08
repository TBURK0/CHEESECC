"""Chip axes -> base_link (REP-103: x forward, y left, z up).

The map is a 3x3 signed permutation given row-major as 9 ints: base = M @ chip.
Default is identity, which is UNVERIFIED: the real IMU mounting is unknown
(see docs/hardware.md -> Axis conventions for the tilt test that determines it).
"""


def parse_axis_map(values):
    v = [int(x) for x in values]
    if len(v) != 9 or any(x not in (-1, 0, 1) for x in v):
        raise ValueError(f'axis map must be 9 values in {{-1,0,1}}, got {values}')
    m = [v[0:3], v[3:6], v[6:9]]
    for i in range(3):
        if sum(abs(m[i][j]) for j in range(3)) != 1 or sum(abs(m[j][i]) for j in range(3)) != 1:
            raise ValueError(f'axis map must have exactly one nonzero per row and column: {values}')
    det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
           - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
           + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
    if det != 1:
        raise ValueError(f'axis map is a reflection (det={det}), not a rotation: {values}')
    return m


def apply(m, vec):
    return tuple(sum(m[i][j] * vec[j] for j in range(3)) for i in range(3))
