"""Triton IR implementations for CANN toolchains lacking external fmod."""
# Keep arithmetic in int32: this CANN compiler cannot lower uint64 promotions.
# Subtract aligned binary mantissas instead of rounding x / y; the latter
# loses the remainder for large quotients and values close to multiples.
FMOD_SOURCE = '''
@triton.jit
def _nt_ascend_fmod(x, y):
    x = tl.cast(x, tl.float32)
    y = tl.cast(y, tl.float32)
    ux = x.to(tl.int32, bitcast=True) + tl.full(y.shape, 0, tl.int32)
    uy = y.to(tl.int32, bitcast=True) + tl.full(x.shape, 0, tl.int32)
    ax = ux & 0x7fffffff
    ay = uy & 0x7fffffff
    active = (ax >= ay) & (ay != 0) & (ax < 0x7f800000)
    xs = tl.where(ax < 0x00800000, tl.abs(x) * 16777216.0, tl.abs(x))
    ys = tl.where(ay < 0x00800000, tl.abs(y) * 16777216.0, tl.abs(y))
    bx = xs.to(tl.int32, bitcast=True)
    by = ys.to(tl.int32, bitcast=True)
    ex = (bx >> 23).to(tl.int32) - tl.where(ax < 0x00800000, 24, 0)
    ey = (by >> 23).to(tl.int32) - tl.where(ay < 0x00800000, 24, 0)
    mx = tl.where(active, (bx & 0x007fffff) | 0x00800000, 0)
    my = (by & 0x007fffff) | 0x00800000
    ex = tl.where(active, ex, ey)
    while tl.sum(((ex > ey) & (mx != 0)).to(tl.int32), 0) > 0:
        step = (ex > ey) & (mx != 0)
        reduced = tl.where(mx >= my, mx - my, mx)
        mx = tl.where(step, reduced << 1, mx)
        ex = tl.where(step, ex - 1, ex)
    mx = tl.where(mx >= my, mx - my, mx)
    while tl.sum(((mx != 0) & (mx < 0x00800000)).to(tl.int32), 0) > 0:
        step = (mx != 0) & (mx < 0x00800000)
        mx = tl.where(step, mx << 1, mx)
        ey = tl.where(step, ey - 1, ey)
    normal = (mx - 0x00800000) | (ey.to(tl.int32) << 23)
    subnormal = mx >> tl.minimum(tl.maximum(1 - ey, 0), 31)
    result_bits = tl.where(ey > 0, normal, subnormal)
    result_bits = tl.where(mx == 0, 0, result_bits) | (ux & (-2147483647 - 1))
    result_bits = tl.where(active, result_bits, ux)
    invalid = (ay == 0) | (ax >= 0x7f800000) | (ay > 0x7f800000)
    result_bits = tl.where(invalid, 0x7fc00000, result_bits).to(tl.int32)
    return result_bits.to(tl.float32, bitcast=True)

@triton.jit
def _nt_ascend_copysign(x, y):
    xb = tl.cast(x, tl.float32).to(tl.int32, bitcast=True)
    yb = tl.cast(y, tl.float32).to(tl.int32, bitcast=True)
    return ((xb & 0x7fffffff) | (yb & (-2147483647 - 1))).to(tl.float32, bitcast=True)
'''
