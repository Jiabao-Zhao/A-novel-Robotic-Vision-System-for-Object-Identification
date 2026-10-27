# The reviewed bound evaluated per query row

This prototype changes no reviewed proof/kernel/probe or production file. It
evaluates the same exact bound B(M,H) from DERIVATION.md with a smaller valid M
for each row, then rounds every evaluation step outward. It is not a new distance
metric or an approximate replacement for the frozen nearest-neighbor outputs.

For one query row use

    M_i = max(max(abs(original_FP64_CAD)), max(abs(original_FP64_query_row))).

Absolute value and maximum choose representable original magnitudes without
arithmetic error. They bound every CAD/query coordinate relevant to that row.
Nonfinite CAD inputs invalidate all rows. Nonfinite query coordinates, negative
or nonfinite magnitudes, M_i>2^60 or unsupported tree depth receive an infinite
threshold and cannot certify. The prototype limits supported integer H to
0..1024 as a resource guard; unsupported H must use the original query.

## Why exact interpolation is legitimate

Fix H. In the reviewed formula, the coordinate difference error is affine:

    e_d = a*M + b.

For FP32 conversion a=4u+2u^2 and b=(3+2u)*eta; for original FP64 inputs
a=2u and b=eta. Therefore

    F = (aM+b)((4+a)M+b) + u*((2+a)M+b)^2 + eta
    V = (1+u)*((2+a)M+b)^2 + eta
    A3 = u*(6+4u+u^2)*V + eta*(3+3u+u^2)
    C = (7u+2u^2)*V + (2+u)*eta
    P_(H+3) = C*sum((1+u)^k, k=0..H+2).

Every coefficient is nonnegative; each expression has degree at most two in M.
The exact combined bound is

    B(M,H) = 3F32 + A32 + 9F64 + 2A64 + P64_(H+3).

Thus B is identically a positive quadratic for all M>=0. Interpolation is done
with unrounded Fractions, using the same reviewed distance_error and recurrence:

    c = B(0,H)
    a = (B(2,H)-2B(1,H)+B(0,H))/2
    b = B(1,H)-B(0,H)-a.

All three exact coefficients are asserted nonnegative. Independent exact values
at1/3,7/5 and2^60 also detect implementation drift. These checks supplement the
degree argument; matching a few points alone would not prove the identity.

## Outward binary64 evaluation

Convert each nonnegative exact coefficient to an upper binary64 value. For
every supported row evaluate

    x2 = up(M_i*M_i)
    q  = up(a_up*x2)
    l  = up(b_up*M_i)
    s  = up(q+l)
    s  = up(s+c_up)
    threshold_i = up(s*2).

Here up(z) is the next representable binary64 toward positive infinity after
that single multiply/add. NumPy materializes each operation separately; no
unreviewed FMA or polynomial reassociation is allowed. The checked CPU state is
round-to-nearest with gradual underflow (FTZ/DAZ disabled), as in the pinned
pilot. Overflow or failed guards invalidate the row. Since all operands and
coefficients are nonnegative, each upward step is monotone and upper-bounds its
exact operation. Induction gives threshold_i >= exact2B(M_i,H), including
subnormal/zero products. Coefficient rounding is also upward, never nearest-only.

The GPU first/second difference remains downward rounded and the certificate
still requires strict gap > threshold_i. The original exact fallback and CPU
distance recomputation obligations are unchanged. New source/tests/CPU-only
analysis must pass independent review before another GPU pilot or integration.

Tests cover exact coefficient identity and outwardness over zeros, subnormal
neighbors, logarithmic/random scales, the2^60 cap and adjacent values, at tree
depths0/1/3/12/31/63/127/255. Invalid per-row inputs and unsupported depths retain
explicit fallback. The separate CPU analysis uses only16,384 deterministic
spread rows from the saved label-free dense query bank; it reads no accuracy
labels and performs no GPU operation or pose fit.
