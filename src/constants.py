import numpy as np


NON_ZERO_TOLERANCE = 1e-6
"""Default value for setting fluxes to zero. If the absolute value of a flux is 
below this number, it is considered to be zero."""


EPS = 1e-9
"""Default vale for safe division."""


FLOAT_ROUNDING = 6
"""Number of decimal points to round all floats for file safety."""


MAX_BRETL_ITERATIONS = 1000
"""Maximum number of iterations allowed to be used by Bretl polytope sampling algorithm."""


CATEGORY_DICT = {
    -3.0: '-', 
    -2.0: '--',
    -1.0: '-0',
    0.0: '0',
    1.0: '0+',
    2.0: '+',
    3.0: '++',
    4.0: '-+',
    5.0: 'err',
    100.0: 'var'
    }
"""Dictionary that maps numeric qualitative codes to symbolic labels. The numeric codes 
are produced by `qual_translate` and represent qualitative flux variability states derived 
from FVA minimum/maximum values."""


Numerical = int | float

Floating = np.ndarray | float