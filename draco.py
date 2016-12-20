"""
This code computes finite difference coefficients of arbitrarily-biased stencils on uniform grids.
While finite difference coefficient tables on Wikipedia are useful, they are only provided for
centered and pure forward/backward differences.
With draco, one can compute coefficients for stencils made of, for instance, two 'forward' points
and three 'backward' points.

Users input two things through the command line, in the following order:

1. 'stencil' - the grid offsets used in the derivative approximation
2. 'derivativeOrder' - the order of the derivative to be approximated

The stencil vector must have unique entries.
For example, [-1, 0, +1] uses a grid point and its two adjacent neighbors (a centered difference).
Similarly the stencil [-1, 0, +1, +2] uses a grid point and its left neighbor and two right neighbors.
The stencil [-2, -1, 0] will make a backward difference and [0, +1, +2] will make a forward difference.

The code will write out two things:

1. The coefficients used to approximate the derivative
2. The order of accuracy of the approximation
"""

import numpy as np
from scipy.misc import factorial
from fractions import Fraction
import argparse

parser = argparse.ArgumentParser(description='Compute finite difference coefficients on '
                                             'uniform grids with arbitrarily-biased stencils. '
                                             'Ex: python3 draco.py --derorder=1 --stencil=[-1,0,+1]')
parser.add_argument('--derorder', dest='derOrder', action='store',
                    help='The desired derivative order')
parser.add_argument('--stencil', dest='stencil', action='store',
                    help='The stencil, written in grid offsets, e.g. [-1,0,+1]')
args = parser.parse_args()

k = int(args.derOrder)
x = eval(args.stencil)
n = len(x)
A = np.ones((n, n))
b = np.zeros((n, 1))
b[k] = 1

for rowIdx in np.arange(2, n+1):
    for colIdx in np.arange(1, n+1):
        A[rowIdx-1, colIdx-1] = x[colIdx-1]**(rowIdx-1) / factorial(rowIdx-1)

coefficients = np.linalg.solve(A, b)

expansions = []

p = 17  # check to 17th order
orderCheckSum = np.zeros((1, p))

for j in np.arange(0, n):
    xj = x[j]
    aj = coefficients[j][0]
    expansionCoefficients = []
    for i in np.arange(0, p):
        expansionCoefficients.append(aj*xj**i/factorial(i))
    expansions.append(expansionCoefficients)
    orderCheckSum += expansionCoefficients

order = 0
for i in np.arange(k+1, p):
    if np.abs(orderCheckSum[0][i]) > 1e-14:
        order = i-k
        break

print('\n\norder of accuracy:', str(order), '\n')
print('grid offset | coefficient\n-------------------------')
for i, c in enumerate(coefficients):
    xloc = str(x[i])
    coeff = str(Fraction.from_float(c[0]).limit_denominator())
    print('    {0:7} |    {1}'.format(xloc, coeff))
print('\n\n')
