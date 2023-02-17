# P-1 MCS 572 due Wed 15 Feb 2023 : muller.py

"""
A basic implementation of Muller's method in Python, using NumPy.
"""

import numpy as np

def muller(f, x0, x1, x2, dxtol=1.0e-8, fxtol=1.0e-8, N=10, verbose=True):
    """
    Applies Muller's method to approximate a root of a polynomial f.

ON ENTRY :
    f          coefficients of a polynomial
    x0,x1,x2   are three distinct start points
    dxtol      tolerance on the forward error
    fxtol      tolerance on the backward error
    N          maximum number of iterations
    verbose    to print results

ON RETURN : (x, absdx, absfx, nbrit, fail)
    x          approximation for a root
    absdx      estimated forward error
    absfx      estimated backward error
    nbrit      number of iterations
    fail       true if tolerances not reached,
               false otherwise.

EXAMPLE:
    import numpy as np
    f = np.random.random(d) + np.random.random(d)*complex(0, 1)
    x0 = np.random.random() + np.random.random()*complex(0, 1)
    x1 = x0 + 1.0e-4*(np.random.random() + np.random.random()*complex(0, 1))
    x2 = x0 + 1.0e-4*(np.random.random() + np.random.random()*complex(0, 1))
    (root, err, res, nit, fail) = muller(f, x0, x1, x2)
"""
    if verbose:
        print("running Muller's method...")
        titlep1 = "        real(root)            imag(root)      "
        titlep2 = "  |dx|     |f(x)|"
        print("step : ", titlep1, titlep2)
    fx0 = np.polyval(f, x0)
    fx1 = np.polyval(f, x1)
    fx2 = np.polyval(f, x2)
    (root, proot, dx) = (x2, fx2, 1)
    for i in range(1, N+1):
        # the quadric q passes through (x0, fx0), (x1, fx1), (x2, fx2)
        (h0, h1) = (x1 - x0, x2 - x1)
        (d0, d1) = ((fx1 - fx0)/h0, (fx2 - fx1)/h1)
        # q = a*x^2 + b*x + c
        a = (d1 - d0)/(h1 + h0)
        b = a*h1 + d1
        c = fx2
        # apply the quadratic formula
        rad = np.sqrt(b*b - 4*a*c)
        # the closest root has the largest denominator
        if(abs(b + rad) > abs(b - rad)):
            den = b + rad
        else:
            den = b - rad
        dx = -2*c/den
        root = x2 + dx
        froot = np.polyval(f, root)
        if verbose:
            stri = "%3d" % i
            strreal = "%+.16e" % root.real
            strimag = "%+.16e" % root.imag
            strdx = "%.2e" % abs(dx)
            strfx = "%.2e" % abs(froot)
            print(stri, " : ", strreal, strimag, strdx, strfx)
        if((abs(dx) < dxtol) or (abs(proot) < fxtol)):
            if verbose:
                print("succeeded after", i, "steps")
            return (root, abs(dx), abs(froot), i, False)
        (x0, fx0) = (x1, fx1)
        (x1, fx1) = (x2, fx2)
        (x2, fx2) = (root, froot)
    if verbose:
        print("failed requirements after", N, "steps")
    return (root, abs(dx), abs(froot), N, True)

def test(deg, verbose=True):
    """
    Tests a basic implementation of Muller's method,
    on a polynomial with random complex coefficients of degree deg.

    If not verbose, then only the roots at the end are printed,
    otherwise each intermediate value is shown.
    """
    pol = np.random.random(deg) + np.random.random(deg)*complex(0, 1)
    pt0 = np.random.random() + np.random.random()*complex(0, 1)
    pt1 = pt0 + 1.0e-4*(np.random.random() + np.random.random()*complex(0, 1))
    pt2 = pt0 + 1.0e-4*(np.random.random() + np.random.random()*complex(0, 1))
    result = muller(pol, pt0, pt1, pt2, verbose=verbose)
    print(" approximation  :", result[0])
    print("  forward error :", result[1])
    print(" backward error :", result[2])
    print("number of steps :", result[3])
    print("         failed :", result[4])

# test(6)  # test polynomial of degree 6

def rank(roots, z, tol=1.0e-4):
    """
    Returns the position of z in the list roots.
    Two complex numbers x and y are considered equal
    if abs(x - y) < tol.
    If z does not appear in roots, then -1 is returned.
    """
    for idx in range(len(roots)):
        if abs(roots[idx] - z) < tol: return idx
    return -1

def main():
    """
    For p = x^8 - 1, makes a matrix plot of the attraction basins
    of the method of muller.
    """
    cff = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    sq2 = np.sqrt(2)/2
    roots = [complex(1), complex(sq2, sq2), complex(0, 1), \
             complex(-sq2, sq2), complex(-1), complex(-sq2, -sq2), \
             complex(0, -1), complex(sq2, -sq2)]
    dim = 501
    (left, right) = (-1.1, +1.1)
    dz = (right - left)/(dim-1)
    mat = np.zeros((dim, dim), dtype=int)
    from os import times
    from random import random
    start = times();
    for i in range(dim):
        for j in range(dim):
            z0 = complex(left + i*dz, left + j*dz)
            if(rank(roots, z0, tol=0.01) != -1):
                mat[i,j] = len(roots)
            else:
                z1 = z0 + 1.0e-4*complex(random(), random())
                z2 = z0 + 1.0e-4*complex(random(), random())
                result = muller(cff, z0, z1, z2, verbose=True)
                mat[i,j] = rank(roots, result[0])
    stop = times();
    print('user cpu time : %.4f' % (stop[0] - start[0]))
    print('  system time : %.4f' % (stop[1] - start[1]))
    print(' elapsed time : %.4f' % (stop[4] - start[4]))
    print(mat)

if __name__ == "__main__":
    main()
