# static_muller.py
import numpy as np
from mpi4py import MPI
COMM = MPI.COMM_WORLD

def manager(dim,sz):
    from os import times
    mat = np.zeros((dim, dim), dtype=int)
    colcnt = 0
    start = times();
    while colcnt+sz-1 < dim:
        for k in range(1,sz):
            COMM.send(colcnt+k-1, dest=k)
        for k in range(1,sz):
            mat[colcnt+k-1,:] = COMM.recv(colcnt+k-1, dest=k)
        colcnt = colcnt+sz-1
    for k in range(1,dim-colcnt):
        COMM.send(colcnt+k-1, dest=k)
    for k in range(1,dim-colcnt):
        mat[colcnt+k-1,:] = COMM.recv(colcnt+k-1, dest=k)
    for k in range(1,sz):
        COMM.send(-1, dest=k)
    stop = times();
    print('user cpu time : %.4f' % (stop[0] - start[0]))
    print('  system time : %.4f' % (stop[1] - start[1]))
    print(' elapsed time : %.4f' % (stop[4] - start[4]))
    print(mat)

def worker(dim,left,dz,roots,tol,cff):
    from random import random
    row = np.zeros((1, dim), dtype=int)
    while 1:
        j = COMM.recv(source=0)
        if j == -1:
            break
        for i in range(dim):
            z0 = complex(left + i*dz, left + j*dz)
            if(rank(roots, z0, tol=0.01) != -1):
                mat[i] = len(roots)
            else:
                z1 = z0 + 1.0e-4*complex(random(), random())
                z2 = z0 + 1.0e-4*complex(random(), random())
                result = muller(cff, z0, z1, z2, verbose=True)
                row[i] = rank(roots, result[0])
    COMM.send(row, dest=0)

def muller(f, x0, x1, x2, dxtol=1.0e-8, fxtol=1.0e-8, N=10, verbose=True):
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

def rank(roots, z, tol=1.0e-4):
    for idx in range(len(roots)):
        if abs(roots[idx] - z) < tol: return idx
    return -1

def main():
    rk = COMM.Get_rank()
    sz = COMM.Get_size()
    cff = np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    sq2 = np.sqrt(2)/2
    roots = [complex(1), complex(sq2, sq2), complex(0, 1), \
             complex(-sq2, sq2), complex(-1), complex(-sq2, -sq2), \
             complex(0, -1), complex(sq2, -sq2)]
    dim = 501
    (left, right) = (-1.1, +1.1)
    dz = (right - left)/(dim-1)
    if rk > 0:
        worker(dim,left,dz,roots,tol,cff)
    else:
        manager(dim,sz)

if __name__ == "__main__":
    main()
