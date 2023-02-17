import numpy as np
from mpi4py import MPI
COMM = MPI.COMM_WORLD

def manager(dim,sz):
    mat = np.zeros((dim, dim), dtype=int)
    rowcnt = 0
    while rowcnt+sz-1 < dim:
        for k in range(1,sz):
            COMM.send(rowcnt+k-1, dest=k)
        for k in range(1,sz):
            mat[rowcnt+k-1,:] = COMM.recv(source=k)
        rowcnt = rowcnt+sz-1
    for k in range(1,dim-rowcnt+1):
        COMM.send(rowcnt+k-1, dest=k)
    for k in range(1,dim-rowcnt+1):
        mat[rowcnt+k-1,:] = COMM.recv(source=k)
    for k in range(1,sz):
        COMM.send(-1, dest=k)

def worker(dim,left,dz,roots,cff):
    from random import random,seed
    row = np.zeros((1, dim), dtype=int)
    while 1:
        i = COMM.recv(source=0)
        if i == -1:
            break
        COMM.send(row, dest=0)

def muller(f, x0, x1, x2, dxtol=1.0e-8, fxtol=1.0e-8, N=10):
    fx0 = np.polyval(f, x0)
    fx1 = np.polyval(f, x1)
    fx2 = np.polyval(f, x2)
    (root, proot, dx) = (x2, fx2, 1)
    for i in range(1, N+1):
        (h0, h1) = (x1 - x0, x2 - x1)
        (d0, d1) = ((fx1 - fx0)/h0, (fx2 - fx1)/h1)
        a = (d1 - d0)/(h1 + h0)
        b = a*h1 + d1
        c = fx2
        rad = np.sqrt(b*b - 4*a*c)
        if(abs(b + rad) > abs(b - rad)):
            den = b + rad
        else:
            den = b - rad
        dx = -2*c/den
        root = x2 + dx
        froot = np.polyval(f, root)
        if((abs(dx) < dxtol) or (abs(proot) < fxtol)):
            return (root, abs(dx), abs(froot), i)
        (x0, fx0) = (x1, fx1)
        (x1, fx1) = (x2, fx2)
        (x2, fx2) = (root, froot)
    return (root, abs(dx), abs(froot), N)

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
        worker(dim,left,dz,roots,cff)
    else:
        manager(dim,sz)

if __name__ == "__main__":
    main()
