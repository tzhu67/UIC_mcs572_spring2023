# dynamic_mandelbrot
from mpi4py import MPI
COMM = MPI.COMM_WORLD

def manager(p, rows, c, d):
    loadcnt = [0,]*(p-1)
    y = d; dy = (d-c)/(rows-1);
    for i in range(1,p):
        COMM.send(y,dest=i)
        y = y - dy
    jobcnt = p-1
    done = 0
    while done < p-1:
        for i in range(1,p):
            if COMM.iprobe(source=i):
                loadcnt[i-1] = loadcnt[i-1] + COMM.recv(source=i)
                if jobcnt >= rows:
                    COMM.send(3, dest=i)
                    done = done + 1
                else:
                    COMM.send(y, dest=i)
                    y = y - dy
                    jobcnt = jobcnt + 1
    print(loadcnt)

def worker(columns, a, b):
    while 1:
        load = 0;
        x = a; dx = (b-a)/(columns-1);
        y = COMM.recv(source=0)
        if y == 3:
            break
        for j in range(columns):
            load = load + iterate(x,y)
            x = x + dx
        COMM.send(load, dest=0)

def iterate(x,y):
    wx = 0.0; wy = 0.0; v = 0.0; k = 0
    while ((v < 4) and (k < 255)):
        xx = wx*wx - wy*wy
        wy = 2.0*wx*wy
        wx = xx + x
        wy = wy + y
        v = wx*wx + wy*wy
        k = k + 1
    return k

def main():
    rank = COMM.Get_rank()
    size = COMM.Get_size()
    rows = 5000; columns = 5000
    a = -2.0; b = 2.0; c = -2.0; d = 2.0
    if rank > 0:
        worker(columns, a, b)
    else:
        manager(size, rows, c, d)

main()
