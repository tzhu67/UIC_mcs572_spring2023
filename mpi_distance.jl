using MPI
using Distributions

function d(t)
    return rand(Uniform(0,1))
end

MPI.Init()
comm = MPI.COMM_WORLD
myid = MPI.Comm_rank(comm)
p = MPI.Comm_size(comm)

n = 16
dn = n/(p-1)
dt = 2*pi/n

if myid == 0
	for i in 0:p-2
		MPI.send(i*dn*dt, i+1, 0, comm)
		# println("Manager sent angles to worker $(i+1) beginning with $(i*dn*dt).")
	end
else
	t0 = MPI.recv(0, 0, comm)
	t0 = t0[1]
	for i in 0:dn-1
		dist = d(t0+i*dt)
		# println("Worker $myid received angle $(t0+i*dt) from manager.")
	end
end

MPI.Finalize()