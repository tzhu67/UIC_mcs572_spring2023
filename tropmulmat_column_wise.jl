using Base.Threads
using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%.2f", f)

function random_ternary_matrix(nbrows::Int,nbcols::Int)
    result = zeros(nbrows, nbcols)
    for i=1:nbrows
        for j=1:nbcols
            result[i, j] = log2(abs(rand(Int) % 3))
        end
    end
    return result
end

function tropical_matrix_multiply(A::Array{Float64,2},B::Array{Float64,2})
    nbrowsA, nbcolsA = size(A)
    nbcolsB, nbrowsB = size(B)
    result = zeros(nbrowsA, nbcolsB)
    @threads for i=1:nbrowsA
        for j=1:nbcolsB
            result[i, j] = -Inf
            for k=1:nbcolsA
                result[i, j] = max(result[i, j], A[i, k] + B[j, k])
            end
        end
    end
    return result
end

function main()
    p = nthreads()
    l = parse(Int, ARGS[1])
    m = parse(Int, ARGS[2])
    n = parse(Int, ARGS[3])
    N = parse(Int, ARGS[4])
    for i = 1:N
        A = random_ternary_matrix(l, m)
        # println("The matrix A :")
        # show(stdout, "text/plain", A); println("");
        B = random_ternary_matrix(m, n)
        # println("The matrix B :")
        # show(stdout, "text/plain", transpose(B)); println("");
        C = tropical_matrix_multiply(A, B)
        # println("The matrix A*B :")
        # show(stdout, "text/plain", C); println("");
    end
end

main()