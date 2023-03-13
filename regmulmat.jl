using Base.Threads
using Printf
Base.show(io::IO, f::Float64) = @printf(io, "%.2f", f)

function random_ternary_matrix(nbrows::Int,nbcols::Int)
    result = zeros(nbrows, nbcols)
    for i=1:nbrows
        for j=1:nbcols
            result[i, j] = log2(abs(rand(Int) % 3) + 1)
        end
    end
    return result
end

function regular_matrix_multiply(A::Array{Float64,2},B::Array{Float64,2})
    nbrowsA, nbcolsA = size(A)
    nbrowsB, nbcolsB = size(B)
    result = zeros(nbrowsA, nbcolsB)
    @threads for i=1:nbrowsA
        for j=1:nbcolsB
            result[i, j] = 0
            for k=1:nbcolsA
                result[i, j] = result[i, j] + A[i, k] * B[k, j]
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
        A = random_ternary_matrix(3, 5)
        # println("The matrix A :")
        # show(stdout, "text/plain", A); println("");
        B = random_ternary_matrix(5, 4)
        # println("The matrix B :")
        # show(stdout, "text/plain", B); println("");
        C = regular_matrix_multiply(A, B)
        # println("The matrix A*B :")
        # show(stdout, "text/plain", C); println("");
    end
end

main()