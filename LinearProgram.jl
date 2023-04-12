#PH Centenaro

using LinearAlgebra
using Match

struct LinearProgram
    c::Vector
    A::Matrix
    b::Vector
    iB::Vector{Int}
    iA::Vector{Int}

    function LinearProgram(c::Vector, A::Matrix, signs::Vector{Symbol}, b::Vector; iB::Vector{Int}=Int[])
        A = copy(A)
        b = copy(b)
        c = copy(c)
        (nrowsA, ncolsA), sizeSigns, sizeb, sizec = size(A), size(signs, 1), size(b, 1), size(c, 1)
        if (nrowsA, nrowsA, ncolsA) != (sizeSigns, sizeb, sizec)
            throw(DimensionMismatch("Dimensions of A, b and c do not match. Have A as a $(nrowsA)x$(ncolsA) matrix and b and c as $(sizeb)- and $(sizec)-dimensional vectors, respectively."))
        end

        slack = createSlackSubmatrix!(c, A, signs, b)
        ncolsA = size(A, 2)
        iA = Int[]
        if isempty(iB)
            artificial = createArtificialSubmatrix!(c, A, slack, iB, iA)
        else
            artificial = Array{}(undef, nrowsA, 0)
            if length(iB) != nrowsA
                throw(DimensionMismatch("The number of variables in the basis must match the number of constraints."))
            end
            if !isnothing(findfirst(x -> x > ncolsA, iB))
                throw(ArgumentError("One or more indices in the basis does not exist."))
            end
        end
        A = [A slack artificial]
        new(c, A, b, iB, iA)
    end
end

function createSlackSubmatrix!(c::Vector, A::Matrix, signs::Vector{Symbol}, b::Vector)
    m = size(A, 1)
    slack = Array{Float64}(undef, m, 0)
    for i in 1:m
        if b[i] < 0
            b[i] = -b[i]
            A[i, :] = -A[i, :]
            signs[i] = @match signs[i] begin
                :less => :greater
                :greater => :less
            end
        end
        @match signs[i] begin
            :less => begin
                xS = zeros(m)
                xS[i] = 1
                slack = [slack xS]
                push!(c, 0)
            end
            :greater => begin 
                xS = zeros(m)
                xS[i] = -1
                slack = [slack xS]
                push!(c, 0)
            end
            :equal => nothing
            _ => throw(ArgumentError("Invalid constraint sign at line $i. The only acceptable values are :equal, :less and :greater."))
        end
    end
    return slack
end

function createArtificialSubmatrix!(c::Vector, A::Matrix, slack::Matrix, iB::Vector{Int}, iA::Vector{Int})
    nslack = size(slack, 2)
    n = size(A, 2)
    m = size(A, 1)
    artificial = Array{Float64}(undef, m, 0)
    if nslack == 0
        artificial = I(m)
        for i in n+1:n+m
            push!(iA, i)
            push!(iB, i)
        end
    else
        row = 1
        indexA = n + nslack + 1 #Index of the current artificial variable to be added to the base.
        for j in 1:nslack
            for i in row:m
                @match slack[i, j] begin
                    0 || -1 => begin 
                        push!(c, 0)
                        xA = zeros(m)
                        xA[i] = 1
                        artificial = [artificial xA]
                        push!(iB, indexA)
                        push!(iA, indexA)
                        indexA = indexA + 1
                        if slack[i, j] == -1
                            row = i + 1
                            if j != n + nslack
                                break
                            end
                        end
                    end
                    _ => begin
                        push!(iB, j + n)
                        row = i + 1
                        if j != n + nslack
                            break
                        end
                    end
                end
            end
        end
    end
    return artificial
end