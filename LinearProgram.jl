#PH Centenaro

##Packages
using LinearAlgebra
using Match

##Linear program definition
#Represents a linear programming problem in canonical form. This means that every inequality constraint
#is turned into an equality constraint through the inclusion of slack variables. Furthermore, if there are
#any negative values in the right-hand-side vector b, they are turned positive, and the respective constraints
#are changed accordingly. If an initial basis vector iB is not specified by the user, artificial variables may
#be added as well. Though some values might be changed in the conversion to canonical form, the arguments
#passed to the constructor are preserved.
struct LinearProgram
    c::Vector #Cost vector.
    A::Matrix #Restrictions matrix.
    b::Vector #Right-hand side values of constraints.
    iB::Vector{Int} #Vector of initial basic variables.
    iA::Vector{Int} #Vector of artificial variables.

    function LinearProgram(c::Vector, A::Matrix, signs::Vector{Symbol}, b::Vector; iB::Vector{Int}=Int[])
        #Ensures that the problem is modeled correctly.
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
    #The following piece of code takes care of adding slack variables to the linear program.
    #Let i be the constraint row currently under analysis. Then we can define our algorithm as follows:
    #   ❤ MAIN STEP
    #       1. If i > m, stop. Return the current slack matrix.
    #       2. Verify the value of bᵢ. If it is negative, multiply bᵢ and Aᵢ by -1. If it's an equality constraint,
    #          keep it as such. Otherwise, if it's a ≥ constraint, make it ≤ and vice versa.
    #       3. Verify the constraint sign. If it's ≤, add an eᵢ (ith unit) vector to the slack matrix. If it's ≥,
    #          add an -eᵢ vector to the slack matrix. In either case, append a 0 to the cost vector.
    #       4. Increment i by one and return to step 1.
    #This procedure ensures the creation of slack variables where necessary. Note that, by adding slack variables,
    #we are producing a different description of the same problem. However, it might be the case that the slack
    #variables aren't enough to get the simplex method going. If by step 3 every constraint turns out to be a ≤
    #constraint, then the slack variables are enough to produce an acceptable initial basic feasible solution.
    #If, however, any constraints are = or ≥, then, in the absence of a user-defined initial basic feasible solution,
    #an artificial submatrix becomes necessary.
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
    #The following piece of code deals with the problem of adding artificial variables to the linear program.
    #This algorithm begins at the first element of the first slack variable column, and it ends at the last
    #element of the last slack variable column.
    #The algorithm can be formulated as follows:
    #   ❤ INITIAL STEP
    #       1. If there aren't any slack variables, set the artificial matrix to I and put all the artificial
    #          variables in the basis, in ascending order of indices. Return the artificial matrix.
    #       2. Add as many zeros to the cost coefficient vector as there are slack variables.
    #       3. Select the first slack variable column in the constraint matrix.
    #       4. Select the first row of this column.
    #       We will henceforth name the current row and column i and j, respectively.
    #       Moreover, let A be an mxn constraint matrix after the inclusion of the slack variables.
    #   ❤ MAIN STEP
    #       1. If i > m, stop. Return the current artificial matrix.
    #       2. If Aᵢⱼ = 1 and j ≠ n, increment j by one, put the slack variable in the basis and go to step 5.
    #       3. Add an eᵢ (ith unit) vector to the artificial matrix, put the artificial variable in the basis
    #          and add a zero to the cost vector.
    #       4. If Aᵢⱼ = -1 and j ≠ n, increment j by one.
    #       5. Increment i by one and return to step 1.
    #This procedure guarantees that, after the concatenation of the constraint matrix and the artificial matrix,
    #there will be at least m columns that, when organized, constitute an identity submatrix. Moreover, by
    #proceeding through the constraint matrix in an orderly fashion, the m desired variables are added to the basis
    #in precisely the order needed to produce the identity matrix. Hence no ordering is needed after the inclusion
    #of the slack variables.
    nslack = size(slack, 2)
    n = size(A, 2)
    m = size(A, 1)
    artificial = Array{Float64}(undef, m, 0)
    if nslack == 0
        artificial = I(m)
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