#PH Centenaro

##Packages

using LinearAlgebra
using DataFrames
using Match
using BenchmarkTools


##Linear program definition

#Represents a linear programming problem in canonical form.
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

##Solution definition

#Represents a solution to a linear programming problem.
mutable struct Solution
    type::Symbol #Whether this is a minimization or maximization problem.
    conclusion::Symbol #Whether the problem is bounded, unbounded or tired.
    ray::Matrix #In case the problem is unbounded, this matrix contains the vectors of the ray sum [bbar; 0] + xk[-yk; ek].
    xB::Vector #Variable values.
    iB::Vector{Int} #Basic variable indices.
    z #Objective value at the optimal point.
    iterations::Int #Number of iterations it took to complete the solution.
        function Solution(type::Symbol, conclusion::Symbol, ray::Matrix, xB::Vector, iB::Vector{Int},
        z, iterations::Int)
        new(type, conclusion, ray, xB, iB, z, iterations)
    end
end

mutable struct TableauSolution
    tableau::Matrix #The optimal solution in tableau format.
    type::Symbol #Whether this is a minimization or maximization problem.
    conclusion::Symbol #Whether the problem is bounded, unbounded or tired.
    ray::Matrix #In case the problem is unbounded, this matrix contains the vectors of the ray sum [bbar; 0] + xk[-yk; ek].
    xB::Vector #Variable values.
    iB::Vector{Int} #Basic variable indices.
    z::Float64 #Objective value at the optimal point.
    iteration::Int #Number of iterations it took to reach this solution.

    function TableauSolution(tableau::Matrix, type::Symbol, conclusion::Symbol, ray::Matrix, xB::Vector,
        iB::Vector{Int}, z::Float64, iteration::Int)
        new(tableau, type, conclusion, ray, xB, iB, z, iteration)
    end
end

TableauSolution(tableau::Matrix, type::Symbol, conclusion::Symbol, ray::Matrix, xB::Vector, 
    iB::Vector{Int}, z::Float64, iteration::Int) =
    TableauSolution(tableau, type, conclusion, ray, xB, iB, z, iteration)

#Produces a solution to the linear programming problem specified.
#The solution contains the most important data upon completion of the simplex method.
#Optionally, this function may produce a dataframe containing data on each iteration of the method.
function simplex(lp::LinearProgram, iB::Vector{Int}, phase::Symbol, type::Symbol, anticycling::Bool, maxiter::Int)
    m = size(lp.A, 1)
    c = []
    @match phase begin
        :one => begin
            nart = length(lp.iA)
            n = size(lp.A, 2)
            c = vcat(zeros(n - nart), ones(nart))
        end
        :two => begin
            n = size(lp.A, 2) - length(lp.iA)
            c = @match type begin
                :max => -lp.c
                :min => lp.c[:]
                _ => throw(ArgumentError("Argument 'type' must be :min for minimization or :max for maximization."))
            end
        end
    end
    A = @view lp.A[:, 1:n]
    b = @view lp.b[:]
    iN = setdiff(1:n, iB)
    solution = Solution(type, :tired, [0. 0], zeros(m), iB, 0., 1)
    while solution.iterations < maxiter
        B, Blu, bbar = initialStep!(lp, c, iB, iN, solution)
        done, zbark, k = pricing!(Blu, lp, c, iB, iN, solution)
        done && break
        done, yk = unboundednessTest!(Blu, lp, iN, k, solution)
        done && break
        r = @match anticycling begin
            false => minRatioTest(bbar, yk)
            true => lexicographicRule(B, bbar, yk, iB, lp)
        end
        dantzigRule!(iB, iN, r, k)
        solution.iterations += 1
    end
    return solution
end

function initialStep!(lp::LinearProgram, c::Vector, iB::Vector{Int}, iN::Vector{Int}, solution)
    B = @view lp.A[:, iB]
    Blu = lu(B)
    solution.xB = Blu\lp.b
    solution.z = c[iB]'solution.xB
    bbar = solution.xB
    return B, Blu, bbar
end

function pricing!(B::LU, lp::LinearProgram, c::Vector, iB::Vector{Int}, iN::Vector{Int}, solution::Solution)
    w = B'\c[iB]
    N = @view lp.A[:, iN]
    zN = N'w - c[iN]
    zbark, k = findmax(zN)
    if zbark <= 0
        solution.conclusion = :bounded
        return true, nothing, nothing
    end
    return false, zbark, k
end

function unboundednessTest!(B, lp::LinearProgram, iN::Vector{Int}, k::Int, solution::Solution)
    ak = @view lp.A[:, iN[k]]
    yk = B\ak
    if maximum(yk) <= 0
        ek = zeros(n-m)
        ek[k] = 1
        ray = [bbar -yk; zeros(n-m) ek]
        val = @match type begin
            :max => Inf
            :min => -Inf
        end
        solution.conclusion = :unbounded
        return true, nothing
    end
    return false, yk
end

function minRatioTest(bbar::Vector, yk::Vector)
    ratios = replace(x -> x < 0 || isnan(x) ? Inf : x, bbar./yk)
    r = findmin(ratios)[2]
    return r
end

function lexicographicRule(B, bbar::Vector, yk::Vector, iB::Vector{Int}, lp::LinearProgram)
    m = size(lp.A, 1)
    ratios = []
    for i in 1:m
        if yk[i] > 0
            push!(ratios, bbar[i]/yk[i])
        else
            push!(ratios, Inf)
        end
    end
    minratio = minimum(ratios)
    Ij = findall(x -> x == minratio, ratios)
    j = 1
    for j in 1:length(iB)
        aj = @view lp.A[:, iB[j]]
        yj = B\aj
        ratios = []
        for i in 1:m
            if yk[i] > 0 && i in Ij
                push!(ratios, yj[i]/yk[i])
            else
                push!(ratios, Inf)
            end
        end
        minratio = minimum(ratios)
        Ij = findall(x -> x == minratio, ratios)
        if length(Ij) == 1
            break
        end
        j = j + 1
    end
    r = Ij[1]
    return r
end

function dantzigRule!(iB::Vector{Int}, iN::Vector{Int}, r::Int, k::Int)
    iB[r], iN[k] = iN[k], iB[r]
end

#Produces a first-phase BFS to the two-phase method in tableau format.
function firstPhaseTableau(lp::LinearProgram, anticycling::Symbol; maxiter::Int)
    iA = @view lp.iA[:]
    if iA == []
        return lp.iB
    end

    iter = 1

    m, n = size(lp.A)

    #Initializes pointers to the linear programming problem, to avoid duplicating data.
    A = @view lp.A[:, :]
    b = @view lp.b[:]
    c = zeros(n)
    c[iA] .= 1
    iB = lp.iB[:]
    iN = setdiff(1:n, iB)
    B = @view lp.A[:, lp.iB]
    N = @view lp.A[:, iN]
    
    c = zeros(n)
    c[iA] .= 1
    x = zeros(n)
    
    #Initializes the variables relevant to the first iteration of the tableau format solution.
    bbar = B\b
    z = c[iB]'x[iB]
    w = B'\c[iB]
    zN = N'w - c[iN]
    yN = B\A[:, iN]

    tableau = [1.; zeros(m, 1)]

    #Initial tableau construction step
    for i in 1:n
        col = []
        if i in iB
            j = findfirst(==(i), iB)
            col = [zeros(j); 1; zeros(m-j)]
        else
            j = findfirst(==(i), iN)
            zj = zN[j]
            yj = yN[:, j]
            col = [zj; yj]
        end
        tableau = [tableau col]
    end
    RHS = [z; bbar]
    tableau = [tableau RHS]

    #Defines relevant names for each part of the tableau, so as to simplify its construction.
    zi = @view tableau[1, 2:end] #Cost associated with variables xi.
    ziRHS = @view tableau[1, 2:end-1] #Cost associated with variables xi and the RHS column.
    Y = @view tableau[2:end, 2:end-1] #Matrix containing the y vectors.
    YRHS = @view tableau[2:end, 2:end] #Matrix containing the y vectors and the RHS column.

    while iter < maxiter
        #Step 1 (determine entering variable based on pricing)
        (kzN, k) = findmax(zi)
        entering = k
        if kzN <= 0
            x = B\b
            return iB
        end
        yk = Y[:, k]
        
        #Step 2 (minimum ratio test)
        ratios = replace(x -> x <= 0 ? Inf : x, YRHS[:, end]./yk)
        if anticycling == :none
            r = findmin(ratios)[2]
            iB[r] = k
        elseif anticycling == :lexicographic
            minratio = minimum(ratios)
            Ij = findall(x -> x == minratio, ratios)
            j = 1
            while length(Ij) > 1
                yj = Y[iN[j]]
                ratios = replace(x -> x < 0 || isnan(x) ? Inf : x, yj./yk)
                minratio = minimum(ratios)
                Ij = findall(x -> x == minratio, ratios)
                j = j + 1
            end
            r = Ij[1]
            leaving = iB[r]
            iB[r] = entering
            iN[k] = leaving
        end

        #Step 3 (pivoting)
        YRHS[r, :] ./= yk[r]
        for i in 1:m
            if i == r
                continue
            end
            YRHS[i, :] -= yk[i]*YRHS[r, :]
        end
        zi[:] += -kzN*YRHS[r, :]

        iter = iter + 1
    end
    return iter, tableau, :tired
end

#Produces a solution to the linear programming problem specified in the form of a tableau.
#Optionally, this function may produce an array of tableaus representing each iteration of the method.
function secondPhaseTableau(lp::LinearProgram, iB::Vector{Int}, type::Symbol, anticycling::Symbol, verbose::Bool, maxiter::Int)
    iter = 1

    #If the problem at hand is a minimization problem, t = 1. If it's a maximization problem, t = -1.
    #This coefficient t multiplies the problem's cost vector in all relevant operations.
    t = 1
    if type == :max
        t = -1
    elseif type != :min
        throw(ArgumentError("Argument 'type' must be :min for minimization or :max for maximization."))
    end
    m = size(lp.A, 1)
    n = size(lp.A, 2) - size(lp.iA, 1)

    #Initializes the indices of the basic and nonbasic variables, if they haven't been specified in the function call.
    if iB == [0]
        iB = [i for i in 1:m]
        iN = [i for i in (m+1):n]
    else
        iB = copy(iB)
        iN = setdiff(1:n, iB)
    end

    #Initializes pointers to the linear programming problem, to avoid duplicating data.
    A = @view lp.A[:, :]
    B = @view lp.A[:, iB]
    N = @view lp.A[:, iN]
    b = @view lp.b[:]
    c = @view lp.c[:]

    x = zeros(n)
    x[iB] = B\b
    
    #Initializes the variables relevant to the first iteration of the tableau format solution.
    bbar = B\b
    z = (t*c[iB])'x[iB]
    w = B'\(t*c[iB])
    zN = N'w - t*c[iN]
    yN = B\A[:, iN]

    tableau = [1.; zeros(m, 1)]

    #Initial tableau construction step
    for i in 1:n
        col = []
        if i in iB
            j = findfirst(==(i), iB)
            col = [zeros(j); 1; zeros(m-j)]
        else
            j = findfirst(==(i), iN)
            zj = zN[j]
            yj = yN[:, j]
            col = [zj; yj]
        end
        tableau = [tableau col]
    end
    RHS = [z; bbar]
    tableau = [tableau RHS]

    #Defines relevant names for each part of the tableau, so as to simplify its construction.
    zi = @view tableau[1, 2:end-1] #Cost associated with variables xi.
    ziRHS = @view tableau[1, 2:end] #Cost associated with variables xi and the RHS column.
    Y = @view tableau[2:end, 2:end-1] #Matrix containing the y vectors.
    YRHS = @view tableau[2:end, 2:end] #Matrix containing the y vectors and the RHS column.

    while iter < maxiter
        #Step 1 (determine entering variable based on pricing)
        (kzN, k) = findmax(zi)
        entering = k
        if kzN <= 0
            return TableauSolution(tableau, type, :bounded, [0. 0], YRHS[:, end], iB, z, iter)
        end
        yk = Y[:, k]
        if maximum(yk) <= 0
            ek = zeros(n-m)
            ek[k] = 1
            ray = [YRHS[:, end] -yk; zeros(n-m) ek]
            return TableauSolution(tableau, type, :unbounded, ray, YRHS[:, end], iB, -t*Inf, iter)
        end
        
        #Step 2 (minimum ratio test)
        ratios = replace(x -> x < 0 ? Inf : x, YRHS[:, end]./yk)
        if anticycling == :none
            r = findmin(ratios)[2]
            iB[r] = k
        elseif anticycling == :lexicographic
            minratio = minimum(ratios)
            Ij = findall(x -> x == minratio, ratios)
            j = 1
            while length(Ij) > 1
                yj = Y[iN[j]]
                ratios = replace(x -> x < 0 || isnan(x) ? Inf : x, yj./yk)
                minratio = minimum(ratios)
                Ij = findall(x -> x == minratio, ratios)
                j = j + 1
            end
            r = Ij[1]
            leaving = iB[r]
            iB[r] = entering
            iN[k] = leaving
        end

        #Step 3 (pivoting)
        YRHS[r, :] ./= yk[r]
        for i in 1:m
            if i == r
                continue
            end
            YRHS[i, :] -= yk[i]*YRHS[r, :]
        end
        ziRHS[:] -= kzN*YRHS[r, :]

        iter = iter + 1
    end
    return iter, tableau, :tired
end

#Solves the linear programming problem in one of the forms specified.
function solve(lp::LinearProgram; iB::Vector{Int}=Int[], type::Symbol=:min, form::Symbol=:default,
    anticycling::Bool=true, maxiter::Int=100)
    if form == :default
        if isempty(iB)
            iB = copy(lp.iB)
        end
        if !isempty(lp.iA)
            firstPhase = simplex(lp, iB, :one, type, anticycling, maxiter)
            iB = firstPhase.iB
        end
        return simplex(lp, iB, :two, type, anticycling, maxiter)
    elseif form == :tableau
        BFS = firstPhaseTableau(lp, anticycling, maxiter=size(lp.iA, 1) + 2)
        return secondPhaseTableau(lp, BFS, type, anticycling, verbose, maxiter)
    else
        throw(ArgumentError("Argument 'form' must be either :default or :tableau."))
    end
end

##Tests
##Lexicographic example
c = [-3/4, 20, -1/2, 6]
A = [1/4 -8 -1 9; 1/2 -12 -1/2 3; 0 0 1 0]
s = [:less, :less, :less]
b = [0., 0, 1]
lp = LinearProgram(c, A, s, b)
sol = solve(lp)

##Two-phase method example
c2 = [1, -2]
A2 = [1 1; -1 1; 0 1]
s2 = [:greater, :greater, :less]
b2 = [2, 1, 3]
lp2 = LinearProgram(c2, A2, s2, b2)
sol2 = solve(lp2)

##