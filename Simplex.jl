#PH Centenaro

using LinearAlgebra
using Match

include("LinearProgram.jl")

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

function pricing!(B, lp::LinearProgram, c::Vector, iB::Vector{Int}, iN::Vector{Int}, solution::Solution)
    w = B'\c[iB]
    N = @view lp.A[:, iN]
    zbar = N'w - c[iN]
    zbark, k = findmax(zbar)
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
    m = size(lp.A, 1)
    ratios = []
    for i in 1:m
        if yk[i] > 0
            push!(ratios, bbar[i]/yk[i])
        else
            push!(ratios, Inf)
        end
    end
    r = findmin(ratios)
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