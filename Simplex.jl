#PH Centenaro

using LinearAlgebra
using Match

include("LinearProgram.jl")

mutable struct Solution
    type::Symbol #Whether this is a minimization or maximization problem.
    conclusion::Symbol #Whether the problem is bounded, unbounded or tired.
    ray::Matrix #In case the problem is unbounded, this matrix contains the vectors of the ray sum [bbar; 0] + xk[-yk; ek].
    xB::Vector{Real} #Variable values.
    iB::Vector{Int} #Basic variable indices.
    z::Real #Objective value at the optimal point.
    iterations::Int #Real of iterations it took to complete the solution.
        function Solution(type::Symbol, conclusion::Symbol, ray::Matrix, xB::Vector, iB::Vector{Int},
        z, iterations::Int)
        new(type, conclusion, ray, xB, iB, z, iterations)
    end
end

mutable struct SimplexData
    A::Matrix{Real}
    b::Vector{Real}
    m::Int
    n::Int
    c::Vector{Real}
    z::Real
    k::Int
    r::Int
    yk::Vector{Real}
    iB::Vector{Int}
    iN::Vector{Int}
    iA::Vector{Int}
    xB::Vector{Real}
    bbar::Vector{Real}
    Blu::LU
    iter::Int
    anticycling::Bool
    type::Symbol
    solution::Solution
    
    function SimplexData()
        new()
    end
end

function simplex(lp::LinearProgram, iB::Vector{Int}, type::Symbol, anticycling::Bool, maxiter::Int)
    data = SimplexData()
    initializeSimplex!(lp, iB, type, anticycling, data)
    while data.iter < maxiter
        initialStep!(data)
        pricing!(data) && return data.solution
        unboundednessTest!(data) && return data.solution
        blockingVariableSelection!(data)
        dantzigRule!(data)
    end
    return data.solution
end

function initializeSimplex!(lp::LinearProgram, iB::Vector{Int}, type::Symbol, anticycling::Bool, d::SimplexData)
    d.m = size(lp.A, 1)
    @match type begin
        :firstPhase => begin
            nart = length(lp.iA)
            d.n = size(lp.A, 2)
            d.c = vcat(zeros(d.n - nart), ones(nart))
        end
        :min => begin
            d.n = size(lp.A, 2) - length(lp.iA)
            d.c = lp.c[:]
        end
        :max => begin
            d.n = size(lp.A, 2) - length(lp.iA)
            d.c = -lp.c[:]
        end
        _ => throw(ArgumentError("Argument 'type' must be :min for minimization, :max for maximization or :firstPhase for a first-phase procedure."))
    end
    d.A = @view lp.A[:, 1:d.n]
    d.b = @view lp.b[:]
    d.iA = @view lp.iA[:]
    d.iB = copy(iB)
    d.iN = setdiff(1:d.n, d.iB)
    d.anticycling = anticycling
    d.type = type
    d.solution = Solution(type, :tired, [0. 0], Real[], iB, 0, 1)
    d.iter = 1
end

function initialStep!(d::SimplexData)
    B = d.A[:, d.iB]
    d.Blu = lu(B)
    d.xB = d.Blu\d.b
    d.z = d.c[d.iB]'d.xB
    d.bbar = d.xB
end

function pricing!(d::SimplexData)
    w = d.Blu'\d.c[d.iB]
    N = @view d.A[:, d.iN]
    zbar = N'w - d.c[d.iN]
    zbark, d.k = findmax(zbar)
    if zbark <= 0
        @match d.type begin
            :firstPhase => begin
                artInBase = findall(x -> x in d.iA, d.iB)
                if !isnothing(findfirst(x -> x > 0, d.xB[artInBase]))
                    d.solution.conclusion = :unfeasible
                else
                    d.solution.conclusion = :feasible
                end
            end
            _ => begin
                d.solution.conclusion = :bounded
            end
        end
        d.solution.xB = d.xB
        d.solution.iB = d.iB
        d.solution.z = d.z
        d.solution.iterations = d.iter
        return true
    end
    return false
end

function unboundednessTest!(d::SimplexData)
    ak = d.A[:, d.iN[d.k]]
    d.yk = d.Blu\ak
    if maximum(d.yk) <= 0
        ek = zeros(d.n-d.m)
        ek[d.k] = 1
        d.solution.conclusion = :unbounded
        d.solution.ray = [d.bbar -d.yk; zeros(d.n-d.m) ek]
        d.solution.z = @match d.type begin
            :max => Inf
            :min => -Inf
        end
        d.solution.xB = d.xB
        d.solution.iB = d.iB
        d.solution.iterations = d.iter
        return true
    end
    return false
end

function blockingVariableSelection!(d::SimplexData)
    @match d.anticycling begin
        true => lexicographicRule!(d)
        false => minRatioTest!(d)
    end
end

function minRatioTest!(d::SimplexData)
    ratios = []
    for i in 1:d.m
        if d.yk[i] > 0
            push!(ratios, d.bbar[i]/d.yk[i])
        else
            push!(ratios, Inf)
        end
    end
    d.r = findmin(ratios)[2]
end

function lexicographicRule!(d::SimplexData)
    ratios = []
    for i in 1:d.m
        if d.yk[i] > 0
            push!(ratios, d.bbar[i]/d.yk[i])
        else
            push!(ratios, Inf)
        end
    end
    minratio = minimum(ratios)
    Ij = findall(x -> x == minratio, ratios)
    j = 1
    while(length(Ij) > 1)
        aj = @view d.A[:, d.iB[j]]
        yj = d.Blu\aj
        ratios = []
        for i in 1:d.m
            if d.yk[i] > 0 && i in Ij
                push!(ratios, yj[i]/d.yk[i])
            else
                push!(ratios, Inf)
            end
        end
        minratio = minimum(ratios)
        Ij = findall(x -> x == minratio, ratios)
        j = j + 1
    end
    d.r = Ij[1]
end

function dantzigRule!(d::SimplexData)
    d.iB[d.r], d.iN[d.k] = d.iN[d.k], d.iB[d.r]
    d.iter += 1
end