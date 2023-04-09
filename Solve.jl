include("Simplex.jl")

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
    end
end