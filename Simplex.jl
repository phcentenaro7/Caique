#PH Centenaro

##Packages
using LinearAlgebra
using Match

##Important files
include("LinearProgram.jl")

##Solution structure
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

##Simplex method
#Produces a solution to the linear programming problem specified.
#The solution contains the most important data upon completion of the simplex method.
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
    #The following piece of code generates variables that are very important to the simplex method.
    #This constitutes the first step of a simplex iteration.
    #The algorithm does the following:
    #   ❤ MAIN STEP
    #       1. Generate the basis matrix from the constraint matrix. Suppose that the basic variables are ordered
    #          from indices 1 to m and let A be the constraint matrix. Then we can define the basis B as
    #          B = [a₁, ..., aₘ].
    #       2. Perform LU factorization on the B matrix. This is important to speed up resolution of subsequent
    #          steps, since they rely on multiple operations upon the basis matrix.
    #       3. Generate the values of the variables in the current basis. Denoting the basis variable vector as
    #          xB, the LU-factorized basis matrix as B̄ and the right-hand-side vector as b, the basis variable
    #          vector is xB = B̄⁻¹b.
    #       4. Generate the current objective value z. Since the nonbasic variables are all equal to zero, to
    #          calculate z we need only consider basic variables and their costs. Mathematically, Denoting
    #          the cost vector for the basic variables as cB and the basic variable vector as xB, we have
    #          z = cBᵀxB.
    #       5. Generate the current basis vector b̄. This is the same as the basis variable vector xB. Though they
    #          are equal, they are symbolically different. We use b̄ as a fixed vector for the current basic variable
    #          values, subject to changes through Dantzig's rule. The vector xB, on the other hand, is used to
    #          represent the basic variables as they suffer updates.
    #       6. Return the basis matrix, the LU-factorized basis matrix and the current basis vector.
    B = @view lp.A[:, iB]
    Blu = lu(B)
    solution.xB = Blu\lp.b
    solution.z = c[iB]'solution.xB
    bbar = solution.xB
    return B, Blu, bbar
end

function pricing!(B, lp::LinearProgram, c::Vector, iB::Vector{Int}, iN::Vector{Int}, solution::Solution)
    #The following piece of code takes care of the pricing operation, which constitutes the second step of
    #the simplex method. If we rewrite the linear programming problem in terms of the nonbasic variables,
    #we get that the objective value is given by z = z₀ - Σz̄ⱼxⱼ, j ∈ J, with J being the set of
    #nonbasic variable indices and z₀ being the objective value under the current basis. As for z̄ⱼ, this is
    #given by z̄ⱼ = zⱼ - cⱼ = cBᵀB⁻¹aⱼ - cⱼ. Logically, z̄ⱼ represents the change in the objective value brought
    #by a unitary increase of xⱼ. More specifically, if we can find a z̄ⱼ such that z̄ⱼ > 0, then increasing the
    #value of xⱼ decreases the value of the objective function. This is what the pricing operation is meant to do.
    #The pricing algorithm is defined as follows:
    #   ❤ MAIN STEP
    #       1. Calculate the simplex multiplier vector w. Denoting the basic cost vector by cB and the basis matrix
    #          by B, the simplex multiplier vector is given by w = cBᵀB⁻¹.
    #       2. Calculate the cost reduction vector z̄. Let N be the nonbasic matrix and cN be the nonbasic cost vector.
    #          Then z̄ can be calculated as z̄ = Nᵀw - cN.
    #       3. Obtain z̄ₖ such that z̄ₖ = max{z̄ⱼ, j ∈ J}.
    #       4. If z̄ₖ ≤ 0, stop. We have reached an optimal basic feasible solution and the problem is bounded.
    #          Specifically, if z̄ₖ < 0, the solution found is the only optimal solution. However, if z̄ₖ = 0, then the
    #          problem has infinitely many solutions along the hyperplane over which xₖ can be increased, so long as
    #          all the other variables' nonnegativity constraints are respected.
    #       5. Return z̄ₖ and k.
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
    #The following piece of code takes care of the unboundedness test. This test constitutes the third step of
    #the simplex method, and it's needed to determine if there is no optimal basic feasible solution to the problem.
    #Essentially, if the problem's polyhedron extends indefinitely along a ray, and if this ray coincides with
    #the opposite of the cost coefficient vector (for a minimization problem), then by increasing the entering variable
    #xₖ indefinitely we can indefinitely reduce the objective value. Alternatively, we can say that z → ∞ as xₖ → ∞.
    #Algebraically, we can verify this characteristic with the following algorithm:
    #   ❤ MAIN STEP
    #       1. Generate the coefficient vector yₖ for the entering variable. This vector contains all the coefficients
    #          for the entering variable xₖ, for each of its constraints, when the problem is represented in terms of the
    #          nonbasic variables. Vector yₖ can be calculated as yₖ = B⁻¹aₖ.
    #       2. If max{yₖ} ≤ 0, stop. This is justified by the fact that we can isolate the basic variable vector in the
    #          nonbasic representation of the linear programming problem, resulting in the expression xB = b̄ - yₖxₖ.
    #          Observe that if all the elements of yₖ are ≤ 0, none of the variables in the base will ever decrease.
    #          Therefore, by increasing xₖ indefinitely, we decrease the objective function value (when minimizing)
    #          indefinitely, and the base remains compliant with the nonnegativity constraints forever.
    #       3. Return yₖ.
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
    #The following piece of code constitutes the minimum ratio test, the fourth step of the simplex method. This test
    #cannot handle problems where degeneration is an issue, so it is generally not recommended. To understand the
    #minimum ratio test, consider only the constraint lines i for which yᵢₖ > 0. The basic variables in the nonbasic
    #representation of the linear programming problem can be written as xB = b̄ - yₖxₖ. In terms of the constraint
    #lines i, this becomes xBi = b̄ᵢ - yᵢₖxₖ. We want to find the first basic variable that becomes equal to zero
    #as we raise xₖ. This variable is known as the blocking variable, and it's the one that leaves the basis.
    #The algorithm to find the blocking variable is as follows:
    #   ❤ MAIN STEP
    #       1. Calculate the vector q = b̄ᵢ/yᵢₖ.
    #       2. Return the index r of the blocking variable, such that b̄ᵣ/yᵣₖ = min{q}.
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
    #The following piece of code constitutes the lexicographic rule, a version of the fourth step of the simplex
    #method that is able to handle degeneracy issues. The reason why this algorithm works would require a lengthy
    #commentary. But it suffices to say that this method ensures that no basis is repeated as the simplex method
    #jumps from one basic feasible solution to another during degeneracy. Since there is a finite number of bases,
    #this guarantees that no cycle will be formed and the degenerate point will be left eventually.
    #Let the constraint lines n be such that yₙₖ > 0. Assuming, without loss of generality, that the basic variables
    #have indices 1 to m, the lexicographic rule is as follows:
    #   ❤ INITIAL STEP
    #       1. Let j = 1. This variable will indicate the current basic variable row in the main step.
    #   ❤ MAIN STEP
    #       1. Calculate the vector q₀ = b̄ₙ/yₙₖ.
    #       2. Let I₀ be the set of indices i such that b̄ᵢ/yᵢₖ = min{q}. If |I₀| = 1, then return r such that I₀ = {r}.
    #       3. Calculate the vector qⱼ = yᵢⱼ/yᵢₖ, with yᵢₖ > 0 and i ∈ Iⱼ₋₁.
    #       4. Let Iⱼ be the set of indices i such that yᵢⱼ/yᵢₖ = min{qⱼ}. If |Iⱼ| = 1, then return r such that Iⱼ = {r}.
    #       5. Increment j by one and repeat step 3.
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
    #This little function takes care of Dantzig's rule, according to which we must swap two variables at the end
    #of each simplex iteration, one being basic and the other being nonbasic. The variable that enters the basis, k,
    #is selected via pricing operation, while the leaving variable r is selected through the minimum ratio test or an
    #alternative method that is capable of handling cycling problems.
    iB[r], iN[k] = iN[k], iB[r]
end
