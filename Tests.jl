##
using BenchmarkTools
include("LinearProgram.jl")
include("Simplex.jl")
include("Solve.jl")
##

#Bazaraa
##Example 4.5
begin
    c = [-1, 2, -3]
    A = [1 1 1; -1 1 2; 0 2 3; 0 0 1]
    s = [:equal, :equal, :equal, :less]
    b = [6, 4, 10, 2]
    lp = LinearProgram(c, A, s, b)
    sol = solve(lp)
end

#Linear Programming With Matlab
##Example 3-4-1
begin
    c = [4, 5]
    A = [1 1; 1 2; 4 2; -1 -1; -1 1]
    s = repeat([:greater], 5)
    b = [-1, 1, 8, -3, 1]
    lp = LinearProgram(c, A, s, b)
    sol = solve(lp, type=:firstPhase)
end

##Exercise 3-4-2.1
begin
    c = [3, 1]
    A = [-1 -1; 2 2]
    s = repeat([:greater], 2)
    b = [-2, 10]
    lp = LinearProgram(c, A, s, b)
    sol = solve(lp)
end

##Exercise 3-4-2.2
begin
    c = [-1, 1]
    A = [2 -1; 1 2]
    s = repeat([:greater], 2)
    b = [1, 2]
    lp = LinearProgram(c, A, s, b)
    sol = solve(lp)
end

##Exercise 3-4-3.1
begin
    c = [-2, -4, -1, -1]
    A = [-1 -3 0 -1
         -2 -1 0  0
         0 -1 -4 -1
         1 1 2 0
         -1 1 4 0]
    s = repeat([:greater], 5)
    b = [-4, -3, -3, 1, 1]
    lp = LinearProgram(c, A, s, b)
    sol = solve(lp)
end

##Exercise 3-4-3.2
begin
    c = [-1, -3, 0]
    A = [-1 0 -1; -1 1 0; 0 -1 1]
    s = repeat([:greater], 3)
    b = [2, -1, 3]
    lp = LinearProgram(c, A, s, b)
    sol = solve(lp)
end

##Exercise 3-4-4
begin
    c = [2, 3, 6, 4]
    A = [1 2 3 1; 1 1 2 3]
    s = [:greater, :greater]
    b = [5, 3]
    lp = LinearProgram(c, A, s, b)
    sol = solve(lp)
end