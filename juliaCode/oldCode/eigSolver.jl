include("sipdg3.jl")
#using Arpack
using IterativeSolvers
##set options in sipdg3

##orthogonalize columns
function MGS(Z)
    n = size(Z,2);

    Q = Z;
    for j=1:n
        Q[:,j] = Z[:,j] / norm(Z[:,j]);
        for k = j+1:n
            Z[:,k] = Z[:,k] - dot(Q[:,j], Z[:,k]) * Q[:,j];
        end
    end

    return Q;
end

##you dont need to check for convergence this often
##note: shift is already incorporated into A
function subspaceIteration(A, Q0)
    println("Starting subspace iteration. note shift already incorporated.")
    maxIter = 50;
    p = size(Q0,2);

    converged = false;
    i=0;
    Q = Q0;
    Y = (A ) \ Q;
    ##Y = fixedPointSolver(A, Q);
    reshape(Y,length(Y),1)
    tol=1e-10;

    while(!converged && i<=maxIter)
        ##Q,R = qr(Y); ##do MGS here
        Q = MGS(Y);
        Y = (A ) \ Q;
        ##Y = fixedPointSolver(A, Q);
        reshape(Y,length(Y),1)

        ##get ritz pairs and compute residual
        H = Q' * (A * Q);
        if(i>0 && i%1==0)
            maxRes=-1;
            println("Convergence check")
            ev = eigvals(H);
            evecs = eigvecs(H);
            for k = 1:p
                ##res = norm(A*Q*evecs[:,k] - ev[k]*Q*evecs[:,k]  );
                res = norm(A*Q - ev[k]*Q  );
                ##println("residual of ritz pair ", k, ": ", res);
                maxRes = max(maxRes,res);
            end

            println("max residual of ritz pair:", maxRes);
            if(maxRes < tol)
                converged=true
            end
        end
        i+=1;
    end

    println("Num subspace Iterations: ", i);

    return Y,Q;
end


struct Operator
    Vexact ##vector of potential values
    L ##laplacian (note this could be a SchurPrecond)

    P3d
    W3d 
    MInv3d 
    σ
end

##(Minv⊗Minv⊗Minv)*(P1d⊗P1d⊗P1d)*(W1d⊗W1d⊗W1d)*Diagonal((P1d⊗P1d⊗P1d)'*Vexact)*( (P1d⊗P1d⊗P1d)' )
function Base.:*(P::Operator, b)
    s = -P.σ * b;
    y = P.P3d' * b;
    y = Diagonal(P.P3d'*P.Vexact) * y;
    y = P.W3d*y;
    y = P.P3d*y;
    y = P.MInv3d*y

    ##apply laplace
    y = y + P.L*b;
    return y + s;
end

function LinearAlgebra.mul!(y::AbstractVector,P::Operator, b)
    y .= P*b;
    return y;
end

function Base.eltype(P::Operator)
    return Float64;
end

function Base.size(P::Operator, d)
    return size(P1d,1)^3;
end

##NOT IMPLEMENTED YET :( )
function LinearAlgebra.ldiv!( y, P::Operator, x )
	# mul!( y, P.U', x );
	# ldiv!( P.T, y );
	# mul!( y, P.U, y );
	return x
end

function LinearAlgebra.ldiv!( P::Operator, x )
	return ldiv!( x, P, x )
end

function applyPotentialAndShift( P::Operator, b)
    s = -P.σ * b;
    y = P.P3d' * b;
    y = Diagonal(P.P3d'*P.Vexact) * y;
    y = P.W3d*y;
    y = P.P3d*y;
    y = P.MInv3d*y

    return y + s;
end

function applyLaplacian( P::Operator, x)
    return P.L * x;
end

##THIS DOES NOT WORK AT ALL. probably need preconditioner
function fixedPointSolver( P::Operator, b, guess)
    println("Fixed Point Solver: ");
    tol = 1e-10;
    x = guess;
    res = norm(P*x - b); 
    println("residual: ", res);
    res = 2*tol;
    maxIter = 10;
    iter=0;
    while(res > tol && iter<maxIter)
        p = applyPotentialAndShift(P, x);
        x1 = (P.L) \ (b-p);
        res = norm(P*x1 - b); 
        x = x1;
        println("residual: ", res);
        iter+=1;
    end

    return x;
end

function Base.:\( P::Operator, b )
    return bicgstabl(P,b;reltol=1e-12,max_mv_products=100)
end


P3d = P1d⊗P1d⊗P1d;
W3d = W1d⊗W1d⊗W1d;
MInv3d = Minv⊗Minv⊗Minv;

Op = Operator(Vexact, S2, P3d, W3d, MInv3d,0);
N = n * (p+1);
N3 = n^3 * (p+1)^3;

# y = rand(N3,);
# println(norm(y));
# mul!(y,Op,bV );
# println(norm(y));
# uu = bicgstabl(Op,bV;verbose=true);

# eigenvalues are just diagonal of D3. We don't actually need to compute these,
## we have that evec for (λ_i + λ_j + λ_k) is v_i ⊗ v_j ⊗ v_k

#1d eigenvalues - we have these from diag of T
#can compute these eigvecs faster if you really want

##note the 1d L1 that we are using is inv(M)*(bilinear form) = point values of f

#=form V matrix. = ∫ ϕ_i ϕ_j v dx
interpolate v = ∑_k v(x_k) ϕ_k
for one element, we have that this is ∑_s,k ϕ_i(x_s) \phi_j(x_s) \phi_k(x_s) w_s v(x_k)
    Let Pi,s = \phi_i(x_s) W = diag(w_s), v_k = v(x_k)
    Then local V is P W diagm(P^T v_k) P^T 
    = P diagm (w_s ∑_k \phi_k(s)v(x_k)) P^T 
    = ∑_s \phi_i(x_s) (w_s ∑_k \phi_k(s)v(x_k)) \phi_j(x_s) for V_ij

    ##if we tensor a single block, this should just be a tensor product of V with the correct potential values in the middle
    ##bc we are computing 2D integrals as products of 1d integrals

    ##however, I belive we need a higher order quadrature rule for this guy.

    now for multiple elements and dimensions, what do we have.
=#
# println("forming system ")
# @time begin
#     V = (Minv⊗Minv⊗Minv)*(P1d⊗P1d⊗P1d)*(W1d⊗W1d⊗W1d)*Diagonal((P1d⊗P1d⊗P1d)'*Vexact)*( (P1d⊗P1d⊗P1d)' );
#     VMat = SparseMatrixCSC(V);
#     A = (L + VMat);
# end

# uV = A \ bV;
# println("max error from exact solution: ", maximum( abs.(uV-uexact) ) )



##lets make multiplication and backslash for this guy.


d1 = sort(diag(T));
println("compute local evecs ")
@time V1 = eigvecs(Matrix(L1));
println("these should be integers")
println(sqrt.(d1./(4*pi^2)));


num_ev=19;

vavg = sum(Vexact) / length(Vexact);

evals_comp = zeros(num_ev,1);
evecs_comp = zeros(ComplexF64,N3,num_ev);
##first do inverse iteration to find smallest eigenvalue:
##evals_comp[1], evecs_comp[:,1] = inverseIteration(A,ones(N3,1)/sqrt(N3), vavg);
##same as doing subspace iteration

# Op = Operator(Vexact, S2, P3d, W3d, MInv3d,vavg);
# Y,Q = subspaceIteration(Op,ones(N3,1)/sqrt(N3) );
# H = Q' * (Op * Q);

# evals_comp[1] = real(H[1,1]) + vavg;
# evecs_comp[:,1] = Q*real(eigvecs(H));
# println("p eval ", evals_comp[1], " val 1 ", evecs_comp[1,1]);

#now do subspace iteration for space associated with first repeated eval:
σ = d1[2] + vavg;
Z0 = zeros(N3,6);
##put either evec 2 or 3 at some spot and evec 1 at other 2 spot
k=1;
for i = 2:3
    for j=1:3
        idx=[1 1 1]; 
        idx[j]=i;
        Z0[:,k] = real(V1[:,idx[1]] ⊗ V1[:,idx[2]] ⊗ V1[:,idx[3]]);
        global k+=1;
    end
end

# Op = Operator(Vexact, S1, P3d, W3d, MInv3d,σ);
# Y,Q = subspaceIteration(A, Z0);

# H = Q' * A * Q;

# # evalsH = eigvals(H);
# # evecsH = eigvecs(H);

# evals_comp[2:7] = real(eigvals(H));
# evecs_comp[:,2:7] = Q*eigvecs(H);

# ##second repeated eval: 12 associated with this guy
# σ = 2* d1[2] + vavg;
# Z0 = zeros(N3,12);
# ##this is evec 1 in one spot and evecs 2 or 3 at other spots
# k=1;
# for i = 1:3
#     for j=2:3
#         for l = 2:3
#             idx=[1 1 1]; 
#             idx[i]=1;
#             idx[(i)%3 + 1]=j;
#             idx[(i+1)%3 + 1]=l;

#             Z0[:,k] = real(V1[:,idx[1]] ⊗ V1[:,idx[2]] ⊗ V1[:,idx[3]]);
#             global k+=1;
#         end
#     end
# end

# Op = Operator(Vexact, S1, P3d, W3d, MInv3d,σ);
# Y,Q = subspaceIteration(A, Z0);

# H = Q' * A * Q;

# evals_comp[8:19] = real(eigvals(H));
# evecs_comp[:,8:19] = Q*eigvecs(H);

# evals = eigvals(Matrix(A));
# evecs = eigvecs(Matrix(A));

# println("Eval errors: ", evals_comp - evals[1:19]) 

