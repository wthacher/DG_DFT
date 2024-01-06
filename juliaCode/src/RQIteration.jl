
using IterativeSolvers
include("laplacePrecond.jl");
#orthogonalize columns with respect to M IP
function MGS(Z, M =I)
    n = size(Z,2);

    Q = Z;
    for j=1:n
        # Q[:,j] = Z[:,j] / norm(Z[:,j]);
        Q[:,j] = Z[:,j] / sqrt(dot(Z[:,j], M*Z[:,j]));
        for k = j+1:n
            Z[:,k] = Z[:,k] - dot(Q[:,j], M* Z[:,k]) * Q[:,j];
        end
    end

    return Q;
end

###Lets do rayleigh quotient iteration. single vector for now
##this is bad bc we have to solve nearly singular system which is causing major problemos
##in general though, this moves pretty fast and could work for getting an inital guess?
function RQIteration(A, Q0)
    maxIter = 10;
    i=0;
    Q = Q0;
    ρOld =dot(Q, A*Q);
    println("initial guess: ",ρOld  )
    #ρOld = A.Pre.vShift;
    ##inital guess should be
    
    tol=1e-12;
    dif= 2*tol;

    while(dif>tol && i<=maxIter)
        setShift(A, ρOld);
        Q = A \ Q;
        Q = Q / norm(Q);
        ρNew = dot(Q, A*Q);
        dif = abs(ρNew-ρOld);
        println("dif = ", dif, " ρ= ",ρNew );
        ρOld=ρNew;
        i+=1;
    end

    println("Num RQ Iterations: ", i);

    return ρOld, Q;
end

mutable struct shiftedOperator
    Op::Operator
    shift::Float64
    # Mhalf::Diagonal
    # Minvhalf::Diagonal

    shiftedOperator(Op, shift) = new(Op, shift); #,Diagonal( sqrt.(diag(Op.M) ) ), Diagonal( sqrt.(diag(Op.Minv) ) )  );
end

function Base.:*(P::shiftedOperator, b)
    return P.Op*b - P.shift*b;
    #return P.Mhalf * ( (P.Op ) * (P.Minvhalf * b) ) - P.shift*b;
end

function LinearAlgebra.mul!(y::AbstractVector,P::shiftedOperator, b)
    y .= P*b;
    return y;
end

function Base.eltype(P::shiftedOperator)
    return Float64;
end

function Base.size(P::shiftedOperator, d)
    return size(P.Op, d);
end

##this is Mhalf (L+V) Minvhalf in tensor coordinates
##needs to be able to multiply and backslash
##owns a prconditioner which owns an operator
mutable struct RQOperator
    Pre::laplacePrecond
    Mhalf::Diagonal
    Minvhalf::Diagonal
    shift::Float64 
    shiftedOp::shiftedOperator

    RQOperator(Pre) = new(Pre, Diagonal( sqrt.(diag(Pre.Op.M) ) ), Diagonal( sqrt.(diag(Pre.Op.Minv) ) )  ,0, shiftedOperator(Pre.Op, 0) )   
end



##note we dont use shift here
function Base.:*(P::RQOperator, b)
    return P.Mhalf * ( (P.Pre.Op ) * (P.Minvhalf * b) ) ;
end

function LinearAlgebra.mul!(y::AbstractVector,P::RQOperator, b)
    y .= P*b;
    return y;
end

function Base.eltype(P::RQOperator)
    return Float64;
end

function Base.size(P::RQOperator, d)
    return size(P.Pre.Op, d);
end

#want to invert Mhalf (L+V - σI) Minvhalf
#invert (L+V)^-1 using bicgstabl
function Base.:\(P::RQOperator, b)
    x = P.Minvhalf * b;
    x = cg(P.shiftedOp, x; Pl=P.Pre, verbose=false,reltol=1e-14,maxiter=10);
    return P.Mhalf * x;
end

function setShift(P::RQOperator, shift)
    P.shift = shift;
    P.shiftedOp.shift = shift;
    setSigma(P.Pre, shift);
end