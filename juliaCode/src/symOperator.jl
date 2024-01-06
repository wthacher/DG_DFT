##solves eigenvalue problem (M ⊗^d M)(L + V) u = λ (M ⊗^d M) u for a few small evals
##uses multigrid preconditioner to approx (L+V)^-1

include("operator.jl");

##implements (M ⊗^d M)(L + V)
##has block versions of apply 
struct symOperator 
    Op::Operator
    PermM::Diagonal
    PermMinv::Diagonal

    symOperator(Op) = new(Op, Diagonal(Op.Perm*Op.M*(Op.PermT) ), Diagonal( (Op.Perm)*Op.Minv*Op.PermT ));
end


function Base.:*(P::symOperator, b)
    if(P.Op.block)
        return P.PermM * (P.Op * b);
    else
        return P.Op.M * (P.Op * b)
    end
    # y = P.PermM * (P.Op * b);
    # return y;
end

function LinearAlgebra.mul!(y::AbstractVector,P::symOperator, b)
    y .= P*b;
    return y;
end

##implements C = C β + AB α, overwrites C
function LinearAlgebra.mul!(C::AbstractArray, P::symOperator, B::AbstractArray, α::Number, β::Number )
    m = size(B,2);
    C .= C*β;
    #println("called with size ", m);
    for i = 1:m
        @views C[:,i] .= C[:,i] + α * (P * B[:,i] );
    end
    return C;
end

function Base.eltype(P::symOperator)
    return Float64;
end

function Base.size(P::symOperator, d)
    return size( (P.Op).Perm,1);
end

function Base.size(P::symOperator)
    return size( (P.Op).Perm);
end

function getMatrix(P::symOperator)
    A = getMatrix(P.Op);
    if(P.Op.block)
        A = P.PermM * A;
    else
        A = P.Op.M * A;
    end

    return A;
end
