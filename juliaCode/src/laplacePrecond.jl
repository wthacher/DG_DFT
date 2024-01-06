include("symOperator.jl");

##inverts (M ⊗^d M)(laplacian + (vShift-σ)I) exactly as preconditioner if sym
##else invert (laplacian + σI)
##note works in block coords
mutable struct laplacePrecond
    Op::Operator
    vShift::Float64
    σ::Float64
    DiagInv::Diagonal
    sym::Bool

    laplacePrecond(Op, vShift, σ, sym) = new(Op, vShift, σ, (vShift-σ)==0 ? Op.L.Dinv : Diagonal( 1 ./ (diag(Op.L.D) .+  (vShift-σ)) ), sym  )
end

##apply P (L + σI)^-1(Minv)(PermT)
##or  P (L + σI)^-1 (PermT)
function LinearAlgebra.ldiv!( Y::AbstractArray, P::laplacePrecond, B::AbstractArray )
    if(P.Op.block)
        Y .= (P.Op.PermT * B);
        if(P.sym)
            Y .= (P.Op.Minv) *  Y;
        end
        Y .= P.Op.L.Vinv * Y;
        Y .= P.DiagInv * Y;
        Y .= P.Op.L.V * Y;
        Y .= P.Op.Perm * Y;
    else
        Y .= B;
        if(P.sym) ##apply Minv
            Y .= (P.Op.Minv) *  Y;
        end
        ##apply Linv
        Y .= P.Op.L.Vinv * Y;
        Y .= P.DiagInv * Y;
        Y .= P.Op.L.V * Y;
    end

	return Y;
end

#put answer in x
function LinearAlgebra.ldiv!( P::laplacePrecond, x )
    # y = 0.0*x;
	# z = ldiv!( y, P, x );
    # x .= z;
    ldiv!(x, P, x );
end

#do backslash
function Base.:\( P::laplacePrecond, b )
    return ldiv!( P, b );
end

function setSigma(P::laplacePrecond, sigma)
    P.σ = sigma;
    P.DiagInv = (P.vShift-sigma)==0 ? P.Op.L.Dinv : Diagonal( 1 ./ (diag(P.Op.L.D) .+ (P.vShift - P.σ) ) )
end

#######LOW RANK VERSION for initial guesses
mutable struct lowRankLaplacePrecond
    Op::Operator
    vShift::Float64
    σ::Float64
    DiagInv::Diagonal
    V ##low rank evecs
    VT
    Mhalf
    Minvhalf
    sym::Bool
    r::Int64

    function lowRankLaplacePrecond(Op, vShift, σ, sym, r)
        L1 = Op.L1;
        M1 = Op.M1;

        Mhalf_L_Minvhalf = Diagonal(sqrt.(diag(M1))) * L1 * Diagonal(sqrt.(1 ./ diag(M1)))
	    ev = eigen(Symmetric(Matrix(Mhalf_L_Minvhalf)));
        V = ev.vectors[:,1:r];
        VT = V';
        D = ev.values[1:r];
        V = V ⊗ V ⊗ V;
        VT = VT ⊗ VT ⊗ VT;
        D = Diagonal( Diagonal(D) ⊕ Diagonal(D) ⊕ Diagonal(D)  );
        Dinv = Diagonal( 1 ./ diag(D) );
        if(abs(ev.values[1]) < 1e-10)
            Dinv[1,1] = 0;
        end

        if(vShift-σ)!=0 
            Dinv = Diagonal( 1 ./ ( diag(D) .+ (vShift-σ) ) );
        end

        println("creating rank ", r, " preconditioner")
        new(Op, vShift, σ, 
        Dinv, V, VT, Diagonal(sqrt.(diag(Op.M) ) ), Diagonal( 1 ./ (sqrt.(diag(Op.M) )) ), 
        sym, r  )
    end

    # lowRankLaplacePrecond(Op, vShift, σ, sym, r) = new(Op, vShift, σ, 
    #     (vShift-σ)==0 ? Op.L.Dinv : Diagonal( 1 ./ (diag(Op.L.D) .+  (vShift-σ)) ), 
    #     sym, r  )
end

##if solving M(L+V), we need to do Minvhalf then precond then Minvhalf
##if solving (L+V), do Mhalf then precond then Minvhalf
function LinearAlgebra.ldiv!( Y::AbstractArray, P::lowRankLaplacePrecond, B::AbstractArray )
    if(P.Op.block)
        X = (P.Op.PermT * B);
        if(P.sym)
            X = (P.Minvhalf) *  X;
        else
            X = (P.Mhalf) * X;
        end
        X = P.VT * X;
        X = P.DiagInv * X;
        X = P.V * X;
        X = P.Minvhalf * X; ##for symmetric or not
        X = P.Op.Perm * X;
        Y .= X;
    else
        X = copy(B);
        if(P.sym)
            X = (P.Minvhalf) *  X;
        else
            X = (P.Mhalf) * X;
        end
        X = P.VT * X;
        X = P.DiagInv * X;
        X = P.V * X;
        X = P.Minvhalf * X; ##for symmetric or not we need this
        Y .= X;
    end

	return Y;
end

#put answer in x
function LinearAlgebra.ldiv!( P::lowRankLaplacePrecond, x )
    # y = 0.0*x;
	# z = ldiv!( y, P, x );
    # x .= z;
    ldiv!(x, P, x );
end

#do backslash
function Base.:\( P::lowRankLaplacePrecond, b )
    return ldiv!( P, b );
end

# function setSigma(P::lowRankLaplacePrecond, sigma)
#     P.σ = sigma;
#     P.DiagInv = (P.vShift-sigma)==0 ? P.Op.L.Dinv : Diagonal( 1 ./ (diag(P.Op.L.D) .+ (P.vShift - P.σ) ) )
# end