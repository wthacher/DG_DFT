include("operatorSetUp.jl");
###############
###Diagonalized structure of operator
##also stores Laplacian as kron product 
struct diagPrecond
	V
	D
	Dinv
	Vinv
    L
end

function LinearAlgebra.ldiv!( y, P::diagPrecond, x )
	mul!( y, P.Vinv, x );
	y .= P.Dinv * y ;
	y .= P.V * y ;
	return y
end

function LinearAlgebra.ldiv!( P::diagPrecond, x )
	return ldiv!( x, P, x )
end

function Base.:\( P::diagPrecond, b )
    return ldiv!( P, b )
end

function Base.:*(P::diagPrecond, b)
	# y = (P.Vinv)*b;
	# y = P.D*y;
	# y = (P.V)*y;
	return (P.L)*b;
end

function getMatrix(P::diagPrecond)
	A = SparseMatrixCSC(P.L);
	return A;
end

################
##Operator structure. This will be used by multigrid class
##note this operator is L  + V
##if block is true we apply it in block form
struct Operator
    Vexact ##vector of potential values
    L ##laplacian (note this will be a diag preconditioner)

	nodeFunc

    LB ##Laplacian block
	MB ##mass mat block

    M ##3d mass mat
	Minv ##mass matrix inverse

    Perm ##permutation from tensor to block
	PermT
    L1 ##1d laplacian
	M1

	block
end

##important: assumes v in the right form already
function Base.:*(P::Operator, b)
    y1 = Diagonal(P.Vexact)  * b;
	y2 = b;
	if(P.block)
    	y2 = P.PermT*y2;
	end

    y2 = P.L*y2;

	if(P.block)
    	y2 = P.Perm*y2;
	end

    y = y1 + y2;
    return y;
end

function LinearAlgebra.mul!(y::AbstractVector,P::Operator, b)
    y .= P*b;
    return y;
end

function Base.eltype(P::Operator)
    return Float64;
end

function Base.size(P::Operator, d)
    return size(P.Perm,1);
end

function Base.size(P::Operator)
    return size(P.Perm);
end

function LinearAlgebra.issymmetric(P::Operator)
	return false;
end

##for low level of multigrid - return operator as matrix in block coords
##assumes we already have v in correct format !!very important
function getMatrix(P::Operator)
    A = getMatrix(P.L);
    V = Diagonal(P.Vexact);
	R = SparseMatrixCSC;
	if(P.block)
		R = SparseMatrixCSC(P.Perm * A * P.PermT + V);
	else
		R = SparseMatrixCSC(A + V);
	end

	return R;

end

##this returns permutation that turns tensor coords into block coords
function getPermutation(n,p)
	##given n grid with order p, convert kron order to block diag order - nodes are numbered within each element
	##indces are now arranged in x dir, then y dir, then z dir
	##we want to move through elements like that, but be in order in each element
	P = Tuple{Int64,Int64,Float64}[];

	N = (p+1)^3 * n^3;
	N1D = (p+1) * n;

	##find element then get idx from position on grid
	##to get element #
	function new_idx(xi,yi,zi)
		nx = floor(Int64,(xi-1) / (p+1));
		ny = floor(Int64,(yi-1) / (p+1));
		nz = floor(Int64,(zi-1) / (p+1));

		nElem = (nx) + n*(ny) + n^2*(nz);

		ix = (xi-1) % (p+1);
		iy = (yi-1) % (p+1);
		iz = (zi-1) % (p+1);

		idx = nElem*(p+1)^3 + (ix+1) + (p+1)*iy + (p+1)^2*iz;

		##println(idx)

		return trunc(Int64,idx);

	end

	c_idx=1;
	for k = 1:N1D
		for j = 1:N1D
			for i = 1:N1D
				push!(P,(new_idx(i,j,k), c_idx, 1))
				c_idx+=1;
			end
		end
	end


	P = sparse( (x->x[1]).(P), (x->x[2]).(P), (x->x[3]).(P), N, N );
	return P;


end

##outputs operator of order p with grid n for data vData, permutation from tensor to block
##nodeFunc is a Function that gives nodes and weights
##opt is SIP or LDG so far
##assumes quad points are same as nodes for now
##symmetric operator is M(L+V)
##BC type is homogeneous dirichlet, neumann, or periodic
function createOperator(n,p, BCtype, vData, nodeFunc, opt, block, LOR)
	##can do other plocals also
	plocal = p;
	if(LOR)
		plocal=1;
	end

	L1, M1, K1, M1inv, LB = setUpPPOperator(n,p,nodeFunc,opt,plocal,BCtype);

	Mhalf_L_Minvhalf = Diagonal(sqrt.(diag(M1))) * L1 * Diagonal(sqrt.(1 ./ diag(M1)))
	# if(LOR)
	# 	Mhalf_L_Minvhalf = sqrt(Matrix(M1)) * L1 * inv(sqrt(Matrix(M1)));
	# end
    # ev = eigen(Matrix(L1));
	# println("asdf", diag(L1*ev.vectors - ev.vectors*Diagonal(ev.values)));
	ev = eigen(Symmetric(Matrix(Mhalf_L_Minvhalf)));
	evecs = Diagonal(sqrt.(1 ./ diag(M1))) * ev.vectors;
	evecsinv = ev.vectors' * Diagonal(sqrt.(diag(M1)));

	# if(LOR)
	# 	evecs = inv(sqrt(Matrix(M1))) * ev.vectors;
	# 	evecsinv = ev.vectors' * sqrt(Matrix(M1));
	# end

	evals = ev.values;
	if(BCtype == "periodic")
		evals[1]=0;
	end

    evecs3 = evecs⊗evecs⊗evecs;
    evals3 = Diagonal(evals)⊕Diagonal(evals)⊕Diagonal(evals);

    evalsinv3 = diag(evals3);
    evalsinv3 = 1 ./ evalsinv3;
    if(BCtype == "periodic")	
		evalsinv3[1] = 0;
	end
    evalsinv3 = Diagonal(evalsinv3);

    evecsinv3=evecsinv⊗evecsinv⊗evecsinv;

	L = kroneckersum(L1,L1,L1);
	#K = K1⊗M1⊗M1 + M1⊗K1⊗M1 + M1⊗M1⊗K1; ##stiffness matrix


    S2 = diagPrecond(evecs3, evals3, evalsinv3, evecsinv3, L) ;

    Perm = getPermutation(n,p);
	PermT = SparseMatrixCSC(Perm');

    #P3d = P1d⊗P1d⊗P1d;
    M = Diagonal(M1⊗M1⊗M1);
	Minv = Diagonal(M1inv⊗M1inv⊗M1inv);

	MB = Diagonal( diag(M)[1:p+1]);

    Op = Operator(vData, S2, nodeFunc, LB, MB, M, Minv, Perm, PermT, L1, M1,block);
	#println("sparsity of L1:", nnz(L1)/(size(L1,1)^2) )

    return Op;

end
