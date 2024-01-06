using LinearAlgebra
using SparseArrays
using IterativeSolvers
using FastGaussQuadrature

####################################
###	Basis functions
###	Legendre polynomials and derivatives of degree p at nodes x
###	@x: points at which to evaluate, @p: max order of poly
###	@RETURN: Matrix of fn vals, Matrix of dfn vals
####################################
function legendre_poly(x, p)
    z = zeros(size(x))
    o = ones(size(x))
    y = hcat(o, x, repeat(z, 1, p-1))
    dy = hcat(z, o, repeat(z, 1, p-1))
    for i = 1:p-1
        @. y[:,i+2] = ((2i+1)*x*y[:,i+1] - i*y[:,i]) / (i+1)
        @. dy[:,i+2] = ((2i+1)*(x*dy[:,i+1] + y[:,i+1]) - i*dy[:,i]) / (i+1)
    end
    return y, dy
end

####################################
###	Gauss-Lobatto quadrature on [-1,1]
###	@n: number of points
###	@RETURN pts, wts
####################################
function lobatto_quad(n)
    return gausslobatto(n)
end


####################################################################################################################

struct Elem
	mID::Int64
	mX::Vector{Float64}
	mNodes::Vector{Float64}
	mIndex::Vector{Int64}
	Elem( ii,x,y,p ) = new( ii,[x,y], zeros(p+1), zeros(Int64,p+1) )
end

function setFn!( f, u, elem::Elem )

	for ii = 1:length( elem.mIndex )
		u[ elem.mIndex[ii] ] = f( elem.mNodes[ii] );
	end

end

######################################################################
###	Set up problem
######################################################################
n = 3;
mesh = [ x for x in LinRange( 0.0, 1.0, n+1 ) ];
p = 4;
nodes, weights = lobatto_quad( p+1 ); x1 = zeros( n*(p+1) );

######################################################################
###	Set up basis functions
######################################################################
polyNodes, dpolyNodes = legendre_poly( nodes, p );

polyNodes_p1, dpolyNodes_p1 = legendre_poly( [-1.0,1.0], 1 );
coeffs_p1 = inv( polyNodes_p1 );

######################################################################
###	Create elements and solution vector
######################################################################
elems = [ Elem( ii, mesh[ii], mesh[ii+1],p ) for ii = 1:n ];
elemnodes = ( nodes .+ 1.0 ) .* 0.5;
for elem in elems
	J = abs( elem.mX[2] - elem.mX[1] );
	elem.mNodes .= elem.mX[1] .+ J * elemnodes;
	for ii = 1:p+1
		elem.mIndex[ii] = (elem.mID-1)*(p+1)+ii;
	end
	x1[(elem.mID-1)*(p+1)+1:elem.mID*(p+1)] .= elem.mNodes;
end
u = zeros( length(elems)*(p+1) );


######################################################################
###	Initial conditions
######################################################################
u0(x) = exp( -60*(x-0.5)^2 );
ddu0(x) = 120 * exp(-15*(1 - 2*x)^2) * (29 - 120*x + 120*x^2);

for elem in elems
	setFn!( u0, u, elem )
end

######################################################################
### Assemble local matrices
######################################################################
Mlocal = zeros( p+1,p+1 ); Llocal = zeros( p+1,p+1 );
bFnGauss = polyNodes_p1 * coeffs_p1; dbFnGauss = dpolyNodes_p1 * coeffs_p1;

for nn = 1:p
	J = abs( elems[1].mNodes[nn+1] - elems[1].mNodes[nn] );

	for kk = 1:2
		Mlocal[nn,nn] += bFnGauss[kk,1] * bFnGauss[kk,1] * 0.5 * J;
		Llocal[nn,nn] += dbFnGauss[kk,1] * dbFnGauss[kk,1] / ( 0.5 * J );

		Mlocal[nn,nn+1] += bFnGauss[kk,1] * bFnGauss[kk,2] * 0.5 * J;
		Llocal[nn,nn+1] += dbFnGauss[kk,1] * dbFnGauss[kk,2] / ( 0.5 * J );

		Mlocal[nn+1,nn] += bFnGauss[kk,2] * bFnGauss[kk,1] * 0.5 * J;
		Llocal[nn+1,nn] += dbFnGauss[kk,2] * dbFnGauss[kk,1] / ( 0.5 * J );

		Mlocal[nn+1,nn+1] += bFnGauss[kk,2] * bFnGauss[kk,2] * 0.5 * J;
		Llocal[nn+1,nn+1] += dbFnGauss[kk,2] * dbFnGauss[kk,2] / ( 0.5 * J );
	end
end

######################################################################
### Assemble global matrices
######################################################################
M = Tuple{Int64,Int64,Float64}[];
L = Tuple{Int64,Int64,Float64}[];
dbFnGauss = dpolyNodes_p1 * coeffs_p1;

for elem in elems

	for ii = 1:length(elem.mIndex)
		for jj = 1:length(elem.mIndex)
			indexi = elem.mIndex[ii];
			indexj = elem.mIndex[jj];
			push!( M, ( indexi, indexj, Mlocal[ii,jj] ) );
			push!( L, ( indexi, indexj, Llocal[ii,jj] ) );
		end
	end
	
	### Flux terms - everyone only do to their left ##
	J = abs( elem.mNodes[2]-elem.mNodes[1] ) * 0.5;

	prevElem = elems[end];
	if elem.mID != 1
		prevElem = elems[elem.mID-1];
	end

	#du-/v-
	row = prevElem.mIndex[end];
	for jj = length(prevElem.mIndex)-1:length(prevElem.mIndex)
		col = prevElem.mIndex[jj];
		val = -0.5*dbFnGauss[2,2-length(prevElem.mIndex)+jj]/J;
		push!( L, (row, col, val ) );
		push!( L, (col, row, val ) );
	end

	#du-/v+
	row = elem.mIndex[1];
	for jj = length(prevElem.mIndex)-1:length(prevElem.mIndex)
		col = prevElem.mIndex[jj];
		val = 0.5*dbFnGauss[2,2-length(prevElem.mIndex)+jj]/J;
		push!( L, (row, col, val ) );
		push!( L, (col, row, val ) );
	end

	#du+/v-
	row = prevElem.mIndex[end];
	for jj = 1:2
		col = elem.mIndex[jj];
		val = -0.5*dbFnGauss[1,jj]/J;
		push!( L, (row, col, val ) );
		push!( L, (col, row, val ) );
	end

	#du+/v+
	row = elem.mIndex[1];
	for jj = 1:2
		col = elem.mIndex[jj];
		val = 0.5*dbFnGauss[1,jj]/J;
		push!( L, (row, col, val ) );
		push!( L, (col, row, val ) );
	end
	
	#Penalty terms
	pen = 0.5/J;

	#u-/v- term
	row = prevElem.mIndex[end]; col = prevElem.mIndex[end];
	push!( L, (row, col, pen ) );

	#u-/v+ term
	row = elem.mIndex[1]; col = prevElem.mIndex[end];
	push!( L, (row, col, -pen ) );

	#u+/v- term
	row = prevElem.mIndex[end]; col = elem.mIndex[1];
	push!( L, (row, col, -pen ) );

	#u+/v+ term
	row = elem.mIndex[1]; col = elem.mIndex[1];
	push!( L, (row, col, pen ) );
	
end

M = sparse( (x->x[1]).(M), (x->x[2]).(M), (x->x[3]).(M), length(u), length(u) ); droptol!( M, 1e-12 );
L = sparse( (x->x[1]).(L), (x->x[2]).(L), (x->x[3]).(L), length(u), length(u) ); droptol!( L, 1e-12 );

######################################################################
### Laplace test
######################################################################

b = zeros( length(u) );
for elem in elems
	setFn!( ddu0, b, elem )
end
L = M \ L;

#=
######################################
### Block diagonals
######################################
struct BlockDiagonalMatrix
	nBlocks::Int
	nBlockSize::Int
	nRows::Int64			
	mBlocks::Vector{Array{Float64,2}}
	mBlockInvs::Vector{Array{Float64,2}}
end

function BlocktoSparse( B::BlockDiagonalMatrix )
	
	A = Tuple{Int64,Int64,Float64}[];
	for nn = 1:B.nBlocks

		block = B.mBlocks[nn]; bId = (nn-1)*B.nBlockSize;
		for col = 1:size(block,2)
			for row = 1:size(block,1)
				push!( A, ( bId+row, bId+col, block[row,col] ) );
			end
		end

	end

	return sparse( (x->x[1]).(A), (x->x[2]).(A), (x->x[3]).(A), B.nRows, B.nRows );

end

function extractBlockDiagonal( bSize::Int, A )

	nBlocks = Int( size(A,1) / bSize ); blocks = Vector{Array{Float64,2}}( undef, nBlocks );
	factorisations = Vector{Array{Float64,2}}( undef, nBlocks );
	
	for nn = 1:nBlocks
		block = zeros( bSize,bSize );
		for col = 1:bSize
			colval = (nn-1)*bSize+col;
			for row = 1:bSize
				rowval = (nn-1)*bSize+row;
				block[row,col] = A[rowval,colval];
			end
		end
		blocks[nn] = block;
		factorisations[nn] = inv(block);
	end

	return BlocktoSparse( BlockDiagonalMatrix( nBlocks, bSize, nBlocks*bSize, blocks, factorisations ) )

end

#######################################################################################
### Block Jacobi
#######################################################################################
struct precond_BlockJacobi
	mA
	mLU

	function precond_BlockJacobi( bSize::Int, A )
		B = extractBlockDiagonal( bSize, A );
		new( B, lu(B) )
	end
end

function LinearAlgebra.ldiv!( y, P::precond_BlockJacobi, x )
	z = P.mLU \ x;
	y .= z;
end

function LinearAlgebra.ldiv!( P::precond_BlockJacobi, x )
	y = zeros( size(x) );
	ldiv!( y, P, x );
	x .= y;
end

function Base.:\( P::precond_BlockJacobi, x )
	y = zeros( size(x) );
	ldiv!( y, P, x );
end

utest = -L \ b; utest .-= utest[1];

######################################################################
### Testing stuff
######################################################################

L = M * L; b = M * b;
L[1,:] .= 0.0;
L[:,1] .= 0.0;
L[1,1] = 1.0;
L[end,:] .= 0.0;
L[:,end] .= 0.0;
L[end,end] = 1.0;
b[1] = 0.0; b[end] = 0.0; droptol!( L,1e-12 );

P = precond_BlockJacobi( p+1, L );

uiter = -cg( L, b; Pl = P, verbose = true, reltol=1e-6 ); uiter .-= uiter[1];
println( maximum( abs.(u-uiter) ) )
=#
