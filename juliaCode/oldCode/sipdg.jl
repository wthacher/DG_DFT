using LinearAlgebra
using SparseArrays
using IterativeSolvers

####################################
###	Get Chebyshev nodes including endpoints
###	@n: number of points >= 2
###	@RETURN: Array of points
####################################
function chebyshevNodes( n )
	#return vcat( [-1.0], [ cos( (k+k-1)*pi/(n+n-4) ) for k = n-2:-1:1 ], [1.0] )
	return ( [ 0.5 * (-cos( pi * i/p ) + 1.0) for i=0:p ] .* 2.0 ) .- 1.0
end

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
###	Gaussian quadrature
###	Gaussian quadrature on [-1,1] for given degree of precision p
###	@p: highest order polynomial exactly integrated
###	@RETURN gpts, gws
####################################
function gauss_quad(p)
    n = ceil((p+1)/2)
    b = 1:n-1
    b = @. b / sqrt(4*b^2 - 1)
    eval, evec = eigen(diagm(1 => b, -1 => b))
    return eval, 2*evec[1,:].^2
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
n = 4;
mesh = [ x for x in LinRange( 0.0, 1.0, n+1 ) ];
p = 4;
nodes = chebyshevNodes( p+1 ); x1 = zeros( n*(p+1) );

######################################################################
###	Set up basis functions
######################################################################
polyNodes, dpolyNodes = legendre_poly( nodes, p );
coeffs = inv( polyNodes );

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
u0(x) = exp( sin( 2*pi*x ) ) - 1.0;
ddu0(x) = -(u0(x) +1.0) * ( 4*pi^2 ) * ( sin(2*pi*x) - cos(2*pi*x)^2 );

for elem in elems
	setFn!( u0, u, elem )
end

######################################################################
### Assemble local matrices
######################################################################
Mlocal = zeros( p+1,p+1 ); Llocal = zeros( p+1,p+1 );
gps, gws = gauss_quad(2p+2);	#Probably overkill but who cares

polyGauss, dpolyGauss = legendre_poly( gps, p );
bFnGauss = polyGauss * coeffs; dbFnGauss = dpolyGauss * coeffs;

for ii = 1:p+1
	for jj = 1:p+1

		Msum = 0.0; Lsum = 0.0;
		for kk = 1:length(gws)
			Msum += bFnGauss[kk,ii] * bFnGauss[kk,jj] * gws[kk];
			Lsum += dbFnGauss[kk,ii] * dbFnGauss[kk,jj] * gws[kk];
		end

		Mlocal[ii,jj] = Msum;
		Llocal[ii,jj] = Lsum;

	end
end
Minvlocal = inv( Mlocal );

######################################################################
### Assemble global matrices
######################################################################
Minv = Tuple{Int64,Int64,Float64}[];
L = Tuple{Int64,Int64,Float64}[];
polyEnds, dpolyEnds = legendre_poly( [-1.0,1.0], p ); dbFnGauss = dpolyEnds * coeffs;

for elem in elems

	J = abs( elem.mX[2]-elem.mX[1] ) * 0.5;
	for ii = 1:length(elem.mIndex)
		for jj = 1:length(elem.mIndex)
			indexi = elem.mIndex[ii];
			indexj = elem.mIndex[jj];
			push!( Minv, ( indexi, indexj, Minvlocal[ii,jj]/J ) );
			push!( L, ( indexi, indexj, Llocal[ii,jj]/J ) );
		end
	end
	
	### Flux terms - everyone only do to their left ##
	prevElem = elems[end];
	if elem.mID != 1
		prevElem = elems[elem.mID-1];
	end

	#du-/v-
	row = prevElem.mIndex[end];
	for (jj,col) in enumerate(prevElem.mIndex)
		val = -0.5*dbFnGauss[2,jj]/J;
		push!( L, (row, col, val ) );
		push!( L, (col, row, val ) );
	end

	#du-/v+
	row = elem.mIndex[1];
	for (jj,col) in enumerate(prevElem.mIndex)
		val = 0.5*dbFnGauss[2,jj]/J;
		push!( L, (row, col, val ) );
		push!( L, (col, row, val ) );
	end

	#du+/v-
	row = prevElem.mIndex[end];
	for (jj,col) in enumerate(elem.mIndex)
		val = -0.5*dbFnGauss[1,jj]/J;
		push!( L, (row, col, val ) );
		push!( L, (col, row, val ) );
	end

	#du+/v+
	row = elem.mIndex[1];
	for (jj,col) in enumerate(elem.mIndex)
		val = 0.5*dbFnGauss[1,jj]/J;
		push!( L, (row, col, val ) );
		push!( L, (col, row, val ) );
	end
	
	#Penalty terms
	pen = (p+1)*4/J;

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

Minv = sparse( (x->x[1]).(Minv), (x->x[2]).(Minv), (x->x[3]).(Minv), length(u), length(u) );
L = sparse( (x->x[1]).(L), (x->x[2]).(L), (x->x[3]).(L), length(u), length(u) );

######################################################################
### Laplace test
######################################################################

f = zeros( length(u) );
for elem in elems
	setFn!( ddu0, f, elem )
end
L = Minv * L;

F = schur( Matrix(L) ); D = real(F.T); V = real(F.Z);
zeroIndex = findmin( real(F.values) )[2]; ord = ones( Bool, length(F.values) ); ord[zeroIndex] = false;
 
F = ordschur( F, ord ); D = real(F.T); V = real(F.Z);
D[end,end] = 1.0;

L[1,:] .= 0.0;
L[:,1] .= 0.0;
L[1,1] = 1.0;
L[end,:] .= 0.0;
L[:,end] .= 0.0;
L[end,end] = 1.0;
f[1] = 0.0; f[end] = 0.0;

######################################################################
### Preconditioner stuff
######################################################################
test = V * ( D \ (V' * L) ); f = V * ( D \ (V' * f) );

uiter = -gmres( test, f; verbose = true, reltol=1e-12 );
println( maximum( abs.(u-uiter) ) )



