using LinearAlgebra
using SparseArrays
using Kronecker
using IterativeSolvers
using IncompleteLU
#using PyPlot
using Plots

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

function uniformNodes( n )
	return LinRange( -1.0, 1.0, n )
end

function setUp1D(n,p)
	mesh = [ x for x in LinRange( 0.0, 1.0, n+1 ) ];
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

	######################################################################
	### Assemble local matrices
	######################################################################
	Mlocal = zeros( p+1,p+1 );
	Llocal = zeros( p+1,p+1 );
	gps, gws = gauss_quad(2p); #this should be fine for mass matrix which has order p times order p
	##this is fine for our order p polynomials	
	gpsP, gwsP = gauss_quad(3p); ##for potential term


	polyGauss, dpolyGauss = legendre_poly( gps, p );
	bFnGauss = polyGauss * coeffs; dbFnGauss = dpolyGauss * coeffs;

	polyGaussP, dpolyGaussP = legendre_poly( gpsP, p );
	bFnGaussP = polyGaussP * coeffs; dbFnGaussP = dpolyGaussP * coeffs;

	Plocal = zeros(p+1, length(gwsP)); ##lets formulate Mlocal - Plocal W Plocal'. Plocal[i,k] = phi_i(x_k), where x_k are quad points
	Wlocal = diagm(gwsP);

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

		for ss = 1:length(gwsP)
			Plocal[ii,ss] = bFnGaussP[ss,ii]; ##function i evaluated at quad point s
		end

	end
	##Mlocal2 = Plocal * Wlocal * Plocal'; ##same thing
	Minvlocal = inv(Mlocal);

	# M2d = Mlocal ⊗ Mlocal;
	# M2d2 = (Plocal ⊗ Plocal) * (Wlocal ⊗ Wlocal) * (Plocal ⊗ Plocal)';


	##Mlocal is phi_i against phi_j
	##Llocal is d phi_i against d_phij

	######################################################################
	### Assemble global matrices in 1D
	######################################################################
	Minv = Tuple{Int64,Int64,Float64}[];
	P1d = Tuple{Int64,Int64,Float64}[];
	W1d = Tuple{Int64,Int64,Float64}[];

	L = Tuple{Int64,Int64,Float64}[];
	polyEnds, dpolyEnds = legendre_poly( [-1.0,1.0], p ); dbFnGauss = dpolyEnds * coeffs;

	for elem in elems
		J = abs( elem.mX[2]-elem.mX[1] ) * 0.5; ##transform 0:h to -1:1
		for ii = 1:length(elem.mIndex)
			indexi = elem.mIndex[ii];
			for jj = 1:length(elem.mIndex)
				indexj = elem.mIndex[jj];
				push!( L, ( indexi, indexj, Llocal[ii,jj]/J ) );
				push!( Minv, ( indexi, indexj, Minvlocal[ii,jj]/J ) );
			end

			for jj = 1:length(gwsP)
				indexj = (elem.mID-1) * (length(gwsP)) + jj; ##size of block 
				push!( P1d, ( indexi, indexj, Plocal[ii,jj] ) );
			end
		end
		
		for jj = 1:length(gwsP)
			indexj = (elem.mID-1) * (length(gwsP)) + jj;
			push!( W1d, ( indexj, indexj, Wlocal[jj,jj]*J ) );
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
		pen = (p+1)^2/J;

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

	Minv = sparse( (x->x[1]).(Minv), (x->x[2]).(Minv), (x->x[3]).(Minv), length(x1), length(x1) );

	P1d = sparse( (x->x[1]).(P1d), (x->x[2]).(P1d), (x->x[3]).(P1d), length(x1), length(gwsP)*n );

	W1d = sparse( (x->x[1]).(W1d), (x->x[2]).(W1d), (x->x[3]).(W1d), length(gwsP)*n, length(gwsP)*n );
	L = sparse( (x->x[1]).(L), (x->x[2]).(L), (x->x[3]).(L), length(x1), length(x1) );
	L1 = Minv * L; id = spdiagm( ones(size(L,1)) );

	######################################################################
	### Kronecker solve setup
	######################################################################

	##why dont we just use eigen decomp???

	function findZeros( x )
		arr = Vector{Int}();
		for ii = 1:length(x)
			if abs(x[ii]) < 1e-10
				push!(arr,ii);
			end
		end
		return arr
	end

	#Schur decomp of 1D operator
	F = schur( Matrix(L1) ); 
	zeroIndex = findZeros( F.values );
	ord = ones( Bool, length(F.values) ); 
	for ii in zeroIndex
		ord[ii] = false;
	end
	F = ordschur( F, ord ); 

	T = real(F.T); V = real(F.Z);
	for jj = 1:size(T,2)
		for ii = 1:size(T,1)
			if abs(T[ii,jj]) < 1e-8 ####huh
				T[ii,jj] = 0.0;
			end
		end
	end

	
	
	return T,V,L1,P1d, W1d, Minv, Plocal,Wlocal,Mlocal;
end

function setUpVecs(n,p)
	x1 = zeros( n*(p+1) );
	mesh = [ x for x in LinRange( 0.0, 1.0, n+1 ) ];
	nodes = chebyshevNodes( p+1 ); x1 = zeros( n*(p+1) );

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
	######################################################################
	### Set up 3d problem
	######################################################################
	### Analytical
	f(x,y,z) = exp( sin( 2*pi*x ) * sin( 2*pi*y ) * sin( 2*pi*z ) ) - 1.0;
	
	##potential function
	v(x,y,z) = 1*(sin(x*y*z) + 2);

	d2f_x(x,y,z) = (f(x,y,z)+1.0) * ( 4*pi^2*sin(2*pi*y)*sin(2*pi*z) ) * ( -sin(2*pi*x) + sin(2*pi*z)*sin(2*pi*y)*cos(2*pi*x)^2 ) ;
	d2f_y(x,y,z) = (f(x,y,z)+1.0) * ( 4*pi^2*sin(2*pi*x)*sin(2*pi*z) ) * ( -sin(2*pi*y) + sin(2*pi*x)*sin(2*pi*z)*cos(2*pi*y)^2 ) ;
	d2f_z(x,y,z) = (f(x,y,z)+1.0) * ( 4*pi^2*sin(2*pi*x)*sin(2*pi*y) ) * ( -sin(2*pi*z) + sin(2*pi*x)*sin(2*pi*y)*cos(2*pi*z)^2 ) ;
	d2f(x,y,z) = d2f_x(x,y,z) + d2f_y(x,y,z) + d2f_z(x,y,z);
	rhsV(x,y,z) = -d2f(x,y,z) + v(x,y,z)*f(x,y,z);

	# f(x,y,z) = 1+x+x^2;
	# v(x,y,z) = 1 + x + x^2;
	# rhsV(x,y,z) = (1 + x + x^2)^2;


	x3 = zeros( length(x1)^3,3 ); b = zeros( size(x3,1) ); uexact = zeros( size(x3,1) );
	bV = zeros( size(x3,1) );
	Vexact = zeros( size(x3,1) );
	# x3size = ( length(x1)^3,3 );
	# vsize = ( length(x1)^3,1 );

	# x3 = fill(0.0,x3size);
	# b = fill(0.0,vsize);
	# uexact = fill(0.0,vsize);


	function setPosition!( x3, b, uexact, Vexact, x1 )
		currIndex = 1;

		for ii = 1:length(x1)
			zpos = x1[ii];
			for jj = 1:length(x1)
				ypos = x1[jj];
				for kk = 1:length(x1)
					xpos = x1[kk];
					x3[currIndex,1] = xpos; x3[currIndex,2] = ypos; x3[currIndex,3] = zpos;

					b[currIndex] = d2f( xpos, ypos, zpos );
					uexact[currIndex] = f( xpos, ypos, zpos );
					bV[currIndex] = rhsV( xpos, ypos, zpos );
					Vexact[currIndex] = v( xpos, ypos, zpos );
					currIndex += 1;
				end
			end
		end
		

	end

	setPosition!(x3, b, uexact, Vexact, x1)
	
	return uexact, b, bV, Vexact;
end


n=3;
p=3;

##get schur factors, local operator, local moment mat, local W and P mat to construct v
T,V,L1,P1d, W1d, Minv, Plocal,Wlocal,Mlocal = setUp1D(n,p);
evecs = eigvecs(Matrix(L1));

##for some reason julia computes complex conjugate pairs of eigenvectors, which we want to fix
for i = 2:(n*(p+1))
	if( norm( real(evecs[:,i-1]) - real(evecs[:,i]) ) < 1e-15  )
		evecs[:,i-1] = real(evecs[:,i-1]) / norm(real(evecs[:,i-1]));
		evecs[:,i] = imag(evecs[:,i]) / norm(imag(evecs[:,i]))
	end
end

evecs = real(evecs);

# display(plot(real(evecs[:,end])));
evals = real(eigvals(Matrix(L1)));
evals[1]=0;

#evecs_ = evecs[:,2:end];
evecsinv=inv(evecs);

##set up exact nodal values of eigenvectors?
uexact, b, bV, Vexact = setUpVecs(n,p);

# V3 = V⊗V⊗V;
# T3 = T⊗T⊗T;
L = collect( kroneckersum(L1,L1,L1) );
# D3 = collect( kroneckersum(T,T,T) ); ##I times I times T + I times T times I + T times I times I

# # println(Base.summarysize(T3) );
# # println(Base.summarysize(D3) );
# ###note we need to have that bottom right corner = 1-4;???
# #####Lewis why?


# @time D3 = collect( kroneckersum(T,T,T) );

# ##Do we need this
# D3[end,end] = 1e-4;

######################################################################
### Preconditioning
######################################################################

struct SchurPrecond
	U
	T
end

function LinearAlgebra.ldiv!( y, P::SchurPrecond, x )
	mul!( y, P.U', x );
	ldiv!( P.T, y );
	mul!( y, P.U, y );
	return y
end

function LinearAlgebra.ldiv!( P::SchurPrecond, x )
	return ldiv!( x, P, x )
end

function Base.:\( P::SchurPrecond, b )
    return ldiv!( P, b )
end

function Base.:*(P::SchurPrecond, b)
	y = (P.U')*b;
	y = P.T*y;
	y = (P.U)*y;
	return y;
end

#S1 = SchurPrecond( V⊗V⊗V, UpperTriangular(D3) );


##try diagonalizing the whole jawn
struct diagPrecond
	V
	D
	Dinv
	Vinv
end

function LinearAlgebra.ldiv!( y, P::diagPrecond, x )
	mul!( y, P.Vinv, x );
	y = P.Dinv * y ;
	y = P.V * y ;
	return y
end

function LinearAlgebra.ldiv!( P::diagPrecond, x )
	return ldiv!( x, P, x )
end

function Base.:\( P::diagPrecond, b )
    return ldiv!( P, b )
end

function Base.:*(P::diagPrecond, b)
	y = (P.Vinv)*b;
	y = P.D*y;
	y = (P.V)*y;
	return y;
end

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

function getMatrix(P::diagPrecond)
	A1 = SparseMatrixCSC(P.Vinv);
	A2 = SparseMatrixCSC(P.D);
	A3 = SparseMatrixCSC(P.V);
	return(A3*A2*A1);
end

evecs3 = evecs⊗evecs⊗evecs;
evals3 = Diagonal(evals)⊕Diagonal(evals)⊕Diagonal(evals);

evalsinv3 = diag(evals3);
evalsinv3 = 1 ./ evalsinv3;
evalsinv3[1] = 0;
evalsinv3 = Diagonal(evalsinv3);

evecsinv3=evecsinv⊗evecsinv⊗evecsinv;

S2 = diagPrecond(evecs3, evals3, evalsinv3, evecsinv3);

Perm = getPermutation(n,p);

b1=copy(b); b2=copy(b); b3=copy(b);

# x = S1 \ zeros(length(b));

# println("schur time ");
# @time v = -( S1 \ b1 );
# v1 = v[1]; v .-= v1;

x = S2*zeros(length(b));
println("mult time ");
@time x = S2*zeros(length(b));

x = S2 \ zeros(length(b));
println("diag time ");
@time v2 = -( S2 \ b2 );
v21 = v2[1]; v2 .-= v21;



# # uL = (-L \ b3);
# # uL1 = uL[1]; uL .-= uL1;


# # @time u = -( S1 \ b );
# # u1 = u[1]; u .-= u1;

# # ##uL = - (L \ b);
# println( maximum( abs.(v-uexact) ) )
println( maximum( abs.(v2-uexact) ) )
##println( maximum( abs.(uL-uexact) ) )


