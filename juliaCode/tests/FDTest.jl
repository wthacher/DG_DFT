##try out low order finite difference preconditioners to speed things up

using FastGaussQuadrature
using IterativeSolvers
using LinearAlgebra
using Plots
using SparseArrays

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

struct Elem
	mID::Int64
	mX::Vector{Float64}
	mNodes::Vector{Float64}
	mIndex::Vector{Int64}
	Elem( ii,x,y,p ) = new( ii,[x,y], zeros(p+1), zeros(Int64,p+1) )
end

function setUp1D(n,p,nodeFunc,type,LOR)
	mesh = [ x for x in LinRange( 0.0, 1.0, n+1 ) ];
	#nodes = chebyshevNodes( p+1 ); 
    x1 = zeros( n*(p+1) );
    nodes, weights = nodeFunc(p+1); ##have flexible choice of nodes

	######################################################################
	###	Set up basis functions
	######################################################################
	polyNodes, dpolyNodes = legendre_poly( nodes, p );
	coeffs = inv( polyNodes );

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

	######################################################################
	### Assemble local matrices
	######################################################################
	Mlocal = zeros( p+1,p+1 );
	Klocal = zeros( p+1,p+1 );
	Krlocal = zeros( p+1,p+1 );

	Mlocal_p1 = zeros( p+1,p+1 );
	Klocal_p1 = zeros( p+1,p+1 );
	Krlocal_p1 = zeros( p+1,p+1 );

	bFnGauss_p1 = polyNodes_p1 * coeffs_p1; 
	dbFnGauss_p1 = dpolyNodes_p1 * coeffs_p1;

    gws = weights;
    gwsP = weights;

	bFnGauss = polyNodes * coeffs; 
	dbFnGauss = dpolyNodes * coeffs;
    #dbFnGauss = dpolyNodes * coeffs;

    bFnGaussP = bFnGauss;

	Plocal = zeros(p+1, length(gwsP)); ##lets formulate Mlocal - Plocal W Plocal'. Plocal[i,k] = phi_i(x_k), where x_k are quad points
	Wlocal = diagm(gws);

	for ii = 1:p+1
		for jj = 1:p+1
			Msum = 0.0; Ksum = 0.0; Krsum = 0.0;
			for kk = 1:length(gws)
				Msum += bFnGauss[kk,ii] * bFnGauss[kk,jj] * gws[kk];
				Ksum += dbFnGauss[kk,ii] * dbFnGauss[kk,jj] * gws[kk];
				Krsum -= dbFnGauss[kk,ii] * bFnGauss[kk,jj] * gws[kk];
			end

			Mlocal[ii,jj] = Msum;
			Klocal[ii,jj] = Ksum;
			Krlocal[ii,jj] = Krsum;
		end

		for ss = 1:length(gwsP)
			Plocal[ii,ss] = bFnGaussP[ss,ii]; ##function i evaluated at quad point s
		end

	end
	
	Minvlocal = inv(Mlocal);

	##set up p1 local
	###NOTE THIS ASSUMES you have endpoints I believe
	for nn = 1:p
		J = abs( elems[1].mNodes[nn+1] - elems[1].mNodes[nn] );
	
		for kk = 1:2
			Mlocal_p1[nn,nn] += bFnGauss_p1[kk,1] * bFnGauss_p1[kk,1] * 0.5 * J;
			Klocal_p1[nn,nn] += dbFnGauss_p1[kk,1] * dbFnGauss_p1[kk,1] / ( 0.5 * J );
			Krlocal_p1[nn,nn] -= dbFnGauss_p1[kk,1] * bFnGauss_p1[kk,1] ;
	
			Mlocal_p1[nn,nn+1] += bFnGauss_p1[kk,1] * bFnGauss_p1[kk,2] * 0.5 * J;
			Klocal_p1[nn,nn+1] += dbFnGauss_p1[kk,1] * dbFnGauss_p1[kk,2] / ( 0.5 * J );
			Krlocal_p1[nn,nn+1] -= dbFnGauss_p1[kk,1] * bFnGauss_p1[kk,2] ;
	
			Mlocal_p1[nn+1,nn] += bFnGauss_p1[kk,2] * bFnGauss_p1[kk,1] * 0.5 * J;
			Klocal_p1[nn+1,nn] += dbFnGauss_p1[kk,2] * dbFnGauss_p1[kk,1] / ( 0.5 * J );
			Krlocal_p1[nn+1,nn] -= dbFnGauss_p1[kk,2] * bFnGauss_p1[kk,1] ;
	
			Mlocal_p1[nn+1,nn+1] += bFnGauss_p1[kk,2] * bFnGauss_p1[kk,2] * 0.5 * J;
			Klocal_p1[nn+1,nn+1] += dbFnGauss_p1[kk,2] * dbFnGauss_p1[kk,2] / ( 0.5 * J );
			Krlocal_p1[nn+1,nn+1] -= dbFnGauss_p1[kk,2] * bFnGauss_p1[kk,2] ;
		end
	end

	Minvlocal_p1 = inv(Mlocal_p1);


	######################################################################
	### Assemble global matrices in 1D
	######################################################################
	M = Tuple{Int64,Int64,Float64}[];
	Minv = Tuple{Int64,Int64,Float64}[];
	# P = Tuple{Int64,Int64,Float64}[];
	# W = Tuple{Int64,Int64,Float64}[];

	K = Tuple{Int64,Int64,Float64}[];
	Kr = Tuple{Int64,Int64,Float64}[];
    
	polyEnds, dpolyEnds = legendre_poly( [-1.0,1.0], p ); 
    dbFnGaussEnds = dpolyEnds * coeffs;
    bFnGaussEnds = polyEnds * coeffs; ##values of basis functions at endpoints. need for fluxes

	M_p1 = Tuple{Int64,Int64,Float64}[];
	Minv_p1 = Tuple{Int64,Int64,Float64}[];
	# P_p1 = Tuple{Int64,Int64,Float64}[];
	# W_p1 = Tuple{Int64,Int64,Float64}[];

	K_p1 = Tuple{Int64,Int64,Float64}[];
	Kr_p1 = Tuple{Int64,Int64,Float64}[];
    
	polyEnds_p1, dpolyEnds_p1 = legendre_poly( [-1.0,1.0], 1 ); 
    dbFnGaussEnds_p1 = dpolyEnds_p1 * coeffs_p1;
    bFnGaussEnds_p1 = polyEnds_p1 * coeffs_p1; ##values of basis functions at endpoints. need for fluxes

	bFnGaussEnds_p1 = zeros(2,p+1);
	bFnGaussEnds_p1[1,1] = 1 + elems[1].mNodes[1] / (elems[1].mNodes[2] - elems[1].mNodes[1]); 
	bFnGaussEnds_p1[2,p+1] = 1 + (mesh[2]-elems[1].mNodes[end]) / (elems[1].mNodes[end] - elems[1].mNodes[end-1] );
	dbFnGaussEnds_p1 = zeros(2,p+1);

	dbFnGaussEnds_p1[1,1] =  -1.0 / ( (elems[1].mNodes[2] - elems[1].mNodes[1]) );
	if(nodes[1] == -1.0) ##if second basis function supported it has opposite sign slope
		dbFnGaussEnds_p1[1,2] = -1 * dbFnGaussEnds_p1[1,1];
	end

	dbFnGaussEnds_p1[2,p+1] = 1.0 / ( (elems[1].mNodes[end] - elems[1].mNodes[end-1]) );
	if(nodes[p+1] == 1.0)
		dbFnGaussEnds_p1[2,p] = -1 * dbFnGaussEnds_p1[2,p+1]; ##if second to last basis function supported has oppositve slope
	end

	for elem in elems
		J = abs( elem.mX[2]-elem.mX[1] ) * 0.5; ##transform 0:h to -1:1
        pen = (p+1)^2/J;
		##not sure if these are correct

        ##mass and stiffy
        prevElem = elems[end];
		if elem.mID != 1
			prevElem = elems[elem.mID-1];
		end

		nextElem = elems[1];
		if elem.mID != length(elems)
			nextElem = elems[elem.mID+1];
		end

		penl_p1 = 1.0 / (abs( elem.mNodes[2]-elem.mNodes[1] ) );
		penr_p1 = 1.0 / (abs( nextElem.mNodes[1] - elem.mNodes[end]) );

		for ii = 1:length(elem.mIndex)
			indexi = elem.mIndex[ii];
			for jj = 1:length(elem.mIndex)
				indexj = elem.mIndex[jj];
                indexjNeigh = prevElem.mIndex[jj];
				push!( K, ( indexi, indexj, Klocal[ii,jj]/J ) );
				push!( Kr, ( indexi, indexj, Krlocal[ii,jj] ) );

				push!( M, ( indexi, indexj, Mlocal[ii,jj]*J ) );
				push!( Minv, ( indexi, indexj, Minvlocal[ii,jj]/J ) );
				
				##jacobians already accounted for
				#println("ADD BACK IN GRAD TERM")
				push!( K_p1, ( indexi, indexj, Klocal_p1[ii,jj] ) );
				push!( Kr_p1, ( indexi, indexj, Krlocal_p1[ii,jj] ) );

				push!( M_p1, ( indexi, indexj, Mlocal_p1[ii,jj] ) );
				push!( Minv_p1, ( indexi, indexj, Minvlocal_p1[ii,jj] ) );

				##FOR SIP::
                ##penalty with yourself: v is i and u is j
                push!( K, ( indexi, indexj, pen*bFnGaussEnds[1,ii] * bFnGaussEnds[1,jj] ) ); ##-v(xl) -u(xl)
                push!( K, ( indexi, indexj, pen*bFnGaussEnds[2,ii] * bFnGaussEnds[2,jj] ) ); ## v(xr) u(xr)

                ##penalty w left neighbor
                push!( K, ( indexi, indexjNeigh, -pen*bFnGaussEnds[1,ii] * bFnGaussEnds[2,jj] ) ); ##-v(xl) u_neigh(xr) 
                push!( K, ( indexjNeigh, indexi, -pen*bFnGaussEnds[1,ii] * bFnGaussEnds[2,jj] ) ); ##v_neigh(xr) -u(xl)

                # #flux with yourself
                push!( K, ( indexi, indexj, (.5/J)*dbFnGaussEnds[1,ii] * bFnGaussEnds[1,jj] ) ); ##v' jump in u on left
                push!( K, ( indexi, indexj, -(.5/J)*dbFnGaussEnds[2,ii] * bFnGaussEnds[2,jj] ) ); ##v' jump in u on right

                push!( K, ( indexi, indexj, (.5/J)*dbFnGaussEnds[1,jj] * bFnGaussEnds[1,ii] ) ); ##u' jump in v on left
                push!( K, ( indexi, indexj, -(.5/J)*dbFnGaussEnds[2,jj] * bFnGaussEnds[2,ii] ) ); ##u' jump in v on right

                # #flux with left neighbor j
                push!( K, ( indexi, indexjNeigh, -(.5/J)*dbFnGaussEnds[1,ii] * bFnGaussEnds[2,jj] ) ); ##v' jump in u left neighbor
                push!( K, ( indexi, indexjNeigh, (.5/J)*dbFnGaussEnds[2,jj] * bFnGaussEnds[1,ii] ) ); ##u' jump in v on left

                push!( K, ( indexjNeigh, indexi, -(.5/J)*dbFnGaussEnds[1,ii] * bFnGaussEnds[2,jj] ) ); ##v' jump in u left neighbor
                push!( K, ( indexjNeigh, indexi, (.5/J)*dbFnGaussEnds[2,jj] * bFnGaussEnds[1,ii] ) ); ##u' jump in v on left


				###LOR##

				##penalty with yourself: v is i and u is j
                push!( K_p1, ( indexi, indexj, penl_p1*bFnGaussEnds_p1[1,ii] * bFnGaussEnds_p1[1,jj] ) ); ##-v(xl) -u(xl)
                push!( K_p1, ( indexi, indexj, penl_p1*bFnGaussEnds_p1[2,ii] * bFnGaussEnds_p1[2,jj] ) ); ## v(xr) u(xr)

                ##penalty w left neighbor
                push!( K_p1, ( indexi, indexjNeigh, -penl_p1*bFnGaussEnds_p1[1,ii] * bFnGaussEnds_p1[2,jj] ) ); ##-v(xl) u_neigh(xr) 
                push!( K_p1, ( indexjNeigh, indexi, -penl_p1*bFnGaussEnds_p1[1,ii] * bFnGaussEnds_p1[2,jj] ) ); ##v_neigh(xr) -u(xl)

                #flux with yourself
                push!( K_p1, ( indexi, indexj, (.5)*dbFnGaussEnds_p1[1,ii] * bFnGaussEnds_p1[1,jj] ) ); ##v' jump in u on left
                push!( K_p1, ( indexi, indexj, -(.5)*dbFnGaussEnds_p1[2,ii] * bFnGaussEnds_p1[2,jj] ) ); ##v' jump in u on right

                push!( K_p1, ( indexi, indexj, (.5)*dbFnGaussEnds_p1[1,jj] * bFnGaussEnds_p1[1,ii] ) ); ##u' jump in v on left
                push!( K_p1, ( indexi, indexj, -(.5)*dbFnGaussEnds_p1[2,jj] * bFnGaussEnds_p1[2,ii] ) ); ##u' jump in v on right

                #flux with left neighbor j
                push!( K_p1, ( indexi, indexjNeigh, -(.5)*dbFnGaussEnds_p1[1,ii] * bFnGaussEnds_p1[2,jj] ) ); ##v' jump in u left neighbor
                push!( K_p1, ( indexi, indexjNeigh, (.5)*dbFnGaussEnds_p1[2,jj] * bFnGaussEnds_p1[1,ii] ) ); ##u' jump in v on left

                push!( K_p1, ( indexjNeigh, indexi, -(.5)*dbFnGaussEnds_p1[1,ii] * bFnGaussEnds_p1[2,jj] ) ); ##v' jump in u left neighbor
                push!( K_p1, ( indexjNeigh, indexi, (.5)*dbFnGaussEnds_p1[2,jj] * bFnGaussEnds_p1[1,ii] ) ); ##u' jump in v on left
				
				

			end

		end

		##LDG fluxes - need to fix this for all node types
		##v left u left for me and v right u left for neighbor..I think this is correct
		for ii = 1:length(elem.mIndex)
			indexi = elem.mIndex[ii];
			for jj = 1:length(elem.mIndex)
				val = -bFnGaussEnds[1,ii] * bFnGaussEnds[1,jj];
				indexj = elem.mIndex[jj];
				push!( Kr, ( indexi, indexj, val ) );
			end
			for jj = 1:length(elem.mIndex)
				val = bFnGaussEnds[2,ii] * bFnGaussEnds[1,jj];
				indexj = nextElem.mIndex[jj];
				push!( Kr, ( indexi, indexj, val ) );
			end
		end

		for ii = 1:length(elem.mIndex)
			indexi = elem.mIndex[ii];
			for jj = 1:length(elem.mIndex)
				val = -bFnGaussEnds_p1[1,ii] * bFnGaussEnds_p1[1,jj];
				indexj = elem.mIndex[jj];
				push!( Kr_p1, ( indexi, indexj, val ) );
			end
			for jj = 1:length(elem.mIndex)
				val = bFnGaussEnds_p1[2,ii] * bFnGaussEnds_p1[1,jj];
				indexj = nextElem.mIndex[jj];
				push!( Kr_p1, ( indexi, indexj, val ) );
			end
		end

		##LDG for p1 fluxes - only use end point nodes
		# push!(Kr_p1, (elem.mIndex[1], elem.mIndex[1], -vlxl*vlxl)  ); ##only left basis function alive there
		# push!(Kr_p1, (elem.mIndex[1], nextElem.mIndex[1], vrxr*vlxl)  ); ##right basis function with right neighbor

	end

    
	Minv = sparse( (x->x[1]).(Minv), (x->x[2]).(Minv), (x->x[3]).(Minv), length(x1), length(x1) ); droptol!( Minv, 1e-12 );
	M = sparse( (x->x[1]).(M), (x->x[2]).(M), (x->x[3]).(M), length(x1), length(x1) ); droptol!( M, 1e-12 );
	
	K = sparse( (x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), length(x1), length(x1) ); droptol!( K, 1e-12 );
	Kright = sparse( (x->x[1]).(Kr), (x->x[2]).(Kr), (x->x[3]).(Kr), length(x1), length(x1) ); droptol!( Kright, 1e-12 );

	#P = sparse( (x->x[1]).(P), (x->x[2]).(P), (x->x[3]).(P), length(x1), length(gwsP)*n ); droptol!( P, 1e-12 );
	#W = sparse( (x->x[1]).(W), (x->x[2]).(W), (x->x[3]).(W), length(gwsP)*n, length(gwsP)*n ); droptol!( W, 1e-12 );

	Minv_p1 = sparse( (x->x[1]).(Minv_p1), (x->x[2]).(Minv_p1), (x->x[3]).(Minv_p1), length(x1), length(x1) ); droptol!( Minv_p1, 1e-12 );
	M_p1 = sparse( (x->x[1]).(M_p1), (x->x[2]).(M_p1), (x->x[3]).(M_p1), length(x1), length(x1) ); droptol!( M_p1, 1e-12 );
	
	K_p1 = sparse( (x->x[1]).(K_p1), (x->x[2]).(K_p1), (x->x[3]).(K_p1), length(x1), length(x1) ); droptol!( K_p1, 1e-12 );
	Kright_p1 = sparse( (x->x[1]).(Kr_p1), (x->x[2]).(Kr_p1), (x->x[3]).(Kr_p1), length(x1), length(x1) ); droptol!( Kright_p1, 1e-12 );


	L = SparseMatrixCSC;
	L_p1 = SparseMatrixCSC;
	if(type == "SIP")
		L = Minv * K;
		L_p1 = Minv_p1 * K_p1;
	elseif(type == "LDG")
		L = Minv * Kright' * Minv * Kright;
		L_p1 = Minv_p1 * Kright_p1' * Minv_p1 * Kright_p1;
	else
		println("not implemented yet");
	end 

    LB = L[1:p+1, 1:p+1];
	LB_p1 = L_p1[1:p+1, 1:p+1];
    #WB = Diagonal( W1d[1:p+1, 1:p+1] );
	
	if(LOR)
		return L_p1, M_p1, LB_p1;
	end
	return L, M, LB; #P, W;
end

function makeFDOperator(p,n)
    h = 1.0 / (n);

    ##generate FD stencil
    w = p / 2;
    xi = -w:w;
    np = length(xi);
    M = reduce(hcat, [ [xi[i]^s for s = 0:p] for i = 1:np ] )'; ##vandermonde
    g = zeros(1,p+1); g[3] = -2; ##want -dxx at 0
    s = g * pinv(M) / h^2; ##stencil
    #s = [1 -16 30 -16 1] / (12*h^2);

    rowidx = reduce(vcat,[repeat([i],np) for i=1:n]);
    colidx = round.(Int64, reduce(vcat,[ (i-w:i+w) for i=1:n]) );
    ##fix periodic points
    colidx = [ (idx<=0 ? n+idx : idx) for idx in colidx];
    colidx = [ (idx>n ? idx%n : idx) for idx in colidx];

    A = sparse( rowidx , colidx , reshape( repeat(s',n),np*n,) ,n,n);

    return A;
end

##make quadratic WLS operator
##for shared nodes, do two-sided fit with 0 jump and use left and right polynomials
##experiment with basis functions to see what works
function makeWLSOperator(n,p,nodeFunc)
    mesh = [ x for x in LinRange( 0.0, 1.0, n+1 ) ];
	#nodes = chebyshevNodes( p+1 ); 
    x1 = zeros( n*(p+1) );
    nodes, weights = nodeFunc(p+1); ##have flexible choice of nodes

    nodesReg = (nodes ./ 2 ) .+ .5;

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

    ##I think we just need to create stencil for one element and then repeat it
    #L_elem = zeros(p+1,p+1); ##local laplacian
    L = zeros(n * (p+1), n*(p+1));
    stencilMat = zeros(p+1, 6 );

	α=2;

	rad=2;

    for i = 1:p
        if(i == 1) ##left stencil, talk to neighbors. try to enforce 0 jump
            # M = zeros(8,6);
			# W = zeros(8,8);
            # r=1;
            # for j = p-1:p+1 ##left hand neighbors
			# 	W[r,r] = 1 / (abs((nodesReg[j]-1)) +1)^α;
            #     for q=0:2
            #         M[r,q+1] = (nodesReg[j]-1)^q;
            #     end
            #     r+=1;
            # end
            # for j = i:i+2
			# 	W[r,r] = 1 / (abs((nodesReg[j])) +1)^α;
            #     for q=0:2
            #         M[r, 3 + q+1] = (nodesReg[j])^q; 
            #     end
            #     r+=1;
            # end
            # ##enforce jump condition at boundary
            # M[7,1]=1;
            # M[7,4]=-1;
			# W[7,7]=1;

			# M[8,2]=1; M[8,5]=-1;
			# W[8,8]=1;

            # sr = [0 0 -2 0 0 0]*pinv(W*M)*W;
            # sl = [0 0 0 0 0 -2]*pinv(W*M)*W;

            # stencilMat[1, 1:6] = sl[1:6]; ##stencil for left hand node
            # stencilMat[p+1, 1:6] = sr[1:6]; ##stencil for right hand node

			##FOR CONTINUOUS BASIS FUNCTIONS:
			WL = zeros(6,6); WR = zeros(6,6);
			M = zeros(6,3);
			r=1;
            for j = p-1:p+1 ##left hand neighbors
                for q=0:2
                    M[r,q+1] = (nodesReg[j]-1)^q;
                end
                r+=1;
            end

			for j = i:i+2
				for q=0:2
					M[r, q+1] = (nodesReg[j])^q; 
				end
				r+=1;
			end

			WL[3,3]=1; WR[2,2]=1;
			WL[4,4]=1; WR[3,3]=1;
			WL[5,5]=1; WR[4,4]=1;

			sr = [0 0 -2]*pinv(WR*M)*WR;
            sl = [0 0 -2]*pinv(WL*M)*WL;

            stencilMat[1, 1:6] = sl[1:6]; ##stencil for left hand node
            stencilMat[p+1, 1:6] = sr[1:6]; ##stencil for right hand node
        else
            M = zeros(3,3);
            for j = i-1:i+1
                for q = 0:2
                    M[j-(i)+2,q+1] = nodesReg[j]^q;
                end
            end

            s = [0 0 -2.0]*inv(M);
            #L_elem[i,i-1:i+1] = s;

            stencilMat[i,1:3] = s;
        end

    end

    h = 1.0 / n;

    ##make operator 
    L = Tuple{Int64,Int64,Float64}[];

    for elem in elems
        prevElem = elems[end];
		if elem.mID != 1
			prevElem = elems[elem.mID-1];
		end

		nextElem = elems[1];
		if elem.mID != length(elems)
			nextElem = elems[elem.mID+1];
		end

        for q=1:3
            push!(L, (elem.mIndex[1], prevElem.mIndex[q + (p-2) ], stencilMat[1, q]) );
            push!(L, (elem.mIndex[1], elem.mIndex[q], stencilMat[1, q+3]) );

            push!(L, (elem.mIndex[p+1], elem.mIndex[q + (p-2)], stencilMat[p+1,q]) );
            push!(L, (elem.mIndex[p+1], nextElem.mIndex[q], stencilMat[p+1,q+3]) );

        end

        for i = 2 : p
            for q = 1:3
                push!(L, (elem.mIndex[i], elem.mIndex[i+q-2],stencilMat[i,q]) );
            end
        end
    end

    L = sparse( (x->x[1]).(L), (x->x[2]).(L), (x->x[3]).(L), n*(p+1), n*(p+1) ); droptol!( L, 1e-12 );

    L = L / h^2; 
    M = I(n * (p+1));
    LB = L[1:p+1, 1:p+1];

    return L,M,LB;

end

##try both higher order and wider stencils as preconditioner
##idea is to use a true MLS basis - find widest rad you need for this order
##and use as weight kernel basically
function makeFlexWLSOperator(n,p,nodeFunc,order)
    mesh = [ x for x in LinRange( 0.0, 1.0, n+1 ) ];
	#nodes = chebyshevNodes( p+1 ); 
    x1 = zeros( n*(p+1) );
    nodes, weights = nodeFunc(p+1); ##have flexible choice of nodes

    nodesReg = (nodes ./ 2 ) .+ .5;

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

    ##I think we just need to create stencil for one element and then repeat it
    #L_elem = zeros(p+1,p+1); ##local laplacian
    L = zeros(n * (p+1), n*(p+1));

	##find stencil width for this order
	##basically we need to find maximum width of interval that contains order+1 points centered at each point
	##use same #of points on each side
	nodesRegPer = [nodesReg .- 1;nodesReg;nodesReg .+ 1];
	maxRad=0;
	for i=p+2:2p+2 ##in the middle
		np = 1;
		radL=0;
		radR=0;
		rc=i+1; lc = i-1;
		while(np < order+1)
			radR=nodesRegPer[rc] - nodesRegPer[i];
			radL=nodesRegPer[i] - nodesRegPer[lc];

			np+=2;
			rc+=1; lc-=1;
		end
		maxRad = max(maxRad, radL, radR);
	end

	α=1;
	stencilMat = zeros(p+1,3p+3); ##stencil matrix

	maxRad *= 1.2; ##so we dont have zeros on the ends


    #w(x) = maxRad==1 ? 1 : (abs(x) >= maxRad ? 0 : exp(α / ( (x^2)/(maxRad^2) - 1.0 ) ) * exp(α)); ##bump function in rad
	
	
	w(x) = maxRad>=1 ? 1 : (abs(x) >= maxRad ? 0 : (1 - 3*(abs(x)/maxRad)^2 + 2*(abs(x)/maxRad)^3  ) );
    dw(x) = maxRad>=1 ? 0 : (abs(x) >= maxRad ? 0 : sign(x)*(-6*(abs(x)/maxRad^2) + 6*(abs(x)^2/maxRad^3)  ) );
	ddw(x) = maxRad>=1 ? 0 : (abs(x) >= maxRad ? 0 : (-12*(1.0/maxRad^2) + 12*(abs(x)/maxRad^3)  ) );##chain rule

	for i=p+2:2p+2
		M = zeros(3p+3,order+1);
		W = zeros(3p+3,3p+3);
		DW = zeros(3p+3,3p+3);
		DDW = zeros(3p+3,3p+3);
        ##find neighborbood of points
		for k = 1:length(nodesRegPer) ##if in neighborhood, put in matrix
			d = abs(nodesRegPer[i] - nodesRegPer[k]);
			if(d <= maxRad)
				for q=0:order
					M[k,q+1] = nodesRegPer[k]^q;
				end
				W[k,k] = w(d);#(d+1)^(-α);#w(d);#exp(-α*d); #(d+1)^(-α);
				DW[k,k] = -dw(d);
				DDW[k,k] = -ddw(d);
			end
		end

		g = zeros(1,order+1); ##dxx
		gd = zeros(1,order+1); ##dx
		gv = zeros(1,order+1); 
		for q=0:order
			g[1,q+1] = q>1 ? q*(q-1)*nodesRegPer[i]^(q-2) : 0.0;
			gd[1,q+1] = q>0 ? q*nodesRegPer[i]^(q-1) : 0.0;
			gv[1,q+1] = nodesRegPer[i]^q;
		end

		A = M' * W * M;
		B = M' * W;
		Ainv = inv(A);
		P = Ainv * B;
		DAinv = -Ainv * (M'*DW*M) * Ainv;
		#DDAinv = -Ainv * (M'*DDW*M) * Ainv;
		DDAinv = -DAinv * (M'*DW*M) * Ainv  -Ainv * (M'*DDW*M) * Ainv - Ainv * (M'*DW*M) * DAinv

        DB = M' * DW;
		DDB = M' * DDW;
		DP = DAinv * B + Ainv * DB;
		DDP = DDAinv * B + DAinv*DB + DAinv*DB + Ainv * DDB;

		s = -g * P; ##2 spatial derivative of polynomial
		#s -= 2 * gd * DP; ##d poly d coef
		#s -= gv*DDP; ##2 derivs of coefficient at stencil center

		stencilMat[i-p-1,:]=s;
		# println(cond(A));
		# println((s*(nodesRegPer.^3)/6));
    end

    h = 1.0 / n;

    ##make operator 
    L = Tuple{Int64,Int64,Float64}[];

    for elem in elems
        prevElem = elems[end];
		if elem.mID != 1
			prevElem = elems[elem.mID-1];
		end

		nextElem = elems[1];
		if elem.mID != length(elems)
			nextElem = elems[elem.mID+1];
		end

        for i = 1 : p+1
            for k=1:p+1
				push!(L, (elem.mIndex[i], prevElem.mIndex[k],stencilMat[i,k]) );
                push!(L, (elem.mIndex[i], elem.mIndex[k],stencilMat[i,k+p+1]) );
				push!(L, (elem.mIndex[i], nextElem.mIndex[k],stencilMat[i,k+2p+2]) );
            end
        end
    end

    L = sparse( (x->x[1]).(L), (x->x[2]).(L), (x->x[3]).(L), n*(p+1), n*(p+1) ); droptol!( L, 1e-12 );

    L = L / h^2; 
    M = I(n * (p+1));
    LB = L[1:p+1, 1:p+1];

    return L,M,LB;

end

##make CG grid and do p=1
function makeCGOperator(n,p,nodeFunc)
	mesh = [ x for x in LinRange( 0.0, 1.0, n+1 ) ];
    x1 = zeros( n*(p+1) );
    nodes, weights = nodeFunc(p+1); ##have flexible choice of nodes

	######################################################################
	###	Set up basis functions
	######################################################################

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
	### Assemble global matrices in 1D
	######################################################################
	M = Tuple{Int64,Int64,Float64}[];

	K = Tuple{Int64,Int64,Float64}[];

	for elem in elems
		J = abs( elem.mX[2]-elem.mX[1] ) * 0.5; ##transform 0:h to -1:1

        prevElem = elems[end];
		if elem.mID != 1
			prevElem = elems[elem.mID-1];
		end

		nextElem = elems[1];
		if elem.mID != length(elems)
			nextElem = elems[elem.mID+1];
		end

		##for each node in mesh, talks to 2 neighbors

		for ii = 1:length(elem.mIndex)
			indexi = elem.mIndex[ii];
			indexL= indexR= nodeL= nodeR=0;
			if ii == 1
				indexL = prevElem.mIndex[end];
				nodeL = prevElem.mNodes[end];
			else
				indexL = elem.mIndex[ii-1]
				nodeL = elem.mNodes[ii-1]
			end

			if ii == length(elem.mIndex)
				indexR = nextElem.mIndex[1];
				nodeR = nextElem.mNodes[1];
			else
				indexR = elem.mIndex[ii+1];
				nodeR = elem.mNodes[ii+1];
			end

			if indexi == 1
				nodeL = nodeL-1;
			end

			if indexi == n*(p+1)
				nodeR = 1 + nodeR;
			end

			node = elem.mNodes[ii];
			
			##integrals over left element
			push!(M, (indexi, indexL, (1/6)*(node - nodeL) ) );
			push!(K, (indexi, indexL, -1.0 / (node - nodeL) ) );
			##integral over right element
			push!(M, (indexi, indexR, (1/6)*(nodeR - node) ) );
			push!(K, (indexi, indexR, -1.0 / (nodeR - node) ) );
			##integral of me against me over both elems
			push!(M, (indexi, indexi, (1/3)*(nodeR - nodeL)));
			push!(K, (indexi, indexi, 1.0 / (node - nodeL)));
			push!(K, (indexi, indexi, 1.0 / (nodeR - node)));

		end
	end

    
	#Minv = sparse( (x->x[1]).(Minv), (x->x[2]).(Minv), (x->x[3]).(Minv), length(x1), length(x1) ); droptol!( Minv, 1e-12 );
	M = sparse( (x->x[1]).(M), (x->x[2]).(M), (x->x[3]).(M), length(x1), length(x1) ); droptol!( M, 1e-12 );
	
	K = sparse( (x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), length(x1), length(x1) ); droptol!( K, 1e-12 );

	return M,K; #P, W;
end

function setUpVecs(n,p,nodeFunc)
	mesh = [ x for x in LinRange( 0.0, 1.0, n+1 ) ];
	#nodes = chebyshevNodes( p+1 ); 
    x1 = zeros( n*(p+1) );
    nodes, weights = nodeFunc(p+1); ##have flexible choice of nodes

    nodesReg = (nodes ./ 2 ) .+ .5;

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

	N = n*(p+1);
	u=zeros(Float64,N,);
	b=zeros(Float64,N,);
	bV=zeros(Float64,N,);
	v=zeros(Float64,N,);

	k = 1;
	uf(x) = exp( sin(2*pi*x*k) ) - 1;# x^2 - x ;#x; #exp( sin(2*pi*x*k) ) - 1;
	rhsf(x) = 4 * k^2 * pi^2 * exp( sin(2*pi*x*k) ) *  (-cos(2*k*pi*x)^2 + sin(2*k*pi*x));
	vf(x) = 1;##.1*exp( sin(x) );

	for elem in elems
		for i=1:p+1
			u[elem.mIndex[i]] = uf(elem.mNodes[i]);
			b[elem.mIndex[i]] = rhsf(elem.mNodes[i]);
			bV[elem.mIndex[i]] = rhsf(elem.mNodes[i]) + vf(elem.mNodes[i]) * uf(elem.mNodes[i]);
			v[elem.mIndex[i]] = vf(elem.mNodes[i]);
		end

	end

	return u,b,bV,v;


end

function jacobiSmoother(A,b);
	Dinv = Diagonal(1 ./ diag(A));
	x = 0.0 * b;
	for i = 1:20
		r = b - A*x;
		println(norm(r));
		x += Dinv*r;
	end

	return x;
end

##outputs p uniform nodes on -1:1
function uniform_nodes(p)
	h = 1.0 / (p);
	nodes = .5*h:h:(1-.5*h);
	return ( (Vector(nodes).*2) .- 1 ), ones(p,)
end


n = 32;
p = 6;
nodeFunc = gaussradau ##make uniform grid to check
type = "SIP" ##need to implement BR2?



L_LOR, M_LOR, LB_LOR = setUp1D(n,p,nodeFunc, type, true);
u_exact, b, bV, v = setUpVecs(n,p,nodeFunc); ##exact solution and potential vals

K_LOR = M_LOR * L_LOR
L, M, LB = setUp1D(n,p,nodeFunc, type, false);

K = M*L;

L_FD = makeFDOperator(2, (p+1)*n)

L_WLS, M_WLS, LB_WLS = makeWLSOperator(n,p,nodeFunc);
L_WLS2, M_WLS2, LB_WLS2 = makeFlexWLSOperator(n,p,nodeFunc,2); ##2 means use quadratics

M_CG, K_CG = makeCGOperator(n,p,nodeFunc);

ev_LOR = eigvals(Matrix(L_LOR))
ev_WLS = eigvals(Matrix(L_WLS));
ev_WLS2 = eigvals(Matrix(L_WLS2));
ev = eigvals(Matrix(L));

u = (L+Diagonal(v) ) \ bV;
u_WLS = (L_WLS+Diagonal(v) ) \ bV;
u_WLS2 = (L_WLS2+Diagonal(v) ) \ bV;
u_LOR = (L_LOR + Diagonal(v)) \ bV;

A = L + Diagonal(v);
A_LOR = L_LOR + Diagonal(v);
A_WLS = L_WLS + Diagonal(v);
A_WLS2 = L_WLS2 + Diagonal(v);

println("Error: ", maximum(abs.(u-u_exact)));
println("Error WLS: ", maximum(abs.(u_WLS-u_exact)));
println("Error WLS2: ", maximum(abs.(u_WLS2-u_exact)));
println("Error LOR: ", maximum(abs.(u_LOR-u_exact)));

Mhalf = Diagonal(sqrt.(diag(M)) );
Minvhalf = Diagonal(sqrt.(1 ./ diag(M)) );

J_WLS = Diagonal(1 ./ diag(A_WLS) ) * A_WLS - I;
J_WLS2 = Diagonal(1 ./ diag(A_WLS2) ) * A_WLS2 - I;
ev_JWLS = eigvals(Matrix(J_WLS));
#J_WLS_MinvHalf = Diagonal(1 ./ diag(Mhalf*A_WLS*Minvhalf) ) * Mhalf*A_WLS*Minvhalf - I;
J = Diagonal(1 ./ diag(A) ) * A - I;

A_pWLS = Matrix(A_WLS) \ Matrix(A);
bV_pWLS = Matrix(A_WLS) \ bV;

A_pLOR = Matrix(A_LOR) \ Matrix(A);
bV_pLOR = Matrix(A_LOR) \ bV;

A_pWLS2 = Matrix(A_WLS2) \ Matrix(A);
bV_pWLS2 = Matrix(A_WLS2) \ bV;
# L_pWLS2 = pinv(Matrix(L_WLS2)) * Matrix(L);
# A_pWLS2 = L_pWLS2 + Diagonal(v);
scatter(real.(eigvals(A_pWLS2)), imag.(eigvals(A_pWLS2)) )
scatter(real.(eigvals(Matrix(A))), imag.(eigvals(Matrix(A))) )

pK = pinv(Matrix(K_CG)) * Matrix(K)

eigvecs_CG  = eigvecs(Matrix(K_CG), Matrix(M_CG));
ev_CG = eigvals(Matrix(K_CG), Matrix(M_CG));

E_CG = L * eigvecs_CG - eigvecs_CG * Diagonal(ev_CG);


# println("precond GMres")
# x = gmres(Diagonal(1 ./ (diag(A_pWLS2)))* A_pWLS2, Diagonal(1 ./ (diag(A_pWLS2)))*bV_pWLS2;verbose=true,reltol=1e-12);
# println(norm(A*x - bV))
# # println("done.")

# println("not precond GMres")
# x = gmres(A, bV;verbose=true,reltol=1e-12);
# println(norm(A*x - bV))

# eigvecs_WLS2 = eigvecs(Matrix(L_WLS2));
# eigvecs_L = eigvecs(Matrix(L));
# P_L = inv(eigvecs_WLS2) * L * eigvecs_WLS2;
# E_WLS2 = L * eigvecs_WLS2 - eigvecs_WLS2 * Diagonal(ev_WLS2);



# Lu = L_WLS2 * u_exact;
# t = Lu - b;
# println("truncation error of laplacian: ", maximum(abs.(t)));



###need to design some scheme that has good small evals
println("seems like for imaginary evecs the imaginary part is just phase shifted");