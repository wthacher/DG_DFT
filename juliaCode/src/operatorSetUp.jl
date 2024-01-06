###sets up operators to solve (L + V) u = f 
using LinearAlgebra
using SparseArrays
using Kronecker
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


struct Elem
	mID::Int64
	mX::Vector{Float64}
	mNodes::Vector{Float64}
	mIndex::Vector{Int64}
	Elem( ii,x,y,p ) = new( ii,[x,y], zeros(p+1), zeros(Int64,p+1) )
end

##make piecewise polynomial DG operator
function makePPLocals(nodes, plocal)
    ##integrate using lagrange polynomials
    numNodes = size(nodes,1);
    numElem = trunc(Int64, (numNodes-1) / plocal);

    Klocal = zeros(numNodes, numNodes);
    Mlocal = zeros(numNodes, numNodes);
    Krlocal = zeros(numNodes, numNodes);
    bEnds = zeros(2,numNodes);
    dBEnds = zeros(2,numNodes);

    for i =1:numElem
        xLeft = i==1 ? 0 : nodes[plocal*(i-1)+1];
        xRight = i==numElem ? 1 : nodes[plocal*(i)+1];

        nodeslocal = nodes[plocal*(i-1)+1:plocal*(i)+1];

        nodeslocal .-= xLeft; ##make xLeft to be zero
        nodeslocal ./= .5*(xRight-xLeft); ##make interval 2 wide
        nodeslocal .-= 1; ##-1 to 1

        ##shift local node set
        polynodes, dpolynodes = legendre_poly(nodeslocal, plocal); ##values and derivs of legendre polynomials at given points in [-1,1]
        coeffs = inv(polynodes);

        gps, gws = gausslegendre(2*plocal); #this should be fine for mass matrix which has order p times order p

        polyGauss, dpolyGauss = legendre_poly( gps, plocal );
        bFnGauss = polyGauss * coeffs; 
        dbFnGauss = dpolyGauss * coeffs;

        J = .5*(xRight - xLeft);
        idx = plocal*(i-1);
        for ii = 1:plocal+1
            for jj = 1:plocal+1

                Msum = 0.0; Ksum = 0.0; Krsum=0;
                for kk = 1:length(gws)
                    Msum += bFnGauss[kk,ii] * bFnGauss[kk,jj] * gws[kk];
                    Ksum += dbFnGauss[kk,ii] * dbFnGauss[kk,jj] * gws[kk];
                    Krsum += dbFnGauss[kk,ii] * bFnGauss[kk,jj] * gws[kk];
                end

                Mlocal[ii+idx,jj+idx] += Msum*J;
                Klocal[ii+idx,jj+idx] += Ksum/J;
                Krlocal[ii+idx,jj+idx] += Krsum;
            end
	    end

        if(i==1)
            polyEndsL, dpolyEndsL = legendre_poly( [-1], plocal ); 
            bEnds[1,1:plocal+1] = polyEndsL * coeffs;
            dBEnds[1,1:plocal+1] = dpolyEndsL * coeffs /J;
        end
        if(i==numElem)
            polyEndsR, dpolyEndsR = legendre_poly( [1], plocal ); 
            bEnds[2,end-plocal:end] = polyEndsR * coeffs;
            dBEnds[2,end-plocal:end] = dpolyEndsR * coeffs /J; 
        end


    end

    return Klocal, Krlocal, Mlocal, bEnds, dBEnds
end
##make sparse operator
##need p % plocal = 0 for this to work
##left and right sub-elements start at element boundaries
function setUpPPOperator(n,p,nodeFunc,type,plocal,bcType)
    @assert (p%plocal==0) "p % plocal != 0";
    mesh = [ x for x in LinRange( 0.0, 1.0, n+1 ) ];
    x1 = zeros( n*(p+1) );
    nodes, weights = nodeFunc(p+1); ##have flexible choice of nodes

	######################################################################
	###	Create elements 
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

    nodesScaled = .5*(nodes .+ 1)

    Klocal, Krlocal, Mlocal, bEnds, dBEnds = makePPLocals(nodesScaled, plocal);
	if(plocal == p)
		Mlocal = Diagonal(weights);
	end

	if(plocal ===1 )
		Mlocal = (Diagonal(reshape( sum(Mlocal, dims=2),p+1,) )); ##lump mass matrix for LOR
	end
	Minvlocal = inv(Mlocal);
	
	######################################################################
	### Assemble global matrices in 1D
	######################################################################
	M = Tuple{Int64,Int64,Float64}[];
	Minv = Tuple{Int64,Int64,Float64}[];

	K = Tuple{Int64,Int64,Float64}[];
	Kr = Tuple{Int64,Int64,Float64}[];

	for elem in elems
		J = abs( elem.mX[2]-elem.mX[1] ) ; 
        pen = 2*(p+1)^2/J;
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


		for ii = 1:length(elem.mIndex)
			indexi = elem.mIndex[ii];
			for jj = 1:length(elem.mIndex)
				indexj = elem.mIndex[jj];
                indexjNeigh = prevElem.mIndex[jj];
				push!( K, ( indexi, indexj, Klocal[ii,jj]/J ) );
				push!( Kr, ( indexi, indexj, -Krlocal[ii,jj] ) );

				push!( M, ( indexi, indexj, Mlocal[ii,jj]*J ) );
				push!( Minv, ( indexi, indexj, Minvlocal[ii,jj]/J ) );

				##FOR SIP::
                ##penalty with yourself: v is i and u is j
                
                push!( K, ( indexi, indexj, pen*bEnds[1,ii] * bEnds[1,jj] ) ); ##-v(xl) -u(xl)
                push!( K, ( indexi, indexj, pen*bEnds[2,ii] * bEnds[2,jj] ) ); ## v(xr) u(xr)

                ##penalty w left neighbor
                if(bcType == "periodic" || elem.mID != 1 )
                    push!( K, ( indexi, indexjNeigh, -pen*bEnds[1,ii] * bEnds[2,jj] ) ); ##-v(xl) u_neigh(xr) 
                    push!( K, ( indexjNeigh, indexi, -pen*bEnds[1,ii] * bEnds[2,jj] ) ); ##v_neigh(xr) -u(xl)
                end

                #flux with yourself
                fWL = .5; fWR = -.5
                if(bcType == "dirichlet" && elem.mID == 1)
                    fWL = 1; ##outward normal flux is -1 but we subtract it in bilinear form
                elseif(bcType == "dirichlet" && elem.mID == n) 
                    fWR = -1;
                end

                push!( K, ( indexi, indexj, (fWL/J)*dBEnds[1,ii] * bEnds[1,jj] ) ); ##v' jump in u on left
                push!( K, ( indexi, indexj, (fWR/J)*dBEnds[2,ii] * bEnds[2,jj] ) ); ##v' jump in u on right

                push!( K, ( indexi, indexj, (fWL/J)*dBEnds[1,jj] * bEnds[1,ii] ) ); ##u' jump in v on left
                push!( K, ( indexi, indexj, (fWR/J)*dBEnds[2,jj] * bEnds[2,ii] ) ); ##u' jump in v on right

                # #flux with left neighbor j
                if(bcType == "periodic" || elem.mID != 1 )
                    push!( K, ( indexi, indexjNeigh, -(.5/J)*dBEnds[1,ii] * bEnds[2,jj] ) ); ##v' jump in u left neighbor
                    push!( K, ( indexi, indexjNeigh, (.5/J)*dBEnds[2,jj] * bEnds[1,ii] ) ); ##u' jump in v on left

                    push!( K, ( indexjNeigh, indexi, -(.5/J)*dBEnds[1,ii] * bEnds[2,jj] ) ); ##v' jump in u left neighbor
                    push!( K, ( indexjNeigh, indexi, (.5/J)*dBEnds[2,jj] * bEnds[1,ii] ) ); ##u' jump in v on left
                end

			end

		end

		##LDG fluxes - need to fix this for all node types
		##v left u left for me and v right u left for neighbor..I think this is correct
		for ii = 1:length(elem.mIndex)
			indexi = elem.mIndex[ii];
			for jj = 1:length(elem.mIndex)
				val = -bEnds[1,ii] * bEnds[1,jj];
				indexj = elem.mIndex[jj];
				push!( Kr, ( indexi, indexj, val ) );
			end
			for jj = 1:length(elem.mIndex)
				val = bEnds[2,ii] * bEnds[1,jj];
				indexj = nextElem.mIndex[jj];
				push!( Kr, ( indexi, indexj, val ) );
			end
		end
	end

    
	Minv = sparse( (x->x[1]).(Minv), (x->x[2]).(Minv), (x->x[3]).(Minv), length(x1), length(x1) ); droptol!( Minv, 1e-12 );
	M = sparse( (x->x[1]).(M), (x->x[2]).(M), (x->x[3]).(M), length(x1), length(x1) ); droptol!( M, 1e-12 );
	
	K = sparse( (x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), length(x1), length(x1) ); droptol!( K, 1e-12 );
	Kright = sparse( (x->x[1]).(Kr), (x->x[2]).(Kr), (x->x[3]).(Kr), length(x1), length(x1) ); droptol!( Kright, 1e-12 );


	L = SparseMatrixCSC;
	if(type == "SIP")
		L = Minv * K; droptol!( L, 1e-8*(n) );
	elseif(type == "LDG")
		L = Minv * Kright' * Minv * Kright;
        if(bcType == "dirichlet")
            println("DIR BCS NOT IMPLMENTED FOR LDG YET");
        end
	else
		println("not implemented yet");
	end 

    LB = L[1:p+1, 1:p+1];

	return L, M, K, Minv, LB; #P, W;

end
