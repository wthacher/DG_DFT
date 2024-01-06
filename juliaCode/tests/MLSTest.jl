##try out low order finite difference preconditioners to speed things up
using FastGaussQuadrature
using IterativeSolvers
using LinearAlgebra
using Plots
using SparseArrays
using IncompleteLU
using Kronecker
using Preconditioners

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

function setUp1D(n,p,nodeFunc,type,LOR,bcType)
	mesh = [ x for x in LinRange( 0.0, 1.0, n+1 ) ];
	#nodes = chebyshevNodes( p+1 ); 
    x1 = zeros( n*(p+1) );
    nodes, weights = nodeFunc(p+1); ##have flexible choice of nodes

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
	Klocal = zeros( p+1,p+1 );
	Krlocal = zeros( p+1,p+1 );


    gws = weights;
    gwsP = weights;

	bFnGauss = polyNodes * coeffs; ##we are using nodes as integration points
	dbFnGauss = dpolyNodes * coeffs;

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

		for ii = 1:length(elem.mIndex)
			indexi = elem.mIndex[ii];
			for jj = 1:length(elem.mIndex)
				indexj = elem.mIndex[jj];
                indexjNeigh = prevElem.mIndex[jj];
				push!( K, ( indexi, indexj, Klocal[ii,jj]/J ) );
				push!( Kr, ( indexi, indexj, Krlocal[ii,jj] ) );

				push!( M, ( indexi, indexj, Mlocal[ii,jj]*J ) );
				push!( Minv, ( indexi, indexj, Minvlocal[ii,jj]/J ) );
				
				##FOR SIP::
                ##penalty with yourself: v is i and u is j
                push!( K, ( indexi, indexj, pen*bFnGaussEnds[1,ii] * bFnGaussEnds[1,jj] ) ); ##-v(xl) -u(xl)
                push!( K, ( indexi, indexj, pen*bFnGaussEnds[2,ii] * bFnGaussEnds[2,jj] ) ); ## v(xr) u(xr)

                ##penalty w left neighbor 
                if(bcType == "periodic" || elem.mID != 1 )
                    push!( K, ( indexi, indexjNeigh, -pen*bFnGaussEnds[1,ii] * bFnGaussEnds[2,jj] ) ); ##-v(xl) u_neigh(xr) 
                    push!( K, ( indexjNeigh, indexi, -pen*bFnGaussEnds[1,ii] * bFnGaussEnds[2,jj] ) ); ##v_neigh(xr) -u(xl)
                end

                # #flux with yourself
                fWL = .5; fWR = -.5
                if(bcType == "dirichlet" && elem.mID == 1)
                    fWL = 1; ##outward normal flux
                elseif(bcType == "dirichlet" && elem.mID == n) 
                    fWR = -1;
                end

                push!( K, ( indexi, indexj, (fWL/J)*dbFnGaussEnds[1,ii] * bFnGaussEnds[1,jj] ) ); ##v' jump in u on left
                push!( K, ( indexi, indexj, (fWR/J)*dbFnGaussEnds[2,ii] * bFnGaussEnds[2,jj] ) ); ##v' jump in u on right

                push!( K, ( indexi, indexj, (fWL/J)*dbFnGaussEnds[1,jj] * bFnGaussEnds[1,ii] ) ); ##u' jump in v on left
                push!( K, ( indexi, indexj, (fWR/J)*dbFnGaussEnds[2,jj] * bFnGaussEnds[2,ii] ) ); ##u' jump in v on right

                # #flux with left neighbor j
                if(bcType == "periodic" || elem.mID != 1 )
                    push!( K, ( indexi, indexjNeigh, -(.5/J)*dbFnGaussEnds[1,ii] * bFnGaussEnds[2,jj] ) ); ##v' jump in u left neighbor
                    push!( K, ( indexi, indexjNeigh, (.5/J)*dbFnGaussEnds[2,jj] * bFnGaussEnds[1,ii] ) ); ##u' jump in v on left

                    push!( K, ( indexjNeigh, indexi, -(.5/J)*dbFnGaussEnds[1,ii] * bFnGaussEnds[2,jj] ) ); ##v' jump in u left neighbor
                    push!( K, ( indexjNeigh, indexi, (.5/J)*dbFnGaussEnds[2,jj] * bFnGaussEnds[1,ii] ) ); ##u' jump in v on left
                end

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
	end

    
	Minv = sparse( (x->x[1]).(Minv), (x->x[2]).(Minv), (x->x[3]).(Minv), length(x1), length(x1) ); droptol!( Minv, 1e-12 );
	M = sparse( (x->x[1]).(M), (x->x[2]).(M), (x->x[3]).(M), length(x1), length(x1) ); droptol!( M, 1e-12 );
	
	K = sparse( (x->x[1]).(K), (x->x[2]).(K), (x->x[3]).(K), length(x1), length(x1) ); droptol!( K, 1e-12 );
	Kright = sparse( (x->x[1]).(Kr), (x->x[2]).(Kr), (x->x[3]).(Kr), length(x1), length(x1) ); droptol!( Kright, 1e-12 );

	L = SparseMatrixCSC;
	L_p1 = SparseMatrixCSC;
	if(type == "SIP")
		L = Minv * K;
	elseif(type == "LDG")
        if(bcType == "dirichlet")
            println("DIR BCS NOT IMPLMENTED FOR LDG YET");
        end
		L = Minv * Kright' * Minv * Kright;
	else
		println("not implemented yet");
	end 

    LB = L[1:p+1, 1:p+1];
    #WB = Diagonal( W1d[1:p+1, 1:p+1] );
	
	
	return L, M, LB; #P, W;
end
##given nodes in element, what is the minimum radius needed for this order of polynomial
function getMLSrad(nodeslocal, plocal)
    ##just sample a bunch of points and see
    maxRad = max(nodeslocal[plocal+1],1 - nodeslocal[end-plocal] )
    maxPoint = 1; ##Left and right boundaries
    for i =1:1000 ##probably a smarter way to do this but who cares
        point = i/1000; #rand(1)[1];
        nodeDist = sort(abs.(nodeslocal .- point));
        if(nodeDist[plocal+1] > maxRad)
            maxPoint = point;
        end
        maxRad = max(maxRad, nodeDist[plocal+1]);
    end

   # println("Max point at ", maxPoint)

    return maxRad*1.05;
end

function makeMLSLocals(nlocal, nodeslocal, plocal, rad)
    #rad=1;
    xilocal = 0:(1.0/nlocal):1.0;
    h = 1.0 / nlocal;
    xclocal = (xilocal[1:end-1]) .+ .5*h; ##cell centers
    # α = 1;
    # w(x) = rad==1 ? 1 : (abs(x) >= rad ? 0 : exp(α / ( (x^2)/(rad^2) - 1.0 ) ) * exp(α)); ##bump function in rad
    
    #w(x) =  rad>=1 ? 1 : 1.0 - (abs(x)/rad)^(2*plocal);
    #w(x) = rad>=1 ? 1 : (abs(x) >= rad ? 0 : 1 - abs(x)/rad); ##hat function
    #dw(x) = rad>=1 ? 1 : (abs(x) >= rad ? 0 : -sign(x)/rad); ##hat function
    
    w(x) = rad>=1 ? 1 : (abs(x) >= rad ? 0 : (1 - 3*(abs(x)/rad)^2 + 2*(abs(x)/rad)^3  ) );
    dw(x) = rad>=1 ? 0 : (abs(x) >= rad ? 0 : sign(x)*(-6*(abs(x)/rad^2) + 6*(abs(x)^2/rad^3)  ) );
    #w(x) = abs(x) >= rad ? 0 : 1.0 / (abs(x))^(plocal+1);

    mom(k) = k >=0 ? (1.0/(k+1)) * ( (.5*h)^(k+1) - (-.5*h)^(k+1)  ) : 0;

    numNodes = size(nodeslocal,1);
    K= zeros(numNodes, numNodes);
    M= zeros(numNodes, numNodes);
    M2= zeros(numNodes, numNodes);
    Kr= zeros(numNodes, numNodes);
    L=zeros(numNodes, numNodes); ##-dxx phi against phi j ##testing if IBP is satisfied

    HK = zeros(plocal+1,plocal+1); ##cell integrators
    HM = zeros(plocal+1,plocal+1);
    HKr = zeros(plocal+1,plocal+1);

    HL = zeros(plocal+1,plocal+1);
    for p = 0:plocal
        for q = 0:plocal
            HM[p+1, q+1] = mom(p+q);
            HK[p+1, q+1] = p*q*mom(p+q-2);
            HKr[p+1,q+1] = p*mom(p+q-1);
            HL[p+1,q+1] = -1*(p-1)*p*mom(p+q-2);
        end
    end

    condAvg = 0;
    endsError = zeros(nlocal+1,numNodes); ##error in all basis functions
    dEnds = zeros(nlocal,);

    gL = [q*(-.5*h)^(q-1) for q in 0:plocal];
    gR = [q*(.5*h)^(q-1) for q in 0:plocal];
    gLv = [(-.5*h)^(q) for q in 0:plocal];
    gRv = [(.5*h)^(q) for q in 0:plocal];

    # MomCell = zeros(numNodes,plocal+1);
    # for i =1:numNodes
    #     for p = 0:plocal
    #         MomCell[i,p+1] = nodeslocal[i]^p;
    #     end
    # end
    qp = (plocal+6); ##nough for mass matrix ish
    nodesCell, weightsCell = gausslegendre(qp); #or something idk
    nodesCell .*= (.5*h);

    avgD=0;
    for i =1:nlocal
        ##create local moment matrix for each cell
        ##shift centering to xc
        W = zeros(numNodes, numNodes);
        
        Mom = zeros(numNodes, plocal+1);
        xc = xclocal[i];

        for j =1:numNodes #node in nodeslocal
            node = nodeslocal[j];
            d = abs(xc - node);
            if(d <= rad)
                W[j,j] = w(node-xc);
                for p = 0:plocal
                    Mom[j,p+1] = (node-xc)^p;
                end
            end
        end

        # if(cond(W*Mom) > 1e5)
        #     println("Asd")
        # end
        condAvg += cond(W*Mom); ##pretty constant it seems
        P = pinv(W*Mom)*W;
        M2 = M2 + P' * HM * P;
        #K  = K + P' * HK * P;
        Kr  = Kr + P' * HKr * P;
        #L = L + P' * HL * P; 

        endsError[i,:] -= ( (gL' * P) .* (gLv' * P) )';
        endsError[i+1,:] += ( (gR' * P) .* (gRv'*P) )';

        # gLCell = [xilocal[i]^q for q =0:plocal];
        # PCell = pinv(Wcell*MomCell) *Wcell;
        # eL = gLCell' * PCell - gL' * P;
        # if(norm(eL) > 1e-12)
        #     println("Asdf")
        # end

        GV = zeros(size(nodesCell,1), numNodes);
        GVd = zeros(size(nodesCell,1), numNodes);

        #note you cant really work in monomial basis!
        ##need to work in legendre polyonmial basis idiot
        ##scale local nodes to [-1,1] and use legendre_poly function
        ##how do we get deriv of basis functions?
        Mom_L, dMom_L = legendre_poly( (nodeslocal .* 2) .- 1 ,plocal); ##just use -1:1 range
        for k = 1:size(nodesCell,1)
            gp = nodesCell[k];

            W = zeros(numNodes, numNodes);
            DW = zeros(numNodes, numNodes);
            Mom = zeros(numNodes, plocal+1);
            DMom = zeros(numNodes, plocal+1);

            for j =1:numNodes #node in nodeslocal
                node = nodeslocal[j];
                d = abs(xc+gp - node);
                if(d <= rad)
                    W[j,j] = w(node - (xc+gp) );
                    DW[j,j] = -dw( node - (xc+gp) );
                    for p = 0:plocal
                        Mom[j,p+1] = (node-(xc+gp) )^p;
                        DMom[j,p+1] = p*(node-(xc+gp) )^(p-1);
                    end
                end
            end

            A = Mom' * W * Mom;
            B = Mom' * W;
            P = pinv(A) * B;
            Ainv = pinv(A);

            A_L = Mom_L' * W * Mom_L;
            B_L = Mom_L' * W;
            P_L = inv(A_L) * B_L;
            Ainv_L = inv(A_L);
            #P2 = pinv( Diagonal(sqrt.(diag(W)))*Mom) * Diagonal(sqrt.(diag(W))); ##should b the same

            # DA = DMom' * W * Mom + Mom' * DW * Mom + Mom' * W * DMom;
            # DAinv = -Ainv * DA * Ainv;
            # DB = Mom' * DW + DMom' * W;
            DAinv = -Ainv * (Mom'*DW*Mom) * Ainv;
            DB = Mom' * DW; ##if you shift over a little bit, how do things change?

            DP = DAinv * B + Ainv * DB;

            DAinv_L = -Ainv_L * (Mom_L'*DW*Mom_L) * Ainv_L;
            DB_L = Mom_L' * DW; ##if you shift over a little bit, how do things change?

            DP_L = DAinv_L * B_L + Ainv_L * DB_L ;

            #GV[k,:] = P[1,:]; ##selects value of basis functions at gauss point
            #GVd[k,:] = DP[1,:] + P[2,:]; ##deriv of coefs + deriv of polynomial part 
            ##need to also derive derivative of basis function at gauss point

            pc,dc = legendre_poly([2*(xc+gp)-1], plocal);
            Gvv = pc * P_L;
            Gvd = dc * P_L * 2; ##need to adjust units
            Gvp = pc * DP_L;

            GV[k,:] = Gvv;
            GVd[k,:] = Gvd + Gvp;

        end

        nNodes, _ = gausslegendre(qp);
        polyNodes, dpolyNodes = legendre_poly( nNodes, qp-1 );
	    # coeffs = inv( polyNodes );
        GVd2 = dpolyNodes * inv( polyNodes ) * GV; ##fit polynomial to points and take derivative
        GVd = GVd*.5*h; ##scale derivatives to -1:1 interval from -h/2 to h/2 interval

        #println(norm(GVd - GVd2) );
        avgD += maximum(abs.(GVd - GVd2));

        ##scale back to actual integral size
        M = M +  GV' * Diagonal(weightsCell*.5*h) * GV;
        K = K +  GVd2' * Diagonal(weightsCell/(.5*h)) * GVd2;

    end

    #println("norm avg ", avgD/nlocal);

    #println("sdfadsf:", maximum(abs.((M2-M))))

    #println(condAvg/nlocal);

    ##do endpoints
    WL = zeros(numNodes, numNodes);
    WR = zeros(numNodes, numNodes);
    ML = zeros(numNodes, plocal+1);
    MR = zeros(numNodes, plocal+1);
    for j =1:numNodes #node in nodeslocal
        node = nodeslocal[j];
        dL = abs(node);
        dR = abs(1-node);
        if(dL <= rad)
            WL[j,j] = w(node);
            for p = 0:plocal
                ML[j,p+1] = (node)^p;
            end
        end

        if(dR <= rad)
            WR[j,j] = w(node-1);
            for p = 0:plocal
                MR[j,p+1] = (node-1)^p;
            end
        end
    end

    ##replace with legendre basis
    PL = inv(ML'*WL*ML)*ML'*WL; #pinv(WL*ML)*WL;
    PR = inv(MR'*WR*MR)*MR'*WR; #pinv(WR*MR)*WR;

    gv = zeros(1,plocal+1); ##value and deriv and endpoint
    gd = zeros(1,plocal+1);

    for p=0:plocal
        gv[1,p+1] = 0.0^p; 
        gd[1,p+1] = p * (0.0^(p-1) ); 
    end

    gd[1,1]=0;

    bEnds = [gv*PL; gv*PR];
    dBEnds = [gd*PL; gd*PR];

    endsError[1,:] += bEnds[1,:] .* dBEnds[1,:];
    endsError[nlocal+1,:] -= bEnds[2,:] .* dBEnds[2,:];

    #println("max end dif ",maximum(abs.(endsError)) );

    L = L - (dBEnds[1,:]) * (bEnds[1,:]');
    L = L + (dBEnds[2,:]) * (bEnds[2,:]');

    #println("IBP error" , norm(L - K)); ###IBP DOESNT HOLD :))
    
    errorSums = sum(abs.(endsError),dims=2)
    badFace = findall(x -> abs(x)>1e-14, errorSums);

    return K, Kr, M, bEnds, dBEnds    

end

##choose order of MLS operator
##this will determine some minimal stencil width needed at boundaries
##each point in the element needs to have plocal+1 points accesible
##so how do we determine stencil radius? 
##Weighting will be cubic C1 spline
function setUpMLSOperator(n,p,nodeFunc,type,plocal,bcType)
	mesh = [ x for x in LinRange( 0.0, 1.0, n+1 ) ];
	#nodes = chebyshevNodes( p+1 ); 
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

    ###create local grid with nlocal cells:
    nodesScaled = .5*(nodes .+ 1)
    ref =1;
    tol = 1e-11; #?? i dont know lets see how this converges
    err = 2*tol; ##measure relative error in integral somehow
    iter=1;
    maxIter=10;
    KlocalOld =  zeros(p+1,p+1);
    MlocalOld = zeros(p+1,p+1);
    Klocal =zeros(p+1,p+1);
    Mlocal =zeros(p+1,p+1);
    Krlocal =zeros(p+1,p+1);
    KrlocalOld =zeros(p+1,p+1);
    bEnds = dBEnds = zeros(2,p+1);

    ##do some richardson extrapolation to get these to converge?
    KArray = Array{Matrix}(undef, maxIter, maxIter);
    MArray = Array{Matrix}(undef, maxIter, maxIter);
    KrArray = Array{Matrix}(undef, maxIter, maxIter);
    
    rad = getMLSrad(nodesScaled, plocal); ##radius for bump functions
    ##Kr is dx phi against phi j
    hlocal=0;
    while err > tol && iter < maxIter
        nlocal = ref*(p+1);
        Klocal, Krlocal, Mlocal, bEnds, dBEnds = makeMLSLocals(nlocal, nodesScaled, plocal, rad);
        KArray[iter,1] = Klocal;
        MArray[iter,1] = Mlocal;
        KrArray[iter,1] = Krlocal;

        ##do richardson extrapolation:
        t=2;
        tM = 2;
        for j = 2:iter
            KArray[iter,j] = (t^j * KArray[iter,j-1] .- KArray[iter-1,j-1] ) / (t^j - 1);
            KrArray[iter,j] = (t^j * KrArray[iter,j-1] .- KrArray[iter-1,j-1] ) / (t^j - 1);
            MArray[iter,j] = (t^j * MArray[iter,j-1] .- MArray[iter-1,j-1] ) / (t^j - 1);
        end
        if(iter != 1)
            errK = maximum(abs.(KArray[iter,1] - KArray[iter-1,1]));
            errKr = maximum(abs.(KrArray[iter,iter-1] - KrArray[iter-1,iter-1]));
            errM = maximum(abs.(MArray[iter,1] - MArray[iter-1,1]));
            println("errK: ", errK, " errM: ", errM, " errKr: ", errKr);
            err = max(errK,errM);
        end

        # KlocalOld = Klocal;
        # KrlocalOld = Krlocal;
        # MlocalOld = Mlocal;

        ref*=2; iter+=1;
        #hlocal = 1.0 / nlocal;
    end

    Klocal = KArray[iter-1,1];
    Krlocal = KrArray[iter-1,1];
    Mlocal = MArray[iter-1,1];

    #Klocal, Krlocal, Mlocal, bEnds, dBEnds = makeMLSLocals(2^14, nodesScaled, plocal, rad);


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
	#nodes = chebyshevNodes( p+1 ); 
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
    #uf(x) = sin(2*pi*x*k);
    #rhsf(x) = 4 * k^2 * pi^2 * sin(2*pi*x*k);
	rhsf(x) = 4 * k^2 * pi^2 * exp( sin(2*pi*x*k) ) *  (-cos(2*k*pi*x)^2 + sin(2*k*pi*x)); #-2; #4 * k^2 * pi^2 * exp( sin(2*pi*x*k) ) *  (-cos(2*k*pi*x)^2 + sin(2*k*pi*x))
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

n = 48
p = 4


nodeFunc = gausslobatto;
type = "SIP"
bcType = "dirichlet" ##homog dir. BC

L, M, LB = setUp1D(n,p,nodeFunc, type, false, bcType);
L_PP, M_PP, K_PP, Minv_PP, LB_PP = setUpPPOperator(n,p,nodeFunc,type,p,bcType);
L_MLS, M_MLS, K_MLS, Minv_MLS, LB_MLS = setUpMLSOperator(n,p,nodeFunc,type,1,bcType);
println("MAKE P=1 CG Operator for gauss radau nodes this should be easy")


uexact,b,bV,v = setUpVecs(n,p,nodeFunc);

K = M*L; droptol!( K, 1e-10 );

ev_K = eigvals(Symmetric(Matrix(K)));
ev_K_MLS = eigvals(Symmetric(Matrix(K_MLS)));
ev_K_PP = eigvals(Symmetric(Matrix(K_PP)));

err_ev_MLS = (ev_K - ev_K_MLS) ./ ev_K;
err_ev_PP = (ev_K - ev_K_PP) ./ ev_K;

ev = eigvals(Matrix(L));
ev_MLS = eigvals(Symmetric(Matrix(K_MLS)), Symmetric(Matrix(M_MLS)) );
ev_PP = eigvals(Symmetric(Matrix(K_PP)), Symmetric(Matrix(M_PP)) );

evec_MLS = eigvecs(Symmetric(Matrix(K_MLS)), Symmetric(Matrix(M_MLS)) );
evec = eigvecs(Symmetric(Matrix(K)), Symmetric(Matrix(M)) );
evec_PP = eigvecs(Symmetric(Matrix(K_PP)), Symmetric(Matrix(M_PP)) );
#evec_MLS' * M * evec

P_K_PP = evec_PP' * M_PP * K * evec_PP;
P_K_MLS = evec_MLS' * M_MLS * K * evec_MLS;

u_MLS = Matrix(K_MLS) \ (M_MLS*b);
#u_MLS .-= u_MLS[1];

u = Matrix(K) \ (M * b);

u_PP = Matrix(K_PP) \ (M_PP*b);
#u .-= u[1];

# evec_L_PP = eigvecs(Matrix(L_PP));
# ev_L_pp = eigvals(Matrix(L_PP));
# ev_L = eigvals(Matrix(L));
# evec_L = eigvecs(Matrix(L));

# D = L*evec_L_PP - Diagonal(ev_L_PP)*evec_L_PP;
# D_PP = L_PP*evec_L - Diagonal(ev_L)*evec_L;

println("Full DG error: ", maximum(abs.(u-uexact)), ", L2: ", sqrt( (u-uexact)' * M * (u-uexact)));
println("MLS error: ", maximum(abs.(u_MLS-uexact)), ", L2: ", sqrt( abs((u_MLS-uexact)' * M_MLS * (u_MLS-uexact))));
println("PP error: ", maximum(abs.(u_PP-uexact)), ", L2: ", sqrt( abs((u_PP-uexact)' * M_PP * (u_PP-uexact))));

# r = rand(size(u,1),);
println("MLS K  norm error : ", sqrt(uexact' * K_MLS*uexact) - sqrt(uexact' * K*uexact));
println("MLS M norm error: ", sqrt(uexact' * M_MLS*uexact) - sqrt(uexact' * M*uexact) );
# MB_MLS = Matrix(M_MLS[1:p+1,1:p+1])
# MB_PP = Matrix(M_PP[1:p+1,1:p+1])
# MB = Matrix(M[1:p+1,1:p+1])

# eigvec_MB_MLS = eigvecs(MB_MLS);
# eigvec_MB = eigvecs(MB);

# println("sol norm: " , sqrt( abs((u_MLS)' * M_MLS * (u_MLS)) ) )

# t = Matrix(M_MLS) \ (K_MLS * uexact) - b;
# println("truncation error max norm: ", maximum(abs.(t)));
# x = cg(K, M*b ; verbose=true);
println("MLS Precon")
ILU_MLS = ilu(K_MLS; τ= .000 * sum(diag(K))/size(K,1) );

x_MLS = cg(K, (M*b) ; Pl = ILU_MLS, verbose=true);

println(norm(K*x_MLS - M*b));

println("PP Precon")
ILU_PP = ilu(K_PP; τ= .000 * sum(diag(K))/size(K,1) );

x_PP = cg(K, (M*b) ; Pl = ILU_PP, verbose=true);

println(norm(K*x_PP - M*b));

println("AMG Precon")
K_AMG = AMGPreconditioner{RugeStuben}(K_PP)
x_ = cg(K, (M*b) ; Pl = K_AMG, verbose=true);

println(norm(K*x_ - M*b));

# K3D_MLSx = K_MLS⊗M_MLS⊗M_MLS;# + M_MLS⊗K_MLS⊗M_MLS + M_MLS⊗M_MLS⊗K_MLS;
# K3D_MLSy = M_MLS⊗K_MLS⊗M_MLS;
# K3D_MLSz = M_MLS⊗M_MLS⊗K_MLS;

# K3Dx = K⊗M⊗M;

# ILUL = ilu(L_MLS);
# x = gmres(L,b;Pl=ILUL,verbose=true)

##experiment with Different preconditioners of different levels!! SEE what works best


##solve eval problem for low order system, then do galerkin projection
##and see what eigvectors you get from that
nev=10;
H = evec_MLS[:,1:nev];
KG  =  H' * K * H;
MG = H' * M * H;
evec_G = eigvecs(KG, MG);
evec_G = evec_G * Diagonal(diag(evec_G) ./ abs.(diag(evec_G)) ) ; #scale columns right
evec_Proj = H*evec_G;
D_Proj_MLS = evec_Proj - evec[:,1:nev];

H = evec_PP[:,1:nev];
KG  =  H' * K * H;
MG = H' * M * H;
evec_G = eigvecs(KG, MG);
evec_G = evec_G * Diagonal(diag(evec_G) ./ abs.(diag(evec_G)) ) ; #scale columns right
evec_Proj = H*evec_G;
D_Proj_PP = evec_Proj - evec[:,1:nev];

PK_PP = Matrix(K_PP) \ Matrix(K);
println("PP precondtioned stiffness cond: " , cond(PK_PP));

PK_MLS = Matrix(K_MLS) \ Matrix(K);
println("MLS precondtioned stiffness cond: " , cond(PK_MLS));

PM_PP = Matrix(M_PP) \ Matrix(M);
println("PP precondtioned mass cond: " , cond(PM_PP));

PM_MLS = Matrix(M_MLS) \ Matrix(M);
println("MLS precondtioned mass cond: " , cond(PM_MLS));