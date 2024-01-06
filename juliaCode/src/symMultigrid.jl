include("laplacePrecond.jl");
using IncompleteLU;
using IterativeSolvers;
##multigrid solver on p hierarchy
##keeps track of time spent in each stage
struct symMultigridPrecond
    p ##order at highest level
    n ##grid spacing at highest level
    OpVect::Array{Operator} ##holds Operator on different levels
    TensorOpVect::Array{Operator} ##holds tensorOperators
    InterpVect::Array{Matrix} ##holds interpolation and restriction. Interpolation is exact transfer to new nodes
    RestrictVect::Array{Matrix}
    numCycles ##number of vCycles
    pVect::Array{Int64} ##values in p hierarchy
    verbose::Bool ##do you want a lot of output
    timings::Vector{Float64} ##vector with timings for dif stages
    eigTimings::Vector{Float64} ##timings for eig solver stages
    LUVect::Array{IncompleteLU.ILUFactorization} ##store all LU decomps at all levels
    EvecVect::Array{Kronecker.KroneckerProduct,2}##eigvectors for blocks
    EvalVect::Array{Vector{Float64} }##diagonal of eigvalues
    ACoarse::LinearAlgebra.LU ##store factorization at bottom level
end
#put answer in y. i guess need to do this component wise
function doSymMultigrid!( y, P::symMultigridPrecond, b )
	numCycles = P.numCycles;

    i = 1;
    rat = 1;
    tol = 1e-14; #or something
    if(any(isnan, y) ) ##because cg is dumb?
        y .= zeros(size(y));
    end

    while (i <= numCycles) && (rat > tol)
        y .= vCycle(y, P,1, b);
        # if(true)
        #     r = b - (P.OpVect[1]) * y;
        #     rat = norm(r)/norm(b);
            
        #     println("norm(r)/norm(b)  ",rat, " norm(r) ", norm(r));
        # end
        i +=1;
    end
	return y;
end

##solve BLOCK system and put answer in y. PY = B
function LinearAlgebra.ldiv!( Y::AbstractArray, P::symMultigridPrecond, B::AbstractArray )
    Z = (P.OpVect[1]).Perm * (P.OpVect[1]).Minv * (P.OpVect[1]).PermT * B;
    m = size(B,2);
    for i = 1:m
        @views doSymMultigrid!(Y[:,i], P, Z[:,i]);
    end
	return Y;
end

#put answer in x
function LinearAlgebra.ldiv!( P::symMultigridPrecond, x )
    y = 0.0*x;
	z = ldiv!( y, P, x );
    x .= z;
end

function Base.:\( P::symMultigridPrecond, b )
    return ldiv!( P, b );
end

##create hierarchy of symOperators with corresponding vData
##interpolation is exact - just evaluate polynomial at that point
##assume data has already been permuted into block format
function initSymMultigridPrecond(n, p, BC, vData, nodeFunc, opt, LOR, numCycles, verbose)
    #pVect = [trunc(Int64,p/(2^i) ) for i =0:floor(Int64, log(2,p)) ];
    pVect = [p,1];
    # if(LOR)
    #     pVect =  [trunc(Int64,p/(2^i) ) for i =0:floor(Int64, log(2,p)) ];
    # end
    numLevels = length(pVect);
    OpVect = Array{Operator}(undef, numLevels);
    TensorOpVect = Array{Operator}(undef, numLevels);
    InterpVect = Array{Matrix}(undef, numLevels-1); ##[i] is interpolating from i+1 up to i
    RestrictVect = Array{Matrix}(undef, numLevels-1);##[i] is restricting from i to i-1
    LUVect = Array{IncompleteLU.ILUFactorization,2}(undef, numLevels-1, n^3);##not on all levels
    EvecVect = Array{Kronecker.KroneckerProduct,2}(undef, numLevels-1, 2); ##store eigenvectors and inverse
    EvalVect = Array{Vector{Float64} }(undef, numLevels-1);

    ##Create exact interplations and restrictions as transposes - per element
    println("p hierarchy levels: ", pVect);
    ACoarse = lu(1);

    vData_ = copy(vData); ##already in block form
    for i = 1:(numLevels)
        Op = createOperator(n, pVect[i],BC, vData_, nodeFunc, opt, true,LOR); ##block operator
        tensorOp = createOperator(n, pVect[i],BC, Op.PermT*vData_, nodeFunc, opt, false,LOR); ##tensor operator. convert data
        OpVect[i] = Op;
        TensorOpVect[i] = tensorOp;

        if(i == numLevels)
            # print("getMatrix");
            # A = getMatrix(Op);
            # print("lu");
            # ACoarse = lu(Matrix(A) );
            continue;
        end

        nodesFine, weightsFine = nodeFunc(pVect[i]+1);
        nodesCoarse, weightsCoarse = nodeFunc(pVect[i+1]+1);

        polyNodesFine, dpolyNodes = legendre_poly( nodesFine, pVect[i+1] );
	    #coeffsFine = inv( polyNodesFine ); ##polynomial coeffs for basis functions

        polyNodesCoarse, dpolyNodes = legendre_poly( nodesCoarse, pVect[i+1] );
	    coeffsCoarse = inv( polyNodesCoarse ); ##polynomial coeffs for basis functions

        T = polyNodesFine * coeffsCoarse; ##coarse basis functions evaled at fine nodes

        Interp = T;

        ##Restriction is M_c^{-1} T' Mf (from rob paper)
        InterpVect[i]=Interp⊗Interp⊗Interp; 

        MinvTM = Diagonal(1 ./ weightsCoarse ) * T' * Diagonal( weightsFine);
        #println(MinvTM)
        RM = (MinvTM⊗MinvTM⊗MinvTM);
        RestrictVect[i]= RM ;
        
        ##will need to do this a little differently
        ##have same nodes coarse and fine 
        ##interpolation is still evaluation, which is just using the piecewise linears. try transpose for restriction
        if(LOR)
            T = zeros(size(nodesFine,1),size(nodesCoarse,1)); ##takes coarse to fine
            for j = 1:size(nodesFine,1)
                for k = 1:size(nodesCoarse,1)
                    if(nodesCoarse[k] == nodesFine[j]) ##equal
                        T[j,k] = 1.0;
                        continue;
                    elseif(k ==1 && nodesFine[j] < nodesCoarse[k]) ##left of first node
                        T[j,k] = (nodesCoarse[k+1] - nodesFine[j]) / (nodesCoarse[k+1] - nodesCoarse[k]);
                        T[j,k+1] = 1 - T[j,k];
                        continue;
                    elseif(k == size(nodesCoarse,1) && nodesFine[j] > nodesCoarse[k]) ##right of last node
                        T[j,k] = (nodesFine[j] - nodesCoarse[k-1]) / (nodesCoarse[k] - nodesCoarse[k-1]);
                        T[j,k-1] = 
                        continue;
                    elseif(nodesCoarse[k] < nodesFine[j] && nodesCoarse[k+1]>nodesFine[j]) ##between
                        T[j,k+1] = (nodesFine[j] - nodesCoarse[k] ) / (nodesCoarse[k+1] - nodesCoarse[k]);
                        T[j,k] = 1 - T[j,k+1];
                        continue;
                    end 
                end
            end

            # RM = 0*T';
            # for j = 1:size(nodesCoarse,1)
            #     for k = 1:size(nodesFine,1)
            #         if(nodesCoarse[j] == nodesFine[k])
            #             RM[j,k] = 1.0;
            #         elseif(nodesFine[k] < nodesCoarse[j] && nodesFine[k+1]>nodesCoarse[j])
            #             RM[j,k+1] = (nodesCoarse[j] - nodesFine[k] ) / (nodesFine[k+1] - nodesFine[k]);
            #             RM[j,k] = 1 - RM[j,k+1];
            #         end 
            #     end
            # end

            Interp = T;
            #println(T);
            InterpVect[i]=Interp⊗Interp⊗Interp;
            RM = Diagonal( reshape(1 ./ sum(T',dims=2),size(T',1), ) ) * T'; ##make rows sum to 1?
            #println(RM);
            RM = RM⊗RM⊗RM;
            RestrictVect[i]=RM;

        end

        ##want to evaluate coarse basis functions at fine points

        vDataCopy = copy(vData_);
        vData_ = zeros(n^3 * (pVect[i+1]+1)^3,);
        pl = pVect[i+1]+1;
        ph = pVect[i]+1;
        
        ix = 1;
        LB = (Op).LB;
        evecs = eigvecs(Matrix(LB));
        MhalfFine = Diagonal( sqrt.(weightsFine) );
        MinvhalfFine = Diagonal( 1 ./ (sqrt.(weightsFine) ) );

        #println("Using MhalfLBMinvhalf for blocks");
        MhalfLBMinvhalf = MhalfFine * LB * MinvhalfFine;
        evs = eigen(Symmetric(Matrix(MhalfLBMinvhalf) ));

        ev = evs.values;

        ##if using actual blocks
        evecs  = MinvhalfFine * evs.vectors; ##to restore to normal
        evecsinv = evs.vectors' * MhalfFine;

        # println("Using MhalfLBMinvhalf for blocks");
        # evecs = evs.vectors;
        # evecsinv = evs.vectors';

        evecs3 = evecs⊗evecs⊗evecs;
        evecsinv3 = evecsinv⊗evecsinv⊗evecsinv;
        #evecsinvMinv3 = evecsinvMinv⊗evecsinvMinv⊗evecsinvMinv;
        ev3 = diag(Diagonal(ev) ⊕ Diagonal(ev) ⊕ Diagonal(ev));

        EvalVect[i] = ev3;
        EvecVect[i,1] = evecs3;
        EvecVect[i,2] = evecsinv3;

        LBB = LB⊕LB⊕LB; ##diag block of L

        
        MhalfLOR = sqrt.( diag(Op.MB) );
        MinvhalfLOR = 1 ./ MhalfLOR;
        MhalfLBMinvhalfLOR = Diagonal(MhalfLOR) * LB * Diagonal(MinvhalfLOR);

        #LBB = MhalfLBMinvhalfLOR⊕MhalfLBMinvhalfLOR⊕MhalfLBMinvhalfLOR;
        for nz=1:n
            for ny=1:n
                for nx=1:n
                    idl = (nx-1)*pl^3 + (ny-1)*pl^3*n + (nz-1)*pl^3*n^2 + 1;
                    idh = (nx-1)*ph^3 + (ny-1)*ph^3*n + (nz-1)*ph^3*n^2 + 1;
                    vData_[idl:idl+pl^3-1] = RM * vDataCopy[idh:idh+ph^3-1];
                    
                    if(LOR)
                        VD = Diagonal(Op.Vexact[idh:idh+(ph)^3-1]);
                        DB = SparseMatrixCSC(LBB + VD);
                        drop_tol = 0;#sum(diag(DB)) / size(DB,1);
                        LUVect[i,ix] = ilu(DB,τ=.01*drop_tol);
                    end

                    ix+=1;
                end
            end
        end
    end
    timings = zeros(8,); ##relax, restrict, interp, bottom solve, compute res, make blockDiag, compute LU, invert
    eigTimings  = zeros(4,);
    P = symMultigridPrecond(p,n,OpVect,TensorOpVect, InterpVect, RestrictVect, numCycles, pVect, verbose,timings,eigTimings,LUVect,EvecVect, EvalVect,ACoarse );
    return P;
end

##order p scheme
function vCycle(u_init, P::symMultigridPrecond, level, b)
    u = copy(u_init);
    if(level == length(P.OpVect) )
        if(P.verbose)
            println("bottom solve using preconditioned cg");
        end
        t = time();        
        
        #A = getMatrix(((P.OpVect)[level]));
        ##Solve at coarse level using laplace precondioners
        Vavg = sum(((P.OpVect)[level]).Vexact) / size(((P.OpVect)[level]).Vexact,1);
        LaplacePre = laplacePrecond(((P.TensorOpVect)[level]), Vavg, 0, false);
        bT = ((P.OpVect)[level]).PermT * b; ##change from block to tensor
        u = cg(((P.TensorOpVect)[level]), bT;Pl = LaplacePre,reltol=1e-12,maxiter=20);
        u =  ((P.OpVect)[level]).Perm * u; ##from tensor to block

        #u = P.ACoarse \ b;
        #u .-= u[1];
        dt = time() - t;
        P.timings[4] += dt;
    else
        if(P.verbose)
            println("relaxing on level ", level);
        end
        t = time();
        u = relax(P, level, u , b);
        dt = time() - t;
        P.timings[1] += dt;

        t = time();
        #print("hereere")
        r = ( (P.OpVect)[level] )*u;
        r = b - r;
        dt = time() - t;
        P.timings[5] += dt;
       
        if(P.verbose)
            println("restricting from ", level, " to ", level+1);
        end
        t = time();
        rRest = restrict(P, level, r);
        dt = time() - t;
        P.timings[2] += dt;

        
        e = zeros(size(rRest,1),); #0 * rRest;
        e = vCycle(e, P, level+1, rRest);

        if(P.verbose)
            println("interpolating from ", level+1, " to ", level);
        end

        t = time();
        eInterp = interpolate(P, level, e);
        u = u + eInterp;
        dt = time() - t;
        P.timings[3] += dt;

        if(P.verbose)
            r = b - ( (P.OpVect)[level] ) *u;
        end

        if(P.verbose)
            println("relaxing on level ", level);
        end

        t = time();
        u = relax(P, level, u , b);
        dt = time() - t;
        P.timings[1] += dt;
    end
    

    return u;
end

##restrict from fine to coarse
function restrict(P::symMultigridPrecond, level, r)
    R = P.RestrictVect[level];
    #println("size of restrictor at level ", size(R), " ",level);
    n = P.n;
    i = level;
    rRest = zeros(n^3 * (P.pVect[i+1]+1)^3);
    pl = P.pVect[i+1]+1;
    ph = P.pVect[i]+1;
    for nz=1:n
        for ny=1:n
            for nx=1:n
                idl = (nx-1)*pl^3 + (ny-1)*pl^3*n + (nz-1)*pl^3*n^2 + 1;
                idh = (nx-1)*ph^3 + (ny-1)*ph^3*n + (nz-1)*ph^3*n^2 + 1;
                rRest[idl:idl+pl^3-1] = R * r[idh:idh+ph^3-1];
            end
        end
    end

    return rRest;

end

#interpolate from level+1 to level (coarse to fine)
function interpolate(P::symMultigridPrecond, level, e)
    IM = P.InterpVect[level];
    n = P.n;
    i = level;
    eInterp = zeros(n^3 * (P.pVect[i]+1)^3);
    pl = P.pVect[i+1]+1;
    ph = P.pVect[i]+1;
    #println(pl, " ", ph)
    #println("size of interpolator at level : ", size(IM), " ",level)
    for nz=1:n
        for ny=1:n
            for nx=1:n
                idl = (nx-1)*pl^3 + (ny-1)*pl^3*n + (nz-1)*pl^3*n^2 + 1;
                idh = (nx-1)*ph^3 + (ny-1)*ph^3*n + (nz-1)*ph^3*n^2 + 1;
                eInterp[idh:idh+ph^3-1] = IM * e[idl:idl+pl^3-1];
            end
        end
    end

    return eInterp;
end
##is there an easy way to apply P, apply block diag, then apply P^T? Yes I think i did this

##apply block jacobi
##the symOperator owns a single element block diag for laplacian and potential
function relax(P::symMultigridPrecond, level, x, b)
    Op = ( (P.OpVect)[level]);
    p = P.pVect[level];
    n = P.n;

    LB  = Op.LB;
    LBD = LB⊕LB⊕LB;
    
    numSmooth = 10; #dont smooth on high levels unless you want your shit slow
    # if(p < 5)
    #     numSmooth=1
    # end

    ##LOOP THROUGH ELEMENTS
    for i=1:numSmooth
        r = b - Op*x;
        if(P.verbose)
            println("residual norm: ", norm(r)); 
        end
        idx=1;
        for nz=1:n
            for ny=1:n
                for nx=1:n
                    idl = (nx-1)*(p+1)^3 + (ny-1)*(p+1)^3*n + (nz-1)*(p+1)^3*n^2 + 1;
                    # VD = Diagonal(Op.Vexact[idl:idl+(p+1)^3-1]);
                    Vavg = sum(Op.Vexact[idl:idl+(p+1)^3-1]) / ( (p+1)^3);
                    # DB = SparseMatrixCSC(LBD + VD);

                    t = time();
                    #x[idl:idl+(p+1)^3-1] += DB \ r[idl:idl+(p+1)^3-1];
                    #this smoother is asymptotically not great but honestly whocares. can also try LU
                    # rhs = P.EvecVect[level,2] * r[idl:idl+(p+1)^3-1]; ##applies inverse of evecs
                    # rhs = Diagonal(1 ./ (P.EvalVect[level] .+ Vavg) )  * rhs;
                    # rhs = P.EvecVect[level,1] * rhs;
                    # x[idl:idl+(p+1)^3-1] += rhs;

                    x[idl:idl+(p+1)^3-1] += P.LUVect[level,idx] \ r[idl:idl+(p+1)^3-1];

                    dt = time() - t;
                    P.timings[8] += dt;
                    idx+=1;

                end
            end
        end
    end

    return x;

end
