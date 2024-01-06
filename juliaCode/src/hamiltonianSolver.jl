#=
This class solves  (-Δ + V)u = b where VFunc(x,y,z) is input function for potential using PCG
BC options: "dirichlet" or "periodic"
grid options: n,p,nodeFunc
    -nodeFunc(q) returns nodes, integration weights in -1:1
    -opt: SIP or LDG for the type of DG scheme we are doing
preconditioner options: "kron" "MG" "LORMG" "LORAMG"
=#

include("multilevelLOBPCG.jl");

function setGridFunc(VFunc,n,p,nodeFunc)
    x1 = zeros( n*(p+1) );
	mesh = [ x for x in LinRange( 0.0, 1.0, n+1 ) ];
    x1 = zeros( n*(p+1) );
    nodes, weights = nodeFunc(p+1);

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
	
    Vvec = zeros( length(x1)^3, );
	currIndex = 1;

    for ii = 1:length(x1)
        zpos = x1[ii];
        for jj = 1:length(x1)
            ypos = x1[jj];
            for kk = 1:length(x1)
                xpos = x1[kk];
                Vvec[currIndex] = VFunc( xpos, ypos, zpos );
                currIndex += 1;
            end
        end
    end
	
	return Vvec;
end

function hamiltonianSolver(VFunc, rhsFunc, BC, n, p, nodeFunc, opt, precon)
    N = n^3 * (p+1)^3; ##total DOF

    Perm = getPermutation(n,p); ##takes you from tensor to block form
    Vvec = setGridFunc(VFunc, n,p,nodeFunc);
    rhsVec = setGridFunc(rhsFunc, n,p,nodeFunc);

    TensorOp = createOperator(n,p, BC,Vvec, nodeFunc, opt, false, false);

    blockTensorOp = createOperator(n,p, BC,Vvec, nodeFunc, opt, true, false);

    symTensorOp = symOperator(TensorOp); ##applies M first

    blockSymTensorOp = symOperator(blockTensorOp); ##applies M first

    LORTensorOp = createOperator(n,p, BC,Vvec, nodeFunc, opt, false, true);
    LORSymTensorOp = symOperator(LORTensorOp);

    numCycles = 5;
    verbose = true;

    if(precon == "kron")
        b = TensorOp.M * rhsVec;
        Vavg = sum(Vvec) / N;
        symLaplacePre = laplacePrecond(TensorOp, Vavg, 0, true);
        x = cg(symTensorOp, b;Pl=symLaplacePre, maxiter=10, reltol=1e-12, verbose=true);
        #println("error:" ,norm(symTensorOp*x - TensorOp.M * rhsVec));
        return x;
    elseif(precon == "MG")
        #AssertionError("This MG is pretty bad. smoothing is hard")
        b = Perm * TensorOp.M * rhsVec;
        symMGPre = initSymMultigridPrecond(n,p, BC, Perm*Vvec,nodeFunc,opt,false,numCycles,verbose);
        x = 0*b;
        ldiv!(x, symMGPre, b);

        #x = gmres(blockSymTensorOp, b;Pl=symMGPre, maxiter=50, reltol=1e-12, verbose=true);
        return TensorOp.PermT * x;
    elseif(precon == "LORMG")
        if(BC == "dirichlet")
            AssertionError("MG for dirichlet not implemented yet");
        end
        b = Perm * TensorOp.M * rhsVec;
        LORsymMGPre = initSymMultigridPrecond(n,p, BC, Perm*Vvec,nodeFunc,opt,true,numCycles,verbose);
        x = 0*b;
        ldiv!(x, LORsymMGPre, b);
        println("MG not working well yet. Incurring error on iterpolation up somehow")
        
        #x = gmres(blockSymTensorOp, b; maxiter=50, reltol=1e-12, verbose=true);
        return TensorOp.PermT * x;
    elseif(precon == "LORAMG")
        b = TensorOp.M * rhsVec;
        

        A = getMatrix(LORSymTensorOp);
        A_AMG = AMGPreconditioner{RugeStuben}(A);

        #x = cg(LORSymTensorOp, LORTensorOp.M*rhsVec; Pl=A_AMG, maxiter=50, reltol=1e-12, verbose=true);
        x = cg(symTensorOp, b; Pl=A_AMG, maxiter=50, reltol=1e-12, verbose=true);
        #println("error:" ,norm(symTensorOp*x - TensorOp.M * rhsVec));
        return x;
    elseif(precon == "none")
        b = TensorOp.M * rhsVec;
        x = cg(symTensorOp, b; maxiter=50, reltol=1e-12, verbose=true);
    else
        AssertionError("PC not implemented")
    end

end

