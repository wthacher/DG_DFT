#=

This class solves eval problem -Δ + V where VFunc(x,y,z) is input function for potential
BC options: "dirichlet" or "periodic"
grid options: n,p,nodeFunc
    -nodeFunc(q) returns nodes, integration weights in -1:1
    -opt: SIP or LDG for the type of DG scheme we are doing
preconditioner options: "kron" "MG" "LORMG" "LORAMG"
solver options: LOBPCG (need to implement Arnoldi as well)
nev: number of eig pairs desired
=#

include("hamiltonianSolver.jl");

function eigSolver(VFunc, BC, n, p, nodeFunc, opt, precon, solver, nev)
    N = n^3 * (p+1)^3; ##total DOF

    Perm = getPermutation(n,p); ##takes you from tensor to block form
    Vvec = setGridFunc(VFunc, n,p,nodeFunc);

    # TensorOp = createOperator(n,p, BC,Vvec, nodeFunc, opt, false, false);
    # LORTensorOp = createOperator(n,p,BC, Vvec, nodeFunc, opt, false, true);

    # symTensorOp = symOperator(TensorOp); ##applies M first
    # LORsymTensorOp = symOperator(LORTensorOp); ##applies M first

    numCycles = 2;
    verbose = false;

    symPre = initSymMultigridPrecond(n,p, BC, Perm*Vvec,nodeFunc,opt,false,numCycles,verbose);
    LORsymPre = initSymMultigridPrecond(n,p, BC, Perm*Vvec,nodeFunc,opt,true,numCycles,verbose);
    #out = LOBPCG(symTensorOp, X0M, TensorOp.M, symPre;m_to_check=7);
    if(solver == "LOBPCG")
        vecs, vals, res = doMultilevelLOBPCG(symPre, LORsymPre, precon, nev, 2*nev); ##returns eigvecs, vals, residuals from iterations
        return vecs, vals, res;
    else
        AssertionError("solver not yet implemented");
    end

end