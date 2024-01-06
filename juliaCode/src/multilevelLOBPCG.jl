include("LOBPCG.jl");
include("symMultigrid.jl");
using Preconditioners; ##for tha AMG

##orthogonalize with respect to M inner product
function MGS(Z, M =I)
    n = size(Z,2);

    Q = Z;
    for j=1:n
        # Q[:,j] = Z[:,j] / norm(Z[:,j]);
        Q[:,j] = Z[:,j] / sqrt(dot(Z[:,j], M*Z[:,j]));
        for k = j+1:n
            Z[:,k] = Z[:,k] - dot(Q[:,j], M* Z[:,k]) * Q[:,j];
        end
    end

    return Q;
end

struct AMGWrap
    AMGPC::AMGPreconditioner;
    A
end

function LinearAlgebra.ldiv!(AMG::AMGWrap, B)
    m = size(B,2);
    #println("try preconditioning K with lumped mass matrix!")
    for i = 1:m
        x = AMG.AMGPC \ B[:,i];
        #x = cg(AMG.A, B[:,i];maxiter=50);
        #x = AMG.A \  B[:,i];
        B[:,i] .= x;
    end
	return B;
end


##run multilevel LOBPCG scheme to get good initial guesses
##nev is size of block, m_to_check is how many you want to converge
##the question is, are the first nev modes captured on all levels?
##my guess for this problem is yes based on kron structure
function doMultilevelLOBPCG( P::symMultigridPrecond, LORP::symMultigridPrecond, precon, num_to_check, nev )
    eigVecsCoarse=Matrix{Float64};
    eigVecsLevel=Matrix{Float64};
    eigValsLevel=Matrix{Float64};
    rLevel=Matrix{Float64};

    for i = size(P.pVect,1):-1:1
        
        if(i == size(P.pVect,1) )
            t = time()
            symTensorOpLevel = symOperator(P.TensorOpVect[i]);
            A = getMatrix(symTensorOpLevel); ##ML
            B = symTensorOpLevel.Op.M;
            eigVecsLevel = eigvecs(Symmetric(Matrix(A)), Symmetric(Matrix(B)));
            eigValsLevel = eigvals(Symmetric(Matrix(A)), Symmetric(Matrix(B)));
            #println("eigvals on level $i ", eigValsLevel[1:num_to_check]);
            dt = time() - t;
            P.eigTimings[1] += dt;
        else
            t = time()
            symTensorOpLevel = symOperator(P.TensorOpVect[i]);
            X0M = MGS(eigVecsCoarse, symTensorOpLevel.Op.M); ##M-Orthogonalize on your level
            B = P.TensorOpVect[i].M;
            Vavg = sum(P.TensorOpVect[i].Vexact) / size(P.TensorOpVect[i].Vexact,1);
            symLaplacePreLevel = laplacePrecond(P.TensorOpVect[i], Vavg, 0, true);
            dt = time() - t;
            P.eigTimings[4] += dt;
            
            t = time()
            if(precon =="kron")
                eigVecsLevel, rLevel = LOBPCG(symTensorOpLevel, X0M[:,1:nev], B, symLaplacePreLevel; m_to_check=num_to_check); ##maybe check more?
            elseif(precon == "MG")
                eigVecsLevel, rLevel = LOBPCG(symTensorOpLevel, X0M[:,1:nev], B, P; m_to_check=num_to_check); ##just use multigrid as the preconditioner
            elseif(precon == "LORMG")
                eigVecsLevel, rLevel = LOBPCG(symTensorOpLevel, X0M[:,1:nev], B, LORP; m_to_check=num_to_check);
            elseif(precon == "LORAMG")
                A = getMatrix(symOperator(LORP.TensorOpVect[i]));
                A_AMG = AMGPreconditioner{RugeStuben}(A);
                AMGPrecon = AMGWrap(A_AMG, A);
                eigVecsLevel, rLevel = LOBPCG(symTensorOpLevel, X0M[:,1:nev], B, AMGPrecon; m_to_check=num_to_check);
            elseif(precon == "none")
                eigVecsLevel, rLevel = LOBPCG(symTensorOpLevel, X0M[:,1:nev], B; m_to_check=num_to_check);
            else
                AssertionError("PC not implmented");
            end
            dt = time() - t;
            P.eigTimings[2] += dt;

            eigValsLevel = eigvals(Symmetric(Matrix(eigVecsLevel' * (symTensorOpLevel * eigVecsLevel) ) )   );
            println("eigvals on level $i ", eigValsLevel[1:num_to_check]);
            println("iterations on level $i: ", size(rLevel, 2) );
        end

        ##interpolate up nev evecs
        ##recall that interpolation works in BLOCK FORM and eigvecs are in tensor form
        if(i>1)
            t = time()
            eigVecsCoarse = zeros( (P.n)^3 * (P.pVect[i-1]+1)^3 , nev );
            for k =1:nev
                @views eigVecsCoarse[:,k] .= P.TensorOpVect[i-1].Perm * interpolate(P, i-1, P.TensorOpVect[i].PermT * (eigVecsLevel[:,k]) ); ##interpolate and take back to tensor
            end
            dt = time() - t;
            P.eigTimings[3]+=dt;
        end


    end

    if(size(P.pVect,1)==1)
        return eigVecsLevel[:,1:nev], eigValsLevel[1:nev], zeros(nev,1);
    end
    
    return eigVecsLevel, eigValsLevel, rLevel;
end