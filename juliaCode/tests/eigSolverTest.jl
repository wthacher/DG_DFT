#=
Test eigenvalue solver for increasing p with various preconditioners: kron, MG, LORMG, LORAMG
=#

include("../src/eigSolver.jl")

using Plots

function setUpVecs(n,p,nodeFunc)
	x1 = zeros( n*(p+1) );
	mesh = [ x for x in LinRange( 0.0, 1.0, n+1 ) ];
	#nodes = chebyshevNodes( p+1 ); 
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
	######################################################################
	### Set up 3d problem
	######################################################################
	### Analytical
	f(x,y,z) = exp( sin( 2*pi*x ) * sin( 2*pi*y ) * sin( 2*pi*z ) ) - 1.0;
	
	##potential function
	v(x,y,z) = (sin(10*pi*x*y*z) + 2);
    #v(x,y,z) = sin(x*y) + cos(y*z) + y*z^3;
    #v(x,y,z) = 100*rand();
    #v(x,y,z) = ( sqrt(x^2+y^2+z^2) + 1e-4)^(-1);

	d2f_x(x,y,z) = (f(x,y,z)+1.0) * ( 4*pi^2*sin(2*pi*y)*sin(2*pi*z) ) * ( -sin(2*pi*x) + sin(2*pi*z)*sin(2*pi*y)*cos(2*pi*x)^2 ) ;
	d2f_y(x,y,z) = (f(x,y,z)+1.0) * ( 4*pi^2*sin(2*pi*x)*sin(2*pi*z) ) * ( -sin(2*pi*y) + sin(2*pi*x)*sin(2*pi*z)*cos(2*pi*y)^2 ) ;
	d2f_z(x,y,z) = (f(x,y,z)+1.0) * ( 4*pi^2*sin(2*pi*x)*sin(2*pi*y) ) * ( -sin(2*pi*z) + sin(2*pi*x)*sin(2*pi*y)*cos(2*pi*z)^2 ) ;
	d2f(x,y,z) = d2f_x(x,y,z) + d2f_y(x,y,z) + d2f_z(x,y,z);
	rhsV(x,y,z) = -d2f(x,y,z) + v(x,y,z)*f(x,y,z);

	pf(x,y,z) = 14*x^4*z^3*y + y^2 + z^2 + x^2*z^2  ;

	# f(x,y,z) = 1+x+x^2;
	# v(x,y,z) = 1 + x + x^2;
	# rhsV(x,y,z) = (1 + x + x^2)^2;


	x3 = zeros( length(x1)^3,3 ); b = zeros( size(x3,1) ); uexact = zeros( size(x3,1) );
	bV = zeros( size(x3,1) );
	p3 = zeros( size(x3,1) );
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

					b[currIndex] = -1*d2f( xpos, ypos, zpos );
					uexact[currIndex] = f( xpos, ypos, zpos );
					bV[currIndex] = rhsV( xpos, ypos, zpos );
					Vexact[currIndex] = v( xpos, ypos, zpos );
					p3[currIndex] = pf( xpos, ypos, zpos );
					currIndex += 1;
				end
			end
		end
		

	end

	setPosition!(x3, b, uexact, Vexact, x1)
	
	return uexact, b, bV, Vexact, p3;
end

function eigvalConvergenceTest(nLevels,nodeFunc,opt,precon)
	num_to_check=20;
	nev = 2*num_to_check;
	n=3;

	resArray = Array{Matrix}(undef,nLevels);
	vals = zeros(nev,nLevels);

	timings = zeros(nLevels,);

	vf(x,y,z) = 1;#(sin(10*pi*x*y*z) + 2);

	for i = 1:nLevels
		p = i;

		t = time();
		vecs, vals[:,i], resArray[i] = eigSolver(vf,"periodic",n,p,nodeFunc,opt,
		precon, "LOBPCG",num_to_check);
		dt = time() - t;

		timings[i]=dt;
	end

	return vals, resArray,timings;
end


nodeFunc = gausslobatto
opt = "SIP";

nLevels=4;

vals, resArray, timings = eigvalConvergenceTest(nLevels,nodeFunc, opt,"LORAMG");

val_dif = abs.(vals[:,1:end-1] - reshape( repeat(vals[:,end],nLevels-1), size(vals,1), nLevels-1) )[1:10,:];
ord = log.(val_dif[:,1:end-1] ./ val_dif[:,2:end])


Ns = 1:size(val_dif,2)
plot(Ns,val_dif',labels=reshape(1:nLevels,1,:), yscale=:log10,marker =:circle);
ylims!(1e-14,1e2);
yticks!([1e-12,1e-8,1e-4,1e0],["1e-12","1e-8","1e-4","1e0"]);
xticks!(1:nLevels, [string(i) for i =1:nLevels])
title!("Eigenvalue Error")
xlabel!("P")
ylabel!("Error")

# res = resArray[7];
# res = res[1:10,:];
# plot(1:size(res,2),res',labels=reshape(1:10,1,:), yscale=:log10,marker =:circle);
# ylims!(1e-12,1e3);
# yticks!([1e-12,1e-8,1e-4,1e0],["1e-12","1e-8","1e-4","1e0"]);
# xticks!(1:size(res,2), [string(i) for i =1:size(res,2)])
# title!("residual convergence")
# xlabel!("Iteration")
# ylabel!("residual")




# errMaxMat = zeros(nLevels,maxP);
# errL2Mat = zeros(nLevels,maxP);
# timingsMat = zeros(nLevels,maxP);
# maxRates = zeros(maxP,);
# L2Rates = zeros(maxP,)
# for p = 1:maxP
# 	errMax, errL2, timings = poissonConvergenceTest(round(Int64,2^p),nLevels,nodeFunc,opt,"mg");
# 	errMaxMat[:,p] = errMax; errL2Mat[:,p] = errL2;
# 	timingsMat[:,p] = timings;
# end


# rhs = TensorOp.M * bV;
# @time x = bicgstabl(symTensorOp, rhs;Pl=symLaplacePre,verbose=true,reltol=1e-10);
# # #x = bicgstabl(Op, rhs2;Pl=laplacePre,verbose=true,reltol=1e-16);
# println(norm( (symTensorOp * x - rhs)) );
# println(maximum( abs.(x - uexact) ));

# LORrhs = LORTensorOp.M * bV;

# #@time LORx = cg(LORsymTensorOp, LORrhs;Pl=LORsymLaplacePre,verbose=true,reltol=1e-10);
#@time x = bicgstabl(symTensorOp, TensorOp.M * bV; Pl=LU, verbose=true,reltol=1e-16);

#@time x = cg((x,y) -> mul!(x,symTensorOp,y),Vector(TensorOp.M * bV);precon=( (x,y) -> ldiv!(x,LU,y) ), tol=1e-10);
# # # #x = bicgstabl(Op, rhs2;Pl=laplacePre,verbose=true,reltol=1e-16);
# println(norm( (LORsymTensorOp * LORx[1] - br)) );
# println(maximum( abs.(LORx - uexact) ));
#println(norm( (symTensorOp * x[1] - TensorOp.M * bV)) );

# println(norm(LORx - x));

# @time LORsymPre = initSymMultigridPrecond(n,p,Perm*vExact,nodeFunc,opt,true,1,false);

# uLOR = 0*uexact;

# #ldiv!(uLOR,LORsymPre,Perm*bV ); ##multiply by Minv. We are working with laplacian bro
# #println(norm((LORTensorOp*(Perm')*uLOR) - bV)) ##block back into tensor

# @time LORx = gmres(LORBlockOp, Perm*bV;Pl=LORsymPre,verbose=true,reltol=1e-12);
# println(norm((LORBlockOp*LORx) - Perm*bV)) ##block back into tensor
# println("High-order multilevel: ")
# @time vecs, vals = doMultilevelLOBPCG(symPre, num_to_check, nev)


