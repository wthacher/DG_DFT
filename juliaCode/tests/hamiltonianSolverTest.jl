include("../src/hamiltonianSolver.jl");


using Plots
using BlockDiagonals

###Test solvers for poisson equation with periodic or dirichlet BCS
##Kron and low order work pretty well

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
	k=1;
	

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

##run accuracy convergence test for periodic poisson
##measure error in max nodal and L2 using mass matrix.
function poissonConvergenceTest(p,nLevels,nodeFunc,opt, BC, precon)
	errMax = zeros(nLevels,);
	errL2 = zeros(nLevels,);
	timings = zeros(nLevels,);

	println("ORDER ", p, " Test");

	k=1;

	uFunc(x,y,z) = exp( sin( k*2*pi*x ) * sin( k*2*pi*y ) * sin( k*2*pi*z ) ) - 1.0;
	
	##potential function
	vFunc(x,y,z) = (sin(x*y*z) + 2);
    #v(x,y,z) = sin(x*y) + cos(y*z) + y*z^3;
    #v(x,y,z) = 100*rand();
    #v(x,y,z) = ( sqrt(x^2+y^2+z^2) + 1e-4)^(-1);

	function rhsV(x,y,z)
		k = 1;
		v(x,y,z) = (sin(x*y*z) + 2);
		f(x,y,z) = exp( sin( k*2*pi*x ) * sin( k*2*pi*y ) * sin( k*2*pi*z ) ) - 1.0;
		
		d2f_x(x,y,z) = (f(x,y,z)+1.0) * ( k^2*4*pi^2*sin(k*2*pi*y)*sin(k*2*pi*z) ) * ( -sin(k*2*pi*x) + sin(k*2*pi*z)*sin(k*2*pi*y)*cos(k*2*pi*x)^2 ) ;
		d2f_y(x,y,z) = (f(x,y,z)+1.0) * ( k^2*4*pi^2*sin(k*2*pi*x)*sin(k*2*pi*z) ) * ( -sin(k*2*pi*y) + sin(k*2*pi*x)*sin(k*2*pi*z)*cos(k*2*pi*y)^2 ) ;
		d2f_z(x,y,z) = (f(x,y,z)+1.0) * ( k^2*4*pi^2*sin(k*2*pi*x)*sin(k*2*pi*y) ) * ( -sin(k*2*pi*z) + sin(k*2*pi*x)*sin(k*2*pi*y)*cos(k*2*pi*z)^2 ) ;
		d2f(x,y,z) = d2f_x(x,y,z) + d2f_y(x,y,z) + d2f_z(x,y,z);
		return -d2f(x,y,z) + v(x,y,z)*f(x,y,z);
	end

	

	for i = 1:nLevels
		n = round(Int64,5^i);
		N = n^3 * (p+1)^3;

		uexact = setGridFunc(uFunc,n,p,nodeFunc);
		
		t = time(); 
		uComp = hamiltonianSolver(vFunc,rhsV, BC, n,4,nodeFunc,opt,precon);

		dt = time() - t;
		timings[i] = dt;
		AssertionError("finish this code")

		TensorOp = createOperator(n,p, BC,zeros(N,), nodeFunc, opt, false, false);
		errMax[i] = maximum(abs.(uexact-uComp));
		errL2[i] = (uexact-uComp)' * (TensorOp.M *(uexact-uComp) ) ;
		if(errMax[i] < 1e-10)
			return errMax, sqrt.(errL2), timings;
		end


		#Rprintln("rhs integral: ", ones(size(uexact,1),)' * (TensorOp.M *(b)) );

	end

	return errMax, sqrt.(errL2), timings;
end

nodeFunc = gausslobatto;
opt = "SIP";

maxP=3;
nLevels=3;
errMaxMat = zeros(nLevels,maxP);
errL2Mat = zeros(nLevels,maxP);
timingsMat = zeros(nLevels,maxP);
maxRates = zeros(maxP,);
L2Rates = zeros(maxP,)
for p = 1:maxP
	errMax, errL2, timings = poissonConvergenceTest(round(Int64,2^(p)),nLevels,nodeFunc,opt,"periodic","LORMG");
	errMaxMat[:,p] = errMax; errL2Mat[:,p] = errL2;
	timingsMat[:,p] = timings;
end

P = [ones(nLevels-1,)';(2:nLevels)']';
maxRates = (pinv(P) * log2.(errMaxMat[2:end,:]))[2,:];
L2Rates = (pinv(P) * log2.(errL2Mat[2:end,:]))[2,:];
timingsRates = (pinv(P) * log2.(timingsMat[2:end,:]))[2,:];

Ns = [2^i for i =1:nLevels];

# plot(Ns,errMaxMat,labels=reshape(2 .^(1:maxP),1,:), yscale=:log10,xscale=:log10,marker =:circle);
# ylims!(1e-12,1e2);
# yticks!([1e-10,1e-6,1e-2,1e2],["1e-10","1e-6","1e-2","1e2"]);
# xticks!([2,4,8,16,32], ["2","4","8","16","32"])
# title!("Max Norm Error")

# plot(Ns,errL2Mat,labels=reshape(2 .^(1:maxP),1,:), yscale=:log10,xscale=:log10,marker =:circle)
# ylims!(1e-13,1e2);
# yticks!([1e-12,1e-8,1e-4,1],["1e-12","1e-8","1e-4","1"]);
# xticks!([2,4,8,16,32], ["2","4","8","16","32"])
# title!("L2 Norm Error")

plot(Ns,timingsMat,labels=reshape(2 .^(1:maxP),1,:), yscale=:log10,xscale=:log10,marker =:circle);
ylims!(1e-5,1e2);
yticks!([1e-5,1e-3,1e-1,1e1,1e3],["1e-5","1e-3","1e-1","1e1","1e3"]);
xticks!([2,4,8,16,32], ["2","4","8","16","32"])
title!("Kron Run Time")

