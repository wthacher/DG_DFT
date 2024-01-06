using FFTW

######################################################################
### Set up 1d problem
######################################################################

p = 16;
nElems = [3,3,3];
domain = [ [0.0, 1.0] [0.0, 1.0] [0.0, 1.0] ];

nPts = [ nElems[1]*(p+1), nElems[2]*(p+1), nElems[3]*(p+1) ];
nPts3d = nPts[1]*nPts[2]*nPts[3];

x1 = zeros( nPts[1] ); dx = ( domain[2,1] - domain[1,1] ) / ( length(x1) );
for ii = 2:length(x1)
	x1[ii] = dx*(ii-1);
end

######################################################################
### Set up 3d problem
######################################################################
### Analytical
f(x,y,z) = exp( sin( 2*pi*x ) * sin( 2*pi*y ) * sin( 2*pi*z ) ) - 1.0;

d2f_x(x,y,z) = (f(x,y,z)+1.0) * ( 4*pi^2*sin(2*pi*y)*sin(2*pi*z) ) * ( -sin(2*pi*x) + sin(2*pi*z)*sin(2*pi*y)*cos(2*pi*x)^2 ) ;
d2f_y(x,y,z) = (f(x,y,z)+1.0) * ( 4*pi^2*sin(2*pi*x)*sin(2*pi*z) ) * ( -sin(2*pi*y) + sin(2*pi*x)*sin(2*pi*z)*cos(2*pi*y)^2 ) ;
d2f_z(x,y,z) = (f(x,y,z)+1.0) * ( 4*pi^2*sin(2*pi*x)*sin(2*pi*y) ) * ( -sin(2*pi*z) + sin(2*pi*x)*sin(2*pi*y)*cos(2*pi*z)^2 ) ;
d2f(x,y,z) = d2f_x(x,y,z) + d2f_y(x,y,z) + d2f_z(x,y,z);

x3 = zeros( length(x1)^3,3 ); currIndex = 1;
for ii = 1:length(x1)
	zpos = x1[ii];
	for jj = 1:length(x1)
		ypos = x1[jj];
		for kk = 1:length(x1)
			global currIndex
			xpos = x1[kk];
			x3[currIndex,1] = xpos; x3[currIndex,2] = ypos; x3[currIndex,3] = zpos; currIndex += 1;
		end
	end
end

###	RHS
b = zeros( size(x3,1) ); uexact = zeros( size(x3,1) );
for ii = 1:size(x3,1)
	b[ii] = d2f( x3[ii,1], x3[ii,2], x3[ii,3] );
	uexact[ii] = f( x3[ii,1], x3[ii,2], x3[ii,3] );
end

######################################################################
###	Solve using FFT
######################################################################
N = length( x1 ); k = fftfreq(N) * N;

B = reshape( b, length(x1), length(x1), length(x1) );
uhat = zeros( ComplexF64, length(k), length(k), length(k) );
ksq = k .^ 2;
ksq_ar = zeros( Float64, length(k), length(k), length(k) );
for ii = 1:length(k)
	for jj = 1:length(k)
		for kk = 1:length(k)
			#uhat[ii,jj,kk] = -bhat[ii,jj,kk] / ( (k[ii]*2*pi)^2 + (k[jj]*2*pi)^2 + (k[kk]*2*pi)^2 );
			ksq_ar[ii,jj,kk] =  ( (ksq[ii]) + (ksq[jj]) + (ksq[kk]) );
			# if k[ii] == 0 && k[jj] == 0 && k[kk] == 0
			# 	uhat[ii,jj,kk] = 0.0;
			# else
			# 	uhat[ii,jj,kk] = -bhat[ii,jj,kk] / ( (k[ii]*2*pi)^2 + (k[jj]*2*pi)^2 + (k[kk]*2*pi)^2 );
			# end
		end
	end
end

@time begin
bhat = fft(B); 
#s = Diagonal( diag() )

uhat = bhat ./ ksq_ar;

# for ii = 1:length(k)
# 	for jj = 1:length(k)
# 		for kk = 1:length(k)
# 			#uhat[ii,jj,kk] = -bhat[ii,jj,kk] / ( (k[ii]*2*pi)^2 + (k[jj]*2*pi)^2 + (k[kk]*2*pi)^2 );
# 			uhat[ii,jj,kk] = -bhat[ii,jj,kk] / ( (ksq[ii]) + (ksq[jj]) + (ksq[kk]) );
# 			# if k[ii] == 0 && k[jj] == 0 && k[kk] == 0
# 			# 	uhat[ii,jj,kk] = 0.0;
# 			# else
# 			# 	uhat[ii,jj,kk] = -bhat[ii,jj,kk] / ( (k[ii]*2*pi)^2 + (k[jj]*2*pi)^2 + (k[kk]*2*pi)^2 );
# 			# end
# 		end
# 	end
# end

uhat[1,1,1] = 0;
uhat /= (-4*pi^2);

usol = ifft( uhat  ); 
end 
u = reshape( usol, length(b), 1 ); u1 = u[1]; u .-= u1;

println( maximum( abs.(u-uexact) ) )

