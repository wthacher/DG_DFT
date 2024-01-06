using FFTW

######################################################################
### Set up 1d problem
######################################################################

p = 4;
nElems = [4,4];
domain = [ [0.0, 1.0] [0.0, 1.0] ];

nPts = [ nElems[1]*(p+1), nElems[2]*(p+1) ];
nPts2d = nPts[1]*nPts[2];

x1 = zeros( nPts[1] ); dx = ( domain[2,1] - domain[1,1] ) / ( length(x1) );
for ii = 2:length(x1)
	x1[ii] = dx*(ii-1);
end

######################################################################
### Set up 2d problem
######################################################################
### Analytical
f(x,y) = exp( sin( 2*pi*x ) * sin( 2*pi*y ) ) - 1.0;
d2f(x,y) = (f(x,y)+1.0) * ( 4*pi^2*sin(2*pi*y) ) * ( -sin(2*pi*x) + sin(2*pi*y)*cos(2*pi*x)^2 ) + (f(x,y)+1.0) * ( 4*pi^2*sin(2*pi*x) ) * ( -sin(2*pi*y) + sin(2*pi*x)*cos(2*pi*y)^2 );

x2 = zeros( length(x1)^2,2 ); currIndex = 1;
for ii = 1:length(x1)
	ypos = x1[ii];
	for jj = 1:length(x1)
		global currIndex
		xpos = x1[jj];
		x2[currIndex,1] = xpos; x2[currIndex,2] = ypos; currIndex += 1;
	end
end

###	RHS
b = zeros( size(x2,1) ); uexact = zeros( size(x2,1) );
for ii = 1:size(x2,1)
	b[ii] = d2f( x2[ii,1], x2[ii,2] );
	uexact[ii] = f( x2[ii,1], x2[ii,2] );
end

######################################################################
###	Solve using FFT
######################################################################
@time begin
N = length( x1 ); k = fftfreq(N) * N;

B = reshape( b, length(x1), length(x1) );
bhat = fft(B); uhat = zeros( ComplexF64, length(k), length(k) );
for ii = 1:length(k)
	for jj = 1:length(k)
		if k[ii] == 0 && k[jj] == 0
			uhat[ii,jj] = 0.0;
		else
			uhat[ii,jj] = -bhat[ii,jj] / ( (k[ii]*2*pi)^2 + (k[jj]*2*pi)^2 );
		end
	end
end

usol = ifft(uhat); u = reshape( usol, length(b), 1 ); u1 = u[1]; u .-= u1;
println( maximum( abs.(u-uexact) ) )
end


