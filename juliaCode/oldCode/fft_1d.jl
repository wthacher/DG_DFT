using FFTW

######################################################################
### Set up 1d problem
######################################################################

p = 4;
nElems = [4];
domain = [ [0.0, 1.0] ];

nPts = [ nElems[1]*(p+1) ];

x1 = zeros( nPts[1] ); dx = ( domain[2,1] - domain[1,1] ) / ( length(x1) );
for ii = 2:length(x1)
	x1[ii] = dx*(ii-1);
end

######################################################################
###	Test function
######################################################################
u0(x) = exp( sin( 2*pi*x ) ) - 1.0;
ddu0(x) = -(u0(x) +1.0) * ( 4*pi^2 ) * ( sin(2*pi*x) - cos(2*pi*x)^2 );

u = zeros( length(x1) ); b = zeros( length(x1) );
for ii = 1:length(x1)
	u[ii] = u0( x1[ii] ); b[ii] = ddu0( x1[ii] );
end

######################################################################
###	Solve using FFT
######################################################################
N = length( x1 ); k = fftfreq(N) * N;

bhat = fft(b); uhat = zeros( ComplexF64, length(k) );
for ii = 1:length(k)
	if k[ii] == 0
		uhat[ii] = 0.0;
	else
		uhat[ii] = -bhat[ii] / (k[ii]*2*pi)^2;
	end
end

usol = ifft( uhat );

u1 = usol[1]; usol .-= u1;
println( maximum( abs.(u-usol) ) )

