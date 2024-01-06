# DG_DFT
DG methods for DFT problems

- Code to experiment with spectral DG methods for DG problems
- Can solve eigenvalue problem (-\Delta  + V)u = \lambda u
- Uses LOBPCG with various preconditioners: kronecker, low order refined (LOR) AMG, GMG 


## To do:
- fix GMG. big residual after bottom solve
- implement potential functions
