using MATLAB

##########################################################################################################
#	------------------------------------------------------------------------------------------
#	HOW TO USE
#	------------------------------------------------------------------------------------------
#		1) Create a new session
#			e.g. s1 = createSession()
#		2) For standard MATLAB plotting functions, call them as per usual with s1 as added 1st argument
#			e.g. clf( s1 ), hold_on( s1 ), xlim( s1, [0,1] )
#		3) For other more specific commands look at the "Plot Julia constructs section"
#		4) Close the session once you are done
#			e.g. closeSession( s1 )
##########################################################################################################

##########################################################################################################
# Start/stop session
##########################################################################################################
function createSession()
	return MSession( 0 )
end

function closeSession(s)
	close( s )
end

##########################################################################################################
# Basic tools
##########################################################################################################
function clf( s )
	eval_string( s1, "clf" )
end

function xlim( s, arr )
	eval_string( s1, string( "xlim(", arr, ")" ) )
end

function ylim( s, arr )
	eval_string( s1, string( "ylim(", arr, ")" ) )
end

function xlabel( s, label )
	eval_string( s1, string( "xlabel(\" ", label, "\" )" ) )
end

function ylabel( s, label )
	eval_string( s1, string( "ylabel(\" ", label, "\" )" ) )
end

function axis_equal( s )
	eval_string( s1, "axis equal" )
end

function hold_on( s )
	eval_string( s1, "hold on" )
end

function hold_off( s )
	eval_string( s1, "hold off" )
end

function colorbar( s )
	eval_string( s1, "colorbar()" )
end

function plot( s, x, y; tags="o" )
	put_variable( s1, :xT, x )
	put_variable( s1, :yT, y )
	eval_string( s1, string( "plot( xT,yT, \" ", tags, " \" )" ) );
end

function plot3( s1, x, y, z; tags="o" )
	put_variable( s1, :xT, x )
	put_variable( s1, :yT, y )
	put_variable( s1, :zT, z )
	eval_string( s1, string( "plot3( xT,yT,zT, \" ", tags, " \" )" ) );
end

function scatter( s1, x, y, z )
	put_variable( s1, :xT, x )
	put_variable( s1, :yT, y )
	put_variable( s1, :zT, z )
	eval_string( s1, "scatter( xT,yT,20,zT )" );
end

##########################################################################################################
# Plot Julia constructs
##########################################################################################################

###	Plot edges and vertices of an input mesh
#	@s: MATLAB session
#	@mesh: Mesh object
function plotMesh( s, mesh::Mesh )

	#Get mesh lines
	xArr = mesh.mX[:,1];
	yArr = mesh.mX[:,2];

	#Send to MATLAB and plot
	put_variable( s1, :xArr, xArr )
	put_variable( s1, :yArr, yArr )
	eval_string( s1, "plot( xArr, yArr, \"ko\" )" ); 

end


###	Plot edges and vertices of an input mesh
#	@s: MATLAB session
#	@mesh: Mesh object
function plotNodes( s, mesh::Mesh )

	#Get mesh lines
	xArr = mesh.mX[:,1];
	yArr = mesh.mX[:,2];

	#Send to MATLAB and plot
	put_variable( s1, :xArr, xArr )
	put_variable( s1, :yArr, yArr )
	eval_string( s1, "plot( xArr, yArr, \"ko\" )" ); 


	#Plot labels
	labels = [string(ii) for ii = 1:length(xArr)]; nNodes = (mesh.p+1)^(mesh.dim); 
	nElems = prod( mesh.nElems1D ); centre = Int(round(nNodes*0.5));

	for ii = 1:nElems
		base = (ii-1)*nNodes; 
		for jj = 1:nNodes
			xArr[base+jj] = 0.8*xArr[base+jj] + 0.2*xArr[base+centre];
			yArr[base+jj] = 0.8*yArr[base+jj] + 0.2*yArr[base+centre];
		end
	end

	put_variable( s1, :xArr, xArr )
	put_variable( s1, :yArr, yArr )
	put_variable( s1, :labels, labels ); eval_string( s1, "text( xArr, yArr, labels )" ); 

end







