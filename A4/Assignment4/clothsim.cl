#define DAMPING 0.02f

#define G_ACCEL (float4)(0.f, -9.81f, 0.f, 0.f)

#define WEIGHT_ORTHO	0.138f
#define WEIGHT_DIAG		0.097f
#define WEIGHT_ORTHO_2	0.069f
#define WEIGHT_DIAG_2	0.048f


#define ROOT_OF_2 1.4142135f
#define DOUBLE_ROOT_OF_2 2.8284271f




///////////////////////////////////////////////////////////////////////////////
// The integration kernel
// Input data:
// width and height - the dimensions of the particle grid
// d_pos - the most recent position of the cloth particle while...
// d_prevPos - ...contains the position from the previous iteration.
// elapsedTime      - contains the elapsed time since the previous invocation of the kernel,
// prevElapsedTime  - contains the previous time step.
// simulationTime   - contains the time elapsed since the start of the simulation (useful for wind)
// All time values are given in seconds.
//
// Output data:
// d_prevPos - Input data from d_pos must be copied to this array
// d_pos     - Updated positions
///////////////////////////////////////////////////////////////////////////////
  __kernel void Integrate(unsigned int width,
						unsigned int height, 
						__global float4* d_pos,
						__global float4* d_prevPos,
						float elapsedTime,
						float prevElapsedTime,
						float simulationTime) {
							
	// Make sure the work-item does not map outside the cloth
    if(get_global_id(0) >= width || get_global_id(1) >= height)
		return;

	unsigned int particleID = get_global_id(0) + get_global_id(1) * width;
	// This is just to keep every 8th particle of the first row attached to the bar
    if(particleID > width-1 || ( particleID & ( 7 )) != 0){


		// ADD YOUR CODE HERE!

		// Read the positions
		// Compute the new one position using the Verlet position integration, taking into account gravity and wind
		// Move the value from d_pos into d_prevPos and store the new one in d_pos

	
		float4 x0 = d_prevPos[particleID];
		float4 x1 = d_pos[particleID];
		
		if (elapsedTime / prevElapsedTime < 1000.0) d_pos[particleID] = x1 + (x1 - x0) * (elapsedTime / prevElapsedTime) * 0.7f + G_ACCEL * elapsedTime * elapsedTime;
		else d_pos[particleID] = x1 + G_ACCEL * elapsedTime * elapsedTime;
		d_prevPos[particleID] = x1;
    }	
}



///////////////////////////////////////////////////////////////////////////////
// Input data:
// pos1 and pos2 - The positions of two particles
// restDistance  - the distance between the given particles at rest
//
// Return data:
// correction vector for particle 1
///////////////////////////////////////////////////////////////////////////////
  float4 SatisfyConstraint(float4 pos1,
						 float4 pos2,
						 float restDistance){
	float4 toNeighbor = pos2 - pos1;
	return (toNeighbor - normalize(toNeighbor) * restDistance);
}

///////////////////////////////////////////////////////////////////////////////
// Input data:
// width and height - the dimensions of the particle grid
// restDistance     - the distance between two orthogonally neighboring particles at rest
// d_posIn          - the input positions
//
// Output data:
// d_posOut - new positions must be written here
///////////////////////////////////////////////////////////////////////////////

#define TILE_X 16 
#define TILE_Y 16
#define HALOSIZE 2

__kernel __attribute__((reqd_work_group_size(TILE_X, TILE_Y, 1)))
__kernel void SatisfyConstraints(unsigned int width,
								unsigned int height, 
								float restDistance,
								__global float4* d_posOut,
								__global float4 const * d_posIn){
    
    if(get_global_id(0) >= width || get_global_id(1) >= height)
		return;
	

	// ADD YOUR CODE HERE!
	// Satisfy all the constraints (structural, shear, and bend).
	// You can use weights defined at the beginning of this file.

	// A ping-pong scheme is needed here, so read the values from d_posIn and store the results in d_posOut

	// Hint: you should use the SatisfyConstraint helper function in the following manner:
	//SatisfyConstraint(pos, neighborpos, restDistance) * WEIGHT_XXX
	
	unsigned int particleID = get_global_id(0) + get_global_id(1) * width;
	float4 cacheIn = d_posIn[particleID];
	float4 cacheOut = cacheIn;

	if(particleID > width-1 || ( particleID & ( 7 )) != 0){
		
		bool lh,gh,lw,gw,llh,ggh,llw,ggw;
		lh = get_global_id(1) < height-1;
		gh = get_global_id(1) >= 1;
		lw = get_global_id(0) < width-1;
		gw = get_global_id(0) >= 1;	
		llh = get_global_id(1) < height-2;
		ggh = get_global_id(1) >= 2;
		llw = get_global_id(0) < width-2;
		ggw = get_global_id(0) >= 2;
	
		float4 correction = (float)(0.f,0.f,0.f,0.f);
		if(lh) correction += SatisfyConstraint(cacheIn, d_posIn[particleID+width], restDistance);
		if(gh) correction += SatisfyConstraint(cacheIn, d_posIn[particleID-width], restDistance);
		if(lw) correction += SatisfyConstraint(cacheIn, d_posIn[particleID+1], restDistance);
		if(gw) correction += SatisfyConstraint(cacheIn, d_posIn[particleID-1], restDistance);
		cacheOut += correction * WEIGHT_ORTHO;

		correction = (float)(0.f,0.f,0.f,0.f);
		if(lh && lw) correction += SatisfyConstraint(cacheIn, d_posIn[particleID+width+1], hypot(restDistance,restDistance));
		if(lh && gw) correction += SatisfyConstraint(cacheIn, d_posIn[particleID+width-1], hypot(restDistance,restDistance));
		if(gh && lw) correction += SatisfyConstraint(cacheIn, d_posIn[particleID-width+1], hypot(restDistance,restDistance));
		if(gh && gw) correction += SatisfyConstraint(cacheIn, d_posIn[particleID-width-1], hypot(restDistance,restDistance));
		cacheOut += correction * WEIGHT_DIAG;

		correction = (float)(0.f,0.f,0.f,0.f);
		if(llh) correction += SatisfyConstraint(cacheIn, d_posIn[particleID+2*width], 2*restDistance);
		if(ggh) correction += SatisfyConstraint(cacheIn, d_posIn[particleID-2*width], 2*restDistance);
		if(llw) correction += SatisfyConstraint(cacheIn, d_posIn[particleID+2], 2*restDistance);
		if(ggw) correction += SatisfyConstraint(cacheIn, d_posIn[particleID-2], 2*restDistance);
		cacheOut += correction * WEIGHT_ORTHO_2;
/*
		correction = (float)(0.f,0.f,0.f,0.f);
		if(llh && llw) correction += SatisfyConstraint(cacheIn, d_posIn[particleID+2*width+2], hypot(2*restDistance,2*restDistance));
		if(llh && ggw) correction += SatisfyConstraint(cacheIn, d_posIn[particleID+2*width-2], hypot(2*restDistance,2*restDistance));
		if(ggh && llw) correction += SatisfyConstraint(cacheIn, d_posIn[particleID-2*width+2], hypot(2*restDistance,2*restDistance));
		if(ggh && ggw) correction += SatisfyConstraint(cacheIn, d_posIn[particleID-2*width-2], hypot(2*restDistance,2*restDistance));
		cacheOut += correction * WEIGHT_DIAG_2;

	
		if (length(cacheOut - cacheIn) > 0.5f * restDistance) {
		cacheOut += normalize(cacheOut - cacheIn) * 0.5f * restDistance;
		}
*/
	}
	d_posOut[particleID] = cacheOut;
}


///////////////////////////////////////////////////////////////////////////////
// Input data:
// width and height - the dimensions of the particle grid
// d_pos            - the input positions
// spherePos        - The position of the sphere (xyz)
// sphereRad        - The radius of the sphere
//
// Output data:
// d_pos            - The updated positions
///////////////////////////////////////////////////////////////////////////////
__kernel void CheckCollisions(unsigned int width,
								unsigned int height, 
								__global float4* d_pos,
								float4 spherePos,
								float sphereRad){
								

	// ADD YOUR CODE HERE!
	// Find whether the particle is inside the sphere.
	// If so, push it outside.
	if(get_global_id(0) >= width || get_global_id(1) >= height)
		return;
	unsigned int particleID = get_global_id(0) + get_global_id(1) * width;
	float4 distanceVector = d_pos[particleID] - spherePos;
	if (length(distanceVector) < sphereRad) d_pos[particleID] += normalize(distanceVector) * (sphereRad - length(distanceVector));
	
}

///////////////////////////////////////////////////////////////////////////////
// There is no need to change this function!
///////////////////////////////////////////////////////////////////////////////
float4 CalcTriangleNormal( float4 p1, float4 p2, float4 p3) {
    float4 v1 = p2-p1;
    float4 v2 = p3-p1;

    return cross( v1, v2);
}

///////////////////////////////////////////////////////////////////////////////
// There is no need to change this kernel!
///////////////////////////////////////////////////////////////////////////////
__kernel void ComputeNormals(unsigned int width,
								unsigned int height, 
								__global float4* d_pos,
								__global float4* d_normal){
								
    int particleID = get_global_id(0) + get_global_id(1) * width;
    float4 normal = (float4)( 0.0f, 0.0f, 0.0f, 0.0f);
    
    int minX, maxX, minY, maxY, cntX, cntY;
    minX = max( (int)(0), (int)(get_global_id(0)-1));
    maxX = min( (int)(width-1), (int)(get_global_id(0)+1));
    minY = max( (int)(0), (int)(get_global_id(1)-1));
    maxY = min( (int)(height-1), (int)(get_global_id(1)+1));
    
    for( cntX = minX; cntX < maxX; ++cntX) {
        for( cntY = minY; cntY < maxY; ++cntY) {
            normal += normalize( CalcTriangleNormal(
                d_pos[(cntX+1)+width*(cntY)],
                d_pos[(cntX)+width*(cntY)],
                d_pos[(cntX)+width*(cntY+1)]));
            normal += normalize( CalcTriangleNormal(
                d_pos[(cntX+1)+width*(cntY+1)],
                d_pos[(cntX+1)+width*(cntY)],
                d_pos[(cntX)+width*(cntY+1)]));
        }
    }
    d_normal[particleID] = normalize( normal);
}
