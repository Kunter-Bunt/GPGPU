
__kernel void MatrixRotNaive(__global const float* M, __global float* MR, uint SizeX, uint SizeY)
{
	int2 GID;

	GID.x = get_global_id(0);
	GID.y = get_global_id(1);

	if (GID.x < SizeX && GID.y < SizeY) {
		MR[GID.x * SizeY + (SizeY - GID.y - 1)] = M[GID.y * SizeX + GID.x];
	}
}

__kernel void MatrixRotOptimized(__global const float* M, __global float* MR, uint SizeX, uint SizeY, __local float* block)
{
	int2 GID;
	int2 LID;
	int2 NGID;
	
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);

	LID.x = get_local_id(0);
	LID.y = get_local_id(1);
		
	NGID.x = (GID.x + GID.y * SizeX) % SizeY;
	NGID.y = (GID.x + GID.y * SizeX) / SizeY;

	if (GID.x < SizeX && GID.y < SizeY) {
		block[LID.x + LID.y * get_local_size(0)] = M[(SizeY - NGID.x - 1) * SizeX + NGID.y];
		barrier(CLK_LOCAL_MEM_FENCE);

		MR[NGID.y * SizeY + NGID.x] = block[LID.y * get_local_size(0) + LID.x];
	}

}
 

	
	
