
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
	int2 NLID;
	int2 ecke;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);
	if (GID.x < SizeX && GID.y < SizeY) {
		LID.x = get_local_id(0);
		LID.y = get_local_id(1);

		block[LID.x * get_local_size(1) + (get_local_size(1) - LID.y - 1)] = M[GID.y * SizeX + GID.x];
		barrier(CLK_LOCAL_MEM_FENCE);

		NLID.x = (LID.x + LID.y * get_local_size(0)) % get_local_size(1);
		NLID.y = (LID.x + LID.y * get_local_size(0)) / get_local_size(1);
		
		ecke.x = SizeY - GID.x - LID.x - get_local_size(1);
		ecke.y = GID.x - LID.x;

		MR[ecke.x + NLID.x + (ecke.y + NLID.y)*SizeY] = block[LID.y * get_local_size(0) + LID.x ];

		

	}
}
 
