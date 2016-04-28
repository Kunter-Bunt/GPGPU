
// Rotate the matrix CLOCKWISE

//naive implementation: move the elements of the matrix directly to their destinations
//this will cause unaligned memory accessed which - as we will see - should be avoided on the GPU

__kernel void MatrixRotNaive(__global const float* M, __global float* MR, uint SizeX, uint SizeY)
{
	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);
	MR[GID.x * SizeX + (SizeY - GID.y - 1)] = M[GID.y * SizeX + GID.x];
}

//this kernel does the same thing, however, the local memory is used to
//transform a small chunk of the matrix locally
//then write it back after synchronization in a coalesced access pattern

__kernel void MatrixRotOptimized(__global const float* M, __global float* MR, uint SizeX, uint SizeY,
							__local float* block)
{
	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);
	int2 LID;
	LID.x = get_local_id(0);
	LID.y = get_local_id(1);

	block[LID.y * get_local_size(0) + LID.x ] = M[GID.y * SizeX + GID.x]
	barrier(CLK_LOCAL_MEM_FENCE);

}
 
