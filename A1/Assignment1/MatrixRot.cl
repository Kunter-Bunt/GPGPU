
__kernel void MatrixRotNaive(__global const float* M, __global float* MR, uint SizeX, uint SizeY)
{
	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);
	MR[GID.x * SizeY + (SizeY - GID.y - 1)] = M[GID.y * SizeX + GID.x];
}

__kernel void MatrixRotOptimized(__global const float* M, __global float* MR, uint SizeX, uint SizeY, __local float* block)
{

}
 
