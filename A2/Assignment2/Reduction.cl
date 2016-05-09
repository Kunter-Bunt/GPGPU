
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_InterleavedAddressing(__global uint* array, uint stride) 
{
	int GID = get_global_id(0);
	array[GID * stride] += array[GID * stride + stride / 2];
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_SequentialAddressing(__global uint* array, uint stride) 
{
	int GID = get_global_id(0);
	array[GID] += array[GID + stride];
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_Decomp(const __global uint* inArray, __global uint* outArray, uint N, __local uint* block)
{
	int LID = get_local_id(0);
	int Elem = get_local_size(0) * get_group_id(0) * 2 + LID;
	int LSize = get_local_size(0);
	block[LID] = inArray[Elem] + inArray[Elem + LSize];
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int stride = LSize/2; stride >= 1; stride /= 2) {
		if (LID < stride) block[LID] += block[LID + stride];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (LID == 0) outArray[get_group_id(0)] = block[0];
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Reduction_DecompUnroll(const __global uint* inArray, __global uint* outArray, uint N, __local uint* localBlock)
{
	// TO DO: Kernel implementation
}
