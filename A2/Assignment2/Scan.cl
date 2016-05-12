


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_Naive(const __global uint* inArray, __global uint* outArray, uint N, uint offset) 
{
	int GID = get_global_id(0);
	if (GID >= offset) outArray[GID] = inArray[GID] + inArray[GID - offset];
	else outArray[GID] = inArray[GID];
}



// Why did we not have conflicts in the Reduction? Because of the sequential addressing (here we use interleaved => we have conflicts).

#define UNROLL
#define NUM_BANKS			32
#define NUM_BANKS_LOG		5
#define SIMD_GROUP_SIZE		32

// Bank conflicts
#define AVOID_BANK_CONFLICTS
#ifdef AVOID_BANK_CONFLICTS
	// TO DO: define your conflict-free macro here
#else
	#define OFFSET(A) (A)
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_WorkEfficient(__global uint* array, __global uint* higherLevelArray, __local uint* block) 
{
	int GID = get_global_id(0);
	int LID = get_local_id(0);
	int GSize = get_global_size(0);
	int LSize = get_local_size(0);

	//UpSweep
	block[LID] = array[GID];
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int stride = 1; stride < LSize; stride *= 2) {
		if ((LID + 1) % (2 * stride) == 0 && LID >= stride) block[LID] += block[LID - stride];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (LID == LSize-1) block[LSize-1] = 0;	
	barrier(CLK_LOCAL_MEM_FENCE);

	//DownSweep
	for (int stride = LSize; stride > 0; stride /= 2) {
		if ((LID + 1) % (2 * stride) == 0) {
			int left = block[LID];
			block[LID] += block[LID - stride];
			block[LID - stride] = left;
			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}
	array[GID] += block[LID];
	
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__kernel void Scan_WorkEfficientAdd(__global uint* higherLevelArray, __global uint* array, __local uint* block) 
{
	// TO DO: Kernel implementation (large arrays)
	// Kernel that should add the group PPS to the local PPS (Figure 14)
}
