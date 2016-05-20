
//Each thread load exactly one halo pixel
//Thus, we assume that the halo size is not larger than the 
//dimension of the work-group in the direction of the kernel

//to efficiently reduce the memory transfer overhead of the global memory
// (each pixel is lodaded multiple times at high overlaps)
// one work-item will compute RESULT_STEPS pixels

//for unrolling loops, these values have to be known at compile time

/* These macros will be defined dynamically during building the program

#define KERNEL_RADIUS 2

//horizontal kernel
#define H_GROUPSIZE_X		32
#define H_GROUPSIZE_Y		4
#define H_RESULT_STEPS		2

//vertical kernel
#define V_GROUPSIZE_X		32
#define V_GROUPSIZE_Y		16
#define V_RESULT_STEPS		3

*/

#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)


//////////////////////////////////////////////////////////////////////////////////////////////////////
// Horizontal convolution filter

/*
c_Kernel stores 2 * KERNEL_RADIUS + 1 weights, use these during the convolution
*/

//require matching work-group size
__kernel __attribute__((reqd_work_group_size(H_GROUPSIZE_X, H_GROUPSIZE_Y, 1)))
void ConvHorizontal(
			__global float* d_Dst,
			__global const float* d_Src,
			__constant float* c_Kernel,
			int Width,
			int Pitch
			)
{
	//The size of the local memory: one value for each work-item.
	//We even load unused pixels to the halo area, to keep the code and local memory access simple.
	//Since these loads are coalesced, they introduce no overhead, except for slightly redundant local memory allocation.
	//Each work-item loads H_RESULT_STEPS values + 2 halo values
	__local float tile[H_GROUPSIZE_Y][(H_RESULT_STEPS + 2) * H_GROUPSIZE_X];

	int2 GID,LID,GRID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);
	LID.x = get_local_id(0);
	LID.y = get_local_id(1);
	GRID.x = get_group_id(0);
	GRID.y = get_group_id(1);
	const int baseX = H_GROUPSIZE_X * GRID.x * H_RESULT_STEPS;
	const int baseY = H_GROUPSIZE_Y * GRID.y;
	
	// Load main data + right halo (check for right bound)
	for (int tileID = -1; tileID <= H_RESULT_STEPS; tileID++) {
		int temp = baseX + tileID * H_GROUPSIZE_X + LID.x;
		if (temp < 0 || temp >= Width) tile[LID.y][LID.x + (tileID + 1) * H_GROUPSIZE_X] = 0;
		else tile[LID.y][LID.x + (tileID + 1) * H_GROUPSIZE_X] = d_Src[(baseY + LID.y) * Pitch + temp];
	}

	// Sync the work-items after loading
	barrier(CLK_LOCAL_MEM_FENCE);

	// Convolve and store the result
	for (int tileID = 0; tileID < H_RESULT_STEPS; tileID++) {
		float sum = 0.0;
		for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++) {
			sum += c_Kernel[KERNEL_RADIUS - i] * tile[LID.y][LID.x + (tileID + 1) * H_GROUPSIZE_X + i];
		}
		int temp = baseX + tileID * H_GROUPSIZE_X + LID.x;
		if (temp < Width) d_Dst[(baseY + LID.y) * Pitch + temp] = sum;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
// Vertical convolution filter

//require matching work-group size
__kernel __attribute__((reqd_work_group_size(V_GROUPSIZE_X, V_GROUPSIZE_Y, 1)))
void ConvVertical(
			__global float* d_Dst,
			__global const float* d_Src,
			__constant float* c_Kernel,
			int Height,
			int Pitch
			)
{
	__local float tile[(V_RESULT_STEPS + 2) * V_GROUPSIZE_Y][V_GROUPSIZE_X];

	int2 GID,LID,GRID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);
	LID.x = get_local_id(0);
	LID.y = get_local_id(1);
	GRID.x = get_group_id(0);
	GRID.y = get_group_id(1);
	const int baseX = V_GROUPSIZE_X * GRID.x;
	const int baseY = V_GROUPSIZE_Y * GRID.y * V_RESULT_STEPS;

	// Load main data + right halo (check for right bound)
	for (int tileID = -1; tileID <= V_RESULT_STEPS; tileID++) {
		int temp = baseY + tileID * V_GROUPSIZE_Y + LID.y;
		if (temp < 0 || temp >= Height) tile[LID.y + (tileID + 1) * V_GROUPSIZE_Y][LID.x] = 0;
		else tile[LID.y + (tileID + 1) * V_GROUPSIZE_Y][LID.x] = d_Src[temp * Pitch + baseX + LID.x];
	}



	// Sync the work-items after loading
	barrier(CLK_LOCAL_MEM_FENCE);

	// Convolve and store the result
	for (int tileID = 0; tileID < V_RESULT_STEPS; tileID++) {
		float sum = 0.0;
		for (int i = -KERNEL_RADIUS; i <= KERNEL_RADIUS; i++) {
			sum += c_Kernel[KERNEL_RADIUS - i] * tile[LID.y + (tileID + 1) * V_GROUPSIZE_Y + i][LID.x];
		}
		int temp = baseY + tileID * V_GROUPSIZE_Y + LID.y;
		if (temp < Height) d_Dst[temp * Pitch + baseX + LID.x] = sum;
	}


}
