/*
We assume a 3x3 (radius: 1) convolution kernel, which is not separable.
Each work-group will process a (TILE_X x TILE_Y) tile of the image.
For coalescing, TILE_X should be multiple of 16.

Instead of examining the image border for each kernel, we recommend to pad the image
to be the multiple of the given tile-size.
*/

//should be multiple of 32 on Fermi and 16 on pre-Fermi...
#define TILE_X 32 

#define TILE_Y 16

// d_Dst is the convolution of d_Src with the kernel c_Kernel
// c_Kernel is assumed to be a float[11] array of the 3x3 convolution constants, one multiplier (for normalization) and an offset (in this order!)
// With & Height are the image dimensions (should be multiple of the tile size)
__kernel __attribute__((reqd_work_group_size(TILE_X, TILE_Y, 1)))
void Convolution(
				__global float* d_Dst,
				__global const float* d_Src,
				__constant float* c_Kernel,
				uint Width,  // Use width to check for image bounds
				uint Height,
				uint Pitch   // Use pitch for offsetting between lines
				)
{
	// OpenCL allows to allocate the local memory from 'inside' the kernel (without using the clSetKernelArg() call)
	// in a similar way to standard C.
	// the size of the local memory necessary for the convolution is the tile size + the halo area
	__local float tile[TILE_Y + 2][TILE_X + 2];
	int2 GID;
	int2 LID;
	int2 LSize;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);
	LID.x = get_local_id(0);
	LID.y = get_local_id(1);
	LSize.x = get_local_size(0);
	LSize.y = get_local_size(1);
	// TO DO...
	

	// Fill the halo with zeros
	tile[LID.y][LID.x] = 0;
	tile[LID.y + 2][LID.x] = 0;	
	tile[LID.y][LID.x + 2] = 0;	
	
	barrier(CLK_LOCAL_MEM_FENCE);

	// Load main filtered area from d_Src
	tile[LID.y + 1][LID.x + 1] = d_Src[GID.y * Pitch + GID.x];
	
	barrier(CLK_LOCAL_MEM_FENCE);

	// Load halo regions from d_Src (edges and corners separately), check for image bounds!
	if ((LID.x == 0 || LID.x + 1 == LSize.x) || (LID.y == 0 || LID.y + 1 == LSize.y)) {
		if (GID.x > 0 && GID.y > 0) 		tile[LID.y][LID.x] = d_Src[(GID.y - 1) * Pitch + GID.x - 1];
		if (GID.x > 0 && GID.y + 1 < Height) 	tile[LID.y + 2][LID.x] = d_Src[(GID.y + 1) * Pitch + GID.x - 1];
		if (GID.x + 1 < Width && GID.y > 0) 	tile[LID.y][LID.x + 2] = d_Src[(GID.y - 1) * Pitch + GID.x + 1];
		if (GID.x + 1 < Width && GID.y + 1 < Height) 	tile[LID.y + 2][LID.x + 2] = d_Src[(GID.y + 1) * Pitch + GID.x + 1];
	}

	// Sync threads
	barrier(CLK_LOCAL_MEM_FENCE);
	//d_Dst[GID.y * Pitch + GID.x] = tile[LID.y + 1][LID.x + 1];	
	//return;

	// Perform the convolution and store the convolved signal to d_Dst.
	float sum = 0.0;
	for (int x = 0; x < 9; x++) {
			sum += c_Kernel[x] * tile[LID.y + x / 3][LID.x + x % 3];
	} 
	d_Dst[GID.y * Pitch + GID.x] = c_Kernel[9] * sum + c_Kernel[10];
}
