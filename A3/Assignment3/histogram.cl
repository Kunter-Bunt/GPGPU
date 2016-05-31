#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
__kernel void
set_array_to_constant(
	__global int *array,
	int num_elements,
	int val
)
{
	// There is no need to touch this kernel
	if(get_global_id(0) < num_elements)
		array[get_global_id(0)] = val;
}

__kernel void
compute_histogram(
	__global int *histogram,   // accumulate histogram here
	__global const float *img, // input image
	int width,                 // image width
	int height,                // image height
	int pitch,                 // image pitch
	int num_hist_bins          // number of histogram bins
)
{
	// Insert your kernel code here
	int2 GID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);

	if (GID.x < width && GID.y < height) {
		float gray = img[GID.y * pitch + GID.x];
		int index = gray * num_hist_bins;
		if (index >= 64) index = 63;
		atomic_inc(&histogram[index]);
	}
} 

__kernel void
compute_histogram_local_memory(
	__global int *histogram,   // accumulate histogram here
	__global const float *img, // input image
	int width,                 // image width
	int height,                // image height
	int pitch,                 // image pitch
	int num_hist_bins,         // number of histogram bins
	__local int *local_hist
)
{
	int2 GID,LID;
	GID.x = get_global_id(0);
	GID.y = get_global_id(1);
	LID.x = get_local_id(0);
	LID.y = get_local_id(1);
	if (LID.x == 0 && LID.y == 0) {	
		for (int i = 0; i < num_hist_bins; i++) local_hist[i] = 0;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (GID.x < width && GID.y < height) {
		float gray = img[GID.y * pitch + GID.x];
		int index = gray * num_hist_bins;
		if (index >= 64) index = 63;
		atomic_inc(&local_hist[index]);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (LID.x == 0 && LID.y == 0) {	
		for (int i = 0; i < num_hist_bins; i++) atomic_add(&histogram[i], local_hist[i]);
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
	
} 
