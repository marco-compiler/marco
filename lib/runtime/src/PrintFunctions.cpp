#include <stdio.h>

extern "C"
{
	void modelicaPrint(char* name, float value)
	{
		printf("%s: %f\n", name, value);
	}

	void modelicaPrintFVector(char* name, float* value, int count)
	{
		printf("%s:\n", name);
		for (int a = 0; a < count; a++)
			printf("\t%f\n", value[a]);
	}

	void modelicaPrintDVector(char* name, double* value, int count)
	{
		printf("%s:\n", name);
		for (int a = 0; a < count; a++)
			printf("\t%12f\n", value[a]);
	}

	void modelicaPrintBVector(char* name, char* value, int count)
	{
		printf("%s:\n", name);
		for (int a = 0; a < count; a++)
		{
			if (value[a])
				printf("\tTrue\n");
			else
				printf("\tFalse\n");
		}
	}

	void modelicaPrintIVector(char* name, int* value, int count)
	{
		printf("%s:\n", name);
		for (int a = 0; a < count; a++)
			printf("\t%d\n", value[a]);
	}

	void fill(float* out, long* outDim, float* filler, long* dim)
	{
		size_t flatSize = 1;
		for (int a = 0; outDim[a] != 0; a++)
			flatSize *= outDim[a];

		for (size_t a = 0; a < flatSize; a++)
			out[a] = filler[0];
	}

	void filld(double* out, long* outDim, double* filler, long* dim)
	{
		size_t flatSize = 1;
		for (int a = 0; outDim[a] != 0; a++)
			flatSize *= outDim[a];

		for (size_t a = 0; a < flatSize; a++)
			out[a] = filler[0];
	}
}
