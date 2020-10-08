#pragma once

extern "C"
{
	void modelicaPrint(char* name, float value);

	void modelicaPrintFVector(char* name, float* value, int count);

	void modelicaPrintBVector(char* name, char* value, int count);

	void modelicaPrintIVector(char* name, int* value, int count);

	void fill(float* out, long* outDim, float* filler, long* dim);
}
