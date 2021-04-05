#pragma once

extern "C"
{
	void modelicaPrint(char* name, float value);

	void modelicaPrintFVector(char* name, float* value, int count);

	void modelicaPrintBVector(char* name, char* value, int count);

	void modelicaPrintIVector(char* name, int* value, int count);

	void fill(float* out, long* outDim, float* filler, long* dim);

	float modelicaPow(float b, float exp);
	double modelicaPowD(double b, double exp);

	void printString(char* str);
	void printI1(bool value);
	void printI32(int value);
	void printI64(long value);
	void printF32(float value);
	void printF64(double value);
}
