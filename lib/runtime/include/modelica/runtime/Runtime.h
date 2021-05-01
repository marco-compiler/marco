#pragma once

#include "ArrayDescriptor.h"

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

	[[maybe_unused]] void _Mfill_ai1_i1(UnsizedArrayDescriptor<bool> array, bool value);
	[[maybe_unused]] void _Mfill_ai32_i32(UnsizedArrayDescriptor<int> array, int value);
	[[maybe_unused]] void _Mfill_ai64_i64(UnsizedArrayDescriptor<long> array, long value);
	[[maybe_unused]] void _Mfill_af32_f32(UnsizedArrayDescriptor<float> array, float value);
	[[maybe_unused]] void _Mfill_af64_f64(UnsizedArrayDescriptor<double> array, double value);

	[[maybe_unused]] void _mlir_ciface__Mfill_ai1_i1(UnsizedArrayDescriptor<bool> array, bool value);
	[[maybe_unused]] void _mlir_ciface__Mfill_ai32_i32(UnsizedArrayDescriptor<int> array, int value);
	[[maybe_unused]] void _mlir_ciface__Mfill_ai64_i64(UnsizedArrayDescriptor<long> array, long value);
	[[maybe_unused]] void _mlir_ciface__Mfill_af32_f32(UnsizedArrayDescriptor<float> array, float value);
	[[maybe_unused]] void _mlir_ciface__Mfill_af64_f64(UnsizedArrayDescriptor<double> array, double value);

	[[maybe_unused]] void _Midentity_ai1(UnsizedArrayDescriptor<bool> array);
	[[maybe_unused]] void _Midentity_ai32(UnsizedArrayDescriptor<int> array);
	[[maybe_unused]] void _Midentity_ai64(UnsizedArrayDescriptor<long> array);
	[[maybe_unused]] void _Midentity_af32(UnsizedArrayDescriptor<float> array);
	[[maybe_unused]] void _Midentity_af64(UnsizedArrayDescriptor<double> array);

	[[maybe_unused]] void _mlir_ciface__Midentity_ai1(UnsizedArrayDescriptor<bool> array);
	[[maybe_unused]] void _mlir_ciface__Midentity_ai32(UnsizedArrayDescriptor<int> array);
	[[maybe_unused]] void _mlir_ciface__Midentity_ai64(UnsizedArrayDescriptor<long> array);
	[[maybe_unused]] void _mlir_ciface__Midentity_af32(UnsizedArrayDescriptor<float> array);
	[[maybe_unused]] void _mlir_ciface__Midentity_af64(UnsizedArrayDescriptor<double> array);
}
