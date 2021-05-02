#pragma once

#include "ArrayDescriptor.h"
#include "Mangling.h"

RUNTIME_FUNC_DECL(fill, void, ARRAY(bool), bool)
RUNTIME_FUNC_DECL(fill, void, ARRAY(int), int)
RUNTIME_FUNC_DECL(fill, void, ARRAY(long), long)
RUNTIME_FUNC_DECL(fill, void, ARRAY(float), float)
RUNTIME_FUNC_DECL(fill, void, ARRAY(double), double)

RUNTIME_FUNC_DECL(identity, void, ARRAY(bool))
RUNTIME_FUNC_DECL(identity, void, ARRAY(int))
RUNTIME_FUNC_DECL(identity, void, ARRAY(long))
RUNTIME_FUNC_DECL(identity, void, ARRAY(float))
RUNTIME_FUNC_DECL(identity, void, ARRAY(double))

RUNTIME_FUNC_DECL(diagonal, void, ARRAY(bool), ARRAY(bool))
RUNTIME_FUNC_DECL(diagonal, void, ARRAY(bool), ARRAY(int))
RUNTIME_FUNC_DECL(diagonal, void, ARRAY(bool), ARRAY(long))
RUNTIME_FUNC_DECL(diagonal, void, ARRAY(bool), ARRAY(float))
RUNTIME_FUNC_DECL(diagonal, void, ARRAY(bool), ARRAY(double))

RUNTIME_FUNC_DECL(diagonal, void, ARRAY(int), ARRAY(bool))
RUNTIME_FUNC_DECL(diagonal, void, ARRAY(int), ARRAY(int))
RUNTIME_FUNC_DECL(diagonal, void, ARRAY(int), ARRAY(long))
RUNTIME_FUNC_DECL(diagonal, void, ARRAY(int), ARRAY(float))
RUNTIME_FUNC_DECL(diagonal, void, ARRAY(int), ARRAY(double))

RUNTIME_FUNC_DECL(diagonal, void, ARRAY(long), ARRAY(bool))
RUNTIME_FUNC_DECL(diagonal, void, ARRAY(long), ARRAY(int))
RUNTIME_FUNC_DECL(diagonal, void, ARRAY(long), ARRAY(long))
RUNTIME_FUNC_DECL(diagonal, void, ARRAY(long), ARRAY(float))
RUNTIME_FUNC_DECL(diagonal, void, ARRAY(long), ARRAY(double))

RUNTIME_FUNC_DECL(diagonal, void, ARRAY(float), ARRAY(bool))
RUNTIME_FUNC_DECL(diagonal, void, ARRAY(float), ARRAY(int))
RUNTIME_FUNC_DECL(diagonal, void, ARRAY(float), ARRAY(long))
RUNTIME_FUNC_DECL(diagonal, void, ARRAY(float), ARRAY(float))
RUNTIME_FUNC_DECL(diagonal, void, ARRAY(float), ARRAY(double))

RUNTIME_FUNC_DECL(diagonal, void, ARRAY(double), ARRAY(bool))
RUNTIME_FUNC_DECL(diagonal, void, ARRAY(double), ARRAY(int))
RUNTIME_FUNC_DECL(diagonal, void, ARRAY(double), ARRAY(long))
RUNTIME_FUNC_DECL(diagonal, void, ARRAY(double), ARRAY(float))
RUNTIME_FUNC_DECL(diagonal, void, ARRAY(double), ARRAY(double))

RUNTIME_FUNC_DECL(zeros, void, ARRAY(bool))
RUNTIME_FUNC_DECL(zeros, void, ARRAY(int))
RUNTIME_FUNC_DECL(zeros, void, ARRAY(long))
RUNTIME_FUNC_DECL(zeros, void, ARRAY(float))
RUNTIME_FUNC_DECL(zeros, void, ARRAY(double))

RUNTIME_FUNC_DECL(ones, void, ARRAY(bool))
RUNTIME_FUNC_DECL(ones, void, ARRAY(int))
RUNTIME_FUNC_DECL(ones, void, ARRAY(long))
RUNTIME_FUNC_DECL(ones, void, ARRAY(float))
RUNTIME_FUNC_DECL(ones, void, ARRAY(double))

RUNTIME_FUNC_DECL(linspace, void, ARRAY(bool), float, float)
RUNTIME_FUNC_DECL(linspace, void, ARRAY(bool), double, double)
RUNTIME_FUNC_DECL(linspace, void, ARRAY(int), float, float)
RUNTIME_FUNC_DECL(linspace, void, ARRAY(int), double, double)
RUNTIME_FUNC_DECL(linspace, void, ARRAY(long), float, float)
RUNTIME_FUNC_DECL(linspace, void, ARRAY(long), double, double)
RUNTIME_FUNC_DECL(linspace, void, ARRAY(float), float, float)
RUNTIME_FUNC_DECL(linspace, void, ARRAY(float), double, double)
RUNTIME_FUNC_DECL(linspace, void, ARRAY(double), float, float)
RUNTIME_FUNC_DECL(linspace, void, ARRAY(double), double, double)

RUNTIME_FUNC_DECL(min, bool, ARRAY(bool))
RUNTIME_FUNC_DECL(min, int, ARRAY(int))
RUNTIME_FUNC_DECL(min, long, ARRAY(long))
RUNTIME_FUNC_DECL(min, float, ARRAY(float))
RUNTIME_FUNC_DECL(min, double, ARRAY(double))

RUNTIME_FUNC_DECL(min, bool, bool, bool)
RUNTIME_FUNC_DECL(min, int, int, int)
RUNTIME_FUNC_DECL(min, long, long, long)
RUNTIME_FUNC_DECL(min, float, float, float)
RUNTIME_FUNC_DECL(min, double, double, double)

RUNTIME_FUNC_DECL(max, bool, ARRAY(bool))
RUNTIME_FUNC_DECL(max, int, ARRAY(int))
RUNTIME_FUNC_DECL(max, long, ARRAY(long))
RUNTIME_FUNC_DECL(max, float, ARRAY(float))
RUNTIME_FUNC_DECL(max, double, ARRAY(double))

RUNTIME_FUNC_DECL(max, bool, bool, bool)
RUNTIME_FUNC_DECL(max, int, int, int)
RUNTIME_FUNC_DECL(max, long, long, long)
RUNTIME_FUNC_DECL(max, float, float, float)
RUNTIME_FUNC_DECL(max, double, double, double)

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
