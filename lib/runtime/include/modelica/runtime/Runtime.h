#pragma once

#include "ArrayDescriptor.h"
#include "Mangling.h"

RUNTIME_FUNC_DECL(fill, void, array(bool), bool)
RUNTIME_FUNC_DECL(fill, void, array(int), int)
RUNTIME_FUNC_DECL(fill, void, array(long), long)
RUNTIME_FUNC_DECL(fill, void, array(float), float)
RUNTIME_FUNC_DECL(fill, void, array(double), double)

RUNTIME_FUNC_DECL(identity, void, array(bool))
RUNTIME_FUNC_DECL(identity, void, array(int))
RUNTIME_FUNC_DECL(identity, void, array(long))
RUNTIME_FUNC_DECL(identity, void, array(float))
RUNTIME_FUNC_DECL(identity, void, array(double))

RUNTIME_FUNC_DECL(diagonal, void, array(bool), array(bool))
RUNTIME_FUNC_DECL(diagonal, void, array(bool), array(int))
RUNTIME_FUNC_DECL(diagonal, void, array(bool), array(long))
RUNTIME_FUNC_DECL(diagonal, void, array(bool), array(float))
RUNTIME_FUNC_DECL(diagonal, void, array(bool), array(double))

RUNTIME_FUNC_DECL(diagonal, void, array(int), array(bool))
RUNTIME_FUNC_DECL(diagonal, void, array(int), array(int))
RUNTIME_FUNC_DECL(diagonal, void, array(int), array(long))
RUNTIME_FUNC_DECL(diagonal, void, array(int), array(float))
RUNTIME_FUNC_DECL(diagonal, void, array(int), array(double))

RUNTIME_FUNC_DECL(diagonal, void, array(long), array(bool))
RUNTIME_FUNC_DECL(diagonal, void, array(long), array(int))
RUNTIME_FUNC_DECL(diagonal, void, array(long), array(long))
RUNTIME_FUNC_DECL(diagonal, void, array(long), array(float))
RUNTIME_FUNC_DECL(diagonal, void, array(long), array(double))

RUNTIME_FUNC_DECL(diagonal, void, array(float), array(bool))
RUNTIME_FUNC_DECL(diagonal, void, array(float), array(int))
RUNTIME_FUNC_DECL(diagonal, void, array(float), array(long))
RUNTIME_FUNC_DECL(diagonal, void, array(float), array(float))
RUNTIME_FUNC_DECL(diagonal, void, array(float), array(double))

RUNTIME_FUNC_DECL(diagonal, void, array(double), array(bool))
RUNTIME_FUNC_DECL(diagonal, void, array(double), array(int))
RUNTIME_FUNC_DECL(diagonal, void, array(double), array(long))
RUNTIME_FUNC_DECL(diagonal, void, array(double), array(float))
RUNTIME_FUNC_DECL(diagonal, void, array(double), array(double))

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
