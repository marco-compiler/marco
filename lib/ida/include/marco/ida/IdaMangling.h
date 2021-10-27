#pragma once

#include <marco/runtime/IdaFunctions.h>
#include <marco/runtime/Mangling.h>

//===----------------------------------------------------------------------===//
// Allocation, initialization, usage and deletion
//===----------------------------------------------------------------------===//

#define allocIdaUserData NAME_MANGLED(allocIdaUserData, PTR(void), int64_t)
#define idaInit NAME_MANGLED(idaInit, bool, PTR(void), int64_t)
#define idaStep NAME_MANGLED(idaStep, bool, PTR(void))
#define freeIdaUserData NAME_MANGLED(freeIdaUserData, bool, PTR(void))
#define addTime NAME_MANGLED(addTime, void, PTR(void), double, double, double)
#define addTolerance NAME_MANGLED(addTolerance, void, PTR(void), double, double)

//===----------------------------------------------------------------------===//
// Equation setters
//===----------------------------------------------------------------------===//

#define addColumnIndex NAME_MANGLED(addColumnIndex, void, PTR(void), int64_t, int64_t)
#define addEqDimension NAME_MANGLED(addEqDimension, void, PTR(void), ARRAY(int64_t), ARRAY(int64_t))
#define addResidual NAME_MANGLED(addResidual, void, PTR(void), int64_t, int64_t)
#define addJacobian NAME_MANGLED(addJacobian, void, PTR(void), int64_t, int64_t)

//===----------------------------------------------------------------------===//
// Variable setters
//===----------------------------------------------------------------------===//

#define addVarOffset NAME_MANGLED(addVarOffset, int64_t, PTR(void), int64_t)
#define addVarDimension NAME_MANGLED(addVarDimension, void, PTR(void), ARRAY(int64_t))
#define addVarAccess NAME_MANGLED(addVarAccess, int64_t, PTR(void), int64_t, ARRAY(int64_t), ARRAY(int64_t))

//===----------------------------------------------------------------------===//
// Getters
//===----------------------------------------------------------------------===//

#define getIdaTime NAME_MANGLED(getIdaTime, double, PTR(void))
#define getIdaVariable NAME_MANGLED(getIdaVariable, double, PTR(void), int64_t)
#define getIdaDerivative NAME_MANGLED(getIdaDerivative, double, PTR(void), int64_t)

//===----------------------------------------------------------------------===//
// Lambda constructions
//===----------------------------------------------------------------------===//

#define lambdaConstant NAME_MANGLED(lambdaConstant, int64_t, PTR(void), double)
#define lambdaTime NAME_MANGLED(lambdaTime, int64_t, PTR(void))
#define lambdaInduction NAME_MANGLED(lambdaInduction, int64_t, PTR(void), int64_t)
#define lambdaVariable NAME_MANGLED(lambdaVariable, int64_t, PTR(void), int64_t)
#define lambdaDerivative NAME_MANGLED(lambdaDerivative, int64_t, PTR(void), int64_t)

#define lambdaNegate NAME_MANGLED(lambdaNegate, int64_t, PTR(void), int64_t)
#define lambdaAdd NAME_MANGLED(lambdaAdd, int64_t, PTR(void), int64_t, int64_t)
#define lambdaSub NAME_MANGLED(lambdaSub, int64_t, PTR(void), int64_t, int64_t)
#define lambdaMul NAME_MANGLED(lambdaMul, int64_t, PTR(void), int64_t, int64_t)
#define lambdaDiv NAME_MANGLED(lambdaDiv, int64_t, PTR(void), int64_t, int64_t)
#define lambdaPow NAME_MANGLED(lambdaPow, int64_t, PTR(void), int64_t, int64_t)
#define lambdaAtan2 NAME_MANGLED(lambdaAtan2, int64_t, PTR(void), int64_t, int64_t)

#define lambdaAbs NAME_MANGLED(lambdaAbs, int64_t, PTR(void), int64_t)
#define lambdaSign NAME_MANGLED(lambdaSign, int64_t, PTR(void), int64_t)
#define lambdaSqrt NAME_MANGLED(lambdaSqrt, int64_t, PTR(void), int64_t)
#define lambdaExp NAME_MANGLED(lambdaExp, int64_t, PTR(void), int64_t)
#define lambdaLog NAME_MANGLED(lambdaLog, int64_t, PTR(void), int64_t)
#define lambdaLog10 NAME_MANGLED(lambdaLog10, int64_t, PTR(void), int64_t)

#define lambdaSin NAME_MANGLED(lambdaSin, int64_t, PTR(void), int64_t)
#define lambdaCos NAME_MANGLED(lambdaCos, int64_t, PTR(void), int64_t)
#define lambdaTan NAME_MANGLED(lambdaTan, int64_t, PTR(void), int64_t)
#define lambdaAsin NAME_MANGLED(lambdaAsin, int64_t, PTR(void), int64_t)
#define lambdaAcos NAME_MANGLED(lambdaAcos, int64_t, PTR(void), int64_t)
#define lambdaAtan NAME_MANGLED(lambdaAtan, int64_t, PTR(void), int64_t)
#define lambdaSinh NAME_MANGLED(lambdaSinh, int64_t, PTR(void), int64_t)
#define lambdaCosh NAME_MANGLED(lambdaCosh, int64_t, PTR(void), int64_t)
#define lambdaTanh NAME_MANGLED(lambdaTanh, int64_t, PTR(void), int64_t)
