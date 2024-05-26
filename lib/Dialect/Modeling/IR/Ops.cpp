#include "marco/Dialect/Modeling/IR/ModelingDialect.h"
#include "marco/Dialect/Modeling/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"

using namespace ::mlir::modeling;

#define GET_OP_CLASSES
#include "marco/Dialect/Modeling/IR/Modeling.cpp.inc"
