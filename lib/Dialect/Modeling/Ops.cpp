#include "marco/Dialect/Modeling/ModelingDialect.h"
#include "marco/Dialect/Modeling/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"

using namespace ::mlir::modeling;

#define GET_OP_CLASSES
#include "marco/Dialect/Modeling/Modeling.cpp.inc"
