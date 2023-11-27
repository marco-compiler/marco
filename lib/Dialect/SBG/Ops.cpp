#include "marco/Dialect/SBG/SBGDialect.h"
#include "marco/Dialect/SBG/Ops.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace ::mlir::sbg;

#define GET_OP_CLASSES
#include "marco/Dialect/SBG/SBG.cpp.inc"