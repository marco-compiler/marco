#ifndef MARCO_DIALECTS_BASEMODELICA_OPS_H
#define MARCO_DIALECTS_BASEMODELICA_OPS_H

#include "marco/Dialect/BaseModelica/Attributes.h"
#include "marco/Dialect/BaseModelica/OpInterfaces.h"
#include "marco/Dialect/BaseModelica/Types.h"
#include "marco/Dialect/BaseModelica/VariableAccess.h"
#include "marco/Dialect/Modeling/Attributes.h"
#include "marco/Modeling/AccessFunction.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/RegionKindInterface.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.h"

#define GET_OP_FWD_DEFINES
#include "marco/Dialect/BaseModelica/BaseModelica.h.inc"

#define GET_OP_CLASSES
#include "marco/Dialect/BaseModelica/BaseModelica.h.inc"

#endif // MARCO_DIALECTS_BASEMODELICA_OPS_H