#ifndef MARCO_DIALECTS_MODELICA_OPINTERFACES_H
#define MARCO_DIALECTS_MODELICA_OPINTERFACES_H

#include "marco/Dialect/Modelica/VariableAccess.h"
#include "marco/Modeling/DimensionAccess.h"
#include "marco/Modeling/DimensionAccessAdd.h"
#include "marco/Modeling/DimensionAccessConstant.h"
#include "marco/Modeling/DimensionAccessDimension.h"
#include "marco/Modeling/DimensionAccessDiv.h"
#include "marco/Modeling/DimensionAccessMul.h"
#include "marco/Modeling/DimensionAccessIndices.h"
#include "marco/Modeling/DimensionAccessRange.h"
#include "marco/Modeling/DimensionAccessSub.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"

#define GET_OP_FWD_DEFINES
#include "marco/Dialect/Modelica/Modelica.h.inc"

namespace mlir::modelica
{
  using DimensionAccess = ::marco::modeling::DimensionAccess;
  using DimensionAccessConstant = ::marco::modeling::DimensionAccessConstant;
  using DimensionAccessDimension = ::marco::modeling::DimensionAccessDimension;
  using DimensionAccessAdd = ::marco::modeling::DimensionAccessAdd;
  using DimensionAccessSub = ::marco::modeling::DimensionAccessSub;
  using DimensionAccessMul = ::marco::modeling::DimensionAccessMul;
  using DimensionAccessDiv = ::marco::modeling::DimensionAccessDiv;
  using DimensionAccessRange = ::marco::modeling::DimensionAccessRange;
  using DimensionAccessIndices = ::marco::modeling::DimensionAccessIndices;
}

#include "marco/Dialect/Modelica/ModelicaOpInterfaces.h.inc"

#endif // MARCO_DIALECTS_MODELICA_OPINTERFACES_H
