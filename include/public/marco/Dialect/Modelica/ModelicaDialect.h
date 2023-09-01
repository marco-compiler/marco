#ifndef MARCO_DIALECTS_MODELICA_MODELICADIALECT_H
#define MARCO_DIALECTS_MODELICA_MODELICADIALECT_H

#include "marco/Dialect/Modelica/Common.h"
#include "marco/Dialect/Modelica/Attributes.h"
#include "marco/Dialect/Modelica/Interfaces.h"
#include "marco/Dialect/Modelica/Ops.h"
#include "marco/Dialect/Modelica/Types.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "marco/Dialect/Modelica/ModelicaDialect.h.inc"

namespace mlir::modelica
{
  mlir::Type getMostGenericType(mlir::Value x, mlir::Value y);

  mlir::Type getMostGenericType(mlir::Type x, mlir::Type y);

  mlir::LogicalResult materializeAffineMap(
      mlir::OpBuilder& builder,
      mlir::Location loc,
      mlir::AffineMap affineMap,
      mlir::ValueRange dimensions,
      llvm::SmallVectorImpl<mlir::Value>& results);

  mlir::Value materializeAffineExpr(
      mlir::OpBuilder& builder,
      mlir::Location loc,
      mlir::AffineExpr expression,
      mlir::ValueRange dimensions);
}

#endif // MARCO_DIALECTS_MODELICA_MODELICADIALECT_H
