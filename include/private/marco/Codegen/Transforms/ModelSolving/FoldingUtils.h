#include "mlir/IR/BuiltinAttributes.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"

#ifndef MARCO_FOLDINGUTILS_H
#define MARCO_FOLDINGUTILS_H

using namespace ::mlir::modelica;

static double getDoubleFromAttribute(mlir::Attribute attribute)
{
  if (auto indexAttr = attribute.dyn_cast<mlir::IntegerAttr>()) {
    return indexAttr.getInt();
  }

  if (auto booleanAttr = attribute.dyn_cast<BooleanAttr>()) {
    return booleanAttr.getValue() ? 1 : 0;
  }

  if (auto integerAttr = attribute.dyn_cast<IntegerAttr>()) {
    return integerAttr.getValue().getSExtValue();
  }

  if (auto realAttr = attribute.dyn_cast<RealAttr>()) {
    return realAttr.getValue().convertToDouble();
  }

  llvm_unreachable("Unknown attribute type");
  return 0;
}

static double recursiveFoldValue(mlir::Operation* op)
{
  if (mlir::isa<ConstantOp>(op)) {
    ConstantOp constantOp = mlir::dyn_cast<ConstantOp>(op);
    return getDoubleFromAttribute(constantOp.getValue());
  } else {
    auto ops = op->getOperands();
    if (mlir::isa<AddOp>(op)) {
      return recursiveFoldValue(ops[0].getDefiningOp()) + recursiveFoldValue(ops[1].getDefiningOp());
    } else if (mlir::isa<SubOp>(op)) {
      return recursiveFoldValue(ops[0].getDefiningOp()) - recursiveFoldValue(ops[1].getDefiningOp());
    } else if (mlir::isa<MulOp>(op)) {
      return recursiveFoldValue(ops[0].getDefiningOp()) * recursiveFoldValue(ops[1].getDefiningOp());
    } else if (mlir::isa<DivOp>(op)) {
      return recursiveFoldValue(ops[0].getDefiningOp()) / recursiveFoldValue(ops[1].getDefiningOp());
    } else if (mlir::isa<NegateOp>(op)) {
      return -recursiveFoldValue(op->getOperands()[0].getDefiningOp());
    } else {
      op->dump();
      llvm_unreachable("Unknown operation reached");
    }
  }
}

#endif//MARCO_FOLDINGUTILS_H
