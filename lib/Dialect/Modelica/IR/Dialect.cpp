#include "marco/Dialect/Modelica/IR/Dialect.h"
#include "mlir/Interfaces/FoldInterfaces.h"

using namespace ::mlir::modelica;

#include "marco/Dialect/Modelica/IR/Dialect.cpp.inc"

namespace {
/// Interface keeping the constants materialized by the folder within the
/// regions owning the expressions using them. Otherwise, constants would be
/// hoisted out of equations, statements and modifications, and even out of
/// the classes, up to the closest region isolated from above.
struct ModelicaFoldInterface : public mlir::DialectFoldInterface {
  using mlir::DialectFoldInterface::DialectFoldInterface;

  bool shouldMaterializeInto(mlir::Region *region) const final {
    mlir::Operation *op = region->getParentOp();

    // Regions containing elements, equations or statements.
    if (mlir::isa<ClassOp, EnumerationOp, EquationSectionOp,
                  AlgorithmSectionOp, ModelOp, FunctionOp, InstanceOp>(op)) {
      return false;
    }

    if (auto forEquationOp = mlir::dyn_cast<ForEquationOp>(op)) {
      return region == &forEquationOp.getRangeRegion();
    }

    if (auto forOp = mlir::dyn_cast<ForOp>(op)) {
      return region == &forOp.getRangeRegion();
    }

    if (auto ifEquationOp = mlir::dyn_cast<IfEquationOp>(op)) {
      return region == &ifEquationOp.getConditionRegion();
    }

    if (auto ifOp = mlir::dyn_cast<IfOp>(op)) {
      return region == &ifOp.getConditionRegion();
    }

    if (auto whenEquationOp = mlir::dyn_cast<WhenEquationOp>(op)) {
      return region == &whenEquationOp.getConditionRegion();
    }

    if (auto whenOp = mlir::dyn_cast<WhenOp>(op)) {
      return region == &whenOp.getConditionRegion();
    }

    if (auto whileOp = mlir::dyn_cast<WhileOp>(op)) {
      return region == &whileOp.getConditionRegion();
    }

    // Regions computing expressions.
    return true;
  }
};
} // namespace

namespace mlir::modelica {
//===-------------------------------------------------------------------===//
// Modelica dialect
//===-------------------------------------------------------------------===//

void ModelicaDialect::initialize() {
  registerTypes();
  registerAttributes();
  registerOperations();

  addInterfaces<ModelicaFoldInterface>();
}

mlir::Operation *ModelicaDialect::materializeConstant(mlir::OpBuilder &builder,
                                                      mlir::Attribute value,
                                                      mlir::Type type,
                                                      mlir::Location loc) {
  if (!mlir::isa<BooleanAttr, IntegerAttr, RealAttr, StringValueAttr,
                 EnumLiteralAttr, ArrayValueAttr>(value)) {
    return nullptr;
  }

  auto typedAttr = mlir::cast<mlir::TypedAttr>(value);

  if (typedAttr.getType() != type) {
    return nullptr;
  }

  return builder.create<ConstantOp>(loc, type, typedAttr);
}
} // namespace mlir::modelica
