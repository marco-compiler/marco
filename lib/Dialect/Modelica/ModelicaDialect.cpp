#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/Modelica/Ops.h"
#include "marco/Dialect/Modelica/Attributes.h"
#include "marco/Dialect/Modelica/Types.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace ::mlir;
using namespace ::mlir::modelica;

#include "marco/Dialect/Modelica/ModelicaDialect.cpp.inc"

namespace
{
  struct ModelicaFoldInterface : public mlir::DialectFoldInterface
  {
    using DialectFoldInterface::DialectFoldInterface;

    bool shouldMaterializeInto(Region *region) const final
    {
      return mlir::isa<
          EquationOp,
          AlgorithmOp>(region->getParentOp());
    }
  };

  /// This class defines the interface for handling inlining with Modelica operations.
  struct ModelicaInlinerInterface : public mlir::DialectInlinerInterface
  {
    using mlir::DialectInlinerInterface::DialectInlinerInterface;

    bool isLegalToInline(
        mlir::Operation* call,
        mlir::Operation* callable,
        bool wouldBeCloned) const final
    {
      if (auto rawFunctionOp = mlir::dyn_cast<RawFunctionOp>(callable)) {
        return rawFunctionOp.shouldBeInlined();
      }

      return true;
    }

    bool isLegalToInline(
        mlir::Operation* op,
        mlir::Region* dest,
        bool wouldBeCloned,
        mlir::BlockAndValueMapping& valueMapping) const final
    {
      return true;
    }

    bool isLegalToInline(
        mlir::Region* dest,
        mlir::Region* src,
        bool wouldBeCloned,
        mlir::BlockAndValueMapping& valueMapping) const final
    {
      return true;
    }

    void handleTerminator(
        mlir::Operation* op,
        llvm::ArrayRef<mlir::Value> valuesToReplace) const final
    {
      // Only "modelica.raw_return" needs to be handled here.
      auto returnOp = cast<RawReturnOp>(op);

      // Replace the values directly with the return operands.
      assert(returnOp.getNumOperands() == valuesToReplace.size());

      for (const auto& operand : llvm::enumerate(returnOp.getOperands())) {
        valuesToReplace[operand.index()].replaceAllUsesWith(operand.value());
      }
    }
  };
}

namespace mlir::modelica
{
  //===----------------------------------------------------------------------===//
  // Modelica dialect
  //===----------------------------------------------------------------------===//

  void ModelicaDialect::initialize()
  {
    registerTypes();
    registerAttributes();

    addOperations<
      #define GET_OP_LIST
      #include "marco/Dialect/Modelica/Modelica.cpp.inc"
        >();

    addInterfaces<
        ModelicaFoldInterface,
        ModelicaInlinerInterface>();
  }

  Operation* ModelicaDialect::materializeConstant(
      mlir::OpBuilder& builder, mlir::Attribute value, mlir::Type type, mlir::Location loc)
  {
    return builder.create<ConstantOp>(loc, type, value);
  }
}
