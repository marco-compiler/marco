#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace ::mlir;
using namespace ::mlir::modelica;

#include "marco/Dialect/Modelica/ModelicaDialect.cpp.inc"

namespace
{
  struct ModelicaOpAsmDialectInterface : public OpAsmDialectInterface
  {
    ModelicaOpAsmDialectInterface(Dialect *dialect)
        : OpAsmDialectInterface(dialect)
    {
    }

    AliasResult getAlias(Attribute attr, raw_ostream &os) const override
    {
      if (attr.isa<EquationPathAttr>()) {
        os << "equation_path";
        return AliasResult::OverridableAlias;
      }

      return AliasResult::NoAlias;
    }

    AliasResult getAlias(Type type, raw_ostream &os) const final
    {
      return AliasResult::NoAlias;
    }
  };

  struct ModelicaFoldInterface : public mlir::DialectFoldInterface
  {
    using DialectFoldInterface::DialectFoldInterface;

    bool shouldMaterializeInto(Region *region) const final
    {
      return mlir::isa<
          AlgorithmOp,
          BindingEquationOp,
          DefaultOp,
          EquationOp,
          EquationTemplateOp,
          InitialAlgorithmOp,
          InitialEquationOp,
          StartOp,
          VariableOp>(region->getParentOp());
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
        mlir::IRMapping& valueMapping) const final
    {
      return true;
    }

    bool isLegalToInline(
        mlir::Region* dest,
        mlir::Region* src,
        bool wouldBeCloned,
        mlir::IRMapping& valueMapping) const final
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
        //ModelicaOpAsmDialectInterface,
        ModelicaFoldInterface,
        ModelicaInlinerInterface>();
  }

  Operation* ModelicaDialect::materializeConstant(
      mlir::OpBuilder& builder, mlir::Attribute value, mlir::Type type, mlir::Location loc)
  {
    return builder.create<ConstantOp>(loc, type, value);
  }
}

namespace mlir::modelica
{
  mlir::Type getMostGenericType(mlir::Value x, mlir::Value y)
  {
    assert(x != nullptr && y != nullptr);
    return getMostGenericType(x.getType(), y.getType());
  }

  mlir::Type getMostGenericType(mlir::Type x, mlir::Type y)
  {
    assert((x.isa<BooleanType, IntegerType, RealType, mlir::IndexType>()));
    assert((y.isa<BooleanType, IntegerType, RealType, mlir::IndexType>()));

    if (x.isa<BooleanType>()) {
      return y;
    }

    if (y.isa<BooleanType>()) {
      return x;
    }

    if (x.isa<RealType>()) {
      return x;
    }

    if (y.isa<RealType>()) {
      return y;
    }

    if (x.isa<IntegerType>()) {
      return y;
    }

    return x;
  }

  mlir::LogicalResult materializeAffineMap(
      mlir::OpBuilder& builder,
      mlir::Location loc,
      mlir::AffineMap affineMap,
      mlir::ValueRange dimensions,
      llvm::SmallVectorImpl<mlir::Value>& results)
  {
    for (size_t i = 0, e = affineMap.getNumResults(); i < e; ++i) {
      mlir::Value result = materializeAffineExpr(
          builder, loc, affineMap.getResult(i), dimensions);

      if (!result) {
        return mlir::failure();
      }

      results.push_back(result);
    }

    return mlir::success();
  }

  mlir::Value materializeAffineExpr(
      mlir::OpBuilder& builder,
      mlir::Location loc,
      mlir::AffineExpr expression,
      mlir::ValueRange dimensions)
  {
    if (auto constantExpr = expression.dyn_cast<mlir::AffineConstantExpr>()) {
      return builder.create<ConstantOp>(
          loc, builder.getIndexAttr(constantExpr.getValue()));
    }

    if (auto dimExpr = expression.dyn_cast<mlir::AffineDimExpr>()) {
      assert(dimExpr.getPosition() < dimensions.size());
      return dimensions[dimExpr.getPosition()];
    }

    if (auto binaryExpr = expression.dyn_cast<mlir::AffineBinaryOpExpr>()) {
      if (binaryExpr.getKind() == mlir::AffineExprKind::Add) {
        mlir::Value lhs = materializeAffineExpr(
            builder, loc, binaryExpr.getLHS(), dimensions);

        mlir::Value rhs = materializeAffineExpr(
            builder, loc, binaryExpr.getRHS(), dimensions);

        if (!lhs || !rhs) {
          return nullptr;
        }

        return builder.create<AddOp>(loc, builder.getIndexType(), lhs, rhs);
      }

      if (binaryExpr.getKind() == mlir::AffineExprKind::Mul) {
        mlir::Value lhs = materializeAffineExpr(
            builder, loc, binaryExpr.getLHS(), dimensions);

        mlir::Value rhs = materializeAffineExpr(
            builder, loc, binaryExpr.getRHS(), dimensions);

        if (!lhs || !rhs) {
          return nullptr;
        }

        return builder.create<MulOp>(loc, builder.getIndexType(), lhs, rhs);
      }

      if (binaryExpr.getKind() == mlir::AffineExprKind::FloorDiv) {
        mlir::Value lhs = materializeAffineExpr(
            builder, loc, binaryExpr.getLHS(), dimensions);

        mlir::Value rhs = materializeAffineExpr(
            builder, loc, binaryExpr.getRHS(), dimensions);

        if (!lhs || !rhs) {
          return nullptr;
        }

        return builder.create<DivOp>(loc, builder.getIndexType(), lhs, rhs);
      }
    }

    return nullptr;
  }
}
