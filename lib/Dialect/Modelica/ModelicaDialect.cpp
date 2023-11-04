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
          ReductionOp,
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
      mlir::OpBuilder& builder,
      mlir::Attribute value,
      mlir::Type type,
      mlir::Location loc)
  {
    return builder.create<ConstantOp>(loc, type, value.cast<mlir::TypedAttr>());
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

  bool isScalar(mlir::Type type)
  {
    if (!type) {
      return false;
    }

    return type.isa<
        BooleanType, IntegerType, RealType,
        mlir::IndexType, mlir::IntegerType, mlir::FloatType>();
  }

  bool isScalar(mlir::Attribute attribute)
  {
    if (!attribute) {
      return false;
    }

    if (auto typedAttr = attribute.dyn_cast<mlir::TypedAttr>()) {
      return isScalar(typedAttr.getType());
    }

    return false;
  }

  bool isScalarIntegerLike(mlir::Type type)
  {
    if (!isScalar(type)) {
      return false;
    }

    return type.isa<
        BooleanType, IntegerType,
        mlir::IndexType, mlir::IntegerType>();
  }

  bool isScalarIntegerLike(mlir::Attribute attribute)
  {
    if (!attribute) {
      return false;
    }

    if (auto typedAttr = attribute.dyn_cast<mlir::TypedAttr>()) {
      return isScalarIntegerLike(typedAttr.getType());
    }

    return false;
  }

  bool isScalarFloatLike(mlir::Type type)
  {
    if (!isScalar(type)) {
      return false;
    }

    return type.isa<RealType, mlir::FloatType>();
  }

  bool isScalarFloatLike(mlir::Attribute attribute)
  {
    if (!attribute) {
      return false;
    }

    if (auto typedAttr = attribute.dyn_cast<mlir::TypedAttr>()) {
      return isScalarFloatLike(typedAttr.getType());
    }

    return false;
  }

  int64_t getScalarIntegerLikeValue(mlir::Attribute attribute)
  {
    assert(isScalarIntegerLike(attribute));

    if (auto booleanAttr = attribute.dyn_cast<BooleanAttr>()) {
      return booleanAttr.getValue();
    }

    if (auto integerAttr = attribute.dyn_cast<IntegerAttr>()) {
      return integerAttr.getValue().getSExtValue();
    }

    return attribute.dyn_cast<mlir::IntegerAttr>().getValue().getSExtValue();
  }

  double getScalarFloatLikeValue(mlir::Attribute attribute)
  {
    assert(isScalarFloatLike(attribute));

    if (auto realAttr = attribute.dyn_cast<RealAttr>()) {
      return realAttr.getValue().convertToDouble();
    }

    return attribute.dyn_cast<mlir::FloatAttr>().getValueAsDouble();
  }

  int64_t getIntegerFromAttribute(mlir::Attribute attribute)
  {
    if (isScalarIntegerLike(attribute)) {
      return getScalarIntegerLikeValue(attribute);
    }

    if (isScalarFloatLike(attribute)) {
      return static_cast<int64_t>(getScalarFloatLikeValue(attribute));
    }

    llvm_unreachable("Unknown attribute type");
    return 0;
  }

  std::unique_ptr<DimensionAccess> getDimensionAccess(
      const llvm::DenseMap<mlir::Value, unsigned int>& explicitInductionsPositionMap,
      const AdditionalInductions& additionalInductions,
      mlir::Value value)
  {
    if (auto definingOp = value.getDefiningOp()) {
      if (auto op = mlir::dyn_cast<ConstantOp>(definingOp)) {
        auto attr = mlir::cast<mlir::Attribute>(op.getValue());

        if (auto rangeAttr = attr.dyn_cast<IntegerRangeAttr>()) {
          assert(rangeAttr.getStep() == 1);

          auto lowerBound = static_cast<Range::data_type>(
              rangeAttr.getLowerBound());

          auto upperBound = static_cast<Range::data_type>(
              rangeAttr.getUpperBound());

          return std::make_unique<DimensionAccessRange>(
              value.getContext(), Range(lowerBound, upperBound));
        }

        if (auto rangeAttr = attr.dyn_cast<RealRangeAttr>()) {
          assert(rangeAttr.getStep().convertToDouble() == 1);

          auto lowerBound = static_cast<Range::data_type>(
              rangeAttr.getLowerBound().convertToDouble());

          auto upperBound = static_cast<Range::data_type>(
              rangeAttr.getUpperBound().convertToDouble());

          return std::make_unique<DimensionAccessRange>(
              value.getContext(), Range(lowerBound, upperBound));
        }

        return std::make_unique<DimensionAccessConstant>(
            value.getContext(), getIntegerFromAttribute(attr));
      }

      if (auto op = mlir::dyn_cast<AddOp>(definingOp)) {
        auto lhs = getDimensionAccess(
            explicitInductionsPositionMap, additionalInductions, op.getLhs());

        auto rhs = getDimensionAccess(
            explicitInductionsPositionMap, additionalInductions, op.getRhs());

        if (!lhs || !rhs) {
          return nullptr;
        }

        return *lhs + *rhs;
      }

      if (auto op = mlir::dyn_cast<SubOp>(definingOp)) {
        auto lhs = getDimensionAccess(
            explicitInductionsPositionMap, additionalInductions, op.getLhs());

        auto rhs = getDimensionAccess(
            explicitInductionsPositionMap, additionalInductions, op.getRhs());

        if (!lhs || !rhs) {
          return nullptr;
        }

        return *lhs - *rhs;
      }

      if (auto op = mlir::dyn_cast<MulOp>(definingOp)) {
        auto lhs = getDimensionAccess(
            explicitInductionsPositionMap, additionalInductions, op.getLhs());

        auto rhs = getDimensionAccess(
            explicitInductionsPositionMap, additionalInductions, op.getRhs());

        if (!lhs || !rhs) {
          return nullptr;
        }

        return *lhs * *rhs;
      }

      if (auto op = mlir::dyn_cast<DivOp>(definingOp)) {
        auto lhs = getDimensionAccess(
            explicitInductionsPositionMap, additionalInductions, op.getLhs());

        auto rhs = getDimensionAccess(
            explicitInductionsPositionMap, additionalInductions, op.getRhs());

        if (!lhs || !rhs) {
          return nullptr;
        }

        return *lhs / *rhs;
      }
    }

    if (auto it = explicitInductionsPositionMap.find(value);
        it != explicitInductionsPositionMap.end()) {
      return std::make_unique<DimensionAccessDimension>(
          value.getContext(), it->getSecond());
    }

    if (additionalInductions.hasInductionVariable(value)) {
      return std::make_unique<DimensionAccessIndices>(
          value.getContext(),
          std::make_shared<IndexSet>(
              additionalInductions.getInductionSpace(value)),
          additionalInductions.getInductionDimension(value),
          additionalInductions.getInductionDependencies(value));
    }

    return nullptr;
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
