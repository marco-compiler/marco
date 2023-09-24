#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/Modelica/Ops.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/FoldUtils.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/ScopeExit.h"
#include <cmath>
#include <stack>

using namespace ::mlir::modelica;

static mlir::Value readValue(mlir::OpBuilder& builder, mlir::Value operand)
{
  if (auto arrayType = operand.getType().dyn_cast<ArrayType>();
      arrayType && arrayType.getRank() == 0) {
    return builder.create<LoadOp>(operand.getLoc(), operand);
  }

  return operand;
}

static mlir::Type convertToRealType(mlir::Type type)
{
  if (auto arrayType = type.dyn_cast<ArrayType>()) {
    return arrayType.toElementType(RealType::get(type.getContext()));
  }

  return RealType::get(type.getContext());
}

static bool isScalar(mlir::Type type)
{
  if (!type) {
    return false;
  }

  return type.isa<
      BooleanType, IntegerType, RealType,
      mlir::IndexType, mlir::IntegerType, mlir::FloatType>();
}

static bool isScalar(mlir::Attribute attribute)
{
  if (!attribute) {
    return false;
  }

  if (auto typedAttr = attribute.dyn_cast<mlir::TypedAttr>()) {
    return isScalar(typedAttr.getType());
  }

  return false;
}

static bool isScalarIntegerLike(mlir::Type type)
{
  if (!isScalar(type)) {
    return false;
  }

  return type.isa<
      BooleanType, IntegerType,
      mlir::IndexType, mlir::IntegerType>();
}

static bool isScalarIntegerLike(mlir::Attribute attribute)
{
  if (!attribute) {
    return false;
  }

  if (auto typedAttr = attribute.dyn_cast<mlir::TypedAttr>()) {
    return isScalarIntegerLike(typedAttr.getType());
  }

  return false;
}

static bool isScalarFloatLike(mlir::Type type)
{
  if (!isScalar(type)) {
    return false;
  }

  return type.isa<RealType, mlir::FloatType>();
}

static bool isScalarFloatLike(mlir::Attribute attribute)
{
  if (!attribute) {
    return false;
  }

  if (auto typedAttr = attribute.dyn_cast<mlir::TypedAttr>()) {
    return isScalarFloatLike(typedAttr.getType());
  }

  return false;
}

static int64_t getScalarIntegerLikeValue(mlir::Attribute attribute)
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

static double getScalarFloatLikeValue(mlir::Attribute attribute)
{
  assert(isScalarFloatLike(attribute));

  if (auto realAttr = attribute.dyn_cast<RealAttr>()) {
    return realAttr.getValue().convertToDouble();
  }

  return attribute.dyn_cast<mlir::FloatAttr>().getValueAsDouble();
}

namespace
{
  template<typename T>
  std::optional<T> getScalarAttributeValue(mlir::Attribute attribute)
  {
    if (isScalarIntegerLike(attribute)) {
      return static_cast<T>(getScalarIntegerLikeValue(attribute));
    } else if (isScalarFloatLike(attribute)) {
      return static_cast<T>(getScalarFloatLikeValue(attribute));
    } else {
      return std::nullopt;
    }
  }

  template<typename T>
  bool getScalarAttributesValues(
      llvm::ArrayRef<mlir::Attribute> attributes,
      llvm::SmallVectorImpl<T>& result)
  {
    for (mlir::Attribute attribute : attributes) {
      if (auto value = getScalarAttributeValue<T>(attribute)) {
        result.push_back(*value);
      } else {
        return false;
      }
    }

    return true;
  }
}

static int64_t getIntegerFromAttribute(mlir::Attribute attribute)
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

static mlir::SymbolRefAttr getSymbolRefFromRoot(mlir::Operation* symbol)
{
  llvm::SmallVector<mlir::FlatSymbolRefAttr> flatSymbolAttrs;

  flatSymbolAttrs.push_back(mlir::FlatSymbolRefAttr::get(
      symbol->getContext(),
      mlir::cast<mlir::SymbolOpInterface>(symbol).getName()));

  mlir::Operation* parent = symbol->getParentOp();

  while (parent != nullptr) {
    if (auto classInterface = mlir::dyn_cast<ClassInterface>(parent)) {
      flatSymbolAttrs.push_back(mlir::FlatSymbolRefAttr::get(
          symbol->getContext(),
          mlir::cast<mlir::SymbolOpInterface>(
              classInterface.getOperation()).getName()));
    }

    parent = parent->getParentOp();
  }

  std::reverse(flatSymbolAttrs.begin(), flatSymbolAttrs.end());

  return mlir::SymbolRefAttr::get(
      symbol->getContext(),
      flatSymbolAttrs[0].getValue(),
      llvm::ArrayRef(flatSymbolAttrs).drop_front());
}

static mlir::Operation* resolveSymbol(
    mlir::ModuleOp moduleOp,
    mlir::SymbolTableCollection& symbolTable,
    mlir::SymbolRefAttr symbol)
{
  mlir::Operation* result =
      symbolTable.lookupSymbolIn(moduleOp, symbol.getRootReference());

  for (mlir::FlatSymbolRefAttr nestedRef : symbol.getNestedReferences()) {
    if (result == nullptr) {
      return nullptr;
    }

    result = symbolTable.lookupSymbolIn(result, nestedRef.getAttr());
  }

  return result;
}

#define GET_OP_CLASSES
#include "marco/Dialect/Modelica/Modelica.cpp.inc"

//===---------------------------------------------------------------------===//
// Iteration space operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// RangeOp

namespace mlir::modelica
{
  mlir::OpFoldResult RangeOp::fold(FoldAdaptor adaptor)
  {
    auto lowerBound = adaptor.getLowerBound();
    auto upperBound = adaptor.getUpperBound();
    auto step = adaptor.getStep();

    if (!lowerBound || !upperBound || !step) {
      return {};
    }

    if (isScalarIntegerLike(lowerBound) &&
        isScalarIntegerLike(upperBound) &&
        isScalarIntegerLike(step)) {
      int64_t lowerBoundValue = getScalarIntegerLikeValue(lowerBound);
      int64_t upperBoundValue = getScalarIntegerLikeValue(upperBound);
      int64_t stepValue = getScalarIntegerLikeValue(step);

      return IntegerRangeAttr::get(
          getContext(), getResult().getType(),
          lowerBoundValue, upperBoundValue, stepValue);
    }

    return {};
  }
}

//===---------------------------------------------------------------------===//
// Array operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// AllocaOp

namespace mlir::modelica
{
  mlir::LogicalResult AllocaOp::verify()
  {
    int64_t dynamicDimensionsAmount = getArrayType().getNumDynamicDims();
    size_t valuesAmount = getDynamicSizes().size();

    if (dynamicDimensionsAmount != static_cast<int64_t>(valuesAmount)) {
      return emitOpError(
          "incorrect number of values for dynamic dimensions (expected " +
          std::to_string(dynamicDimensionsAmount) + ", got " +
          std::to_string(valuesAmount) + ")");
    }

    return mlir::success();
  }

  void AllocaOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (auto arrayType = getResult().getType().dyn_cast<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::AutomaticAllocationScopeResource::get());
    }
  }

  mlir::ValueRange AllocaOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    return builder.clone(*getOperation())->getResults();
  }

  void AllocaOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    // The dimensions must not be derived.
  }

  void AllocaOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// AllocOp

namespace mlir::modelica
{
  mlir::LogicalResult AllocOp::verify()
  {
    int64_t dynamicDimensionsAmount = getArrayType().getNumDynamicDims();
    size_t valuesAmount = getDynamicSizes().size();

    if (dynamicDimensionsAmount != static_cast<int64_t>(valuesAmount)) {
      return emitOpError(
          "incorrect number of values for dynamic dimensions (expected " +
          std::to_string(dynamicDimensionsAmount) + ", got " +
          std::to_string(valuesAmount) + ")");
    }

    return mlir::success();
  }

  void AllocOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (auto arrayType = getResult().getType().dyn_cast<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange AllocOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    return builder.clone(*getOperation())->getResults();
  }

  void AllocOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    // The dimensions must not be derived.
  }

  void AllocOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// ArrayFromElementsOp

namespace mlir::modelica
{
  mlir::LogicalResult ArrayFromElementsOp::verify()
  {
    if (!getArrayType().hasStaticShape()) {
      return emitOpError("the shape must be fixed");
    }

    int64_t arrayFlatSize = getArrayType().getNumElements();
    size_t numOfValues = getValues().size();

    if (arrayFlatSize != static_cast<int64_t>(numOfValues)) {
      return emitOpError(
          "incorrect number of values (expected " +
          std::to_string(arrayFlatSize) + ", got " +
          std::to_string(numOfValues) + ")");
    }

    return mlir::success();
  }

  mlir::OpFoldResult ArrayFromElementsOp::fold(FoldAdaptor adaptor)
  {
    if (llvm::all_of(adaptor.getOperands(), [](mlir::Attribute attr) {
          return attr != nullptr;
        })) {
      ArrayType arrayType = getArrayType();

      if (!arrayType.hasStaticShape()) {
        return {};
      }

      mlir::Type elementType = arrayType.getElementType();

      if (elementType.isa<BooleanType>()) {
        llvm::SmallVector<bool> casted;

        if (!getScalarAttributesValues(adaptor.getOperands(), casted)) {
          return {};
        }

        return BooleanArrayAttr::get(arrayType, casted);
      }

      if (elementType.isa<IntegerType>()) {
        llvm::SmallVector<int64_t> casted;

        if (!getScalarAttributesValues(adaptor.getOperands(), casted)) {
          return {};
        }

        return IntegerArrayAttr::get(arrayType, casted);
      }

      if (elementType.isa<RealType>()) {
        llvm::SmallVector<double> casted;

        if (!getScalarAttributesValues(adaptor.getOperands(), casted)) {
          return {};
        }

        return RealArrayAttr::get(arrayType, casted);
      }
    }

    return {};
  }

  void ArrayFromElementsOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(
        mlir::MemoryEffects::Allocate::get(),
        getResult(),
        mlir::SideEffects::DefaultResource::get());

    effects.emplace_back(
        mlir::MemoryEffects::Write::get(),
        getResult(),
        mlir::SideEffects::DefaultResource::get());
  }

  mlir::ValueRange ArrayFromElementsOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    llvm::SmallVector<mlir::Value> derivedValues;

    for (mlir::Value value : getValues()) {
      derivedValues.push_back(derivatives.lookup(value));
    }

    auto derivedOp = builder.create<ArrayFromElementsOp>(
        getLoc(),
        getArrayType().toElementType(RealType::get(builder.getContext())),
        derivedValues);

    return derivedOp->getResults();
  }

  void ArrayFromElementsOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    for (mlir::Value value : getValues()) {
      toBeDerived.push_back(value);
    }
  }

  void ArrayFromElementsOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// ArrayBroadcastOp

namespace mlir::modelica
{
  void ArrayBroadcastOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(
        mlir::MemoryEffects::Allocate::get(),
        getResult(),
        mlir::SideEffects::DefaultResource::get());

    effects.emplace_back(
        mlir::MemoryEffects::Write::get(),
        getResult(),
        mlir::SideEffects::DefaultResource::get());
  }

  mlir::ValueRange ArrayBroadcastOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    mlir::Value derivedValue = derivatives.lookup(getValue());

    auto derivedOp = builder.create<ArrayFromElementsOp>(
        getLoc(),
        getArrayType().toElementType(RealType::get(builder.getContext())),
        derivedValue);

    return derivedOp->getResults();
  }

  void ArrayBroadcastOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getValue());
  }

  void ArrayBroadcastOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// FreeOp

namespace mlir::modelica
{
  void FreeOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(
        mlir::MemoryEffects::Free::get(),
        getArray(),
        mlir::SideEffects::DefaultResource::get());
  }
}

//===---------------------------------------------------------------------===//
// DimOp

namespace
{
  struct DimOpStaticDimensionPattern
      : public mlir::OpRewritePattern<DimOp>
  {
    using mlir::OpRewritePattern<DimOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        DimOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto constantOp = op.getDimension().getDefiningOp<ConstantOp>();

      if (!constantOp) {
        return mlir::failure();
      }

      ArrayType arrayType = op.getArray().getType();

      int64_t dimSize = arrayType.getDimSize(
          constantOp.getValue().cast<mlir::IntegerAttr>().getInt());

      if (dimSize == ArrayType::kDynamic) {
        return mlir::failure();
      }

      rewriter.replaceOpWithNewOp<ConstantOp>(
          op, rewriter.getIndexAttr(dimSize));

      return mlir::success();
    }
  };
}

namespace mlir::modelica
{
  void DimOp::getCanonicalizationPatterns(
      mlir::RewritePatternSet& patterns, mlir::MLIRContext* context)
  {
    patterns.add<DimOpStaticDimensionPattern>(context);
  }
}

//===---------------------------------------------------------------------===//
// LoadOp

namespace
{
  struct MergeSubscriptionsIntoLoadPattern
      : public mlir::OpRewritePattern<LoadOp>
  {
    using mlir::OpRewritePattern<LoadOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        LoadOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto subscriptionOp = op.getArray().getDefiningOp<SubscriptionOp>();

      if (!subscriptionOp) {
        return mlir::failure();
      }

      std::stack<SubscriptionOp> subscriptionOps;

      while (subscriptionOp) {
        subscriptionOps.push(subscriptionOp);

        subscriptionOp =
            subscriptionOp.getSource().getDefiningOp<SubscriptionOp>();
      }

      assert(!subscriptionOps.empty());
      mlir::Value source = subscriptionOps.top().getSource();
      llvm::SmallVector<mlir::Value, 3> indices;

      while (!subscriptionOps.empty()) {
        SubscriptionOp current = subscriptionOps.top();
        indices.append(current.getIndices().begin(),
                       current.getIndices().end());
        subscriptionOps.pop();
      }

      indices.append(op.getIndices().begin(), op.getIndices().end());
      rewriter.replaceOpWithNewOp<LoadOp>(op, source, indices);
      return mlir::success();
    }
  };
}

namespace mlir::modelica
{
  mlir::ParseResult LoadOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    auto loc = parser.getCurrentLocation();
    mlir::OpAsmParser::UnresolvedOperand array;
    mlir::Type arrayType;
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 3> indices;
    llvm::SmallVector<mlir::Type, 3> indicesTypes;

    if (parser.parseOperand(array) ||
        parser.parseOperandList(
            indices, mlir::OpAsmParser::Delimiter::Square) ||
        parser.parseColonType(arrayType) ||
        parser.resolveOperand(array, arrayType, result.operands)) {
      return mlir::failure();
    }

    indicesTypes.resize(
        indices.size(),
        mlir::IndexType::get(result.getContext()));

    size_t i = 0;

    while (mlir::succeeded(parser.parseOptionalComma())) {
      if (parser.parseType(indicesTypes[i++])) {
        return mlir::failure();
      }
    }

    if (parser.resolveOperands(indices, indicesTypes, loc, result.operands)) {
      return mlir::failure();
    }

    result.addTypes(arrayType.cast<ArrayType>().getElementType());
    return mlir::success();
  }

  void LoadOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " " << getArray() << "[" << getIndices() << "]";
    printer.printOptionalAttrDict(getOperation()->getAttrs());
    printer << " : " << getArray().getType();

    if (!llvm::all_of(getIndices(), [](mlir::Value index) {
          return index.getType().isa<mlir::IndexType>();
        })) {
      for (mlir::Value index : getIndices()) {
        printer << ", " << index.getType();
      }
    }
  }

  mlir::LogicalResult LoadOp::verify()
  {
    size_t indicesAmount = getIndices().size();
    int64_t rank = getArrayType().getRank();

    if (rank != static_cast<int64_t>(indicesAmount)) {
      return emitOpError()
          << "incorrect number of indices for store (expected " << rank
          << ", got " << indicesAmount << ")";
    }

    for (size_t i = 0; i < indicesAmount; ++i) {
      if (auto constantOp = getIndices()[i].getDefiningOp<ConstantOp>()) {
        if (auto index = getScalarAttributeValue<int64_t>(
                constantOp.getValue())) {
          if (*index < 0) {
            return emitOpError() << "invalid index (" << *index << ")";
          }

          if (int64_t dimSize = getArrayType().getDimSize(i);
              *index >= dimSize) {
            return emitOpError()
                << "out of bounds access (index = " << *index
                << ", dimension = " << dimSize << ")";
          }
        }
      }
    }

    return mlir::success();
  }

  void LoadOp::getCanonicalizationPatterns(
      mlir::RewritePatternSet& patterns, mlir::MLIRContext* context)
  {
    patterns.add<MergeSubscriptionsIntoLoadPattern>(context);
  }

  void LoadOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(
        mlir::MemoryEffects::Read::get(),
        getArray(),
        mlir::SideEffects::DefaultResource::get());
  }

  mlir::ValueRange LoadOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    auto derivedOp = builder.create<LoadOp>(
        getLoc(), derivatives.lookup(getArray()), getIndices());

    return derivedOp->getResults();
  }

  void LoadOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getArray());
  }

  void LoadOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// StoreOp

namespace mlir::modelica
{
  mlir::ParseResult StoreOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    auto loc = parser.getCurrentLocation();
    mlir::OpAsmParser::UnresolvedOperand array;
    mlir::Type arrayType;
    mlir::OpAsmParser::UnresolvedOperand value;
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 3> indices;
    llvm::SmallVector<mlir::Type, 3> indicesTypes;

    if (parser.parseOperand(array) ||
        parser.parseOperandList(
            indices, mlir::OpAsmParser::Delimiter::Square) ||
        parser.parseComma() ||
        parser.parseOperand(value) ||
        parser.parseColonType(arrayType) ||
        parser.resolveOperand(value, arrayType.cast<ArrayType>().getElementType(), result.operands) ||
        parser.resolveOperand(array, arrayType, result.operands)) {
      return mlir::failure();
    }

    indicesTypes.resize(
        indices.size(),
        mlir::IndexType::get(result.getContext()));

    size_t i = 0;

    while (mlir::succeeded(parser.parseOptionalComma())) {
      if (parser.parseType(indicesTypes[i++])) {
        return mlir::failure();
      }
    }

    if (parser.resolveOperands(indices, indicesTypes, loc, result.operands)) {
      return mlir::failure();
    }

    return mlir::success();
  }

  void StoreOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " " << getArray() << "[" << getIndices() << "]"
            << ", " << getValue();

    printer.printOptionalAttrDict(getOperation()->getAttrs());
    printer << " : " << getArray().getType();

    if (!llvm::all_of(getIndices(), [](mlir::Value index) {
          return index.getType().isa<mlir::IndexType>();
        })) {
      for (mlir::Value index : getIndices()) {
        printer << ", " << index.getType();
      }
    }
  }

  mlir::LogicalResult StoreOp::verify()
  {
    size_t indicesAmount = getIndices().size();
    int64_t rank = getArrayType().getRank();

    if (rank != static_cast<int64_t>(indicesAmount)) {
      return emitOpError()
          << "incorrect number of indices for store (expected " << rank
          << ", got " << indicesAmount << ")";
    }

    for (size_t i = 0; i < indicesAmount; ++i) {
      if (auto constantOp = getIndices()[i].getDefiningOp<ConstantOp>()) {
        if (auto index = getScalarAttributeValue<int64_t>(
                constantOp.getValue())) {
          if (*index < 0) {
            return emitOpError() << "invalid index (" << *index << ")";
          }

          if (int64_t dimSize = getArrayType().getDimSize(i);
              *index >= dimSize) {
            return emitOpError()
                << "out of bounds access (index = " << *index
                << ", dimension = " << dimSize << ")";
          }
        }
      }
    }

    return mlir::success();
  }

  void StoreOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(
        mlir::MemoryEffects::Write::get(),
        getArray(),
        mlir::SideEffects::DefaultResource::get());
  }

  mlir::ValueRange StoreOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    auto derivedOp = builder.create<StoreOp>(
        getLoc(),
        derivatives.lookup(getValue()),
        derivatives.lookup(getArray()),
        getIndices());

    return derivedOp->getResults();
  }

  void StoreOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getArray());
    toBeDerived.push_back(getValue());
  }

  void StoreOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// SubscriptionOp

namespace
{
  struct MergeSubscriptionsPattern
      : public mlir::OpRewritePattern<SubscriptionOp>
  {
    using mlir::OpRewritePattern<SubscriptionOp>::OpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        SubscriptionOp op, mlir::PatternRewriter& rewriter) const override
    {
      auto subscriptionOp = op.getSource().getDefiningOp<SubscriptionOp>();

      if (!subscriptionOp) {
        return mlir::failure();
      }

      std::stack<SubscriptionOp> subscriptionOps;

      while (subscriptionOp) {
        subscriptionOps.push(subscriptionOp);

        subscriptionOp =
            subscriptionOp.getSource().getDefiningOp<SubscriptionOp>();
      }

      assert(!subscriptionOps.empty());
      mlir::Value source = subscriptionOps.top().getSource();
      llvm::SmallVector<mlir::Value, 3> indices;

      while (!subscriptionOps.empty()) {
        SubscriptionOp current = subscriptionOps.top();
        indices.append(current.getIndices().begin(),
                       current.getIndices().end());
        subscriptionOps.pop();
      }

      indices.append(op.getIndices().begin(), op.getIndices().end());
      rewriter.replaceOpWithNewOp<SubscriptionOp>(op, source, indices);
      return mlir::success();
    }
  };
}

namespace mlir::modelica
{
  void SubscriptionOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      mlir::Value source,
      mlir::ValueRange indices)
  {
    build(builder, state,
          inferResultType(source.getType().cast<ArrayType>(), indices),
          source, indices);
  }

  mlir::LogicalResult SubscriptionOp::verify()
  {
    ArrayType sourceType = getSourceArrayType();
    ArrayType resultType = getResultArrayType();
    ArrayType expectedResultType = inferResultType(sourceType, getIndices());

    if (resultType.getRank() != expectedResultType.getRank()) {
      return emitOpError() << "incompatible result rank";
    }

    for (int64_t i = 0, e = resultType.getRank(); i < e; ++i) {
      int64_t actualDimSize = resultType.getDimSize(i);
      int64_t expectedDimSize = expectedResultType.getDimSize(i);

      if (actualDimSize != ArrayType::kDynamic &&
          actualDimSize != expectedDimSize) {
        return emitOpError() << "incompatible size for dimension " << i
                             << " (expected " << expectedDimSize << ", got "
                             << actualDimSize << ")";
      }
    }

    return mlir::success();
  }

  void SubscriptionOp::getCanonicalizationPatterns(
      mlir::RewritePatternSet& patterns, mlir::MLIRContext* context)
  {
    patterns.add<MergeSubscriptionsPattern>(context);
  }

  mlir::ValueRange SubscriptionOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    auto derivedOp = builder.create<SubscriptionOp>(
        getLoc(), derivatives.lookup(getSource()), getIndices());

    return derivedOp->getResults();
  }

  void SubscriptionOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    // The indices must not be derived.
    toBeDerived.push_back(getSource());
  }

  void SubscriptionOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }

  ArrayType SubscriptionOp::inferResultType(
      ArrayType source, mlir::ValueRange indices)
  {
    llvm::SmallVector<int64_t> shape;
    size_t numOfSubscriptions = indices.size();

    for (size_t i = 0; i < numOfSubscriptions; ++i) {
      mlir::Value index = indices[i];

      if (index.getType().isa<IterableType>()) {
        int64_t dimension = ArrayType::kDynamic;

        if (auto constantOp = index.getDefiningOp<ConstantOp>()) {
          mlir::Attribute indexAttr = constantOp.getValue();

          if (auto rangeAttr = mlir::dyn_cast<RangeAttrInterface>(indexAttr)) {
            dimension = rangeAttr.getNumOfElements();
          }
        }

        shape.push_back(dimension);
      }
    }

    for (int64_t dimension :
         source.getShape().drop_front(numOfSubscriptions)) {
      shape.push_back(dimension);
    }

    return source.withShape(shape);
  }
}

//===---------------------------------------------------------------------===//
// ArrayFillOp

namespace mlir::modelica
{
  void ArrayFillOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(
        mlir::MemoryEffects::Write::get(),
        getArray(),
        mlir::SideEffects::DefaultResource::get());
  }
}

//===---------------------------------------------------------------------===//
// ArrayCopyOp

namespace mlir::modelica
{
  void ArrayCopyOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(
        mlir::MemoryEffects::Read::get(),
        getSource(),
        mlir::SideEffects::DefaultResource::get());

    effects.emplace_back(
        mlir::MemoryEffects::Write::get(),
        getDestination(),
        mlir::SideEffects::DefaultResource::get());
  }
}

//===---------------------------------------------------------------------===//
// Variable operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// VariableOp

namespace mlir::modelica
{
  void VariableOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      llvm::StringRef name,
      VariableType variableType)
  {
    llvm::SmallVector<mlir::Attribute, 3> constraints(
        variableType.getNumDynamicDims(),
        builder.getStringAttr(kDimensionConstraintUnbounded));

    build(builder, state, name, variableType,
          builder.getArrayAttr(constraints));
  }

  mlir::ParseResult VariableOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    auto& builder = parser.getBuilder();

    // Variable name.
    mlir::StringAttr nameAttr;

    if (parser.parseSymbolName(
            nameAttr,
            mlir::SymbolTable::getSymbolAttrName(),
            result.attributes)) {
      return mlir::failure();
    }

    // Attributes.
    if (parser.parseOptionalAttrDict(result.attributes)) {
      return mlir::failure();
    }

    // Variable type.
    mlir::Type type;

    if (parser.parseColonType(type)) {
      return mlir::failure();
    }

    result.attributes.append(
        getTypeAttrName(result.name),
        mlir::TypeAttr::get(type));

    // Dimensions constraints.
    llvm::SmallVector<llvm::StringRef> dimensionsConstraints;

    if (mlir::succeeded(parser.parseOptionalLSquare())) {
      do {
        if (mlir::succeeded(
                parser.parseOptionalKeyword(kDimensionConstraintUnbounded))) {
          dimensionsConstraints.push_back(kDimensionConstraintUnbounded);
        } else {
          if (parser.parseKeyword(kDimensionConstraintFixed)) {
            return mlir::failure();
          }

          dimensionsConstraints.push_back(kDimensionConstraintFixed);
        }
      } while (mlir::succeeded(parser.parseOptionalComma()));

      if (parser.parseRSquare()) {
        return mlir::failure();
      }
    }

    result.attributes.append(
        getDimensionsConstraintsAttrName(result.name),
        builder.getStrArrayAttr(dimensionsConstraints));

    // Region for the dimensions constraints.
    mlir::Region* constraintsRegion = result.addRegion();

    mlir::OptionalParseResult constraintsRegionParseResult =
        parser.parseOptionalRegion(*constraintsRegion);

    if (constraintsRegionParseResult.has_value() &&
        failed(*constraintsRegionParseResult)) {
      return mlir::failure();
    }

    return mlir::success();
  }

  void VariableOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " ";
    printer.printSymbolName(getSymName());

    llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
    elidedAttrs.push_back(mlir::SymbolTable::getSymbolAttrName());
    elidedAttrs.push_back(getTypeAttrName());
    elidedAttrs.push_back(getDimensionsConstraintsAttrName());

    printer.printOptionalAttrDict(getOperation()->getAttrs(), elidedAttrs);

    printer << " : " << getType();

    auto dimConstraints =
        getDimensionsConstraints().getAsRange<mlir::StringAttr>();

    if (llvm::any_of(dimConstraints, [](mlir::StringAttr constraint) {
          return constraint == kDimensionConstraintFixed;
        })) {
      printer << " [";

      for (const auto& constraint : llvm::enumerate(dimConstraints)) {
        if (constraint.index() != 0) {
          printer << ", ";
        }

        printer << constraint.value().getValue();
      }

      printer << "] ";
    }

    if (mlir::Region& region = getConstraintsRegion(); !region.empty()) {
      printer.printRegion(region);
    }
  }

  mlir::LogicalResult VariableOp::verify()
  {
    // Verify the semantics for fixed dimensions constraints.
    size_t numOfFixedDims = getNumOfFixedDimensions();
    mlir::Region& constraintsRegion = getConstraintsRegion();
    size_t numOfConstraints = 0;

    if (!constraintsRegion.empty()) {
      auto yieldOp = mlir::cast<YieldOp>(
          constraintsRegion.back().getTerminator());

      numOfConstraints = yieldOp.getValues().size();
    }

    if (numOfFixedDims != numOfConstraints) {
      return emitOpError(
          "not enough constraints for dynamic dimension constraints have been "
          "provided (expected " + std::to_string(numOfFixedDims) + ", got " +
          std::to_string(numOfConstraints) + ")");
    }

    if (!constraintsRegion.empty()) {
      auto yieldOp = mlir::cast<YieldOp>(
          constraintsRegion.back().getTerminator());

      // Check that the amount of values is the same of the fixed dimensions.
      if (yieldOp.getValues().size() != getNumOfFixedDimensions()) {
        return mlir::failure();
      }

      // Check that all the dimensions have 'index' type.
      if (llvm::any_of(yieldOp.getValues(), [](mlir::Value value) {
            return !value.getType().isa<mlir::IndexType>();
          })) {
        return emitOpError(
            "constraints for dynamic dimensions must have 'index' type");
      }
    }

    return mlir::success();
  }

  size_t VariableOp::getNumOfUnboundedDimensions()
  {
    return llvm::count_if(
        getDimensionsConstraints().getAsRange<mlir::StringAttr>(),
        [](mlir::StringAttr dimensionConstraint) {
          return dimensionConstraint.getValue() ==
              kDimensionConstraintUnbounded;
        });
  }

  size_t VariableOp::getNumOfFixedDimensions()
  {
    return llvm::count_if(
        getDimensionsConstraints().getAsRange<mlir::StringAttr>(),
        [](mlir::StringAttr dimensionConstraint) {
            return dimensionConstraint.getValue() ==
              kDimensionConstraintFixed;
        });
  }

  IndexSet VariableOp::getIndices()
  {
    VariableType variableType = getVariableType();

    if (variableType.isScalar()) {
      return {};
    }

    llvm::SmallVector<Range> ranges;

    for (int64_t dimension : variableType.getShape()) {
      ranges.push_back(Range(0, dimension));
    }

    return IndexSet(MultidimensionalRange(ranges));
  }
}

//===---------------------------------------------------------------------===//
// VariableGetOp

namespace mlir::modelica
{
  void VariableGetOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      VariableOp variableOp)
  {
    auto variableType = variableOp.getVariableType();
    auto variableName = variableOp.getSymName();
    build(builder, state, variableType.unwrap(), variableName);
  }

  mlir::LogicalResult VariableGetOp::verifySymbolUses(
      mlir::SymbolTableCollection& symbolTableCollection)
  {
    auto parentClass = getOperation()->getParentOfType<ClassInterface>();

    if (!parentClass) {
      return emitOpError() << "the operation must be used inside a class";
    }

    mlir::Operation* symbol =
        symbolTableCollection.lookupSymbolIn(parentClass, getVariableAttr());

    if (!symbol) {
      return emitOpError()
          << "variable " << getVariable() << " has not been declared";
    }

    if (!mlir::isa<VariableOp>(symbol)) {
      return emitOpError()
          << "symbol " << getVariable() << " is not a variable";
    }

    auto variableOp = mlir::cast<VariableOp>(symbol);
    mlir::Type unwrappedType = variableOp.getVariableType().unwrap();

    if (unwrappedType != getResult().getType()) {
      return emitOpError() << "result type does not match the variable type";
    }

    return mlir::success();
  }

  mlir::ValueRange VariableGetOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    auto derivativeSymbolIt = symbolDerivatives.find(getVariableAttr());

    if (derivativeSymbolIt == symbolDerivatives.end()) {
      return std::nullopt;
    }

    auto derivedOp = builder.create<VariableGetOp>(
        getLoc(), getResult().getType(),
        derivativeSymbolIt->getSecond().getValue());

    return derivedOp->getResults();
  }

  void VariableGetOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    // No operands to be derived.
  }

  void VariableGetOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// VariableSetOp

namespace mlir::modelica
{
  void VariableSetOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      VariableOp variableOp,
      mlir::Value value)
  {
    auto variableName = variableOp.getSymName();
    build(builder, state, variableName, value);
  }

  mlir::LogicalResult VariableSetOp::verifySymbolUses(
      mlir::SymbolTableCollection& symbolTableCollection)
  {
    auto parentClass = getOperation()->getParentOfType<ClassInterface>();

    if (!parentClass) {
      return emitOpError("the operation must be used inside a class");
    }

    mlir::Operation* symbol =
        symbolTableCollection.lookupSymbolIn(parentClass, getVariableAttr());

    if (!symbol) {
      return emitOpError(
          "variable " + getVariable() + " has not been declared");
    }

    auto variableOp = mlir::dyn_cast<VariableOp>(symbol);

    if (!variableOp) {
      return emitOpError("symbol " + getVariable() + " is not a variable");
    }

    if (variableOp.isInput()) {
      return emitOpError("can't set a value for an input variable");
    }

    return mlir::success();
  }

  mlir::ValueRange VariableSetOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    auto derivativeSymbolIt = symbolDerivatives.find(getVariableAttr());

    if (derivativeSymbolIt == symbolDerivatives.end()) {
      return std::nullopt;
    }

    auto derivedOp = builder.create<VariableSetOp>(
        getLoc(),
        derivativeSymbolIt->getSecond().getValue(),
        derivatives.lookup(getValue()));

    return derivedOp->getResults();
  }

  void VariableSetOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getValue());
  }

  void VariableSetOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// ComponentGetOp

namespace mlir::modelica
{
  mlir::LogicalResult ComponentGetOp::verifySymbolUses(
      mlir::SymbolTableCollection& symbolTableCollection)
  {
    // TODO
    return mlir::success();
  }
}

//===---------------------------------------------------------------------===//
// ComponentSetOp

namespace mlir::modelica
{
  mlir::LogicalResult ComponentSetOp::verifySymbolUses(
      mlir::SymbolTableCollection& symbolTableCollection)
  {
    // TODO
    return mlir::success();
  }
}

//===---------------------------------------------------------------------===//
// GlobalVariableOp

namespace mlir::modelica
{
  void GlobalVariableOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      mlir::StringAttr name,
      mlir::TypeAttr type)
  {
    build(builder, state, name, type, nullptr);
  }

  void GlobalVariableOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      llvm::StringRef name,
      mlir::Type type)
  {
    build(builder, state, name, type, nullptr);
  }
}


//===---------------------------------------------------------------------===//
// GlobalVariableGetOp

namespace mlir::modelica
{
  void GlobalVariableGetOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      GlobalVariableOp globalVariableOp)
  {
    auto type = globalVariableOp.getType();
    auto name = globalVariableOp.getSymName();
    build(builder, state, type, name);
  }

  mlir::LogicalResult GlobalVariableGetOp::verifySymbolUses(
      mlir::SymbolTableCollection& symbolTableCollection)
  {
    auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();

    mlir::Operation* symbol =
        symbolTableCollection.lookupSymbolIn(moduleOp, getVariableAttr());

    if (!symbol) {
      return emitOpError()
          << "global variable " << getVariable() << " has not been declared";
    }

    if (!mlir::isa<GlobalVariableOp>(symbol)) {
      return emitOpError()
          << "symbol " << getVariable() << " is not a global variable";
    }

    auto globalVariableOp = mlir::cast<GlobalVariableOp>(symbol);

    if (globalVariableOp.getType() != getResult().getType()) {
      return emitOpError()
          << "result type does not match the global variable type";
    }

    return mlir::success();
  }
}

//===---------------------------------------------------------------------===//
// Math operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// ConstantOp

namespace mlir::modelica
{
  mlir::OpFoldResult ConstantOp::fold(FoldAdaptor adaptor)
  {
    return getValue();
  }

  mlir::ValueRange ConstantOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    // D[const] = 0

    auto derivedOp = builder.create<ConstantOp>(
        getLoc(), getZeroAttr(getResult().getType()));

    return derivedOp->getResults();
  }

  void ConstantOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    // No operands to be derived.
  }

  void ConstantOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// NegateOp

namespace mlir::modelica
{
  mlir::OpFoldResult NegateOp::fold(FoldAdaptor adaptor)
  {
    auto operand = adaptor.getOperand();

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        return getAttr(resultType, -1 * getScalarIntegerLikeValue(operand));
      }

      if (isScalarFloatLike(operand)) {
        return getAttr(resultType, -1 * getScalarFloatLikeValue(operand));
      }
    }

    return {};
  }

  void NegateOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getOperand(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::Value NegateOp::inverse(
      mlir::OpBuilder& builder,
      unsigned int argumentIndex,
      mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (argumentIndex > 0) {
      emitOpError() << "Index out of bounds: " << argumentIndex << ".";
      return nullptr;
    }

    if (size_t size = currentResult.size(); size != 1) {
      emitOpError() << "Invalid amount of values to be nested: " << size
                    << " (expected 1).";

      return nullptr;
    }

    mlir::Value toNest = currentResult[0];
    mlir::Value nestedOperand = readValue(builder, toNest);

    auto right = builder.create<NegateOp>(
        getLoc(), getOperand().getType(), nestedOperand);

    return right.getResult();
  }

  mlir::LogicalResult NegateOp::distribute(
      llvm::SmallVectorImpl<mlir::Value>& results, mlir::OpBuilder& builder)
  {
    mlir::Value operand = getOperand();
    mlir::Operation* operandOp = operand.getDefiningOp();

    if (operandOp) {
      if (auto negDistributionInt =
              mlir::dyn_cast<NegateOpDistributionInterface>(operandOp)) {
        if (mlir::succeeded(negDistributionInt.distributeNegateOp(
            results, builder, getResult().getType()))) {
          return mlir::success();
        }
      }
    }

    // The operation can't be propagated because the child doesn't know how to
    // distribute the negation to its children.
    results.push_back(getResult());
    return mlir::failure();
  }

  mlir::LogicalResult NegateOp::distributeNegateOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType)
  {
    mlir::Value operand = getOperand();
    bool operandDistributed = false;
    mlir::Operation* operandOp = operand.getDefiningOp();

    if (operandOp) {
      if (auto negDistributionInt =
              mlir::dyn_cast<NegateOpDistributionInterface>(operandOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(negDistributionInt.distributeNegateOp(
                childResults, builder, resultType))
            && childResults.size() == 1) {
          operand = childResults[0];
          operandDistributed = true;
        }
      }
    }

    if (!operandDistributed) {
      auto newOperandOp = builder.create<NegateOp>(
          getLoc(), operand.getType(), operand);

      operand = newOperandOp.getResult();
    }

    auto resultOp = builder.create<NegateOp>(getLoc(), resultType, operand);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::LogicalResult NegateOp::distributeMulOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType,
      mlir::Value value)
  {
    mlir::Value operand = getOperand();
    bool operandDistributed = false;
    mlir::Operation* operandOp = operand.getDefiningOp();

    if (operandOp) {
      if (auto mulDistributionInt =
              mlir::dyn_cast<MulOpDistributionInterface>(operandOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(mulDistributionInt.distributeMulOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          operand = childResults[0];
          operandDistributed = true;
        }
      }
    }

    if (!operandDistributed) {
      auto newOperandOp = builder.create<MulOp>(
          getLoc(), operand.getType(), operand, value);

      operand = newOperandOp.getResult();
    }

    auto resultOp = builder.create<NegateOp>(getLoc(), resultType, operand);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::LogicalResult NegateOp::distributeDivOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType,
      mlir::Value value)
  {
    mlir::Value operand = getOperand();
    bool operandDistributed = false;
    mlir::Operation* operandOp = operand.getDefiningOp();

    if (operandOp) {
      if (auto divDistributionInt =
              mlir::dyn_cast<DivOpDistributionInterface>(operandOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(divDistributionInt.distributeDivOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          operand = childResults[0];
          operandDistributed = true;
        }
      }
    }

    if (!operandDistributed) {
      auto newOperandOp = builder.create<DivOp>(
          getLoc(), operand.getType(), operand, value);

      operand = newOperandOp.getResult();
    }

    auto resultOp = builder.create<NegateOp>(getLoc(), resultType, operand);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::ValueRange NegateOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    mlir::Value derivedOperand = derivatives.lookup(getOperand());

    auto derivedOp = builder.create<NegateOp>(
        getLoc(), convertToRealType(getResult().getType()), derivedOperand);

    return derivedOp->getResults();
  }

  void NegateOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
  }

  void NegateOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// AddOp

namespace
{
  struct AddOpIterableOrderingPattern : public mlir::OpRewritePattern<AddOp>
  {
    using mlir::OpRewritePattern<AddOp>::OpRewritePattern;

    mlir::LogicalResult match(AddOp op) const override
    {
      mlir::Value lhs = op.getLhs();
      mlir::Value rhs = op.getRhs();

      return mlir::LogicalResult::success(
          !lhs.getType().isa<IterableType>() &&
          rhs.getType().isa<IterableType>());
    }

    void rewrite(
        AddOp op, mlir::PatternRewriter& rewriter) const override
    {
      // Swap the operands.
      rewriter.replaceOpWithNewOp<AddOp>(
          op, op.getResult().getType(), op.getRhs(), op.getLhs());
    }
  };
}

namespace mlir::modelica
{
  mlir::OpFoldResult AddOp::fold(FoldAdaptor adaptor)
  {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        int64_t lhsValue = getScalarIntegerLikeValue(lhs);
        int64_t rhsValue = getScalarIntegerLikeValue(rhs);
        return getAttr(resultType, lhsValue + rhsValue);
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, lhsValue + rhsValue);
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, lhsValue + rhsValue);
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
        return getAttr(resultType, lhsValue + rhsValue);
      }
    }

    if (auto lhsRange = lhs.dyn_cast<IntegerRangeAttr>();
        lhsRange && isScalar(rhs)) {
      if (isScalarIntegerLike(rhs)) {
        int64_t rhsValue = getScalarIntegerLikeValue(rhs);
        int64_t lowerBound = lhsRange.getLowerBound() + rhsValue;
        int64_t upperBound = lhsRange.getUpperBound() + rhsValue;
        int64_t step = lhsRange.getStep();

        return IntegerRangeAttr::get(
            getContext(), lowerBound, upperBound, step);
      }

      if (isScalarFloatLike(rhs)) {
        double rhsValue = getScalarFloatLikeValue(rhs);

        double lowerBound =
            static_cast<double>(lhsRange.getLowerBound()) + rhsValue;

        double upperBound =
            static_cast<double>(lhsRange.getUpperBound()) + rhsValue;

        double step = static_cast<double>(lhsRange.getStep());
        return RealRangeAttr::get(getContext(), lowerBound, upperBound, step);
      }
    }

    if (auto lhsRange = lhs.dyn_cast<RealRangeAttr>();
        lhsRange && isScalar(rhs)) {
      if (isScalarIntegerLike(rhs)) {
        auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));

        double lowerBound =
            lhsRange.getLowerBound().convertToDouble() + rhsValue;

        double upperBound =
            lhsRange.getUpperBound().convertToDouble() + rhsValue;

        double step = lhsRange.getStep().convertToDouble();

        return RealRangeAttr::get(
            getContext(), lowerBound, upperBound, step);
      }

      if (isScalarFloatLike(rhs)) {
        double rhsValue = getScalarFloatLikeValue(rhs);

        double lowerBound =
            lhsRange.getLowerBound().convertToDouble() + rhsValue;

        double upperBound =
            lhsRange.getUpperBound().convertToDouble() + rhsValue;

        double step = lhsRange.getStep().convertToDouble();
        return RealRangeAttr::get(getContext(), lowerBound, upperBound, step);
      }
    }

    return {};
  }

  void AddOp::getCanonicalizationPatterns(
      mlir::RewritePatternSet& patterns, mlir::MLIRContext* context)
  {
    patterns.add<AddOpIterableOrderingPattern>(context);
  }

  void AddOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getLhs().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getLhs(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getRhs().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getRhs(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::Value AddOp::inverse(
      mlir::OpBuilder& builder,
      unsigned int argumentIndex,
      mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (size_t size = currentResult.size(); size != 1) {
      emitOpError() << "Invalid amount of values to be nested: " << size
                    << " (expected 1).";

      return nullptr;
    }

    mlir::Value toNest = currentResult[0];

    if (argumentIndex == 0) {
      mlir::Value nestedOperand = readValue(builder, toNest);

      auto right = builder.create<SubOp>(
          getLoc(), getLhs().getType(), nestedOperand, getRhs());

      return right.getResult();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);

      auto right = builder.create<SubOp>(
          getLoc(), getRhs().getType(), nestedOperand, getLhs());

      return right.getResult();
    }

    emitOpError() << "Can't invert the operand #" << argumentIndex
                  << ". The operation has 2 operands.";

    return nullptr;
  }

  mlir::LogicalResult AddOp::distributeNegateOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    bool lhsDistributed = false;
    bool rhsDistributed = false;

    if (lhsOp) {
      if (auto negDistributionInt =
              mlir::dyn_cast<NegateOpDistributionInterface>(lhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(negDistributionInt.distributeNegateOp(
                childResults, builder, resultType))
            && childResults.size() == 1) {
          lhs = childResults[0];
          lhsDistributed = true;
        }
      }
    }

    if (rhsOp) {
      if (auto negDistributionInt =
              mlir::dyn_cast<NegateOpDistributionInterface>(rhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(negDistributionInt.distributeNegateOp(
                childResults, builder, resultType))
            && childResults.size() == 1) {
          rhs = childResults[0];
          rhsDistributed = true;
        }
      }
    }

    if (!lhsDistributed) {
      auto newLhsOp = builder.create<NegateOp>(
          lhs.getLoc(), lhs.getType(), lhs);

      lhs = newLhsOp.getResult();
    }

    if (!rhsDistributed) {
      auto newRhsOp = builder.create<NegateOp>(
          rhs.getLoc(), rhs.getType(), rhs);

      rhs = newRhsOp.getResult();
    }

    auto resultOp = builder.create<AddOp>(getLoc(), resultType, lhs, rhs);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::LogicalResult AddOp::distributeMulOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType,
      mlir::Value value)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    bool lhsDistributed = false;
    bool rhsDistributed = false;

    if (lhsOp) {
      if (auto mulDistributionInt =
              mlir::dyn_cast<MulOpDistributionInterface>(lhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(mulDistributionInt.distributeMulOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          lhs = childResults[0];
          lhsDistributed = true;
        }
      }
    }

    if (rhsOp) {
      if (auto mulDistributionInt =
              mlir::dyn_cast<MulOpDistributionInterface>(rhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(mulDistributionInt.distributeMulOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          rhs = childResults[0];
          rhsDistributed = true;
        }
      }
    }

    if (!lhsDistributed) {
      auto newLhsOp = builder.create<MulOp>(
          lhs.getLoc(), lhs.getType(), lhs, value);

      lhs = newLhsOp.getResult();
    }

    if (!rhsDistributed) {
      auto newRhsOp = builder.create<MulOp>(
          rhs.getLoc(), rhs.getType(), rhs, value);

      rhs = newRhsOp.getResult();
    }

    auto resultOp = builder.create<AddOp>(getLoc(), resultType, lhs, rhs);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::LogicalResult AddOp::distributeDivOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType,
      mlir::Value value)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    bool lhsDistributed = false;
    bool rhsDistributed = false;

    if (lhsOp) {
      if (auto divDistributionInt =
              mlir::dyn_cast<DivOpDistributionInterface>(lhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(divDistributionInt.distributeDivOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          lhs = childResults[0];
          lhsDistributed = true;
        }
      }
    }

    if (rhsOp) {
      if (auto divDistributionInt =
              mlir::dyn_cast<DivOpDistributionInterface>(rhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(divDistributionInt.distributeDivOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          rhs = childResults[0];
          rhsDistributed = true;
        }
      }
    }

    if (!lhsDistributed) {
      auto newLhsOp = builder.create<DivOp>(
          lhs.getLoc(), lhs.getType(), lhs, value);

      lhs = newLhsOp.getResult();
    }

    if (!rhsDistributed) {
      auto newRhsOp = builder.create<DivOp>(
          rhs.getLoc(), rhs.getType(), rhs, value);

      rhs = newRhsOp.getResult();
    }

    auto resultOp = builder.create<AddOp>(getLoc(), resultType, lhs, rhs);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::ValueRange AddOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    auto loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(getLhs());
    mlir::Value derivedRhs = derivatives.lookup(getRhs());

    auto derivedOp = builder.create<AddOp>(
        loc, convertToRealType(getResult().getType()), derivedLhs, derivedRhs);

    return derivedOp->getResults();
  }

  void AddOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getLhs());
    toBeDerived.push_back(getRhs());
  }

  void AddOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// AddEWOp

namespace mlir::modelica
{
  mlir::OpFoldResult AddEWOp::fold(FoldAdaptor adaptor)
  {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        int64_t lhsValue = getScalarIntegerLikeValue(lhs);
        int64_t rhsValue = getScalarIntegerLikeValue(rhs);
        return getAttr(resultType, lhsValue + rhsValue);
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, lhsValue + rhsValue);
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, lhsValue + rhsValue);
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
        return getAttr(resultType, lhsValue + rhsValue);
      }
    }

    return {};
  }

  void AddEWOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (getLhs().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getLhs(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getRhs().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getRhs(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::Value AddEWOp::inverse(
      mlir::OpBuilder& builder,
      unsigned int argumentIndex,
      mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (size_t size = currentResult.size(); size != 1) {
      emitOpError() << "Invalid amount of values to be nested: " << size
                    << " (expected 1).";
    }

    mlir::Value toNest = currentResult[0];

    if (argumentIndex == 0) {
      mlir::Value nestedOperand = readValue(builder, toNest);

      auto right = builder.create<SubEWOp>(
          getLoc(), getLhs().getType(), nestedOperand, getRhs());

      return right.getResult();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);

      auto right = builder.create<SubEWOp>(
          getLoc(), getRhs().getType(), nestedOperand, getLhs());

      return right.getResult();
    }

    emitOpError() << "Can't invert the operand #" << argumentIndex
                  << ". The operation has 2 operands.";

    return nullptr;
  }

  mlir::LogicalResult AddEWOp::distributeNegateOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    bool lhsDistributed = false;
    bool rhsDistributed = false;

    if (lhsOp) {
      if (auto negDistributionOp =
              mlir::dyn_cast<NegateOpDistributionInterface>(lhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(negDistributionOp.distributeNegateOp(
                childResults, builder, resultType))
            && childResults.size() == 1) {
          lhs = childResults[0];
          lhsDistributed = true;
        }
      }
    }

    if (rhsOp) {
      if (auto negDistributionOp =
              mlir::dyn_cast<NegateOpDistributionInterface>(rhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(negDistributionOp.distributeNegateOp(
                childResults, builder, resultType))
            && childResults.size() == 1) {
          rhs = childResults[0];
          rhsDistributed = true;
        }
      }
    }

    if (!lhsDistributed) {
      auto newLhsOp = builder.create<NegateOp>(
          lhs.getLoc(), lhs.getType(), lhs);

      lhs = newLhsOp.getResult();
    }

    if (!rhsDistributed) {
      auto newRhsOp = builder.create<NegateOp>(
          rhs.getLoc(), rhs.getType(), rhs);

      rhs = newRhsOp.getResult();
    }

    auto resultOp = builder.create<AddEWOp>(getLoc(), resultType, lhs, rhs);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::LogicalResult AddEWOp::distributeMulOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType,
      mlir::Value value)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    bool lhsDistributed = false;
    bool rhsDistributed = false;

    if (lhsOp) {
      if (auto mulDistributionInt =
              mlir::dyn_cast<MulOpDistributionInterface>(lhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(mulDistributionInt.distributeMulOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          lhs = childResults[0];
          lhsDistributed = true;
        }
      }
    }

    if (rhsOp) {
      if (auto mulDistributionInt =
              mlir::dyn_cast<MulOpDistributionInterface>(rhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(mulDistributionInt.distributeMulOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          rhs = childResults[0];
          rhsDistributed = true;
        }
      }
    }

    if (!lhsDistributed) {
      auto newLhsOp = builder.create<MulOp>(
          lhs.getLoc(), lhs.getType(), lhs, value);

      lhs = newLhsOp.getResult();
    }

    if (!rhsDistributed) {
      auto newRhsOp = builder.create<MulOp>(
          rhs.getLoc(), rhs.getType(), rhs, value);

      rhs = newRhsOp.getResult();
    }

    auto resultOp = builder.create<AddEWOp>(getLoc(), resultType, lhs, rhs);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::LogicalResult AddEWOp::distributeDivOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType,
      mlir::Value value)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    bool lhsDistributed = false;
    bool rhsDistributed = false;

    if (lhsOp) {
      if (auto divDistributionInt =
              mlir::dyn_cast<DivOpDistributionInterface>(lhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(divDistributionInt.distributeDivOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          lhs = childResults[0];
          lhsDistributed = true;
        }
      }
    }

    if (rhsOp) {
      if (auto divDistributionInt =
              mlir::dyn_cast<DivOpDistributionInterface>(rhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(divDistributionInt.distributeDivOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          rhs = childResults[0];
          rhsDistributed = true;
        }
      }
    }

    if (!lhsDistributed) {
      auto newLhsOp = builder.create<DivOp>(
          lhs.getLoc(), lhs.getType(), lhs, value);

      lhs = newLhsOp.getResult();
    }

    if (!rhsDistributed) {
      auto newRhsOp = builder.create<DivOp>(
          rhs.getLoc(), rhs.getType(), rhs, value);

      rhs = newRhsOp.getResult();
    }

    auto resultOp = builder.create<AddEWOp>(getLoc(), resultType, lhs, rhs);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::ValueRange AddEWOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    auto loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(getLhs());
    mlir::Value derivedRhs = derivatives.lookup(getRhs());

    auto derivedOp = builder.create<AddEWOp>(
        loc, convertToRealType(getResult().getType()), derivedLhs, derivedRhs);

    return derivedOp->getResults();
  }

  void AddEWOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getLhs());
    toBeDerived.push_back(getRhs());
  }

  void AddEWOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// SubOp

namespace mlir::modelica
{
  mlir::OpFoldResult SubOp::fold(FoldAdaptor adaptor)
  {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        int64_t lhsValue = getScalarIntegerLikeValue(lhs);
        int64_t rhsValue = getScalarIntegerLikeValue(rhs);
        return getAttr(resultType, lhsValue - rhsValue);
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, lhsValue - rhsValue);
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, lhsValue - rhsValue);
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
        return getAttr(resultType, lhsValue - rhsValue);
      }
    }

    return {};
  }

  void SubOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getLhs().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getLhs(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getRhs().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getRhs(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::Value SubOp::inverse(
      mlir::OpBuilder& builder,
      unsigned int argumentIndex,
      mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (size_t size = currentResult.size(); size != 1) {
      emitOpError() << "Invalid amount of values to be nested: " << size
                    << " (expected 1).";

      return nullptr;
    }

    mlir::Value toNest = currentResult[0];

    if (argumentIndex == 0) {
      mlir::Value nestedOperand = readValue(builder, toNest);

      auto right = builder.create<AddOp>(
          getLoc(), getLhs().getType(), nestedOperand, getRhs());

      return right.getResult();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);

      auto right = builder.create<SubOp>(
          getLoc(), getRhs().getType(), getLhs(), nestedOperand);

      return right.getResult();
    }

    emitOpError() << "Can't invert the operand #" << argumentIndex
                  << ". The operation has 2 operands.";

    return nullptr;
  }

  mlir::LogicalResult SubOp::distributeNegateOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    bool lhsDistributed = false;
    bool rhsDistributed = false;

    if (lhsOp) {
      if (auto negDistributionInt =
              mlir::dyn_cast<NegateOpDistributionInterface>(lhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(negDistributionInt.distributeNegateOp(
                childResults, builder, resultType))
            && childResults.size() == 1) {
          lhs = childResults[0];
          lhsDistributed = true;
        }
      }
    }

    if (rhsOp) {
      if (auto negDistributionInt =
              mlir::dyn_cast<NegateOpDistributionInterface>(rhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(negDistributionInt.distributeNegateOp(
                childResults, builder, resultType))
            && childResults.size() == 1) {
          rhs = childResults[0];
          rhsDistributed = true;
        }
      }
    }

    if (!lhsDistributed) {
      auto newLhsOp = builder.create<NegateOp>(
          lhs.getLoc(), lhs.getType(), lhs);

      lhs = newLhsOp.getResult();
    }

    if (!rhsDistributed) {
      auto newRhsOp = builder.create<NegateOp>(
          rhs.getLoc(), rhs.getType(), rhs);

      rhs = newRhsOp.getResult();
    }

    auto resultOp = builder.create<SubOp>(getLoc(), resultType, lhs, rhs);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::LogicalResult SubOp::distributeMulOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType,
      mlir::Value value)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    bool lhsDistributed = false;
    bool rhsDistributed = false;

    if (lhsOp) {
      if (auto mulDistributionInt =
              mlir::dyn_cast<MulOpDistributionInterface>(lhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(mulDistributionInt.distributeMulOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          lhs = childResults[0];
          lhsDistributed = true;
        }
      }
    }

    if (rhsOp) {
      if (auto mulDistributionInt =
              mlir::dyn_cast<MulOpDistributionInterface>(rhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(mulDistributionInt.distributeMulOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          rhs = childResults[0];
          rhsDistributed = true;
        }
      }
    }

    if (!lhsDistributed) {
      auto newLhsOp = builder.create<MulOp>(
          lhs.getLoc(), lhs.getType(), lhs, value);

      lhs = newLhsOp.getResult();
    }

    if (!rhsDistributed) {
      auto newRhsOp = builder.create<MulOp>(
          rhs.getLoc(), rhs.getType(), rhs, value);

      rhs = newRhsOp.getResult();
    }

    auto resultOp = builder.create<SubOp>(getLoc(), resultType, lhs, rhs);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::LogicalResult SubOp::distributeDivOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType,
      mlir::Value value)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    bool lhsDistributed = false;
    bool rhsDistributed = false;

    if (lhsOp) {
      if (auto divDistributionInt =
              mlir::dyn_cast<DivOpDistributionInterface>(lhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(divDistributionInt.distributeDivOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          lhs = childResults[0];
          lhsDistributed = true;
        }
      }
    }

    if (rhsOp) {
      if (auto divDistributionInt =
              mlir::dyn_cast<DivOpDistributionInterface>(rhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(divDistributionInt.distributeDivOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          rhs = childResults[0];
          rhsDistributed = true;
        }
      }
    }

    if (!lhsDistributed) {
      auto newLhsOp = builder.create<DivOp>(
          lhs.getLoc(), lhs.getType(), lhs, value);

      lhs = newLhsOp.getResult();
    }

    if (!rhsDistributed) {
      auto newRhsOp = builder.create<DivOp>(
          rhs.getLoc(), rhs.getType(), rhs, value);

      rhs = newRhsOp.getResult();
    }

    auto resultOp = builder.create<SubOp>(getLoc(), resultType, lhs, rhs);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::ValueRange SubOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    mlir::Location loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(getLhs());
    mlir::Value derivedRhs = derivatives.lookup(getRhs());

    auto derivedOp = builder.create<SubOp>(
        loc, convertToRealType(getResult().getType()), derivedLhs, derivedRhs);

    return derivedOp->getResults();
  }

  void SubOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getLhs());
    toBeDerived.push_back(getRhs());
  }

  void SubOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// SubEWOp

namespace mlir::modelica
{
  mlir::OpFoldResult SubEWOp::fold(FoldAdaptor adaptor)
  {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        int64_t lhsValue = getScalarIntegerLikeValue(lhs);
        int64_t rhsValue = getScalarIntegerLikeValue(rhs);
        return getAttr(resultType, lhsValue - rhsValue);
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, lhsValue - rhsValue);
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, lhsValue - rhsValue);
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
        return getAttr(resultType, lhsValue - rhsValue);
      }
    }

    return {};
  }

  void SubEWOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getLhs().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getLhs(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getRhs().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getRhs(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::Value SubEWOp::inverse(
      mlir::OpBuilder& builder,
      unsigned int argumentIndex,
      mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (size_t size = currentResult.size(); size != 1) {
      emitOpError() << "Invalid amount of values to be nested: " << size
                    << " (expected 1).";

      return nullptr;
    }

    mlir::Value toNest = currentResult[0];

    if (argumentIndex == 0) {
      mlir::Value nestedOperand = readValue(builder, toNest);

      auto right = builder.create<AddEWOp>(
          getLoc(), getLhs().getType(), nestedOperand, getRhs());

      return right.getResult();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);

      auto right = builder.create<SubEWOp>(
          getLoc(), getRhs().getType(), getLhs(), nestedOperand);

      return right.getResult();
    }

    emitOpError() << "Can't invert the operand #" << argumentIndex
                  << ". The operation has 2 operands.";

    return nullptr;
  }

  mlir::LogicalResult SubEWOp::distributeNegateOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    bool lhsDistributed = false;
    bool rhsDistributed = false;

    if (lhsOp) {
      if (auto negDistributionInt =
              mlir::dyn_cast<NegateOpDistributionInterface>(lhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(negDistributionInt.distributeNegateOp(
                childResults, builder, resultType))
            && childResults.size() == 1) {
          lhs = childResults[0];
          lhsDistributed = true;
        }
      }
    }

    if (rhsOp) {
      if (auto negDistributionInt =
              mlir::dyn_cast<NegateOpDistributionInterface>(rhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(negDistributionInt.distributeNegateOp(
                childResults, builder, resultType))
            && childResults.size() == 1) {
          rhs = childResults[0];
          rhsDistributed = true;
        }
      }
    }

    if (!lhsDistributed) {
      auto newLhsOp = builder.create<NegateOp>(
          lhs.getLoc(), lhs.getType(), lhs);

      lhs = newLhsOp.getResult();
    }

    if (!rhsDistributed) {
      auto newRhsOp = builder.create<NegateOp>(
          rhs.getLoc(), rhs.getType(), rhs);

      rhs = newRhsOp.getResult();
    }

    auto resultOp = builder.create<SubEWOp>(getLoc(), resultType, lhs, rhs);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::LogicalResult SubEWOp::distributeMulOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType,
      mlir::Value value)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    bool lhsDistributed = false;
    bool rhsDistributed = false;

    if (lhsOp) {
      if (auto mulDistributionInt =
              mlir::dyn_cast<MulOpDistributionInterface>(lhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(mulDistributionInt.distributeMulOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          lhs = childResults[0];
          lhsDistributed = true;
        }
      }
    }

    if (rhsOp) {
      if (auto mulDistributionInt =
              mlir::dyn_cast<MulOpDistributionInterface>(rhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(mulDistributionInt.distributeMulOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          rhs = childResults[0];
          rhsDistributed = true;
        }
      }
    }

    if (!lhsDistributed) {
      auto newLhsOp = builder.create<MulOp>(
          lhs.getLoc(), lhs.getType(), lhs, value);

      lhs = newLhsOp.getResult();
    }

    if (!rhsDistributed) {
      auto newRhsOp = builder.create<MulOp>(
          rhs.getLoc(), rhs.getType(), rhs, value);

      rhs = newRhsOp.getResult();
    }

    auto resultOp = builder.create<SubEWOp>(getLoc(), resultType, lhs, rhs);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::LogicalResult SubEWOp::distributeDivOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType,
      mlir::Value value)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    bool lhsDistributed = false;
    bool rhsDistributed = false;

    if (lhsOp) {
      if (auto divDistributionInt =
              mlir::dyn_cast<DivOpDistributionInterface>(lhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(divDistributionInt.distributeDivOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          lhs = childResults[0];
          lhsDistributed = true;
        }
      }
    }

    if (rhsOp) {
      if (auto divDistributionInt =
              mlir::dyn_cast<DivOpDistributionInterface>(rhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(divDistributionInt.distributeDivOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          rhs = childResults[0];
          rhsDistributed = true;
        }
      }
    }

    if (!lhsDistributed) {
      auto newLhsOp = builder.create<DivOp>(
          lhs.getLoc(), lhs.getType(), lhs, value);

      lhs = newLhsOp.getResult();
    }

    if (!rhsDistributed) {
      auto newRhsOp = builder.create<DivOp>(
          rhs.getLoc(), rhs.getType(), rhs, value);

      rhs = newRhsOp.getResult();
    }

    auto resultOp = builder.create<SubEWOp>(getLoc(), resultType, lhs, rhs);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::ValueRange SubEWOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    mlir::Location loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(getLhs());
    mlir::Value derivedRhs = derivatives.lookup(getRhs());

    auto derivedOp = builder.create<SubEWOp>(
        loc,
        convertToRealType(getResult().getType()),
        derivedLhs, derivedRhs);

    return derivedOp->getResults();
  }

  void SubEWOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getLhs());
    toBeDerived.push_back(getRhs());
  }

  void SubEWOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// MulOp

namespace mlir::modelica
{
  mlir::OpFoldResult MulOp::fold(FoldAdaptor adaptor)
  {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        int64_t lhsValue = getScalarIntegerLikeValue(lhs);
        int64_t rhsValue = getScalarIntegerLikeValue(rhs);
        return getAttr(resultType, lhsValue * rhsValue);
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, lhsValue * rhsValue);
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, lhsValue * rhsValue);
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
        return getAttr(resultType, lhsValue * rhsValue);
      }
    }

    return {};
  }

  void MulOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getLhs().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getLhs(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getRhs().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getRhs(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::Value MulOp::inverse(
      mlir::OpBuilder& builder,
      unsigned int argumentIndex,
      mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (size_t size = currentResult.size(); size != 1) {
      emitOpError() << "Invalid amount of values to be nested: " << size
                    << " (expected 1).";

      return nullptr;
    }

    mlir::Value toNest = currentResult[0];

    if (argumentIndex == 0) {
      mlir::Value nestedOperand = readValue(builder, toNest);

      auto right = builder.create<DivOp>(
          getLoc(), getLhs().getType(), nestedOperand, getRhs());

      return right.getResult();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);

      auto right = builder.create<DivOp>(
          getLoc(), getRhs().getType(), nestedOperand, getLhs());

      return right.getResult();
    }

    emitOpError() << "Can't invert the operand #" << argumentIndex
                  << ". The operation has 2 operands.";

    return nullptr;
  }

  mlir::LogicalResult MulOp::distribute(
      llvm::SmallVectorImpl<mlir::Value>& results, mlir::OpBuilder& builder)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    if (lhsOp) {
      if (auto mulDistributionInt =
              mlir::dyn_cast<MulOpDistributionInterface>(lhsOp)) {
        mlir::Value toDistribute = rhs;
        results.clear();

        if (mlir::succeeded(mulDistributionInt.distributeMulOp(
            results, builder, getResult().getType(), toDistribute))) {
          return mlir::success();
        }
      }
    }

    if (rhsOp) {
      if (auto mulDistributionInt =
              mlir::dyn_cast<MulOpDistributionInterface>(rhsOp)) {
        mlir::Value toDistribute = lhs;
        results.clear();

        if (mlir::succeeded(mulDistributionInt.distributeMulOp(
            results, builder, getResult().getType(), toDistribute))) {
          return mlir::success();
        }
      }
    }

    // The operation can't be propagated because none of the children
    // know how to distribute the multiplication to their children.
    results.push_back(getResult());
    return mlir::failure();
  }

  mlir::LogicalResult MulOp::distributeNegateOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    if (lhsOp) {
      if (auto negDistributionInt =
              mlir::dyn_cast<NegateOpDistributionInterface>(lhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(negDistributionInt.distributeNegateOp(
                childResults, builder, resultType))
            && childResults.size() == 1) {
          lhs = childResults[0];

          auto resultOp = builder.create<MulOp>(
              getLoc(), resultType, lhs, rhs);

          results.push_back(resultOp.getResult());
          return mlir::success();
        }
      }
    }

    if (rhsOp) {
      if (auto negDistributionInt =
              mlir::dyn_cast<NegateOpDistributionInterface>(rhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(negDistributionInt.distributeNegateOp(
                childResults, builder, resultType))
            && childResults.size() == 1) {
          rhs = childResults[0];

          auto resultOp = builder.create<MulOp>(
              getLoc(), resultType, lhs, rhs);

          results.push_back(resultOp.getResult());
          return mlir::success();
        }
      }
    }

    auto lhsNewOp = builder.create<NegateOp>(getLoc(), lhs.getType(), lhs);
    lhs = lhsNewOp.getResult();

    auto resultOp = builder.create<MulOp>(getLoc(), resultType, lhs, rhs);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::LogicalResult MulOp::distributeMulOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType,
      mlir::Value value)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    if (lhsOp) {
      if (auto mulDistributionInt =
              mlir::dyn_cast<MulOpDistributionInterface>(lhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(mulDistributionInt.distributeMulOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          lhs = childResults[0];

          auto resultOp = builder.create<MulOp>(
              getLoc(), resultType, lhs, rhs);

          results.push_back(resultOp.getResult());
          return mlir::success();
        }
      }
    }

    if (rhsOp) {
      if (auto mulDistributionInt =
              mlir::dyn_cast<MulOpDistributionInterface>(rhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(mulDistributionInt.distributeMulOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          rhs = childResults[0];

          auto resultOp = builder.create<MulOp>(
              getLoc(), resultType, lhs, rhs);

          results.push_back(resultOp.getResult());
          return mlir::success();
        }
      }
    }

    auto lhsNewOp = builder.create<MulOp>(getLoc(), lhs.getType(), lhs, value);
    lhs = lhsNewOp.getResult();

    auto resultOp = builder.create<MulOp>(getLoc(), resultType, lhs, rhs);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::LogicalResult MulOp::distributeDivOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType,
      mlir::Value value)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    if (lhsOp) {
      if (auto divDistributionInt =
              mlir::dyn_cast<DivOpDistributionInterface>(lhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(divDistributionInt.distributeDivOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          lhs = childResults[0];

          auto resultOp = builder.create<MulOp>(
              getLoc(), resultType, lhs, rhs);

          results.push_back(resultOp.getResult());
          return mlir::success();
        }
      }
    }

    if (rhsOp) {
      if (auto divDistributionInt =
              mlir::dyn_cast<DivOpDistributionInterface>(rhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(divDistributionInt.distributeDivOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          rhs = childResults[0];

          auto resultOp = builder.create<MulOp>(
              getLoc(), resultType, lhs, rhs);

          results.push_back(resultOp.getResult());
          return mlir::success();
        }
      }
    }

    auto lhsNewOp = builder.create<DivOp>(getLoc(), lhs.getType(), lhs, value);
    lhs = lhsNewOp.getResult();

    auto resultOp = builder.create<MulOp>(getLoc(), resultType, lhs, rhs);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::ValueRange MulOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    mlir::Location loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(getLhs());
    mlir::Value derivedRhs = derivatives.lookup(getRhs());

    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value firstMul = builder.create<MulOp>(
        loc, type, derivedLhs, getRhs());

    mlir::Value secondMul = builder.create<MulOp>(
        loc, type, getLhs(), derivedRhs);

    auto derivedOp = builder.create<AddOp>(loc, type, firstMul, secondMul);

    return derivedOp->getResults();
  }

  void MulOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getLhs());
    toBeDerived.push_back(getRhs());
  }

  void MulOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// MulEWOp

namespace mlir::modelica
{
  mlir::OpFoldResult MulEWOp::fold(FoldAdaptor adaptor)
  {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        int64_t lhsValue = getScalarIntegerLikeValue(lhs);
        int64_t rhsValue = getScalarIntegerLikeValue(rhs);
        return getAttr(resultType, lhsValue * rhsValue);
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, lhsValue * rhsValue);
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, lhsValue * rhsValue);
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
        return getAttr(resultType, lhsValue * rhsValue);
      }
    }

    return {};
  }

  void MulEWOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getLhs().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getLhs(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getRhs().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getRhs(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::Value MulEWOp::inverse(
      mlir::OpBuilder& builder,
      unsigned int argumentIndex,
      mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (size_t size = currentResult.size(); size != 1) {
      emitOpError() << "Invalid amount of values to be nested: " << size
                    << " (expected 1).";

      return nullptr;
    }

    mlir::Value toNest = currentResult[0];

    if (argumentIndex == 0) {
      mlir::Value nestedOperand = readValue(builder, toNest);

      auto right = builder.create<DivEWOp>(
          getLoc(), getLhs().getType(), nestedOperand, getRhs());

      return right.getResult();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);

      auto right = builder.create<DivEWOp>(
          getLoc(), getRhs().getType(), nestedOperand, getLhs());

      return right.getResult();
    }

    emitOpError() << "Can't invert the operand #" << argumentIndex
                  << ". The operation has 2 operands.";

    return nullptr;
  }

  mlir::LogicalResult MulEWOp::distribute(
      llvm::SmallVectorImpl<mlir::Value>& results, mlir::OpBuilder& builder)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    if (lhsOp) {
      if (auto mulDistributionInt =
              mlir::dyn_cast<MulOpDistributionInterface>(lhsOp)) {
        mlir::Value toDistribute = rhs;
        results.clear();

        if (mlir::succeeded(mulDistributionInt.distributeMulOp(
                results, builder, getResult().getType(), toDistribute))) {
          return mlir::success();
        }
      }
    }

    if (rhsOp) {
      if (auto mulDistributionInt =
              mlir::dyn_cast<MulOpDistributionInterface>(rhsOp)) {
        mlir::Value toDistribute = lhs;
        results.clear();

        if (mlir::succeeded(mulDistributionInt.distributeMulOp(
                results, builder, getResult().getType(), toDistribute))) {
          return mlir::success();
        }
      }
    }

    // The operation can't be propagated because none of the children
    // know how to distribute the multiplication to their children.
    results.push_back(getResult());
    return mlir::failure();
  }

  mlir::LogicalResult MulEWOp::distributeNegateOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    if (lhsOp) {
      if (auto negDistributionInt =
              mlir::dyn_cast<NegateOpDistributionInterface>(lhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(negDistributionInt.distributeNegateOp(
                childResults, builder, resultType))
            && childResults.size() == 1) {
          lhs = childResults[0];

          auto resultOp = builder.create<MulEWOp>(
              getLoc(), resultType, lhs, rhs);

          results.push_back(resultOp.getResult());
          return mlir::success();
        }
      }
    }

    if (rhsOp) {
      if (auto negDistributionInt =
              mlir::dyn_cast<NegateOpDistributionInterface>(rhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(negDistributionInt.distributeNegateOp(
                childResults, builder, resultType))
            && childResults.size() == 1) {
          rhs = childResults[0];

          auto resultOp = builder.create<MulEWOp>(
              getLoc(), resultType, lhs, rhs);

          results.push_back(resultOp.getResult());
          return mlir::success();
        }
      }
    }

    auto lhsNewOp = builder.create<NegateOp>(getLoc(), lhs.getType(), lhs);
    lhs = lhsNewOp.getResult();

    auto resultOp = builder.create<MulEWOp>(getLoc(), resultType, lhs, rhs);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::LogicalResult MulEWOp::distributeMulOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    if (lhsOp) {
      if (auto mulDistributionInt =
              mlir::dyn_cast<MulOpDistributionInterface>(lhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(mulDistributionInt.distributeMulOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          lhs = childResults[0];

          auto resultOp = builder.create<MulEWOp>(
              getLoc(), resultType, lhs, rhs);

          results.push_back(resultOp.getResult());
          return mlir::success();
        }
      }
    }

    if (rhsOp) {
      if (auto mulDistributionInt =
              mlir::dyn_cast<MulOpDistributionInterface>(rhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(mulDistributionInt.distributeMulOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          rhs = childResults[0];

          auto resultOp = builder.create<MulEWOp>(
              getLoc(), resultType, lhs, rhs);

          results.push_back(resultOp.getResult());
          return mlir::success();
        }
      }
    }

    auto lhsNewOp = builder.create<MulOp>(getLoc(), lhs.getType(), lhs, value);
    lhs = lhsNewOp.getResult();

    auto resultOp = builder.create<MulEWOp>(getLoc(), resultType, lhs, rhs);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::LogicalResult MulEWOp::distributeDivOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType,
      mlir::Value value)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    if (lhsOp) {
      if (auto divDistributionInt =
              mlir::dyn_cast<DivOpDistributionInterface>(lhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(divDistributionInt.distributeDivOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          lhs = childResults[0];

          auto resultOp = builder.create<MulEWOp>(
              getLoc(), resultType, lhs, rhs);

          results.push_back(resultOp.getResult());
          return mlir::success();
        }
      }
    }

    if (rhsOp) {
      if (auto divDistributionInt =
              mlir::dyn_cast<DivOpDistributionInterface>(rhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(divDistributionInt.distributeDivOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          rhs = childResults[0];

          auto resultOp = builder.create<MulEWOp>(
              getLoc(), resultType, lhs, rhs);

          results.push_back(resultOp.getResult());
          return mlir::success();
        }
      }
    }

    auto lhsNewOp = builder.create<DivOp>(getLoc(), lhs.getType(), lhs, value);
    lhs = lhsNewOp.getResult();

    auto resultOp = builder.create<MulEWOp>(getLoc(), resultType, lhs, rhs);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::ValueRange MulEWOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    mlir::Location loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(getLhs());
    mlir::Value derivedRhs = derivatives.lookup(getRhs());

    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value firstMul = builder.create<MulEWOp>(
        loc, type, derivedLhs, getRhs());

    mlir::Value secondMul = builder.create<MulEWOp>(
        loc, type, getLhs(), derivedRhs);

    auto derivedOp = builder.create<AddEWOp>(loc, type, firstMul, secondMul);

    return derivedOp->getResults();
  }

  void MulEWOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getLhs());
    toBeDerived.push_back(getRhs());
  }

  void MulEWOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// DivOp

namespace mlir::modelica
{
  mlir::OpFoldResult DivOp::fold(FoldAdaptor adaptor)
  {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
        auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
        return getAttr(resultType, lhsValue / rhsValue);
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, lhsValue / rhsValue);
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, lhsValue / rhsValue);
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
        return getAttr(resultType, lhsValue / rhsValue);
      }
    }

    return {};
  }

  void DivOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getLhs().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getLhs(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getRhs().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getRhs(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::Value DivOp::inverse(
      mlir::OpBuilder& builder,
      unsigned int argumentIndex,
      mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (size_t size = currentResult.size(); size != 1) {
      emitOpError() << "Invalid amount of values to be nested: " << size
                    << " (expected 1).";

      return nullptr;
    }

    mlir::Value toNest = currentResult[0];

    if (argumentIndex == 0) {
      mlir::Value nestedOperand = readValue(builder, toNest);

      auto right = builder.create<MulOp>(
          getLoc(), getLhs().getType(), nestedOperand, getRhs());

      return right.getResult();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<DivOp>(
          getLoc(), getRhs().getType(), getLhs(), nestedOperand);

      return right.getResult();
    }

    emitOpError() << "Can't invert the operand #" << argumentIndex
                  << ". The operation has 2 operands.";

    return nullptr;
  }

  mlir::LogicalResult DivOp::distribute(
      llvm::SmallVectorImpl<mlir::Value>& results, mlir::OpBuilder& builder)
  {
    mlir::Value lhs = getLhs();
    mlir::Operation* lhsOp = lhs.getDefiningOp();

    if (lhsOp) {
      if (auto divDistributionInt =
              mlir::dyn_cast<DivOpDistributionInterface>(lhsOp)) {
        mlir::Value toDistribute = getRhs();

        if (mlir::succeeded(divDistributionInt.distributeDivOp(
            results, builder, getResult().getType(), toDistribute))) {
          return mlir::success();
        }
      }
    }

    // The operation can't be propagated because the dividend does not know
    // how to distribute the division to their children.
    results.push_back(getResult());
    return mlir::success();
  }

  mlir::LogicalResult DivOp::distributeNegateOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    if (lhsOp) {
      if (auto negDistributionInt =
              mlir::dyn_cast<NegateOpDistributionInterface>(lhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(negDistributionInt.distributeNegateOp(
                childResults, builder, resultType))
            && childResults.size() == 1) {
          lhs = childResults[0];

          auto resultOp = builder.create<DivOp>(
              getLoc(), resultType, lhs, rhs);

          results.push_back(resultOp.getResult());
          return mlir::success();
        }
      }
    }

    if (rhsOp) {
      if (auto negDistributionInt =
              mlir::dyn_cast<NegateOpDistributionInterface>(rhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(negDistributionInt.distributeNegateOp(
                childResults, builder, resultType))
            && childResults.size() == 1) {
          rhs = childResults[0];

          auto resultOp = builder.create<DivOp>(
              getLoc(), resultType, lhs, rhs);

          results.push_back(resultOp.getResult());
          return mlir::success();
        }
      }
    }

    auto lhsNewOp = builder.create<NegateOp>(getLoc(), lhs.getType(), lhs);
    lhs = lhsNewOp.getResult();

    auto resultOp = builder.create<DivOp>(getLoc(), resultType, lhs, rhs);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::LogicalResult DivOp::distributeMulOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType,
      mlir::Value value)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    if (lhsOp) {
      if (auto mulDistributionInt =
              mlir::dyn_cast<MulOpDistributionInterface>(lhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(mulDistributionInt.distributeMulOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          lhs = childResults[0];

          auto resultOp = builder.create<DivOp>(
              getLoc(), resultType, lhs, rhs);

          results.push_back(resultOp.getResult());
          return mlir::success();
        }
      }
    }

    if (rhsOp) {
      if (auto divDistributionInt =
              mlir::dyn_cast<DivOpDistributionInterface>(rhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(divDistributionInt.distributeDivOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          rhs = childResults[0];

          auto resultOp = builder.create<DivOp>(
              getLoc(), resultType, lhs, rhs);

          results.push_back(resultOp.getResult());
          return mlir::success();
        }
      }
    }

    auto lhsNewOp = builder.create<MulOp>(getLoc(), lhs.getType(), lhs, value);
    lhs = lhsNewOp.getResult();

    auto resultOp = builder.create<DivOp>(getLoc(), resultType, lhs, rhs);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::LogicalResult DivOp::distributeDivOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType,
      mlir::Value value)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    if (lhsOp) {
      if (auto divDistributionInt = mlir::dyn_cast<DivOpDistributionInterface>(
              lhs.getDefiningOp())) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(divDistributionInt.distributeDivOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          lhs = childResults[0];

          auto resultOp = builder.create<DivOp>(
              getLoc(), resultType, lhs, rhs);

          results.push_back(resultOp.getResult());
          return mlir::success();
        }
      }
    }

    if (rhsOp) {
      if (auto mulDistributionInt = mlir::dyn_cast<MulOpDistributionInterface>(
              rhs.getDefiningOp())) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(mulDistributionInt.distributeMulOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          rhs = childResults[0];

          auto resultOp = builder.create<DivOp>(
              getLoc(), resultType, lhs, rhs);

          results.push_back(resultOp.getResult());
          return mlir::success();
        }
      }
    }

    auto lhsNewOp = builder.create<DivOp>(getLoc(), lhs.getType(), lhs, value);
    lhs = lhsNewOp.getResult();

    auto resultOp = builder.create<DivOp>(getLoc(), resultType, lhs, rhs);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::ValueRange DivOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    mlir::Location loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(getLhs());
    mlir::Value derivedRhs = derivatives.lookup(getRhs());

    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value firstMul = builder.create<MulOp>(
        loc, type, derivedLhs, getRhs());

    mlir::Value secondMul = builder.create<MulOp>(
        loc, type, getLhs(), derivedRhs);

    mlir::Value numerator = builder.create<SubOp>(
        loc, type, firstMul, secondMul);

    mlir::Value two = builder.create<ConstantOp>(
        loc, RealAttr::get(getContext(), 2));

    mlir::Value denominator = builder.create<PowOp>(
        loc, convertToRealType(getRhs().getType()), getRhs(), two);

    auto derivedOp = builder.create<DivOp>(loc, type, numerator, denominator);

    return derivedOp->getResults();
  }

  void DivOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getLhs());
    toBeDerived.push_back(getRhs());
  }

  void DivOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// DivEWOp

namespace mlir::modelica
{
  mlir::OpFoldResult DivEWOp::fold(FoldAdaptor adaptor)
  {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
        auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
        return getAttr(resultType, lhsValue / rhsValue);
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, lhsValue / rhsValue);
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, lhsValue / rhsValue);
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
        return getAttr(resultType, lhsValue / rhsValue);
      }
    }

    return {};
  }

  void DivEWOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getLhs().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getLhs(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getRhs().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getRhs(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::Value DivEWOp::inverse(
      mlir::OpBuilder& builder,
      unsigned int argumentIndex,
      mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (size_t size = currentResult.size(); size != 1) {
      emitOpError() << "Invalid amount of values to be nested: " << size
                    << " (expected 1)";

      return nullptr;
    }

    mlir::Value toNest = currentResult[0];

    if (argumentIndex == 0) {
      mlir::Value nestedOperand = readValue(builder, toNest);

      auto right = builder.create<MulEWOp>(
          getLoc(), getLhs().getType(), nestedOperand, getRhs());

      return right.getResult();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);

      auto right = builder.create<DivEWOp>(
          getLoc(), getRhs().getType(), getLhs(), nestedOperand);

      return right.getResult();
    }

    emitOpError() << "Can't invert the operand #" << argumentIndex
                  << ". The operation has 2 operands.";

    return nullptr;
  }

  mlir::LogicalResult DivEWOp::distribute(
      llvm::SmallVectorImpl<mlir::Value>& results, mlir::OpBuilder& builder)
  {
    mlir::Value lhs = getLhs();
    mlir::Operation* lhsOp = lhs.getDefiningOp();

    if (lhsOp) {
      if (auto divDistributionInt =
              mlir::dyn_cast<DivOpDistributionInterface>(lhsOp)) {
        mlir::Value toDistribute = getRhs();

        if (mlir::succeeded(divDistributionInt.distributeDivOp(
                results, builder, getResult().getType(), toDistribute))) {
          return mlir::success();
        }
      }
    }

    // The operation can't be propagated because the dividend does not know
    // how to distribute the division to their children.
    results.push_back(getResult());
    return mlir::failure();
  }

  mlir::LogicalResult DivEWOp::distributeNegateOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    if (lhsOp) {
      if (auto negDistributionInt =
              mlir::dyn_cast<NegateOpDistributionInterface>(lhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(negDistributionInt.distributeNegateOp(
                childResults, builder, resultType))
            && childResults.size() == 1) {
          lhs = childResults[0];

          auto resultOp = builder.create<DivEWOp>(
              getLoc(), resultType, lhs, rhs);

          results.push_back(resultOp.getResult());
          return mlir::success();
        }
      }
    }

    if (rhsOp) {
      if (auto negDistributionInt =
              mlir::dyn_cast<NegateOpDistributionInterface>(rhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(negDistributionInt.distributeNegateOp(
                childResults, builder, resultType))
            && childResults.size() == 1) {
          rhs = childResults[0];

          auto resultOp = builder.create<DivEWOp>(
              getLoc(), resultType, lhs, rhs);

          results.push_back(resultOp.getResult());
          return mlir::success();
        }
      }
    }

    auto lhsNewOp = builder.create<NegateOp>(getLoc(), lhs.getType(), lhs);
    lhs = lhsNewOp.getResult();

    auto resultOp = builder.create<DivEWOp>(getLoc(), resultType, lhs, rhs);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::LogicalResult DivEWOp::distributeMulOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType,
      mlir::Value value)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    if (lhsOp) {
      if (auto mulDistributionInt =
              mlir::dyn_cast<MulOpDistributionInterface>(lhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(mulDistributionInt.distributeMulOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          lhs = childResults[0];

          auto resultOp = builder.create<DivEWOp>(
              getLoc(), resultType, lhs, rhs);

          results.push_back(resultOp.getResult());
          return mlir::success();
        }
      }
    }

    if (rhsOp) {
      if (auto divDistributionInt =
              mlir::dyn_cast<DivOpDistributionInterface>(rhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(divDistributionInt.distributeDivOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          rhs = childResults[0];

          auto resultOp = builder.create<DivEWOp>(
              getLoc(), resultType, lhs, rhs);

          results.push_back(resultOp.getResult());
          return mlir::success();
        }
      }
    }

    auto lhsNewOp = builder.create<MulOp>(getLoc(), lhs.getType(), lhs, value);
    lhs = lhsNewOp.getResult();

    auto resultOp = builder.create<DivEWOp>(getLoc(), resultType, lhs, rhs);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::LogicalResult DivEWOp::distributeDivOp(
      llvm::SmallVectorImpl<mlir::Value>& results,
      mlir::OpBuilder& builder,
      mlir::Type resultType,
      mlir::Value value)
  {
    mlir::Value lhs = getLhs();
    mlir::Value rhs = getRhs();

    mlir::Operation* lhsOp = lhs.getDefiningOp();
    mlir::Operation* rhsOp = rhs.getDefiningOp();

    if (lhsOp) {
      if (auto divDistributionInt =
              mlir::dyn_cast<DivOpDistributionInterface>(lhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(divDistributionInt.distributeDivOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          lhs = childResults[0];

          auto resultOp = builder.create<DivEWOp>(
              getLoc(), resultType, lhs, rhs);

          results.push_back(resultOp.getResult());
          return mlir::success();
        }
      }
    }

    if (rhsOp) {
      if (auto mulDistributionInt =
              mlir::dyn_cast<MulOpDistributionInterface>(rhsOp)) {
        llvm::SmallVector<mlir::Value, 1> childResults;

        if (mlir::succeeded(mulDistributionInt.distributeMulOp(
                childResults, builder, resultType, value))
            && childResults.size() == 1) {
          rhs = childResults[0];

          auto resultOp = builder.create<DivEWOp>(
              getLoc(), resultType, lhs, rhs);

          results.push_back(resultOp.getResult());
          return mlir::success();
        }
      }
    }

    auto lhsNewOp = builder.create<DivOp>(getLoc(), lhs.getType(), lhs, value);
    lhs = lhsNewOp.getResult();

    auto resultOp = builder.create<DivEWOp>(getLoc(), resultType, lhs, rhs);
    results.push_back(resultOp.getResult());

    return mlir::success();
  }

  mlir::ValueRange DivEWOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    mlir::Location loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(getLhs());
    mlir::Value derivedRhs = derivatives.lookup(getRhs());

    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value firstMul = builder.create<MulEWOp>(
        loc, type, derivedLhs, getRhs());

    mlir::Value secondMul = builder.create<MulEWOp>(
        loc, type, getLhs(), derivedRhs);

    mlir::Value numerator = builder.create<SubEWOp>(
        loc, type, firstMul, secondMul);

    mlir::Value two = builder.create<ConstantOp>(
        loc, RealAttr::get(getContext(), 2));

    mlir::Value denominator = builder.create<PowEWOp>(
        loc, convertToRealType(getRhs().getType()), getRhs(), two);

    auto derivedOp = builder.create<DivEWOp>(
        loc, type, numerator, denominator);

    return derivedOp->getResults();
  }

  void DivEWOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getLhs());
    toBeDerived.push_back(getRhs());
  }

  void DivEWOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// PowOp

namespace mlir::modelica
{
  mlir::OpFoldResult PowOp::fold(FoldAdaptor adaptor)
  {
    auto base = adaptor.getBase();
    auto exponent = adaptor.getExponent();

    if (!base || !exponent) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(base) && isScalar(exponent)) {
      if (isScalarIntegerLike(base) && isScalarIntegerLike(exponent)) {
        auto baseValue = static_cast<double>(getScalarIntegerLikeValue(base));

        auto exponentValue =
            static_cast<double>(getScalarIntegerLikeValue(exponent));

        return getAttr(resultType, std::pow(baseValue, exponentValue));
      }

      if (isScalarFloatLike(base) && isScalarFloatLike(exponent)) {
        double baseValue = getScalarFloatLikeValue(base);
        double exponentValue = getScalarFloatLikeValue(exponent);
        return getAttr(resultType, std::pow(baseValue, exponentValue));
      }

      if (isScalarIntegerLike(base) && isScalarFloatLike(exponent)) {
        auto baseValue = static_cast<double>(getScalarIntegerLikeValue(base));
        double exponentValue = getScalarFloatLikeValue(exponent);
        return getAttr(resultType, std::pow(baseValue, exponentValue));
      }

      if (isScalarFloatLike(base) && isScalarIntegerLike(exponent)) {
        double baseValue = getScalarFloatLikeValue(base);

        auto exponentValue =
            static_cast<double>(getScalarIntegerLikeValue(exponent));

        return getAttr(resultType, std::pow(baseValue, exponentValue));
      }
    }

    return {};
  }

  void PowOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getBase().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getBase(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getExponent().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getExponent(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange PowOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    // D[x ^ y] = (x ^ (y - 1)) * (y * x' + x * ln(x) * y')

    mlir::Location loc = getLoc();

    mlir::Value derivedBase = derivatives.lookup(getBase());
    mlir::Value derivedExponent = derivatives.lookup(getExponent());

    mlir::Type type = convertToRealType(getResult().getType());

    if (auto constantExponent = getExponent().getDefiningOp<ConstantOp>()) {
      mlir::Value one = builder.create<ConstantOp>(
          loc, RealAttr::get(builder.getContext(), 1));

      mlir::Value exponent = builder.createOrFold<SubOp>(
          loc, getExponent().getType(), getExponent(), one);

      mlir::Value pow = builder.create<PowOp>(loc, type, getBase(), exponent);
      auto derivedOp = builder.create<MulOp>(loc, type, pow, derivedBase);

      return derivedOp->getResults();

    } else {
      mlir::Value one = builder.create<ConstantOp>(
          loc, RealAttr::get(builder.getContext(), 1));

      mlir::Value exponent = builder.create<SubOp>(
          loc, getExponent().getType(), getExponent(), one);

      mlir::Value pow = builder.create<PowOp>(loc, type, getBase(),exponent);

      mlir::Value firstMul = builder.create<MulOp>(
          loc, type, getExponent(), derivedBase);

      mlir::Value ln = builder.create<LogOp>(loc, type, getBase());
      mlir::Value secondMul = builder.create<MulOp>(loc, type, getBase(), ln);

      mlir::Value thirdMul = builder.create<MulOp>(
          loc, type, secondMul, derivedExponent);

      mlir::Value sum = builder.create<AddOp>(loc, type, firstMul, thirdMul);
      auto derivedOp = builder.create<MulOp>(loc, type, pow, sum);

      return derivedOp->getResults();
    }
  }

  void PowOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getBase());
    toBeDerived.push_back(getExponent());
  }

  void PowOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// PowEWOp

namespace mlir::modelica
{
  mlir::OpFoldResult PowEWOp::fold(FoldAdaptor adaptor)
  {
    auto base = adaptor.getBase();
    auto exponent = adaptor.getExponent();

    if (!base || !exponent) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(base) && isScalar(exponent)) {
      if (isScalarIntegerLike(base) && isScalarIntegerLike(exponent)) {
        auto baseValue = static_cast<double>(getScalarIntegerLikeValue(base));

        auto exponentValue =
            static_cast<double>(getScalarIntegerLikeValue(exponent));

        return getAttr(resultType, std::pow(baseValue, exponentValue));
      }

      if (isScalarFloatLike(base) && isScalarFloatLike(exponent)) {
        double baseValue = getScalarFloatLikeValue(base);
        double exponentValue = getScalarFloatLikeValue(exponent);
        return getAttr(resultType, std::pow(baseValue, exponentValue));
      }

      if (isScalarIntegerLike(base) && isScalarFloatLike(exponent)) {
        auto baseValue = static_cast<double>(getScalarIntegerLikeValue(base));
        double exponentValue = getScalarFloatLikeValue(exponent);
        return getAttr(resultType, std::pow(baseValue, exponentValue));
      }

      if (isScalarFloatLike(base) && isScalarIntegerLike(exponent)) {
        double baseValue = getScalarFloatLikeValue(base);

        auto exponentValue =
            static_cast<double>(getScalarIntegerLikeValue(exponent));

        return getAttr(resultType, std::pow(baseValue, exponentValue));
      }
    }

    return {};
  }

  void PowEWOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getBase().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getBase(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getExponent().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getExponent(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange PowEWOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    // D[x ^ y] = (x ^ y) * (y' * ln(x) + (y * x') / x)

    mlir::Location loc = getLoc();

    mlir::Value derivedBase = derivatives.lookup(getBase());
    mlir::Value derivedExponent = derivatives.lookup(getExponent());

    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value pow = builder.create<PowEWOp>(
        loc, type, getBase(), getExponent());

    mlir::Value ln = builder.create<LogOp>(loc, type, getBase());

    mlir::Value firstOperand = builder.create<MulEWOp>(
        loc, type, derivedExponent, ln);

    mlir::Value numerator = builder.create<MulEWOp>(
        loc, type, getExponent(), derivedBase);

    mlir::Value secondOperand = builder.create<DivEWOp>(
        loc, type, numerator, getBase());

    mlir::Value sum = builder.create<AddEWOp>(
        loc, type, firstOperand, secondOperand);

    auto derivedOp = builder.create<MulEWOp>(loc, type, pow, sum);

    return derivedOp->getResults();
  }

  void PowEWOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getBase());
    toBeDerived.push_back(getExponent());
  }

  void PowEWOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// Comparison operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// EqOp

namespace mlir::modelica
{
  mlir::OpFoldResult EqOp::fold(FoldAdaptor adaptor)
  {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        int64_t lhsValue = getScalarIntegerLikeValue(lhs);
        int64_t rhsValue = getScalarIntegerLikeValue(rhs);
        return getAttr(resultType, static_cast<int64_t>(lhsValue == rhsValue));
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, static_cast<int64_t>(lhsValue == rhsValue));
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, static_cast<int64_t>(lhsValue == rhsValue));
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
        return getAttr(resultType, static_cast<int64_t>(lhsValue == rhsValue));
      }
    }

    return {};
  }
}

//===---------------------------------------------------------------------===//
// NotEqOp

namespace mlir::modelica
{
  mlir::OpFoldResult NotEqOp::fold(FoldAdaptor adaptor)
  {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        int64_t lhsValue = getScalarIntegerLikeValue(lhs);
        int64_t rhsValue = getScalarIntegerLikeValue(rhs);
        return getAttr(resultType, static_cast<int64_t>(lhsValue != rhsValue));
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, static_cast<int64_t>(lhsValue != rhsValue));
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, static_cast<int64_t>(lhsValue != rhsValue));
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
        return getAttr(resultType, static_cast<int64_t>(lhsValue != rhsValue));
      }
    }

    return {};
  }
}

//===---------------------------------------------------------------------===//
// GtOp

namespace mlir::modelica
{
  mlir::OpFoldResult GtOp::fold(FoldAdaptor adaptor)
  {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        int64_t lhsValue = getScalarIntegerLikeValue(lhs);
        int64_t rhsValue = getScalarIntegerLikeValue(rhs);
        return getAttr(resultType, static_cast<int64_t>(lhsValue > rhsValue));
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, static_cast<int64_t>(lhsValue > rhsValue));
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, static_cast<int64_t>(lhsValue > rhsValue));
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
        return getAttr(resultType, static_cast<int64_t>(lhsValue > rhsValue));
      }
    }

    return {};
  }
}

//===---------------------------------------------------------------------===//
// GteOp

namespace mlir::modelica
{
  mlir::OpFoldResult GteOp::fold(FoldAdaptor adaptor)
  {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        int64_t lhsValue = getScalarIntegerLikeValue(lhs);
        int64_t rhsValue = getScalarIntegerLikeValue(rhs);
        return getAttr(resultType, static_cast<int64_t>(lhsValue >= rhsValue));
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, static_cast<int64_t>(lhsValue >= rhsValue));
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, static_cast<int64_t>(lhsValue >= rhsValue));
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
        return getAttr(resultType, static_cast<int64_t>(lhsValue >= rhsValue));
      }
    }

    return {};
  }
}

//===---------------------------------------------------------------------===//
// LtOp

namespace mlir::modelica
{
  mlir::OpFoldResult LtOp::fold(FoldAdaptor adaptor)
  {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        int64_t lhsValue = getScalarIntegerLikeValue(lhs);
        int64_t rhsValue = getScalarIntegerLikeValue(rhs);
        return getAttr(resultType, static_cast<int64_t>(lhsValue < rhsValue));
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, static_cast<int64_t>(lhsValue < rhsValue));
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, static_cast<int64_t>(lhsValue < rhsValue));
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
        return getAttr(resultType, static_cast<int64_t>(lhsValue < rhsValue));
      }
    }

    return {};
  }
}

//===---------------------------------------------------------------------===//
// LteOp

namespace mlir::modelica
{
  mlir::OpFoldResult LteOp::fold(FoldAdaptor adaptor)
  {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        int64_t lhsValue = getScalarIntegerLikeValue(lhs);
        int64_t rhsValue = getScalarIntegerLikeValue(rhs);
        return getAttr(resultType, static_cast<int64_t>(lhsValue <= rhsValue));
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, static_cast<int64_t>(lhsValue <= rhsValue));
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
        double rhsValue = getScalarFloatLikeValue(rhs);
        return getAttr(resultType, static_cast<int64_t>(lhsValue <= rhsValue));
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
        return getAttr(resultType, static_cast<int64_t>(lhsValue <= rhsValue));
      }
    }

    return {};
  }
}

//===---------------------------------------------------------------------===//
// Logic operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// NotOp

namespace mlir::modelica
{
  mlir::OpFoldResult NotOp::fold(FoldAdaptor adaptor)
  {
    auto operand = adaptor.getOperand();

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        int64_t operandValue = getScalarIntegerLikeValue(operand);
        return getAttr(resultType, static_cast<int64_t>(operandValue == 0));
      }

      if (isScalarFloatLike(operand)) {
        double operandValue = getScalarFloatLikeValue(operand);
        return getAttr(resultType, static_cast<int64_t>(operandValue == 0));
      }
    }

    return {};
  }

  void NotOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getOperand(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }
}

//===---------------------------------------------------------------------===//
// AndOp

namespace mlir::modelica
{
  mlir::OpFoldResult AndOp::fold(FoldAdaptor adaptor)
  {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        int64_t lhsValue = getScalarIntegerLikeValue(lhs);
        int64_t rhsValue = getScalarIntegerLikeValue(rhs);

        return getAttr(
            resultType,
            static_cast<int64_t>(lhsValue != 0 && rhsValue != 0));
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        double rhsValue = getScalarFloatLikeValue(rhs);

        return getAttr(
            resultType,
            static_cast<int64_t>(lhsValue != 0 && rhsValue != 0));
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        int64_t lhsValue = getScalarIntegerLikeValue(lhs);
        double rhsValue = getScalarFloatLikeValue(rhs);

        return getAttr(
            resultType,
            static_cast<int64_t>(lhsValue != 0 && rhsValue != 0));
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        int64_t rhsValue = getScalarIntegerLikeValue(rhs);

        return getAttr(
            resultType,
            static_cast<int64_t>(lhsValue != 0 && rhsValue != 0));
      }
    }

    return {};
  }

  void AndOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getLhs().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getLhs(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getRhs().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getRhs(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }
}

//===---------------------------------------------------------------------===//
// OrOp

namespace mlir::modelica
{
  mlir::OpFoldResult OrOp::fold(FoldAdaptor adaptor)
  {
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();

    if (!lhs || !rhs) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(lhs) && isScalar(rhs)) {
      if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
        int64_t lhsValue = getScalarIntegerLikeValue(lhs);
        int64_t rhsValue = getScalarIntegerLikeValue(rhs);

        return getAttr(
            resultType,
            static_cast<int64_t>(lhsValue != 0 || rhsValue != 0));
      }

      if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        double rhsValue = getScalarFloatLikeValue(rhs);

        return getAttr(
            resultType,
            static_cast<int64_t>(lhsValue != 0 || rhsValue != 0));
      }

      if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
        int64_t lhsValue = getScalarIntegerLikeValue(lhs);
        double rhsValue = getScalarFloatLikeValue(rhs);

        return getAttr(
            resultType,
            static_cast<int64_t>(lhsValue != 0 || rhsValue != 0));
      }

      if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
        double lhsValue = getScalarFloatLikeValue(lhs);
        int64_t rhsValue = getScalarIntegerLikeValue(rhs);

        return getAttr(
            resultType,
            static_cast<int64_t>(lhsValue != 0 || rhsValue != 0));
      }
    }

    return {};
  }

  void OrOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getLhs().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getLhs(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getRhs().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getRhs(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }
}

//===---------------------------------------------------------------------===//
// SelectOp

namespace mlir::modelica
{
  mlir::ParseResult SelectOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::OpAsmParser::UnresolvedOperand condition;
    mlir::Type conditionType;

    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 1> trueValues;
    llvm::SmallVector<mlir::Type, 1> trueValuesTypes;

    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 1> falseValues;
    llvm::SmallVector<mlir::Type, 1> falseValuesTypes;

    llvm::SmallVector<mlir::Type, 1> resultTypes;

    if (parser.parseLParen() ||
        parser.parseOperand(condition) ||
        parser.parseColonType(conditionType) ||
        parser.parseRParen() ||
        parser.resolveOperand(condition, conditionType, result.operands)) {
      return mlir::failure();
    }

    if (parser.parseComma()) {
      return mlir::failure();
    }

    auto trueValuesLoc = parser.getCurrentLocation();

    if (parser.parseLParen() ||
        parser.parseOperandList(trueValues) ||
        parser.parseColonTypeList(trueValuesTypes) ||
        parser.parseRParen() ||
        parser.resolveOperands(
            trueValues, trueValuesTypes, trueValuesLoc, result.operands)) {
      return mlir::failure();
    }

    if (parser.parseComma()) {
      return mlir::failure();
    }

    auto falseValuesLoc = parser.getCurrentLocation();

    if (parser.parseLParen() ||
        parser.parseOperandList(falseValues) ||
        parser.parseColonTypeList(falseValuesTypes) ||
        parser.parseRParen() ||
        parser.resolveOperands(
            falseValues, falseValuesTypes, falseValuesLoc, result.operands)) {
      return mlir::failure();
    }

    if (parser.parseArrowTypeList(resultTypes)) {
      return mlir::failure();
    }

    result.addTypes(resultTypes);
    return mlir::success();
  }

  void SelectOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " ";
    printer << "(" << getCondition() << " : "
            << getCondition().getType()  << ")";

    printer << ", ";

    printer << "(" <<  getTrueValues() << " : "
            << getTrueValues().getTypes() << ")";

    printer << ", ";

    printer << "(" <<  getFalseValues() << " : "
            << getFalseValues().getTypes() << ")";

    printer.printOptionalAttrDict(getOperation()->getAttrs());
    printer << " -> ";

    auto resultTypes = getResultTypes();

    if (resultTypes.size() == 1) {
      printer << resultTypes;
    } else {
      printer << "(" << resultTypes << ")";
    }
  }
}

//===---------------------------------------------------------------------===//
// Built-in operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// AbsOp

namespace mlir::modelica
{
  mlir::OpFoldResult AbsOp::fold(FoldAdaptor adaptor)
  {
    auto operand = adaptor.getOperand();

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        int64_t operandValue = getScalarIntegerLikeValue(operand);
        return getAttr(resultType, std::abs(operandValue));
      }

      if (isScalarFloatLike(operand)) {
        double operandValue = getScalarFloatLikeValue(operand);
        return getAttr(resultType, std::abs(operandValue));
      }
    }

    return {};
  }

  void AbsOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getOperand(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange AbsOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int AbsOp::getArgExpectedRank(
      unsigned int argIndex, mlir::SymbolTableCollection& symbolTable)
  {
    return 0;
  }

  mlir::ValueRange AbsOp::scalarize(
      mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType =
        getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(
        getLoc(), getOperand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<AbsOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }
}

//===---------------------------------------------------------------------===//
// AcosOp

namespace mlir::modelica
{
  mlir::OpFoldResult AcosOp::fold(FoldAdaptor adaptor)
  {
    auto operand = adaptor.getOperand();

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        auto operandValue =
            static_cast<double>(getScalarIntegerLikeValue(operand));

        return getAttr(resultType, std::acos(operandValue));
      }

      if (isScalarFloatLike(operand)) {
        double operandValue = getScalarFloatLikeValue(operand);
        return getAttr(resultType, std::acos(operandValue));
      }
    }

    return {};
  }

  void AcosOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getOperand(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange AcosOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int AcosOp::getArgExpectedRank(
      unsigned int argIndex, mlir::SymbolTableCollection& symbolTable)
  {
    return 0;
  }

  mlir::ValueRange AcosOp::scalarize(
      mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType =
        getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(
        getLoc(), getOperand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<AcosOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange AcosOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    // D[acos(x)] = -x' / sqrt(1 - x^2)

    mlir::Location loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(getOperand());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value one = builder.create<ConstantOp>(
        loc, RealAttr::get(getContext(), 1));

    mlir::Value two = builder.create<ConstantOp>(
        loc, RealAttr::get(getContext(), 2));

    mlir::Value argSquared = builder.create<PowEWOp>(
        loc, type, getOperand(), two);

    mlir::Value sub = builder.create<SubEWOp>(loc, type, one, argSquared);
    mlir::Value denominator = builder.create<SqrtOp>(loc, type, sub);

    mlir::Value div = builder.create<DivEWOp>(
        loc, type, derivedOperand, denominator);

    auto derivedOp = builder.create<NegateOp>(loc, type, div);

    return derivedOp->getResults();
  }

  void AcosOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
  }

  void AcosOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// AsinOp

namespace mlir::modelica
{
  mlir::OpFoldResult AsinOp::fold(FoldAdaptor adaptor)
  {
    auto operand = adaptor.getOperand();

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        auto operandValue =
            static_cast<double>(getScalarIntegerLikeValue(operand));

        return getAttr(resultType, std::asin(operandValue));
      }

      if (isScalarFloatLike(operand)) {
        double operandValue = getScalarFloatLikeValue(operand);
        return getAttr(resultType, std::asin(operandValue));
      }
    }

    return {};
  }

  void AsinOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getOperand(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange AsinOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int AsinOp::getArgExpectedRank(
      unsigned int argIndex, mlir::SymbolTableCollection& symbolTable)
  {
    return 0;
  }

  mlir::ValueRange AsinOp::scalarize(
      mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType =
        getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(
        getLoc(), getOperand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<AsinOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange AsinOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    // D[arcsin(x)] = x' / sqrt(1 - x^2)

    mlir::Location loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(getOperand());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value one = builder.create<ConstantOp>(
        loc, RealAttr::get(getContext(), 1));

    mlir::Value two = builder.create<ConstantOp>(
        loc, RealAttr::get(getContext(), 2));

    mlir::Value argSquared = builder.create<PowEWOp>(
        loc, type, getOperand(), two);

    mlir::Value sub = builder.create<SubEWOp>(loc, type, one, argSquared);
    mlir::Value denominator = builder.create<SqrtOp>(loc, type, sub);

    auto derivedOp = builder.create<DivEWOp>(
        loc, type, derivedOperand, denominator);

    return derivedOp->getResults();
  }

  void AsinOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
  }

  void AsinOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// AtanOp

namespace mlir::modelica
{
  mlir::OpFoldResult AtanOp::fold(FoldAdaptor adaptor)
  {
    auto operand = adaptor.getOperand();

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        auto operandValue =
            static_cast<double>(getScalarIntegerLikeValue(operand));

        return getAttr(resultType, std::atan(operandValue));
      }

      if (isScalarFloatLike(operand)) {
        double operandValue = getScalarFloatLikeValue(operand);
        return getAttr(resultType, std::atan(operandValue));
      }
    }

    return {};
  }

  void AtanOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getOperand(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange AtanOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int AtanOp::getArgExpectedRank(
      unsigned int argIndex, mlir::SymbolTableCollection& symbolTable)
  {
    return 0;
  }

  mlir::ValueRange AtanOp::scalarize(
      mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType =
        getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(
        getLoc(), getOperand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<AtanOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange AtanOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    // D[atan(x)] = x' / (1 + x^2)

    mlir::Location loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(getOperand());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value one = builder.create<ConstantOp>(
        loc, RealAttr::get(getContext(), 1));

    mlir::Value two = builder.create<ConstantOp>(
        loc, RealAttr::get(getContext(), 2));

    mlir::Value argSquared = builder.create<PowEWOp>(
        loc, type, getOperand(), two);

    mlir::Value denominator = builder.create<AddEWOp>(
        loc, type, one, argSquared);

    auto derivedOp = builder.create<DivEWOp>(
        loc, type, derivedOperand, denominator);

    return derivedOp->getResults();
  }

  void AtanOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
  }

  void AtanOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// Atan2Op

namespace mlir::modelica
{
  void Atan2Op::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getY().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getY(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getX().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getX(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange Atan2Op::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int Atan2Op::getArgExpectedRank(
      unsigned int argIndex, mlir::SymbolTableCollection& symbolTable)
  {
    return 0;
  }

  mlir::ValueRange Atan2Op::scalarize(
      mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType =
        getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newY = builder.create<SubscriptionOp>(
        getLoc(), getY(), indexes);

    if (auto arrayType = newY.getType().dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newY = builder.create<LoadOp>(getLoc(), newY);
    }

    mlir::Value newX = builder.create<SubscriptionOp>(
        getLoc(), getX(), indexes);

    if (auto arrayType = newX.getType().dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newX = builder.create<LoadOp>(getLoc(), newX);
    }

    auto op = builder.create<Atan2Op>(getLoc(), newResultType, newY, newX);
    return op->getResults();
  }

  mlir::ValueRange Atan2Op::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    // D[atan2(y, x)] = (y' * x - y * x') / (y^2 + x^2)

    mlir::Location loc = getLoc();

    mlir::Value derivedY = derivatives.lookup(getY());
    mlir::Value derivedX = derivatives.lookup(getX());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value firstMul = builder.create<MulEWOp>(
        loc, type, derivedY, getX());

    mlir::Value secondMul = builder.create<MulEWOp>(
        loc, type, getY(), derivedX);

    mlir::Value numerator = builder.create<SubEWOp>(
        loc, type, firstMul, secondMul);

    mlir::Value firstSquared = builder.create<MulEWOp>(
        loc, type, getY(), getY());

    mlir::Value secondSquared = builder.create<MulEWOp>(
        loc, type, getX(), getX());

    mlir::Value denominator = builder.create<AddEWOp>(
        loc, type, firstSquared, secondSquared);

    auto derivedOp = builder.create<DivEWOp>(
        loc, type, numerator, denominator);

    return derivedOp->getResults();
  }

  void Atan2Op::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    llvm_unreachable("Not implemented");
  }

  void Atan2Op::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// CeilOp

namespace mlir::modelica
{
  mlir::OpFoldResult CeilOp::fold(FoldAdaptor adaptor)
  {
    auto operand = adaptor.getOperand();

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        int64_t operandValue = getScalarIntegerLikeValue(operand);
        return getAttr(resultType, operandValue);
      }

      if (isScalarFloatLike(operand)) {
        double operandValue = getScalarFloatLikeValue(operand);
        return getAttr(resultType, std::ceil(operandValue));
      }
    }

    return {};
  }

  void CeilOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getOperand(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange CeilOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int CeilOp::getArgExpectedRank(
      unsigned int argIndex, mlir::SymbolTableCollection& symbolTable)
  {
    return 0;
  }

  mlir::ValueRange CeilOp::scalarize(
      mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType =
        getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(
        getLoc(), getOperand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<CeilOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }
}

//===---------------------------------------------------------------------===//
// CosOp

namespace mlir::modelica
{
  mlir::OpFoldResult CosOp::fold(FoldAdaptor adaptor)
  {
    auto operand = adaptor.getOperand();

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        auto operandValue =
            static_cast<double>(getScalarIntegerLikeValue(operand));

        return getAttr(resultType, std::cos(operandValue));
      }

      if (isScalarFloatLike(operand)) {
        double operandValue = getScalarFloatLikeValue(operand);
        return getAttr(resultType, std::cos(operandValue));
      }
    }

    return {};
  }

  void CosOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getOperand(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange CosOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int CosOp::getArgExpectedRank(
      unsigned int argIndex, mlir::SymbolTableCollection& symbolTable)
  {
    return 0;
  }

  mlir::ValueRange CosOp::scalarize(
      mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType =
        getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(
        getLoc(), getOperand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<CosOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange CosOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    // D[cos(x)] = -x' * sin(x)

    mlir::Location loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(getOperand());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value sin = builder.create<SinOp>(loc, type, getOperand());
    mlir::Value negatedSin = builder.create<NegateOp>(loc, type, sin);

    auto derivedOp = builder.create<MulEWOp>(
        loc, type, negatedSin, derivedOperand);

    return derivedOp->getResults();
  }

  void CosOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
  }

  void CosOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// CoshOp

namespace mlir::modelica
{
  mlir::OpFoldResult CoshOp::fold(FoldAdaptor adaptor)
  {
    auto operand = adaptor.getOperand();

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        auto operandValue =
            static_cast<double>(getScalarIntegerLikeValue(operand));

        return getAttr(resultType, std::cosh(operandValue));
      }

      if (isScalarFloatLike(operand)) {
        double operandValue = getScalarFloatLikeValue(operand);
        return getAttr(resultType, std::cosh(operandValue));
      }
    }

    return {};
  }

  void CoshOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getOperand(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange CoshOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int CoshOp::getArgExpectedRank(
      unsigned int argIndex, mlir::SymbolTableCollection& symbolTable)
  {
    return 0;
  }

  mlir::ValueRange CoshOp::scalarize(
      mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType =
        getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(
        getLoc(), getOperand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<CoshOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange CoshOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    // D[cosh(x)] = x' * sinh(x)

    mlir::Location loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(getOperand());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value sinh = builder.create<SinhOp>(loc, type, getOperand());
    auto derivedOp = builder.create<MulEWOp>(loc, type, sinh, derivedOperand);

    return derivedOp->getResults();
  }

  void CoshOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
  }

  void CoshOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// DiagonalOp

namespace mlir::modelica
{
  void DiagonalOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(
        mlir::MemoryEffects::Read::get(),
        getValues(),
        mlir::SideEffects::DefaultResource::get());

    effects.emplace_back(
        mlir::MemoryEffects::Allocate::get(),
        getResult(),
        mlir::SideEffects::DefaultResource::get());

    effects.emplace_back(
        mlir::MemoryEffects::Write::get(),
        getResult(),
        mlir::SideEffects::DefaultResource::get());
  }
}

//===---------------------------------------------------------------------===//
// DivTruncOp

namespace mlir::modelica
{
  mlir::OpFoldResult DivTruncOp::fold(FoldAdaptor adaptor)
  {
    auto x = adaptor.getX();
    auto y = adaptor.getY();

    if (!x || !y) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(x) && isScalar(y)) {
      if (isScalarIntegerLike(x) && isScalarIntegerLike(y)) {
        auto xValue = static_cast<double>(getScalarIntegerLikeValue(x));
        auto yValue = static_cast<double>(getScalarIntegerLikeValue(y));
        return getAttr(resultType, xValue / yValue);
      }

      if (isScalarFloatLike(x) && isScalarFloatLike(y)) {
        double xValue = getScalarFloatLikeValue(x);
        double yValue = getScalarFloatLikeValue(y);
        return getAttr(resultType, std::trunc(xValue / yValue));
      }

      if (isScalarIntegerLike(x) && isScalarFloatLike(y)) {
        auto xValue = static_cast<double>(getScalarIntegerLikeValue(x));
        double yValue = getScalarFloatLikeValue(y);
        return getAttr(resultType, std::trunc(xValue / yValue));
      }

      if (isScalarFloatLike(x) && isScalarIntegerLike(y)) {
        double xValue = getScalarFloatLikeValue(x);
        auto yValue = static_cast<double>(getScalarIntegerLikeValue(y));
        return getAttr(resultType, std::trunc(xValue / yValue));
      }
    }

    return {};
  }
}

//===---------------------------------------------------------------------===//
// ExpOp

namespace mlir::modelica
{
  mlir::OpFoldResult ExpOp::fold(FoldAdaptor adaptor)
  {
    auto operand = adaptor.getExponent();

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        auto operandValue =
            static_cast<double>(getScalarIntegerLikeValue(operand));

        return getAttr(resultType, std::exp(operandValue));
      }

      if (isScalarFloatLike(operand)) {
        double operandValue = getScalarFloatLikeValue(operand);
        return getAttr(resultType, std::exp(operandValue));
      }
    }

    return {};
  }

  void ExpOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getExponent().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getExponent(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange ExpOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int ExpOp::getArgExpectedRank(
      unsigned int argIndex, mlir::SymbolTableCollection& symbolTable)
  {
    return 0;
  }

  mlir::ValueRange ExpOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType =
        getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(
        getLoc(), getExponent(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<ExpOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange ExpOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    // D[e^x] = x' * e^x

    mlir::Location loc = getLoc();
    mlir::Value derivedExponent = derivatives.lookup(getExponent());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value pow = builder.create<ExpOp>(loc, type, getExponent());
    auto derivedOp = builder.create<MulEWOp>(loc, type, pow, derivedExponent);

    return derivedOp->getResults();
  }

  void ExpOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getExponent());
  }

  void ExpOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// FillOp

namespace mlir::modelica
{
  void FillOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(
        mlir::MemoryEffects::Allocate::get(),
        getResult(),
        mlir::SideEffects::DefaultResource::get());

    effects.emplace_back(
        mlir::MemoryEffects::Write::get(),
        getResult(),
        mlir::SideEffects::DefaultResource::get());
  }
}

//===---------------------------------------------------------------------===//
// FloorOp

namespace mlir::modelica
{
  mlir::OpFoldResult FloorOp::fold(FoldAdaptor adaptor)
  {
    auto operand = adaptor.getOperand();

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        int64_t operandValue = getScalarIntegerLikeValue(operand);
        return getAttr(resultType, operandValue);
      }

      if (isScalarFloatLike(operand)) {
        double operandValue = getScalarFloatLikeValue(operand);
        return getAttr(resultType, std::floor(operandValue));
      }
    }

    return {};
  }

  void FloorOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getOperand(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange FloorOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int FloorOp::getArgExpectedRank(
      unsigned int argIndex, mlir::SymbolTableCollection& symbolTable)
  {
    return 0;
  }

  mlir::ValueRange FloorOp::scalarize(
      mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType =
        getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(
        getLoc(), getOperand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<FloorOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }
}

//===---------------------------------------------------------------------===//
// IdentityOp

namespace mlir::modelica
{
  void IdentityOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(
        mlir::MemoryEffects::Allocate::get(),
        getResult(),
        mlir::SideEffects::DefaultResource::get());

    effects.emplace_back(
        mlir::MemoryEffects::Write::get(),
        getResult(),
        mlir::SideEffects::DefaultResource::get());
  }
}

//===---------------------------------------------------------------------===//
// IntegerOp

namespace mlir::modelica
{
  mlir::OpFoldResult IntegerOp::fold(FoldAdaptor adaptor)
  {
    auto operand = adaptor.getOperand();

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        int64_t operandValue = getScalarIntegerLikeValue(operand);
        return getAttr(resultType, operandValue);
      }

      if (isScalarFloatLike(operand)) {
        double operandValue = getScalarFloatLikeValue(operand);
        return getAttr(resultType, std::floor(operandValue));
      }
    }

    return {};
  }

  void IntegerOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getOperand(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange IntegerOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int IntegerOp::getArgExpectedRank(
      unsigned int argIndex, mlir::SymbolTableCollection& symbolTable)
  {
    return 0;
  }

  mlir::ValueRange IntegerOp::scalarize(
      mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType =
        getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(
        getLoc(), getOperand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<IntegerOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }
}

//===---------------------------------------------------------------------===//
// LinspaceOp

namespace mlir::modelica
{
  void LinspaceOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(
        mlir::MemoryEffects::Allocate::get(),
        getResult(),
        mlir::SideEffects::DefaultResource::get());

    effects.emplace_back(
        mlir::MemoryEffects::Write::get(),
        getResult(),
        mlir::SideEffects::DefaultResource::get());
  }
}

//===---------------------------------------------------------------------===//
// LogOp

namespace mlir::modelica
{
  mlir::OpFoldResult LogOp::fold(FoldAdaptor adaptor)
  {
    auto operand = adaptor.getOperand();

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        auto operandValue =
            static_cast<double>(getScalarIntegerLikeValue(operand));

        return getAttr(resultType, std::log(operandValue));
      }

      if (isScalarFloatLike(operand)) {
        double operandValue = getScalarFloatLikeValue(operand);
        return getAttr(resultType, std::log(operandValue));
      }
    }

    return {};
  }

  void LogOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getOperand(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange LogOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int LogOp::getArgExpectedRank(
      unsigned int argIndex, mlir::SymbolTableCollection& symbolTable)
  {
    return 0;
  }

  mlir::ValueRange LogOp::scalarize(
      mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType =
        getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(
        getLoc(), getOperand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<LogOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange LogOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    // D[ln(x)] = x' / x

    mlir::Value derivedOperand = derivatives.lookup(getOperand());

    auto derivedOp = builder.create<DivEWOp>(
        getLoc(),
        convertToRealType(
            getResult().getType()), derivedOperand, getOperand());

    return derivedOp->getResults();
  }

  void LogOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
  }

  void LogOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// Log10Op

namespace mlir::modelica
{
  mlir::OpFoldResult Log10Op::fold(FoldAdaptor adaptor)
  {
    auto operand = adaptor.getOperand();

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        auto operandValue =
            static_cast<double>(getScalarIntegerLikeValue(operand));

        return getAttr(resultType, std::log10(operandValue));
      }

      if (isScalarFloatLike(operand)) {
        double operandValue = getScalarFloatLikeValue(operand);
        return getAttr(resultType, std::log10(operandValue));
      }
    }

    return {};
  }

  void Log10Op::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getOperand(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange Log10Op::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int Log10Op::getArgExpectedRank(
      unsigned int argIndex, mlir::SymbolTableCollection& symbolTable)
  {
    return 0;
  }

  mlir::ValueRange Log10Op::scalarize(
      mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType =
        getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(
        getLoc(), getOperand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<Log10Op>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange Log10Op::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    // D[log10(x)] = x' / (x * ln(10))

    mlir::Location loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(getOperand());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value ten = builder.create<ConstantOp>(
        loc, RealAttr::get(getContext(), 10));

    mlir::Value log = builder.create<LogOp>(
        loc, RealType::get(getContext()), ten);

    mlir::Value mul = builder.create<MulEWOp>(loc, type, getOperand(), log);

    auto derivedOp = builder.create<DivEWOp>(
        loc, getResult().getType(), derivedOperand, mul);

    return derivedOp->getResults();
  }

  void Log10Op::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
  }

  void Log10Op::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// MaxOp

namespace mlir::modelica
{
  mlir::ParseResult MaxOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::OpAsmParser::UnresolvedOperand first;
    mlir::Type firstType;

    mlir::OpAsmParser::UnresolvedOperand second;
    mlir::Type secondType;

    size_t numOperands = 1;

    if (parser.parseOperand(first)) {
      return mlir::failure();
    }

    if (mlir::succeeded(parser.parseOptionalComma())) {
      numOperands = 2;

      if (parser.parseOperand(second)) {
        return mlir::failure();
      }
    }

    if (parser.parseOptionalAttrDict(result.attributes)) {
      return mlir::failure();
    }

    if (parser.parseColon()) {
      return mlir::failure();
    }

    if (numOperands == 1) {
      if (parser.parseType(firstType) ||
          parser.resolveOperand(first, firstType, result.operands)) {
        return mlir::failure();
      }
    } else {
      if (parser.parseLParen() ||
          parser.parseType(firstType) ||
          parser.resolveOperand(first, firstType, result.operands) ||
          parser.parseComma() ||
          parser.parseType(secondType) ||
          parser.resolveOperand(second, secondType, result.operands) ||
          parser.parseRParen()) {
        return mlir::failure();
      }
    }

    mlir::Type resultType;

    if (parser.parseArrow() ||
        parser.parseType(resultType)) {
      return mlir::failure();
    }

    result.addTypes(resultType);

    return mlir::success();
  }

  void MaxOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << getFirst();

    if (getOperation()->getNumOperands() == 2) {
      printer << ", " << getSecond();
    }

    printer.printOptionalAttrDict(getOperation()->getAttrs());
    printer << " : ";

    if (getOperation()->getNumOperands() == 1) {
      printer << getFirst().getType();
    } else {
      printer << "(" << getFirst().getType() << ", "
              << getSecond().getType() << ")";
    }

    printer << " -> " << getResult().getType();
  }

  mlir::OpFoldResult MaxOp::fold(FoldAdaptor adaptor)
  {
    if (adaptor.getOperands().size() == 2) {
      auto first = adaptor.getFirst();
      auto second = adaptor.getSecond();

      if (!first || !second) {
        return {};
      }

      auto resultType = getResult().getType();

      if (isScalar(first) && isScalar(second)) {
        if (isScalarIntegerLike(first) && isScalarIntegerLike(second)) {
          int64_t firstValue = getScalarIntegerLikeValue(first);
          int64_t secondValue = getScalarIntegerLikeValue(second);
          return getAttr(resultType, std::max(firstValue, secondValue));
        }

        if (isScalarFloatLike(first) && isScalarFloatLike(second)) {
          double firstValue = getScalarFloatLikeValue(first);
          double secondValue = getScalarFloatLikeValue(second);
          return getAttr(resultType, std::max(firstValue, secondValue));
        }

        if (isScalarIntegerLike(first) && isScalarFloatLike(second)) {
          auto firstValue =
              static_cast<double>(getScalarIntegerLikeValue(first));

          double secondValue = getScalarFloatLikeValue(second);

          if (firstValue >= secondValue) {
            return getAttr(resultType, firstValue);
          } else {
            return getAttr(resultType, secondValue);
          }
        }

        if (isScalarFloatLike(first) && isScalarIntegerLike(second)) {
          double firstValue = getScalarFloatLikeValue(first);

          auto secondValue =
              static_cast<double>(getScalarIntegerLikeValue(second));

          if (firstValue >= secondValue) {
            return getAttr(resultType, firstValue);
          } else {
            return getAttr(resultType, secondValue);
          }
        }
      }
    }

    return {};
  }

  void MaxOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getFirst().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getFirst(),
          mlir::SideEffects::DefaultResource::get());
    }
  }
}

//===---------------------------------------------------------------------===//
// MinOp

namespace mlir::modelica
{
  mlir::ParseResult MinOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::OpAsmParser::UnresolvedOperand first;
    mlir::Type firstType;

    mlir::OpAsmParser::UnresolvedOperand second;
    mlir::Type secondType;

    size_t numOperands = 1;

    if (parser.parseOperand(first)) {
      return mlir::failure();
    }

    if (mlir::succeeded(parser.parseOptionalComma())) {
      numOperands = 2;

      if (parser.parseOperand(second)) {
        return mlir::failure();
      }
    }

    if (parser.parseOptionalAttrDict(result.attributes)) {
      return mlir::failure();
    }

    if (parser.parseColon()) {
      return mlir::failure();
    }

    if (numOperands == 1) {
      if (parser.parseType(firstType) ||
          parser.resolveOperand(first, firstType, result.operands)) {
        return mlir::failure();
      }
    } else {
      if (parser.parseLParen() ||
          parser.parseType(firstType) ||
          parser.resolveOperand(first, firstType, result.operands) ||
          parser.parseComma() ||
          parser.parseType(secondType) ||
          parser.resolveOperand(second, secondType, result.operands) ||
          parser.parseRParen()) {
        return mlir::failure();
      }
    }

    mlir::Type resultType;

    if (parser.parseArrow() ||
        parser.parseType(resultType)) {
      return mlir::failure();
    }

    result.addTypes(resultType);

    return mlir::success();
  }

  void MinOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << getFirst();

    if (getOperation()->getNumOperands() == 2) {
      printer << ", " << getSecond();
    }

    printer.printOptionalAttrDict(getOperation()->getAttrs());
    printer << " : ";

    if (getOperation()->getNumOperands() == 1) {
      printer << getFirst().getType();
    } else {
      printer << "(" << getFirst().getType() << ", "
              << getSecond().getType() << ")";
    }

    printer << " -> " << getResult().getType();
  }

  mlir::OpFoldResult MinOp::fold(FoldAdaptor adaptor)
  {
    if (adaptor.getOperands().size() == 2) {
      auto first = adaptor.getFirst();
      auto second = adaptor.getSecond();

      if (!first || !second) {
        return {};
      }

      auto resultType = getResult().getType();

      if (isScalar(first) && isScalar(second)) {
        if (isScalarIntegerLike(first) && isScalarIntegerLike(second)) {
          int64_t firstValue = getScalarIntegerLikeValue(first);
          int64_t secondValue = getScalarIntegerLikeValue(second);
          return getAttr(resultType, std::min(firstValue, secondValue));
        }

        if (isScalarFloatLike(first) && isScalarFloatLike(second)) {
          double firstValue = getScalarFloatLikeValue(first);
          double secondValue = getScalarFloatLikeValue(second);
          return getAttr(resultType, std::min(firstValue, secondValue));
        }

        if (isScalarIntegerLike(first) && isScalarFloatLike(second)) {
          auto firstValue =
              static_cast<double>(getScalarIntegerLikeValue(first));

          double secondValue = getScalarFloatLikeValue(second);

          if (firstValue <= secondValue) {
            return getAttr(resultType, firstValue);
          } else {
            return getAttr(resultType, secondValue);
          }
        }

        if (isScalarFloatLike(first) && isScalarIntegerLike(second)) {
          double firstValue = getScalarFloatLikeValue(first);

          auto secondValue =
              static_cast<double>(getScalarIntegerLikeValue(second));

          if (firstValue <= secondValue) {
            return getAttr(resultType, firstValue);
          } else {
            return getAttr(resultType, secondValue);
          }
        }
      }
    }

    return {};
  }

  void MinOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getFirst().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getFirst(),
          mlir::SideEffects::DefaultResource::get());
    }
  }
}

//===---------------------------------------------------------------------===//
// ModOp

namespace mlir::modelica
{
  mlir::OpFoldResult ModOp::fold(FoldAdaptor adaptor)
  {
    auto x = adaptor.getX();
    auto y = adaptor.getY();

    if (!x || !y) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(x) && isScalar(y)) {
      if (isScalarIntegerLike(x) && isScalarIntegerLike(y)) {
        auto xValue = static_cast<double>(getScalarIntegerLikeValue(x));
        auto yValue = static_cast<double>(getScalarIntegerLikeValue(y));

        return getAttr(
            resultType,
            xValue - std::floor(xValue / yValue) * yValue);
      }

      if (isScalarFloatLike(x) && isScalarFloatLike(y)) {
        double xValue = getScalarFloatLikeValue(x);
        double yValue = getScalarFloatLikeValue(y);

        return getAttr(
            resultType,
            xValue - std::floor(xValue / yValue) * yValue);
      }

      if (isScalarIntegerLike(x) && isScalarFloatLike(y)) {
        auto xValue = static_cast<double>(getScalarIntegerLikeValue(x));
        double yValue = getScalarFloatLikeValue(y);

        return getAttr(
            resultType,
            xValue - std::floor(xValue / yValue) * yValue);
      }

      if (isScalarFloatLike(x) && isScalarIntegerLike(y)) {
        double xValue = getScalarFloatLikeValue(x);
        auto yValue = static_cast<double>(getScalarIntegerLikeValue(y));

        return getAttr(
            resultType,
            xValue - std::floor(xValue / yValue) * yValue);
      }
    }

    return {};
  }
}

//===---------------------------------------------------------------------===//
// OnesOp

namespace mlir::modelica
{
  void OnesOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(
        mlir::MemoryEffects::Allocate::get(),
        getResult(),
        mlir::SideEffects::DefaultResource::get());

    effects.emplace_back(
        mlir::MemoryEffects::Write::get(),
        getResult(),
        mlir::SideEffects::DefaultResource::get());
  }
}

//===---------------------------------------------------------------------===//
// ProductOp

namespace mlir::modelica
{
  void ProductOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(
        mlir::MemoryEffects::Read::get(),
        getArray(),
        mlir::SideEffects::DefaultResource::get());
  }
}

//===---------------------------------------------------------------------===//
// RemOp

namespace mlir::modelica
{
  mlir::OpFoldResult RemOp::fold(FoldAdaptor adaptor)
  {
    auto x = adaptor.getX();
    auto y = adaptor.getY();

    if (!x || !y) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(x) && isScalar(y)) {
      if (isScalarIntegerLike(x) && isScalarIntegerLike(y)) {
        int64_t xValue = getScalarIntegerLikeValue(x);
        int64_t yValue = getScalarIntegerLikeValue(y);
        return getAttr(resultType, xValue % yValue);
      }

      if (isScalarFloatLike(x) && isScalarFloatLike(y)) {
        double xValue = getScalarFloatLikeValue(x);
        double yValue = getScalarFloatLikeValue(y);
        return getAttr(resultType, std::fmod(xValue, yValue));
      }

      if (isScalarIntegerLike(x) && isScalarFloatLike(y)) {
        auto xValue = static_cast<double>(getScalarIntegerLikeValue(x));
        double yValue = getScalarFloatLikeValue(y);
        return getAttr(resultType, std::fmod(xValue, yValue));
      }

      if (isScalarFloatLike(x) && isScalarIntegerLike(y)) {
        double xValue = getScalarFloatLikeValue(x);
        auto yValue = static_cast<double>(getScalarIntegerLikeValue(y));
        return getAttr(resultType, std::fmod(xValue, yValue));
      }
    }

    return {};
  }
}

//===---------------------------------------------------------------------===//
// SignOp

namespace mlir::modelica
{
  mlir::OpFoldResult SignOp::fold(FoldAdaptor adaptor)
  {
    auto operand = adaptor.getOperand();

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        int64_t value = getScalarIntegerLikeValue(operand);

        if (value == 0) {
          return getAttr(resultType, static_cast<int64_t>(0));
        } else if (value > 0) {
          return getAttr(resultType, static_cast<int64_t>(1));
        } else {
          return getAttr(resultType, static_cast<int64_t>(-1));
        }
      }

      if (isScalarFloatLike(operand)) {
        double value = getScalarFloatLikeValue(operand);

        if (value == 0) {
          return getAttr(resultType, static_cast<int64_t>(0));
        } else if (value > 0) {
          return getAttr(resultType, static_cast<int64_t>(1));
        } else {
          return getAttr(resultType, static_cast<int64_t>(-1));
        }
      }
    }

    return {};
  }

  void SignOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getOperand(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange SignOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int SignOp::getArgExpectedRank(
      unsigned int argIndex, mlir::SymbolTableCollection& symbolTable)
  {
    return 0;
  }

  mlir::ValueRange SignOp::scalarize(
      mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType =
        getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>();
        arrayType.getRank() == 0)
      newResultType = arrayType.getElementType();

    mlir::Value newOperand = builder.create<SubscriptionOp>(
        getLoc(), getOperand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>();
        arrayType.getRank() == 0)
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);

    auto op = builder.create<SignOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }
}

//===---------------------------------------------------------------------===//
// SinOp

namespace mlir::modelica
{
  mlir::OpFoldResult SinOp::fold(FoldAdaptor adaptor)
  {
    auto operand = adaptor.getOperand();

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        auto operandValue =
            static_cast<double>(getScalarIntegerLikeValue(operand));

        return getAttr(resultType, std::sin(operandValue));
      }

      if (isScalarFloatLike(operand)) {
        double operandValue = getScalarFloatLikeValue(operand);
        return getAttr(resultType, std::sin(operandValue));
      }
    }

    return {};
  }

  void SinOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getOperand(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange SinOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int SinOp::getArgExpectedRank(
      unsigned int argIndex, mlir::SymbolTableCollection& symbolTable)
  {
    return 0;
  }

  mlir::ValueRange SinOp::scalarize(
      mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType =
        getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>();
        arrayType.getRank() == 0)
      newResultType = arrayType.getElementType();

    mlir::Value newOperand = builder.create<SubscriptionOp>(
        getLoc(), getOperand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>();
        arrayType.getRank() == 0)
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);

    auto op = builder.create<SinOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange SinOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    // D[sin(x)] = x' * cos(x)

    mlir::Location loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(getOperand());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value cos = builder.create<CosOp>(loc, type, getOperand());
    auto derivedOp = builder.create<MulEWOp>(loc, type, cos, derivedOperand);

    return derivedOp->getResults();
  }

  void SinOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
  }

  void SinOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// SinhOp

namespace mlir::modelica
{
  mlir::OpFoldResult SinhOp::fold(FoldAdaptor adaptor)
  {
    auto operand = adaptor.getOperand();

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        auto operandValue =
            static_cast<double>(getScalarIntegerLikeValue(operand));

        return getAttr(resultType, std::sinh(operandValue));
      }

      if (isScalarFloatLike(operand)) {
        double operandValue = getScalarFloatLikeValue(operand);
        return getAttr(resultType, std::sinh(operandValue));
      }
    }

    return {};
  }

  void SinhOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getOperand(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange SinhOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int SinhOp::getArgExpectedRank(
      unsigned int argIndex, mlir::SymbolTableCollection& symbolTable)
  {
    return 0;
  }

  mlir::ValueRange SinhOp::scalarize(
      mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType =
        getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(
        getLoc(), getOperand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<SinhOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange SinhOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    // D[sinh(x)] = x' * cosh(x)

    mlir::Location loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(getOperand());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value cosh = builder.create<CoshOp>(loc, type, getOperand());
    auto derivedOp = builder.create<MulEWOp>(loc, type, cosh, derivedOperand);

    return derivedOp->getResults();
  }

  void SinhOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
  }

  void SinhOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// SizeOp

namespace mlir::modelica
{
  mlir::ParseResult SizeOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::OpAsmParser::UnresolvedOperand array;
    mlir::Type arrayType;

    mlir::OpAsmParser::UnresolvedOperand dimension;
    mlir::Type dimensionType;

    size_t numOperands = 1;

    if (parser.parseOperand(array)) {
      return mlir::failure();
    }

    if (mlir::succeeded(parser.parseOptionalComma())) {
      numOperands = 2;

      if (parser.parseOperand(dimension)) {
        return mlir::failure();
      }
    }

    if (parser.parseOptionalAttrDict(result.attributes)) {
      return mlir::failure();
    }

    if (parser.parseColon()) {
      return mlir::failure();
    }

    if (numOperands == 1) {
      if (parser.parseType(arrayType) ||
          parser.resolveOperand(array, arrayType, result.operands)) {
        return mlir::failure();
      }
    } else {
      if (parser.parseLParen() ||
          parser.parseType(arrayType) ||
          parser.resolveOperand(array, arrayType, result.operands) ||
          parser.parseComma() ||
          parser.parseType(dimensionType) ||
          parser.resolveOperand(dimension, dimensionType, result.operands) ||
          parser.parseRParen()) {
        return mlir::failure();
      }
    }

    mlir::Type resultType;

    if (parser.parseArrow() ||
        parser.parseType(resultType)) {
      return mlir::failure();
    }

    result.addTypes(resultType);

    return mlir::success();
  }

  void SizeOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " " << getArray();

    if (getOperation()->getNumOperands() == 2) {
      printer << ", " << getDimension();
    }

    printer.printOptionalAttrDict(getOperation()->getAttrs());
    printer << " : ";

    if (getOperation()->getNumOperands() == 1) {
      printer << getArray().getType();
    } else {
      printer << "(" << getArray().getType() << ", "
              << getDimension().getType() << ")";
    }

    printer << " -> " << getResult().getType();
  }

  void SizeOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(
        mlir::MemoryEffects::Read::get(),
        getArray(),
        mlir::SideEffects::DefaultResource::get());

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }
}

//===---------------------------------------------------------------------===//
// SqrtOp

namespace mlir::modelica
{
  mlir::OpFoldResult SqrtOp::fold(FoldAdaptor adaptor)
  {
    auto operand = adaptor.getOperand();

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        auto operandValue =
            static_cast<double>(getScalarIntegerLikeValue(operand));

        return getAttr(resultType, std::sqrt(operandValue));
      }

      if (isScalarFloatLike(operand)) {
        double operandValue = getScalarFloatLikeValue(operand);
        return getAttr(resultType, std::sqrt(operandValue));
      }
    }

    return {};
  }

  mlir::ValueRange SqrtOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int SqrtOp::getArgExpectedRank(
      unsigned int argIndex, mlir::SymbolTableCollection& symbolTable)
  {
    return 0;
  }

  mlir::ValueRange SqrtOp::scalarize(
      mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType =
        getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(
        getLoc(), getOperand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<SqrtOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  void SqrtOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getOperand(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange SqrtOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    // D[sqrt(x)] = x' / sqrt(x) / 2

    mlir::Location loc = getLoc();

    mlir::Value derivedOperand = derivatives.lookup(getOperand());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value sqrt = builder.create<SqrtOp>(loc, type, getOperand());

    mlir::Value numerator = builder.create<DivEWOp>(
        loc, type, derivedOperand, sqrt);

    mlir::Value two = builder.create<ConstantOp>(
        loc, RealAttr::get(getContext(), 2));

    auto derivedOp = builder.create<DivEWOp>(loc, type, numerator, two);

    return derivedOp->getResults();
  }

  void SqrtOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
  }

  void SqrtOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// SumOp

namespace mlir::modelica
{
  void SumOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(
        mlir::MemoryEffects::Read::get(),
        getArray(),
        mlir::SideEffects::DefaultResource::get());
  }
}

//===---------------------------------------------------------------------===//
// SymmetricOp

namespace mlir::modelica
{
  void SymmetricOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(
        mlir::MemoryEffects::Read::get(),
        getMatrix(),
        mlir::SideEffects::DefaultResource::get());

    effects.emplace_back(
        mlir::MemoryEffects::Allocate::get(),
        getResult(),
        mlir::SideEffects::DefaultResource::get());

    effects.emplace_back(
        mlir::MemoryEffects::Write::get(),
        getResult(),
        mlir::SideEffects::DefaultResource::get());
  }
}

//===---------------------------------------------------------------------===//
// TanOp

namespace mlir::modelica
{
  mlir::OpFoldResult TanOp::fold(FoldAdaptor adaptor)
  {
    auto operand = adaptor.getOperand();

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        auto operandValue =
            static_cast<double>(getScalarIntegerLikeValue(operand));

        return getAttr(resultType, std::tan(operandValue));
      }

      if (isScalarFloatLike(operand)) {
        double operandValue = getScalarFloatLikeValue(operand);
        return getAttr(resultType, std::tan(operandValue));
      }
    }

    return {};
  }

  void TanOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getOperand(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange TanOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int TanOp::getArgExpectedRank(
      unsigned int argIndex, mlir::SymbolTableCollection& symbolTable)
  {
    return 0;
  }

  mlir::ValueRange TanOp::scalarize(
      mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType =
        getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(
        getLoc(), getOperand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<TanOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange TanOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    // D[tan(x)] = x' / (cos(x))^2

    mlir::Location loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(getOperand());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value cos = builder.create<CosOp>(loc, type, getOperand());

    mlir::Value two = builder.create<ConstantOp>(
        loc, RealAttr::get(getContext(), 2));

    mlir::Value denominator = builder.create<PowEWOp>(loc, type, cos, two);

    auto derivedOp = builder.create<DivEWOp>(
        loc, type, derivedOperand, denominator);

    return derivedOp->getResults();
  }

  void TanOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
  }

  void TanOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// TanhOp

namespace mlir::modelica
{
  mlir::OpFoldResult TanhOp::fold(FoldAdaptor adaptor)
  {
    auto operand = adaptor.getOperand();

    if (!operand) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(operand)) {
      if (isScalarIntegerLike(operand)) {
        auto operandValue =
            static_cast<double>(getScalarIntegerLikeValue(operand));

        return getAttr(resultType, std::tanh(operandValue));
      }

      if (isScalarFloatLike(operand)) {
        double operandValue = getScalarFloatLikeValue(operand);
        return getAttr(resultType, std::tanh(operandValue));
      }
    }

    return {};
  }

  void TanhOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getOperand().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getOperand(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange TanhOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int TanhOp::getArgExpectedRank(
      unsigned int argIndex, mlir::SymbolTableCollection& symbolTable)
  {
    return 0;
  }

  mlir::ValueRange TanhOp::scalarize(
      mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType =
        getResult().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(
        getLoc(), getOperand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>();
        arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<TanhOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange TanhOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    // D[tanh(x)] = x' / (cosh(x))^2

    mlir::Location loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(getOperand());
    mlir::Type type = convertToRealType(getResult().getType());

    mlir::Value cosh = builder.create<CoshOp>(loc, type, getOperand());

    mlir::Value two = builder.create<ConstantOp>(
        loc, RealAttr::get(getContext(), 2));

    mlir::Value pow = builder.create<PowEWOp>(loc, type, cosh, two);
    auto derivedOp = builder.create<DivEWOp>(loc, type, derivedOperand, pow);

    return derivedOp->getResults();
  }

  void TanhOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getOperand());
  }

  void TanhOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// TransposeOp

namespace mlir::modelica
{
  void TransposeOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getMatrix().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getMatrix(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getResult().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getResult(),
          mlir::SideEffects::DefaultResource::get());
    }
  }
}

//===---------------------------------------------------------------------===//
// ZerosOp

namespace mlir::modelica
{
  void ZerosOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(
        mlir::MemoryEffects::Allocate::get(),
        getResult(),
        mlir::SideEffects::DefaultResource::get());

    effects.emplace_back(
        mlir::MemoryEffects::Write::get(),
        getResult(),
        mlir::SideEffects::DefaultResource::get());
  }
}

//===---------------------------------------------------------------------===//
// ReductionOp

namespace mlir::modelica
{
  mlir::ParseResult ReductionOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    auto loc = parser.getCurrentLocation();
    std::string action;

    if (parser.parseString(&action) ||
        parser.parseComma()) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> iterables;
    llvm::SmallVector<mlir::Type> iterablesTypes;

    llvm::SmallVector<mlir::OpAsmParser::Argument> inductions;

    mlir::Region* expressionRegion = result.addRegion();
    mlir::Type resultType;

    if (parser.parseKeyword("iterables") ||
        parser.parseEqual() ||
        parser.parseOperandList(
            iterables, mlir::AsmParser::Delimiter::Square) ||
        parser.parseComma() ||
        parser.parseKeyword("inductions") ||
        parser.parseEqual() ||
        parser.parseArgumentList(
            inductions, mlir::AsmParser::Delimiter::Square) ||
        parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
        parser.parseRegion(*expressionRegion, inductions) ||
        parser.parseColon() ||
        parser.parseLParen() ||
        parser.parseTypeList(iterablesTypes) ||
        parser.parseRParen() ||
        parser.resolveOperands(
            iterables, iterablesTypes, loc, result.operands) ||
        parser.parseArrow() ||
        parser.parseType(resultType)) {
      return mlir::failure();
    }

    return mlir::success();
  }

  void ReductionOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " " << getAction()
            << ", iterables = [" << getIterables()
            << "], inductions = [";

    for (size_t i = 0, e = getInductions().size(); i < e; ++i) {
      if (i != 0) {
        printer << ", ";
      }

      printer.printRegionArgument(getInductions()[i]);
    }

    printer << "] ";

    llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
    elidedAttrs.push_back(getActionAttrName().getValue());

    printer.printOptionalAttrDictWithKeyword(
        getOperation()->getAttrs(), elidedAttrs);

    printer.printRegion(getExpressionRegion(), false);
    printer << " : ";

    auto iterables = getIterables();
    printer << "(";

    for (size_t i = 0, e = iterables.size(); i < e; ++i) {
      if (i != 0) {
        printer << ", ";
      }

      printer << iterables[i].getType();
    }

    printer << ") -> ";

    printer << getResult().getType();
  }

  mlir::Block* ReductionOp::createExpressionBlock(mlir::OpBuilder& builder)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    llvm::SmallVector<mlir::Type> argTypes;
    llvm::SmallVector<mlir::Location> argLocs;

    for (mlir::Value iterable : getIterables()) {
      auto iterableType = iterable.getType().cast<IterableType>();
      argTypes.push_back(iterableType.getInductionType());
      argLocs.push_back(builder.getUnknownLoc());
    }

    return builder.createBlock(&getExpressionRegion(), {}, argTypes, argLocs);
  }
}

//===---------------------------------------------------------------------===//
// Modeling operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// PackageOp

namespace mlir::modelica
{
  void PackageOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      llvm::StringRef name)
  {
    state.addRegion()->emplaceBlock();

    state.attributes.push_back(builder.getNamedAttr(
        mlir::SymbolTable::getSymbolAttrName(),
        builder.getStringAttr(name)));
  }

  mlir::ParseResult PackageOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::StringAttr nameAttr;

    if (parser.parseSymbolName(
            nameAttr,
            mlir::SymbolTable::getSymbolAttrName(),
            result.attributes) ||
        parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
      return mlir::failure();
    }

    mlir::Region* bodyRegion = result.addRegion();

    if (parser.parseRegion(*bodyRegion)) {
      return mlir::failure();
    }

    if (bodyRegion->empty()) {
      bodyRegion->emplaceBlock();
    }

    return mlir::success();
  }

  void PackageOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " ";
    printer.printSymbolName(getSymName());
    printer << " ";

    llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
    elidedAttrs.push_back(mlir::SymbolTable::getSymbolAttrName());

    printer.printOptionalAttrDictWithKeyword(
        getOperation()->getAttrs(), elidedAttrs);

    printer.printRegion(getBodyRegion());
  }

  mlir::Block* PackageOp::bodyBlock()
  {
    assert(getBodyRegion().hasOneBlock());
    return &getBodyRegion().front();
  }
}

//===---------------------------------------------------------------------===//
// ModelOp

namespace mlir::modelica
{
  void ModelOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      llvm::StringRef name)
  {
    build(builder, state, name, builder.getArrayAttr({}));
  }

  mlir::RegionKind ModelOp::getRegionKind(unsigned index)
  {
    return mlir::RegionKind::Graph;
  }


  void ModelOp::collectVariables(llvm::SmallVectorImpl<VariableOp>& variables)
  {
    for (VariableOp variableOp : getVariables()) {
      variables.push_back(variableOp);
    }
  }

  void ModelOp::collectEquations(
      llvm::SmallVectorImpl<EquationInstanceOp>& initialEquations,
      llvm::SmallVectorImpl<EquationInstanceOp>& equations)
  {
    for (EquationInstanceOp op : getOps<EquationInstanceOp>()) {
      if (op.getInitial()) {
        initialEquations.push_back(op);
      } else {
        equations.push_back(op);
      }
    }
  }

  void ModelOp::collectEquations(
      llvm::SmallVectorImpl<MatchedEquationInstanceOp>& initialEquations,
      llvm::SmallVectorImpl<MatchedEquationInstanceOp>& equations)
  {
    for (MatchedEquationInstanceOp op : getOps<MatchedEquationInstanceOp>()) {
      if (op.getInitial()) {
        initialEquations.push_back(op);
      } else {
        equations.push_back(op);
      }
    }
  }

  void ModelOp::collectSCCs(
      llvm::SmallVectorImpl<SCCOp>& initialSCCs,
      llvm::SmallVectorImpl<SCCOp>& SCCs)
  {
    for (SCCOp op : getOps<SCCOp>()) {
      if (op.getInitial()) {
        initialSCCs.push_back(op);
      } else {
        SCCs.push_back(op);
      }
    }
  }

  void ModelOp::collectAlgorithms(
      llvm::SmallVectorImpl<AlgorithmOp>& initialAlgorithms,
      llvm::SmallVectorImpl<AlgorithmOp>& algorithms)
  {
    for (AlgorithmOp op : getOps<AlgorithmOp>()) {
      if (op.getInitial()) {
        initialAlgorithms.push_back(op);
      } else {
        algorithms.push_back(op);
      }
    }
  }
}

//===---------------------------------------------------------------------===//
// OperatorRecordOp

namespace mlir::modelica
{
  void OperatorRecordOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      llvm::StringRef name)
  {
    state.addRegion()->emplaceBlock();

    state.attributes.push_back(builder.getNamedAttr(
        mlir::SymbolTable::getSymbolAttrName(),
        builder.getStringAttr(name)));
  }

  mlir::ParseResult OperatorRecordOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::StringAttr nameAttr;

    if (parser.parseSymbolName(
            nameAttr,
            mlir::SymbolTable::getSymbolAttrName(),
            result.attributes)) {
      return mlir::failure();
    }

    mlir::Region* bodyRegion = result.addRegion();

    if (parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
        parser.parseRegion(*bodyRegion)) {
      return mlir::failure();
    }

    if (bodyRegion->empty()) {
      bodyRegion->emplaceBlock();
    }

    return mlir::success();
  }

  void OperatorRecordOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " ";
    printer.printSymbolName(getSymName());
    printer << " ";

    llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
    elidedAttrs.push_back(mlir::SymbolTable::getSymbolAttrName());

    printer.printOptionalAttrDictWithKeyword(
        getOperation()->getAttrs(), elidedAttrs);

    printer.printRegion(getBodyRegion());
  }

  mlir::Block* OperatorRecordOp::bodyBlock()
  {
    assert(getBodyRegion().hasOneBlock());
    return &getBodyRegion().front();
  }
}

//===---------------------------------------------------------------------===//
// StartOp

namespace mlir::modelica
{
  VariableOp StartOp::getVariableOp(mlir::SymbolTableCollection& symbolTable)
  {
    auto cls = getOperation()->getParentOfType<ClassInterface>();
    return symbolTable.lookupSymbolIn<VariableOp>(cls, getVariableAttr());
  }
}

//===---------------------------------------------------------------------===//
// DefaultOp

namespace mlir::modelica
{
  VariableOp DefaultOp::getVariableOp(mlir::SymbolTableCollection& symbolTable)
  {
    auto cls = getOperation()->getParentOfType<ClassInterface>();
    return symbolTable.lookupSymbolIn<VariableOp>(cls, getVariableAttr());
  }
}

//===---------------------------------------------------------------------===//
// BindingEquationOp

namespace mlir::modelica
{
  VariableOp BindingEquationOp::getVariableOp(
      mlir::SymbolTableCollection& symbolTable)
  {
    auto cls = getOperation()->getParentOfType<ClassInterface>();
    return symbolTable.lookupSymbolIn<VariableOp>(cls, getVariableAttr());
  }
}

//===---------------------------------------------------------------------===//
// ForEquationOp

namespace
{
  struct EmptyForEquationOpErasePattern
      : public mlir::OpRewritePattern<ForEquationOp>
  {
    using mlir::OpRewritePattern<ForEquationOp>::OpRewritePattern;

    mlir::LogicalResult match(ForEquationOp op) const override
    {
      return mlir::LogicalResult::success(op.getOps().empty());
    }

    void rewrite(
        ForEquationOp op, mlir::PatternRewriter& rewriter) const override
    {
      rewriter.eraseOp(op);
    }
  };
}

namespace mlir::modelica
{
  void ForEquationOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      long from,
      long to,
      long step)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    state.addAttribute(
        getFromAttrName(state.name), builder.getIndexAttr(from));

    state.addAttribute(getToAttrName(state.name), builder.getIndexAttr(to));

    state.addAttribute(
        getStepAttrName(state.name), builder.getIndexAttr(step));

    mlir::Region* bodyRegion = state.addRegion();

    builder.createBlock(
        bodyRegion, {}, builder.getIndexType(), builder.getUnknownLoc());
  }

  void ForEquationOp::getCanonicalizationPatterns(
      mlir::RewritePatternSet& patterns, mlir::MLIRContext* context)
  {
    patterns.add<EmptyForEquationOpErasePattern>(context);
  }

  mlir::Block* ForEquationOp::bodyBlock()
  {
    assert(getBodyRegion().getBlocks().size() == 1);
    return &getBodyRegion().front();
  }

  mlir::Value ForEquationOp::induction()
  {
    assert(getBodyRegion().getNumArguments() != 0);
    return getBodyRegion().getArgument(0);
  }

  mlir::ParseResult ForEquationOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    auto& builder = parser.getBuilder();

    mlir::OpAsmParser::Argument induction;

    long from;
    long to;
    long step = 1;

    if (parser.parseArgument(induction) ||
        parser.parseEqual() ||
        parser.parseInteger(from) ||
        parser.parseKeyword("to") ||
        parser.parseInteger(to)) {
      return mlir::failure();
    }

    if (mlir::succeeded(parser.parseOptionalKeyword("step"))) {
      if (parser.parseInteger(step)) {
        return mlir::failure();
      }
    }

    induction.type = builder.getIndexType();

    result.attributes.append("from", builder.getIndexAttr(from));
    result.attributes.append("to", builder.getIndexAttr(to));
    result.attributes.append("step", builder.getIndexAttr(step));

    if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
      return mlir::failure();
    }

    mlir::Region* bodyRegion = result.addRegion();

    if (parser.parseRegion(*bodyRegion, induction)) {
      return mlir::failure();
    }

    return mlir::success();
  }

  void ForEquationOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " " << induction() << " = " << getFrom() << " to " << getTo();

    if (auto step = getStep(); step != 1) {
      printer << " step " << step;
    }

    printer.printOptionalAttrDictWithKeyword(
        getOperation()->getAttrs(), {"from", "to", "step"});

    printer << " ";
    printer.printRegion(getBodyRegion(), false);
  }
}

//===---------------------------------------------------------------------===//
// EquationTemplateOp

namespace mlir::modelica
{
  mlir::ParseResult EquationTemplateOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    llvm::SmallVector<mlir::OpAsmParser::Argument, 3> inductions;
    mlir::Region* bodyRegion = result.addRegion();

    if (parser.parseKeyword("inductions") ||
        parser.parseEqual() ||
        parser.parseArgumentList(
            inductions, mlir::OpAsmParser::Delimiter::Square) ||
        parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
      return mlir::failure();
    }

    for (auto& induction : inductions) {
      induction.type = parser.getBuilder().getIndexType();
    }

    if (parser.parseRegion(*bodyRegion, inductions)) {
      return mlir::failure();
    }

    if (bodyRegion->empty()) {
      mlir::OpBuilder builder(bodyRegion);

      llvm::SmallVector<mlir::Type, 3> argTypes(
          inductions.size(), builder.getIndexType());

      llvm::SmallVector<mlir::Location, 3> argLocs(
          inductions.size(), builder.getUnknownLoc());

      builder.createBlock(bodyRegion, {}, argTypes, argLocs);
    }

    result.addTypes(EquationType::get(parser.getContext()));
    return mlir::success();
  }

  void EquationTemplateOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " ";
    printer << "inductions = [";
    printer.printOperands(getInductionVariables());
    printer << "]";
    printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
    printer << " ";
    printer.printRegion(getBodyRegion(), false);
  }

  mlir::Block* EquationTemplateOp::createBody(unsigned int numOfInductions)
  {
    mlir::OpBuilder builder(getContext());

    llvm::SmallVector<mlir::Type, 3> argTypes(
        numOfInductions, builder.getIndexType());

    llvm::SmallVector<mlir::Location, 3> argLocs(
        numOfInductions, builder.getUnknownLoc());

    return builder.createBlock(&getBodyRegion(), {}, argTypes, argLocs);
  }

  mlir::ValueRange EquationTemplateOp::getInductionVariables()
  {
    return getBodyRegion().getArguments();
  }

  uint64_t EquationTemplateOp::getNumOfImplicitInductionVariables(
      uint64_t viewElementIndex)
  {
    // Checking lhs or rhs is the same.
    mlir::Value lhs = getValueAtPath(
        EquationPath(EquationPath::LEFT, viewElementIndex));

    size_t implicitIterationVariables = 0;

    if (auto arrayType = lhs.getType().dyn_cast<ArrayType>()) {
      implicitIterationVariables = arrayType.getRank();
    }

    return implicitIterationVariables;
  }

  std::optional<mlir::modeling::IndexSet>
  EquationTemplateOp::computeImplicitIterationSpace(uint64_t viewElementIndex)
  {
    mlir::Value lhs = getValueAtPath(
        EquationPath(EquationPath::LEFT, viewElementIndex));

    mlir::Value rhs = getValueAtPath(
        EquationPath(EquationPath::RIGHT, viewElementIndex));

    auto lhsArrayType = lhs.getType().dyn_cast<ArrayType>();
    auto rhsArrayType = rhs.getType().dyn_cast<ArrayType>();

    llvm::SmallVector<int64_t> shape;

    if (lhsArrayType && rhsArrayType) {
      assert(lhsArrayType.getRank() == rhsArrayType.getRank());

      for (int64_t i = 0, rank = lhsArrayType.getRank(); i < rank; ++i) {
        int64_t lhsDim = lhsArrayType.getDimSize(i);

        if (lhsDim == ArrayType::kDynamic) {
          int64_t rhsDim = rhsArrayType.getDimSize(i);
          assert(rhsDim != ArrayType::kDynamic);
          shape.push_back(rhsDim);
        } else {
          shape.push_back(lhsDim);
        }
      }
    }

    if (shape.empty()) {
      return std::nullopt;
    }

    if (!lhsArrayType) {
      return std::nullopt;
    }

    llvm::SmallVector<Range, 3> ranges;

    for (int64_t dimension : shape) {
      ranges.push_back(mlir::modeling::Range(0, dimension));
    }

    return IndexSet(MultidimensionalRange(ranges));
  }

  llvm::DenseMap<mlir::Value, unsigned int>
  EquationTemplateOp::getInductionsPositionMap()
  {
    mlir::ValueRange inductionVariables = getInductionVariables();
    llvm::DenseMap<mlir::Value, unsigned int> inductionsPositionMap;

    for (auto inductionVariable : llvm::enumerate(inductionVariables)) {
      inductionsPositionMap[inductionVariable.value()] =
          inductionVariable.index();
    }

    return inductionsPositionMap;
  }

  mlir::LogicalResult EquationTemplateOp::getAccesses(
      llvm::SmallVectorImpl<VariableAccess>& result,
      mlir::SymbolTableCollection& symbolTable,
      uint64_t elementIndex)
  {
    auto equationSides =
        mlir::cast<EquationSidesOp>(getBody()->getTerminator());

    // Get the induction variables and number them.
    auto inductionsPositionMap = getInductionsPositionMap();

    // Search the accesses starting from the left-hand side of the equation.
    if (mlir::failed(searchAccesses(
            result, symbolTable, inductionsPositionMap,
            equationSides.getLhsValues()[elementIndex],
            EquationPath(EquationPath::LEFT, elementIndex)))) {
      return mlir::failure();
    }

    // Search the accesses starting from the right-hand side of the equation.
    if (mlir::failed(searchAccesses(
            result, symbolTable, inductionsPositionMap,
            equationSides.getRhsValues()[elementIndex],
            EquationPath(EquationPath::RIGHT, elementIndex)))) {
      return mlir::failure();
    }

    return mlir::success();
  }

  mlir::LogicalResult EquationTemplateOp::getWriteAccesses(
      llvm::SmallVectorImpl<VariableAccess>& result,
      const IndexSet& equationIndices,
      llvm::ArrayRef<VariableAccess> accesses,
      const VariableAccess& matchedAccess)
  {
    const AccessFunction& matchedAccessFunction =
        matchedAccess.getAccessFunction();

    IndexSet matchedVariableIndices =
        matchedAccessFunction.map(equationIndices);

    for (const VariableAccess& access : accesses) {
      if (access.getVariable() != matchedAccess.getVariable()) {
        continue;
      }

      const AccessFunction& accessFunction = access.getAccessFunction();

      IndexSet accessedVariableIndices =
          accessFunction.map(equationIndices);

      if (matchedVariableIndices.empty() && accessedVariableIndices.empty()) {
        result.push_back(access);
      } else if (matchedVariableIndices.overlaps(accessedVariableIndices)) {
        result.push_back(access);
      }
    }

    return mlir::success();
  }

  mlir::LogicalResult EquationTemplateOp::getReadAccesses(
      llvm::SmallVectorImpl<VariableAccess>& result,
      const IndexSet& equationIndices,
      llvm::ArrayRef<VariableAccess> accesses,
      const VariableAccess& matchedAccess)
  {
    const AccessFunction& matchedAccessFunction =
        matchedAccess.getAccessFunction();

    IndexSet matchedVariableIndices =
        matchedAccessFunction.map(equationIndices);

    for (const VariableAccess& access : accesses) {
      if (access.getVariable() != matchedAccess.getVariable()) {
        result.push_back(access);
      } else {
        const AccessFunction& accessFunction =
            access.getAccessFunction();

        IndexSet accessedVariableIndices =
            accessFunction.map(equationIndices);

        if (!matchedVariableIndices.empty() &&
            !accessedVariableIndices.empty()) {
          if (!matchedVariableIndices.contains(accessedVariableIndices)) {
            result.push_back(access);
          }
        }
      }
    }

    return mlir::success();
  }

  mlir::Value EquationTemplateOp::getValueAtPath(const EquationPath& path)
  {
    mlir::Block* bodyBlock = getBody();
    EquationPath::EquationSide side = path.getEquationSide();

    auto equationSidesOp =
        mlir::cast<EquationSidesOp>(bodyBlock->getTerminator());

    mlir::Value value = side == EquationPath::LEFT
        ? equationSidesOp.getLhsValues()[path[0]]
        : equationSidesOp.getRhsValues()[path[0]];

    for (size_t i = 1, e = path.size(); i < e; ++i) {
      mlir::Operation* op = value.getDefiningOp();
      assert(op != nullptr && "Invalid equation path");
      value = op->getOperand(path[i]);
    }

    return value;
  }

  std::optional<VariableAccess> EquationTemplateOp::getAccessAtPath(
      mlir::SymbolTableCollection& symbolTable,
      const EquationPath& path)
  {
    // Get the induction variables and number them.
    mlir::ValueRange inductionVariables = getInductionVariables();
    llvm::DenseMap<mlir::Value, unsigned int> inductionsPositionMap;

    for (auto inductionVariable : llvm::enumerate(inductionVariables)) {
      inductionsPositionMap[inductionVariable.value()] =
          inductionVariable.index();
    }

    // Get the access.
    llvm::SmallVector<VariableAccess, 1> accesses;
    mlir::Value access = getValueAtPath(path);

    if (mlir::failed(searchAccesses(
            accesses, symbolTable, inductionsPositionMap, access, path))) {
      return std::nullopt;
    }

    assert(accesses.size() == 1);
    return accesses[0];
  }

  std::optional<mlir::AffineMap> EquationTemplateOp::getAccessFunction(
      llvm::ArrayRef<mlir::Value> indices)
  {
    // Get the induction variables and number them.
    mlir::ValueRange inductionVariables = getInductionVariables();
    llvm::DenseMap<mlir::Value, unsigned int> inductionsPositionMap;

    for (const auto& inductionVariable : llvm::enumerate(inductionVariables)) {
      inductionsPositionMap[inductionVariable.value()] =
          inductionVariable.index();
    }

    llvm::SmallVector<mlir::AffineExpr, 3> expressions;

    for (mlir::Value index : indices) {
      if (auto expression = getAffineExpr(inductionsPositionMap, index)) {
        expressions.push_back(*expression);
      } else {
        return std::nullopt;
      }
    }

    return mlir::AffineMap::get(
        inductionVariables.size(), 0, expressions, getContext());
  }

  mlir::LogicalResult EquationTemplateOp::searchAccesses(
      llvm::SmallVectorImpl<VariableAccess>& accesses,
      mlir::SymbolTableCollection& symbolTable,
      llvm::DenseMap<mlir::Value, unsigned int>& inductionsPositionMap,
      mlir::Value value,
      EquationPath path)
  {
    llvm::SmallVector<mlir::AffineExpr, 10> dimensionAccesses;

    return searchAccesses(accesses, symbolTable, inductionsPositionMap,
                   value, dimensionAccesses, std::move(path));
  }

  mlir::LogicalResult EquationTemplateOp::searchAccesses(
      llvm::SmallVectorImpl<VariableAccess>& accesses,
      mlir::SymbolTableCollection& symbolTable,
      llvm::DenseMap<mlir::Value, unsigned int>& inductionsPositionMap,
      mlir::Value value,
      llvm::SmallVectorImpl<mlir::AffineExpr>& dimensionAccesses,
      EquationPath path)
  {
    if (mlir::Operation* definingOp = value.getDefiningOp()) {
      if (mlir::failed(searchAccesses(
              accesses, symbolTable, inductionsPositionMap,
              definingOp, dimensionAccesses, std::move(path)))) {
        return mlir::failure();
      }
    }

    return mlir::success();
  }

  mlir::LogicalResult EquationTemplateOp::searchAccesses(
      llvm::SmallVectorImpl<VariableAccess>& accesses,
      mlir::SymbolTableCollection& symbolTable,
      llvm::DenseMap<mlir::Value, unsigned int>& inductionsPositionMap,
      mlir::Operation* op,
      llvm::SmallVectorImpl<mlir::AffineExpr>& dimensionAccesses,
      EquationPath path)
  {
    if (auto variableGetOp = mlir::dyn_cast<VariableGetOp>(op)) {
      llvm::SmallVector<mlir::AffineExpr, 3> reverted(
          dimensionAccesses.rbegin(), dimensionAccesses.rend());

      uint64_t numOfExplicitInductions =
          static_cast<uint64_t>(inductionsPositionMap.size());

      uint64_t numOfImplicitInductions =
          getNumOfImplicitInductionVariables(path[0]);

      uint64_t numOfInductions =
          numOfExplicitInductions + numOfImplicitInductions;

      if (path.size() == 1) {
        for (uint64_t i = numOfExplicitInductions; i < numOfInductions; ++i) {
          reverted.push_back(mlir::getAffineDimExpr(i, getContext()));
        }

        auto affineMap = mlir::AffineMap::get(
            numOfInductions, 0, reverted, getContext());

        accesses.push_back(VariableAccess(
            std::move(path),
            mlir::SymbolRefAttr::get(variableGetOp.getVariableAttr()),
            AccessFunction::build(affineMap)));
      } else {
        if (auto arrayType = variableGetOp.getType().dyn_cast<ArrayType>();
            arrayType &&
            arrayType.getRank() > static_cast<int64_t>(reverted.size())) {
          // Access to each scalar variable.
          llvm::SmallVector<Range, 3> ranges;

          for (int64_t i = static_cast<int64_t>(reverted.size()),
                       rank = arrayType.getRank(); i < rank; ++i) {
            int64_t dimension = arrayType.getDimSize(i);
            assert(dimension != ArrayType::kDynamic);
            ranges.push_back(Range(0, dimension));
          }

          MultidimensionalRange indices(ranges);

          for (Point point : indices) {
            llvm::SmallVector<mlir::AffineExpr, 3> extendedExpressions(
                reverted.begin(), reverted.end());

            for (int64_t index : point) {
              extendedExpressions.push_back(
                  mlir::getAffineConstantExpr(index, getContext()));
            }

            auto affineMap = mlir::AffineMap::get(
                numOfInductions, 0, extendedExpressions, getContext());

            accesses.push_back(VariableAccess(
                std::move(path),
                mlir::SymbolRefAttr::get(variableGetOp.getVariableAttr()),
                AccessFunction::build(affineMap)));
          }
        } else {
          auto affineMap = mlir::AffineMap::get(
              numOfInductions, 0, reverted, getContext());

          accesses.push_back(VariableAccess(
              std::move(path),
              mlir::SymbolRefAttr::get(variableGetOp.getVariableAttr()),
              AccessFunction::build(affineMap)));
        }
      }

      return mlir::success();
    }

    if (auto loadOp = mlir::dyn_cast<LoadOp>(op)) {
      for (size_t i = 0, e = loadOp.getIndices().size(); i < e; ++i) {
        mlir::Value index = loadOp.getIndices()[e - 1 - i];
        auto expression = getAffineExpr(inductionsPositionMap, index);

        if (!expression) {
          loadOp.emitOpError() << "Can't compute access";
          return mlir::failure();
        }

        dimensionAccesses.push_back(*expression);
      }

      return searchAccesses(
          accesses, symbolTable, inductionsPositionMap,
          loadOp.getArray(), dimensionAccesses, std::move(path));
    }

    if (auto subscriptionOp = mlir::dyn_cast<SubscriptionOp>(op)) {
      for (size_t i = 0, e = subscriptionOp.getIndices().size(); i < e; ++i) {
        mlir::Value index = subscriptionOp.getIndices()[e - 1 - i];
        auto expression = getAffineExpr(inductionsPositionMap, index);

        if (!expression) {
          subscriptionOp.emitOpError() << "Can't compute access";
          return mlir::failure();
        }

        dimensionAccesses.push_back(*expression);
      }

      return searchAccesses(
          accesses, symbolTable, inductionsPositionMap,
          subscriptionOp.getSource(), dimensionAccesses, std::move(path));
    }

    for (size_t i = 0, e = op->getNumOperands(); i < e; ++i) {
      EquationPath::Guard guard(path);
      path += i;

      if (mlir::failed(searchAccesses(
              accesses, symbolTable, inductionsPositionMap,
              op->getOperand(i), path))) {
        return mlir::failure();
      }
    }

    return mlir::success();
  }

  std::optional<mlir::AffineExpr> EquationTemplateOp::getAffineExpr(
      llvm::DenseMap<mlir::Value, unsigned int>& inductionsPositionMap,
      mlir::Value index)
  {
    if (auto definingOp = index.getDefiningOp()) {
      if (auto op = mlir::dyn_cast<ConstantOp>(definingOp)) {
        return mlir::getAffineConstantExpr(
            getIntegerFromAttribute(op.getValue()), index.getContext());
      }

      if (auto op = mlir::dyn_cast<AddOp>(definingOp)) {
        auto lhs = getAffineExpr(inductionsPositionMap, op.getLhs());
        auto rhs = getAffineExpr(inductionsPositionMap, op.getRhs());

        if (!lhs || !rhs) {
          return std::nullopt;
        }

        return *lhs + *rhs;
      }

      if (auto op = mlir::dyn_cast<SubOp>(definingOp)) {
        auto lhs = getAffineExpr(inductionsPositionMap, op.getLhs());
        auto rhs = getAffineExpr(inductionsPositionMap, op.getRhs());

        if (!lhs || !rhs) {
          return std::nullopt;
        }

        return *lhs - *rhs;
      }

      if (auto op = mlir::dyn_cast<MulOp>(definingOp)) {
        auto lhs = getAffineExpr(inductionsPositionMap, op.getLhs());
        auto rhs = getAffineExpr(inductionsPositionMap, op.getRhs());

        if (!lhs || !rhs) {
          return std::nullopt;
        }

        return *lhs * *rhs;
      }

      if (auto op = mlir::dyn_cast<DivOp>(definingOp)) {
        auto lhs = getAffineExpr(inductionsPositionMap, op.getLhs());
        auto rhs = getAffineExpr(inductionsPositionMap, op.getRhs());

        if (!lhs || !rhs) {
          return std::nullopt;
        }

        return lhs->floorDiv(*rhs);
      }
    }

    if (auto it = inductionsPositionMap.find(index);
        it != inductionsPositionMap.end()) {
      return mlir::getAffineDimExpr(it->second, getContext());
    }

    return std::nullopt;
  }

  mlir::LogicalResult EquationTemplateOp::cloneWithReplacedAccess(
      mlir::RewriterBase& rewriter,
      std::optional<std::reference_wrapper<const IndexSet>> equationIndices,
      const VariableAccess& access,
      EquationTemplateOp replacementEquation,
      const VariableAccess& replacementAccess,
      llvm::SmallVectorImpl<std::pair<IndexSet, EquationTemplateOp>>& results)
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);

    // Erase the operations in case of unrecoverable failure.
    auto cleanOnFailure = llvm::make_scope_exit([&]() {
      for (const auto& result : results) {
        rewriter.eraseOp(result.second);
      }
    });

    // The set of indices that are yet to be processed.
    IndexSet remainingEquationIndices;

    if (equationIndices) {
      remainingEquationIndices = equationIndices->get();
    }

    // Determine the access functions.
    mlir::Value destinationValue = getValueAtPath(access.getPath());
    int64_t destinationRank = 0;

    if (auto destinationArrayType =
            destinationValue.getType().dyn_cast<ArrayType>()) {
      destinationRank = destinationArrayType.getRank();
    }

    mlir::Value sourceValue =
        replacementEquation.getValueAtPath(replacementAccess.getPath());

    int64_t sourceRank = 0;

    if (auto sourceArrayType = sourceValue.getType().dyn_cast<ArrayType>()) {
      sourceRank = sourceArrayType.getRank();
    }

    if (destinationRank > sourceRank) {
      // The access to be replaced requires indices of the variables that are
      // potentially not handled by the source equation.
      return mlir::failure();
    }

    mlir::AffineMap destinationMap = access.getAccessFunction().getAffineMap();

    // The extra subscription indices to be applied to the replacement value.
    llvm::SmallVector<mlir::Value> additionalSubscriptionIndices;

    if (destinationRank < sourceRank) {
      // The access to be replaced specifies more indices than the ones given
      // by the source equation. This means that the source equation writes to
      // more indices than the requested ones. Inlining the source equation
      // results in possibly wasted additional computations, but does lead to
      // a correct result.

      destinationMap = mlir::AffineMap::get(
          destinationMap.getNumDims(),
          destinationMap.getNumSymbols(),
          destinationMap.getResults().drop_back(
              sourceRank - destinationRank),
          destinationMap.getContext());

      // If the destination access has more indices than the source one,
      // then collect the additional ones and apply them to the
      // replacement value.
      int64_t rankDifference = sourceRank - destinationRank;
      mlir::Operation* replacedValueOp = destinationValue.getDefiningOp();

      auto allAdditionalIndicesCollected = [&]() -> bool {
        return rankDifference ==
            static_cast<int64_t>(additionalSubscriptionIndices.size());
      };

      while (mlir::isa<LoadOp, SubscriptionOp>(replacedValueOp) &&
             !allAdditionalIndicesCollected()) {
        if (auto loadOp =
                mlir::dyn_cast<LoadOp>(replacedValueOp)) {
          size_t numOfIndices = loadOp.getIndices().size();

          for (size_t i = 0; i < numOfIndices &&
               !allAdditionalIndicesCollected(); ++i) {
            additionalSubscriptionIndices.push_back(
                loadOp.getIndices()[numOfIndices - i - 1]);
          }

          replacedValueOp = loadOp.getArray().getDefiningOp();
          continue;
        }

        if (auto subscriptionOp =
                mlir::dyn_cast<SubscriptionOp>(replacedValueOp)) {
          size_t numOfIndices = subscriptionOp.getIndices().size();

          for (size_t i = 0; i < numOfIndices &&
               !allAdditionalIndicesCollected(); ++i) {
            additionalSubscriptionIndices.push_back(
                subscriptionOp.getIndices()[numOfIndices - i - 1]);
          }

          replacedValueOp =
              subscriptionOp.getSource().getDefiningOp();

          continue;
        }

        return mlir::failure();
      }

      assert(allAdditionalIndicesCollected());

      // Indices have been collected in reverse order, due to the bottom-up
      // visit of the operations tree.
      std::reverse(additionalSubscriptionIndices.begin(),
                   additionalSubscriptionIndices.end());
    }

    VariableAccess destinationAccess(
        access.getPath(), access.getVariable(),
        AccessFunction::build(destinationMap));

    // Try to perform a vectorized replacement first.
    if (mlir::failed(cloneWithReplacedVectorizedAccess(
            rewriter, equationIndices, access, replacementEquation,
            replacementAccess, additionalSubscriptionIndices, results,
            remainingEquationIndices))) {
      return mlir::failure();
    }

    // Perform scalar replacements on the remaining equation indices.
    // TODO
    /*
    for (Point scalarEquationIndices : remainingEquationIndices) {
    }
     */

    if (remainingEquationIndices.empty()) {
      cleanOnFailure.release();
      return mlir::success();
    }

    return mlir::failure();
  }

  mlir::LogicalResult EquationTemplateOp::cloneWithReplacedVectorizedAccess(
      mlir::RewriterBase& rewriter,
      std::optional<std::reference_wrapper<const IndexSet>> equationIndices,
      const VariableAccess& access,
      EquationTemplateOp replacementEquation,
      const VariableAccess& replacementAccess,
      llvm::ArrayRef<mlir::Value> additionalSubscriptions,
      llvm::SmallVectorImpl<
          std::pair<IndexSet, EquationTemplateOp>>& results,
      IndexSet& remainingEquationIndices)
  {
    const AccessFunction& destinationAccessFunction =
        access.getAccessFunction();

    const AccessFunction& sourceAccessFunction =
        replacementAccess.getAccessFunction();

    auto transformation = getReplacementTransformationAccess(
        destinationAccessFunction, sourceAccessFunction);

    if (transformation) {
      if (mlir::failed(cloneWithReplacedVectorizedAccess(
              rewriter, equationIndices, access, replacementEquation,
              replacementAccess, *transformation, additionalSubscriptions,
              results, remainingEquationIndices))) {
        return mlir::failure();
      }
    }

    return mlir::success();
  }

  mlir::LogicalResult EquationTemplateOp::cloneWithReplacedVectorizedAccess(
      mlir::RewriterBase& rewriter,
      std::optional<std::reference_wrapper<const IndexSet>> equationIndices,
      const VariableAccess& access,
      EquationTemplateOp replacementEquation,
      const VariableAccess& replacementAccess,
      const AccessFunction& transformation,
      llvm::ArrayRef<mlir::Value> additionalSubscriptions,
      llvm::SmallVectorImpl<std::pair<IndexSet, EquationTemplateOp>>& results,
      IndexSet& remainingEquationIndices)
  {
    if (equationIndices && !equationIndices->get().empty()) {
      for (const MultidimensionalRange& range : llvm::make_range(
               equationIndices->get().rangesBegin(),
               equationIndices->get().rangesEnd())) {
        if (mlir::failed(cloneWithReplacedVectorizedAccess(
                rewriter, std::reference_wrapper(range), access,
                replacementEquation, replacementAccess, transformation,
                additionalSubscriptions, results, remainingEquationIndices))) {
          return mlir::failure();
        }
      }

      return mlir::success();
    }

    return cloneWithReplacedVectorizedAccess(
        rewriter,
        std::optional<
            std::reference_wrapper<const MultidimensionalRange>>(std::nullopt),
        access, replacementEquation, replacementAccess, transformation,
        additionalSubscriptions, results, remainingEquationIndices);
  }

  mlir::LogicalResult EquationTemplateOp::cloneWithReplacedVectorizedAccess(
      mlir::RewriterBase& rewriter,
      std::optional<
          std::reference_wrapper<const MultidimensionalRange>> equationIndices,
      const VariableAccess& access,
      EquationTemplateOp replacementEquation,
      const VariableAccess& replacementAccess,
      const AccessFunction& transformation,
      llvm::ArrayRef<mlir::Value> additionalSubscriptions,
      llvm::SmallVectorImpl<std::pair<IndexSet, EquationTemplateOp>>& results,
      IndexSet& remainingEquationIndices)
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(getOperation());

    // Create the equation template.
    auto equationTemplateOp = rewriter.create<EquationTemplateOp>(getLoc());
    equationTemplateOp->setAttrs(getOperation()->getAttrDictionary());

    if (equationIndices) {
      remainingEquationIndices -= equationIndices->get();
      results.emplace_back(IndexSet(equationIndices->get()), equationTemplateOp);
    } else {
      results.emplace_back(IndexSet(), equationTemplateOp);
    }

    mlir::Block* equationBodyBlock =
        equationTemplateOp.createBody(getInductionVariables().size());

    rewriter.setInsertionPointToStart(equationBodyBlock);

    // Clone the operations composing the replacement equation.
    mlir::IRMapping replacementMapping;

    if (mlir::failed(mapInductionVariables(
            rewriter, replacementEquation.getLoc(),
            replacementMapping, replacementEquation, equationTemplateOp,
            transformation))) {
      return mlir::failure();
    }

    for (auto& op : replacementEquation.getOps()) {
      if (!mlir::isa<EquationSideOp, EquationSidesOp>(op)) {
        rewriter.clone(op, replacementMapping);
      }
    }

    // Compute the replacement value.
    mlir::Value mappedReplacement =
        replacementMapping.lookup(replacementEquation.getValueAtPath(
            EquationPath(EquationPath::RIGHT, access.getPath()[0])));

    // The optional additional subscription indices.
    llvm::SmallVector<mlir::Value, 3> additionalMappedSubscriptions;

    // Clone the operations composing the destination equation.
    mlir::IRMapping destinationMapping;

    for (auto [oldInduction, newInduction] : llvm::zip(
             getInductionVariables(),
             equationTemplateOp.getInductionVariables())) {
      destinationMapping.map(oldInduction, newInduction);
    }

    mlir::Value originalReplacedValue = getValueAtPath(access.getPath());

    for (auto& op : getOps()) {
      mlir::Operation* clonedOp = rewriter.clone(op, destinationMapping);
      for (auto [oldResult, newResult] :
           llvm::zip(op.getResults(), clonedOp->getResults())) {
        if (oldResult == originalReplacedValue) {
          // The value to be replaced has been found.
          mlir::Value replacement = mappedReplacement;

          if (!additionalSubscriptions.empty()) {
            additionalMappedSubscriptions.clear();

            for (mlir::Value index : additionalSubscriptions) {
              additionalMappedSubscriptions.push_back(
                  destinationMapping.lookup(index));
            }

            replacement = rewriter.create<SubscriptionOp>(
                replacement.getLoc(), replacement,
                additionalMappedSubscriptions);

            if (auto arrayType = replacement.getType().dyn_cast<ArrayType>();
                arrayType && arrayType.isScalar()) {
              replacement = rewriter.create<LoadOp>(
                  replacement.getLoc(), replacement);
            }
          }

          destinationMapping.map(oldResult, replacement);
        } else {
          destinationMapping.map(oldResult, newResult);
        }
      }
    }

    return mlir::success();
  }

  std::unique_ptr<AccessFunction>
  EquationTemplateOp::getReplacementTransformationAccess(
      const AccessFunction& destinationAccess,
      const AccessFunction& sourceAccess)
  {
    if (auto sourceInverseAccess = sourceAccess.inverse()) {
      return destinationAccess.combine(*sourceInverseAccess);
    }

    // Check if the source access is invertible by removing the constant
    // accesses.

    // Determine the constant results to be removed.
    mlir::AffineMap sourceAffineMap = sourceAccess.getAffineMap();
    llvm::SmallVector<int64_t, 3> constantExprPositions;

    for (size_t i = 0, e = sourceAffineMap.getNumResults(); i < e; ++i) {
      if (sourceAffineMap.getResult(i).isa<mlir::AffineConstantExpr>()) {
        constantExprPositions.push_back(i);
      }
    }

    // Compute the reduced access functions.
    auto reducedSourceAccessFunction =
        AccessFunction::build(mlir::compressUnusedDims(
            sourceAccess.getAffineMap().dropResults(constantExprPositions)));

    auto reducedSourceInverseAccessFunction =
        reducedSourceAccessFunction->inverse();

    if (!reducedSourceInverseAccessFunction) {
      return nullptr;
    }

    auto reducedDestinationAccessFunction = AccessFunction::build(
        destinationAccess.getAffineMap().dropResults(constantExprPositions));

    auto combinedReducedAccess = reducedDestinationAccessFunction->combine(
        *reducedSourceInverseAccessFunction);

    mlir::AffineMap combinedAffineMap = combinedReducedAccess->getAffineMap();

    return AccessFunction::build(mlir::AffineMap::get(
        destinationAccess.getNumOfDims(), 0,
        combinedAffineMap.getResults(), combinedAffineMap.getContext()));
  }

  mlir::LogicalResult EquationTemplateOp::mapInductionVariables(
      mlir::OpBuilder& builder,
      mlir::Location loc,
      mlir::IRMapping& mapping,
      EquationTemplateOp source,
      EquationTemplateOp destination,
      const AccessFunction& transformation)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(destination.getBody());

    mlir::AffineMap affineMap = transformation.getAffineMap();

    if (affineMap.getNumResults() < source.getInductionVariables().size()) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Value> affineMapResults;

    if (mlir::failed(materializeAffineMap(
            builder, loc, affineMap, destination.getInductionVariables(),
            affineMapResults))) {
      return mlir::failure();
    }

    auto sourceInductionVariables = source.getInductionVariables();

    for (size_t i = 0, e = sourceInductionVariables.size(); i < e; ++i) {
      mapping.map(sourceInductionVariables[i], affineMapResults[i]);
    }

    return mlir::success();
  }

  IndexSet EquationTemplateOp::applyAccessFunction(
      const AccessFunction& accessFunction,
      std::optional<MultidimensionalRange> explicitEquationIndices,
      std::optional<MultidimensionalRange> implicitEquationIndices,
      const EquationPath& path)
  {
    IndexSet result;

    if (explicitEquationIndices) {
      result = accessFunction.map(IndexSet(*explicitEquationIndices));
    }

    if (path.size() == 1) {
      // Leaf of the equation. Add the implicit inductions.
      if (implicitEquationIndices) {
        result.append(IndexSet(*implicitEquationIndices));
      }
    }

    return result;
  }

  mlir::LogicalResult EquationTemplateOp::explicitate(
      mlir::RewriterBase& rewriter,
      mlir::SymbolTableCollection& symbolTableCollection,
      std::optional<MultidimensionalRange> explicitEquationIndices,
      std::optional<MultidimensionalRange> implicitEquationIndices,
      const EquationPath& path)
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);

    // Get all the paths that lead to accesses with the same accessed variable
    // and function.
    auto requestedAccess = getAccessAtPath(symbolTableCollection, path);

    if (!requestedAccess) {
      return mlir::failure();
    }

    const AccessFunction& requestedAccessFunction =
        requestedAccess->getAccessFunction();

    IndexSet requestedIndices = applyAccessFunction(
        requestedAccessFunction,
        explicitEquationIndices,
        implicitEquationIndices,
        path);

    llvm::SmallVector<VariableAccess, 10> accesses;

    if (mlir::failed(getAccesses(accesses, symbolTableCollection, path[0]))) {
      return mlir::failure();
    }

    llvm::SmallVector<VariableAccess, 5> filteredAccesses;

    for (const VariableAccess& access : accesses) {
      if (requestedAccess->getVariable() != access.getVariable()) {
        continue;
      }

      const AccessFunction& currentAccessFunction = access.getAccessFunction();

      IndexSet currentIndices = applyAccessFunction(
          currentAccessFunction,
          explicitEquationIndices,
          implicitEquationIndices,
          access.getPath());

      if (requestedIndices != currentIndices &&
          requestedIndices.overlaps(currentIndices)) {
        return mlir::failure();
      }

      assert(requestedIndices == currentIndices ||
             !requestedIndices.overlaps(currentIndices));

      if (requestedIndices == currentIndices) {
        filteredAccesses.push_back(access);
      }
    }

    assert(!filteredAccesses.empty());

    // If there is only one access, then it is sufficient to follow the path
    // and invert the operations.

    auto terminator =
        mlir::cast<EquationSidesOp>(getBody()->getTerminator());

    auto lhsOp = terminator.getLhs().getDefiningOp();
    auto rhsOp = terminator.getRhs().getDefiningOp();

    rewriter.setInsertionPoint(lhsOp);

    if (rhsOp->isBeforeInBlock(lhsOp)) {
      rewriter.setInsertionPoint(rhsOp);
    }

    if (filteredAccesses.size() == 1) {
      for (size_t i = 1, e = path.size(); i < e; ++i) {
        if (mlir::failed(explicitateLeaf(
                rewriter, path[0], path[i], path.getEquationSide()))) {
          return mlir::failure();
        }
      }

      if (path.getEquationSide() == EquationPath::RIGHT) {
        llvm::SmallVector<mlir::Value, 1> lhsValues;
        llvm::SmallVector<mlir::Value, 1> rhsValues;

        rewriter.setInsertionPointAfter(terminator);

        rewriter.create<EquationSidesOp>(
            terminator->getLoc(), terminator.getRhs(), terminator.getLhs());

        rewriter.eraseOp(terminator);
      }
    } else {
      // If there are multiple accesses, then we must group all of them and
      // extract the common multiplying factor.

      if (mlir::failed(groupLeftHandSide(
              rewriter, symbolTableCollection,
              explicitEquationIndices, implicitEquationIndices,
              *requestedAccess))) {
        return mlir::failure();
      }
    }

    return mlir::success();
  }

  EquationTemplateOp EquationTemplateOp::cloneAndExplicitate(
      mlir::RewriterBase& rewriter,
      mlir::SymbolTableCollection& symbolTableCollection,
      std::optional<MultidimensionalRange> explicitEquationIndices,
      std::optional<MultidimensionalRange> implicitEquationIndices,
      const EquationPath& path)
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(getOperation());

    auto clonedOp =
        mlir::cast<EquationTemplateOp>(rewriter.clone(*getOperation()));

    auto cleanOnFailure = llvm::make_scope_exit([&]() {
      rewriter.eraseOp(clonedOp);
    });

    if (mlir::failed(clonedOp.explicitate(
            rewriter,
            symbolTableCollection,
            explicitEquationIndices,
            implicitEquationIndices,
            path))) {
      return nullptr;
    }

    cleanOnFailure.release();
    return clonedOp;
  }

  mlir::LogicalResult EquationTemplateOp::explicitateLeaf(
      mlir::RewriterBase& rewriter,
      uint64_t viewElementIndex,
      size_t argumentIndex,
      EquationPath::EquationSide side)
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);

    auto equationSidesOp =
        mlir::cast<EquationSidesOp>(getBody()->getTerminator());

    mlir::ValueRange oldLhsValues = equationSidesOp.getLhsValues();
    mlir::ValueRange oldRhsValues = equationSidesOp.getRhsValues();

    assert(viewElementIndex < oldLhsValues.size());
    assert(viewElementIndex < oldRhsValues.size());

    mlir::Value toExplicitate = side == EquationPath::LEFT
        ? oldLhsValues[viewElementIndex] : oldRhsValues[viewElementIndex];

    mlir::Value otherExp = side == EquationPath::RIGHT
        ? oldLhsValues[viewElementIndex] : oldRhsValues[viewElementIndex];

    mlir::Operation* op = toExplicitate.getDefiningOp();
    auto invertibleOp = mlir::dyn_cast<InvertibleOpInterface>(op);

    if (!invertibleOp) {
      return mlir::failure();
    }

    rewriter.setInsertionPoint(invertibleOp);

    if (auto otherExpOp = otherExp.getDefiningOp();
        otherExpOp && invertibleOp->isBeforeInBlock(otherExpOp)) {
      rewriter.setInsertionPointAfter(otherExpOp);
    }

    mlir::Value invertedOpResult =
        invertibleOp.inverse(rewriter, argumentIndex, otherExp);

    if (!invertedOpResult) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Value, 1> newLhsValues;
    llvm::SmallVector<mlir::Value, 1> newRhsValues;

    for (size_t i = 0, e = oldLhsValues.size(); i < e; ++i) {
      if (i == viewElementIndex) {
        if (side == EquationPath::LEFT) {
          newLhsValues.push_back(op->getOperand(argumentIndex));
        } else {
          newLhsValues.push_back(invertedOpResult);
        }
      } else {
        newLhsValues.push_back(oldLhsValues[i]);
      }
    }

    for (size_t i = 0, e = oldRhsValues.size(); i < e; ++i) {
      if (i == viewElementIndex) {
        if (side == EquationPath::LEFT) {
          newRhsValues.push_back(invertedOpResult);
        } else {
          newRhsValues.push_back(op->getOperand(argumentIndex));
        }
      } else {
        newRhsValues.push_back(oldRhsValues[i]);
      }
    }

    // Create the new terminator.
    rewriter.setInsertionPoint(equationSidesOp);

    auto oldLhs = mlir::cast<EquationSideOp>(
        equationSidesOp.getLhs().getDefiningOp());

    auto oldRhs = mlir::cast<EquationSideOp>(
        equationSidesOp.getRhs().getDefiningOp());

    rewriter.replaceOpWithNewOp<EquationSideOp>(oldLhs, newLhsValues);
    rewriter.replaceOpWithNewOp<EquationSideOp>(oldRhs, newRhsValues);

    return mlir::success();
  }

  static mlir::LogicalResult removeSubtractions(
      mlir::RewriterBase& rewriter, mlir::Operation* root)
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    mlir::Operation* op = root;

    if (!op) {
      return mlir::success();
    }

    if (!mlir::isa<SubscriptionOp>(op) && !mlir::isa<LoadOp>(op)) {
      for (mlir::Value operand : op->getOperands()) {
        if (mlir::failed(removeSubtractions(
                rewriter, operand.getDefiningOp()))) {
          return mlir::failure();
        }
      }
    }

    if (auto subOp = mlir::dyn_cast<SubOp>(op)) {
      rewriter.setInsertionPoint(subOp);
      mlir::Value rhs = subOp.getRhs();

      mlir::Value negatedRhs = rewriter.create<NegateOp>(
          rhs.getLoc(), rhs.getType(), rhs);

      rewriter.replaceOpWithNewOp<AddOp>(
          subOp, subOp.getResult().getType(), subOp.getLhs(), negatedRhs);
    }

    return mlir::success();
  }

  static mlir::LogicalResult distributeMulAndDivOps(
      mlir::RewriterBase& rewriter, mlir::Operation* root)
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    mlir::Operation* op = root;

    if (!op) {
      return mlir::success();
    }

    for (auto operand : op->getOperands()) {
      if (mlir::failed(distributeMulAndDivOps(
              rewriter, operand.getDefiningOp()))) {
        return mlir::failure();
      }
    }

    if (auto distributableOp = mlir::dyn_cast<DistributableOpInterface>(op)) {
      if (!mlir::isa<NegateOp>(op)) {
        rewriter.setInsertionPoint(distributableOp);
        llvm::SmallVector<mlir::Value, 1> results;

       if (mlir::succeeded(distributableOp.distribute(results, rewriter))) {
          for (size_t i = 0, e = distributableOp->getNumResults();
               i < e; ++i) {
            mlir::Value oldValue = distributableOp->getResult(i);
            mlir::Value newValue = results[i];
            rewriter.replaceAllUsesWith(oldValue, newValue);
          }
       }
      }
    }

    return mlir::success();
  }

  static mlir::LogicalResult pushNegateOps(
      mlir::RewriterBase& rewriter, mlir::Operation* root)
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    mlir::Operation* op = root;

    if (!op) {
      return mlir::success();
    }

    for (mlir::Value operand : op->getOperands()) {
      if (mlir::failed(pushNegateOps(rewriter, operand.getDefiningOp()))) {
        return mlir::failure();
      }
    }

    if (auto distributableOp = mlir::dyn_cast<NegateOp>(op)) {
      rewriter.setInsertionPoint(distributableOp);
      llvm::SmallVector<mlir::Value, 1> results;

      if (mlir::succeeded(distributableOp.distribute(results, rewriter))) {
        rewriter.replaceOp(distributableOp, results);
      }
    }

    return mlir::success();
  }

  mlir::LogicalResult EquationTemplateOp::collectSummedValues(
      llvm::SmallVectorImpl<std::pair<mlir::Value, EquationPath>>& result,
      mlir::Value root,
      EquationPath path)
  {
    if (auto definingOp = root.getDefiningOp()) {
      if (auto addOp = mlir::dyn_cast<AddOp>(definingOp)) {
        if (mlir::failed(collectSummedValues(
                result, addOp.getLhs(), path + 0))) {
          return mlir::failure();
        }

        if (mlir::failed(collectSummedValues(
                result, addOp.getRhs(), path + 1))) {
          return mlir::failure();
        }

        return mlir::success();
      }
    }

    result.push_back(std::make_pair(root, path));
    return mlir::success();
  }

  static void foldValue(
      mlir::RewriterBase& rewriter,
      mlir::Value value,
      mlir::Block* block)
  {
    mlir::OperationFolder helper(value.getContext());
    std::stack<mlir::Operation*> visitStack;
    llvm::SmallVector<mlir::Operation*, 3> ops;
    llvm::DenseSet<mlir::Operation*> processed;

    if (auto definingOp = value.getDefiningOp()) {
      visitStack.push(definingOp);
    }

    while (!visitStack.empty()) {
      auto op = visitStack.top();
      visitStack.pop();

      ops.push_back(op);

      for (const auto& operand : op->getOperands()) {
        if (auto definingOp = operand.getDefiningOp()) {
          visitStack.push(definingOp);
        }
      }
    }

    llvm::SmallVector<mlir::Operation*, 3> constants;

    for (mlir::Operation* op : llvm::reverse(ops)) {
      if (processed.contains(op)) {
        continue;
      }

      processed.insert(op);

      if (mlir::failed(helper.tryToFold(op))) {
        break;
      }
    }

    for (auto* op : llvm::reverse(constants)) {
      op->moveBefore(block, block->begin());
    }
  }

  static std::optional<bool> isZeroAttr(mlir::Attribute attribute)
  {
    if (auto booleanAttr = attribute.dyn_cast<BooleanAttr>()) {
      return !booleanAttr.getValue();
    }

    if (auto integerAttr = attribute.dyn_cast<IntegerAttr>()) {
      return integerAttr.getValue() == 0;
    }

    if (auto realAttr = attribute.dyn_cast<RealAttr>()) {
      return realAttr.getValue().isZero();
    }

    if (auto integerAttr = attribute.cast<mlir::IntegerAttr>()) {
      return integerAttr.getValue() == 0;
    }

    if (auto floatAttr = attribute.cast<mlir::FloatAttr>()) {
      return floatAttr.getValueAsDouble() == 0;
    }

    return std::nullopt;
  }

  std::optional<std::pair<unsigned int, mlir::Value>>
  EquationTemplateOp::getMultiplyingFactor(
      mlir::OpBuilder& builder,
      mlir::SymbolTableCollection& symbolTableCollection,
      llvm::DenseMap<mlir::Value, unsigned int>& inductionsPositionMap,
      const IndexSet& equationIndices,
      mlir::Value value,
      llvm::StringRef variable,
      const IndexSet& variableIndices,
      EquationPath path)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto isAccessToVarFn = [&](mlir::Value value, llvm::StringRef variable) {
      mlir::Operation* definingOp = value.getDefiningOp();

      if (!definingOp) {
        return false;
      }

      while (definingOp) {
        if (auto op = mlir::dyn_cast<VariableGetOp>(definingOp)) {
          return op.getVariable() == variable;
        }

        if (auto op = mlir::dyn_cast<GlobalVariableGetOp>(definingOp)) {
          return op.getVariable() == variable;
        }

        if (auto op = mlir::dyn_cast<LoadOp>(definingOp)) {
          definingOp = op.getArray().getDefiningOp();
          continue;
        }

        if (auto op = mlir::dyn_cast<SubscriptionOp>(definingOp)) {
          definingOp = op.getSource().getDefiningOp();
          continue;
        }

        return false;
      }

      return false;
    };

    if (isAccessToVarFn(value, variable)) {
      llvm::SmallVector<VariableAccess> accesses;

      if (mlir::failed(searchAccesses(
              accesses, symbolTableCollection, inductionsPositionMap,
              value, path)) || accesses.size() != 1) {
        return std::nullopt;
      }

      if (accesses[0].getVariable().getRootReference() == variable) {
        const AccessFunction& accessFunction = accesses[0].getAccessFunction();
        auto accessedIndices = accessFunction.map(equationIndices);

        if (variableIndices == accessedIndices) {
          mlir::Value one = builder.create<ConstantOp>(
              value.getLoc(), getOneAttr(value.getType()));

          return std::make_pair(static_cast<unsigned int>(1), one);
        }
      }
    }

    mlir::Operation* op = value.getDefiningOp();

    if (auto constantOp = mlir::dyn_cast<ConstantOp>(op)) {
      return std::make_pair(
          static_cast<unsigned int>(0),
          constantOp.getResult());
    }

    if (auto negateOp = mlir::dyn_cast<NegateOp>(op)) {
      auto operand = getMultiplyingFactor(
          builder, symbolTableCollection, inductionsPositionMap,
          equationIndices, negateOp.getOperand(), variable, variableIndices,
          path + 0);

      if (!operand) {
        return std::nullopt;
      }

      if (!operand->second) {
        return std::nullopt;
      }

      mlir::Value result = builder.create<NegateOp>(
          negateOp.getLoc(), negateOp.getResult().getType(), operand->second);

      return std::make_pair(operand->first, result);
    }

    if (auto mulOp = mlir::dyn_cast<MulOp>(op)) {
      auto lhs = getMultiplyingFactor(
          builder, symbolTableCollection, inductionsPositionMap,
          equationIndices, mulOp.getLhs(), variable, variableIndices,
          path + 0);

      auto rhs = getMultiplyingFactor(
          builder, symbolTableCollection, inductionsPositionMap,
          equationIndices, mulOp.getRhs(), variable, variableIndices,
          path + 1);

      if (!lhs || !rhs) {
        return std::nullopt;
      }

      if (!lhs->second || !rhs->second) {
        return std::make_pair(static_cast<unsigned int>(0), mlir::Value());
      }

      mlir::Value result = builder.create<MulOp>(
          mulOp.getLoc(), mulOp.getResult().getType(),
          lhs->second, rhs->second);

      return std::make_pair(lhs->first + rhs->first, result);
    }

    auto hasAccessToVar = [&](mlir::Value value,
                              EquationPath path) -> std::optional<bool> {
      llvm::SmallVector<VariableAccess> accesses;

      if (mlir::failed(searchAccesses(
              accesses, symbolTableCollection, inductionsPositionMap,
              value, path))) {
        return std::nullopt;
      }

      bool hasAccess = llvm::any_of(accesses, [&](const VariableAccess& access) {
        if (access.getVariable().getRootReference().getValue() != variable) {
          return false;
        }

        const AccessFunction& accessFunction = access.getAccessFunction();
        IndexSet accessedIndices = accessFunction.map(equationIndices);

        if (accessedIndices.empty() && variableIndices.empty()) {
          return true;
        }

        return accessedIndices.overlaps(variableIndices);
      });

      if (hasAccess) {
        return true;
      }

      return false;
    };

    if (auto divOp = mlir::dyn_cast<DivOp>(op)) {
      auto dividend = getMultiplyingFactor(
          builder, symbolTableCollection, inductionsPositionMap,
          equationIndices, divOp.getLhs(), variable, variableIndices,
          path + 0);

      if (!dividend) {
        return std::nullopt;
      }

      if (!dividend->second) {
        return dividend;
      }

      // Check that the right-hand side value has no access to the variable
      // of interest.
      auto rhsHasAccess = hasAccessToVar(divOp.getRhs(), path + 1);

      if (!rhsHasAccess || *rhsHasAccess) {
        return std::nullopt;
      }

      mlir::Value result = builder.create<DivOp>(
          divOp.getLoc(), divOp.getResult().getType(), dividend->second,
          divOp.getRhs());

      return std::make_pair(dividend->first, result);
    }

    // Check that the value is not the result of an operation using the
    // variable of interest. If it has such access, then we are not able to
    // extract the multiplying factor.
    if (hasAccessToVar(value, path)) {
      return std::make_pair(static_cast<unsigned int>(1), mlir::Value());
    }

    return std::make_pair(static_cast<unsigned int>(0), value);
  }

  bool EquationTemplateOp::checkAccessEquivalence(
      const IndexSet& equationIndices,
      const VariableAccess& firstAccess,
      const VariableAccess& secondAccess)
  {
    const AccessFunction& firstAccessFunction =
        firstAccess.getAccessFunction();

    const AccessFunction& secondAccessFunction =
        secondAccess.getAccessFunction();

    IndexSet firstIndices = firstAccessFunction.map(equationIndices);
    IndexSet secondIndices = secondAccessFunction.map(equationIndices);

    if (firstIndices.empty() && secondIndices.empty()) {
      return true;
    }

    if (firstAccessFunction.getAffineMap() ==
        secondAccessFunction.getAffineMap()) {
      return true;
    }

    if (firstIndices.flatSize() == 1 && firstIndices == secondIndices) {
      return true;
    }

    return false;
  }

  mlir::LogicalResult EquationTemplateOp::groupLeftHandSide(
      mlir::RewriterBase& rewriter,
      mlir::SymbolTableCollection& symbolTableCollection,
      std::optional<MultidimensionalRange> explicitEquationIndices,
      std::optional<MultidimensionalRange> implicitEquationIndices,
      const VariableAccess& requestedAccess)
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    auto inductionsPositionMap = getInductionsPositionMap();
    uint64_t viewElementIndex = requestedAccess.getPath()[0];

    IndexSet equationIndices;

    if (explicitEquationIndices) {
      MultidimensionalRange extendedIndices = *explicitEquationIndices;

      if (implicitEquationIndices) {
        extendedIndices = extendedIndices.append(*implicitEquationIndices);
      }

      equationIndices += extendedIndices;
    } else if (implicitEquationIndices) {
      equationIndices += *implicitEquationIndices;
    }

    auto requestedValue = getValueAtPath(requestedAccess.getPath());

    // Determine whether the access to be grouped is inside both the equation's
    // sides or just one of them. When the requested access is found, also
    // check that the path goes through linear operations. If not,
    // explicitation is not possible.
    bool lhsHasAccess = false;
    bool rhsHasAccess = false;

    llvm::SmallVector<VariableAccess> accesses;

    if (mlir::failed(getAccesses(
            accesses, symbolTableCollection, viewElementIndex))) {
      return mlir::failure();
    }

    const AccessFunction& requestedAccessFunction =
        requestedAccess.getAccessFunction();

    auto requestedIndices = requestedAccessFunction.map(equationIndices);

    for (const VariableAccess& access : accesses) {
      if (access.getVariable() != access.getVariable()) {
        continue;
      }

      const AccessFunction& currentAccessFunction = access.getAccessFunction();
      auto currentAccessIndices = currentAccessFunction.map(equationIndices);

      if ((requestedIndices.empty() && currentAccessIndices.empty()) ||
          requestedIndices.overlaps(currentAccessIndices)) {
        if (!checkAccessEquivalence(
                equationIndices, requestedAccess, access)) {
          return mlir::failure();
        }

        EquationPath::EquationSide side = access.getPath().getEquationSide();
        lhsHasAccess |= side == EquationPath::LEFT;
        rhsHasAccess |= side == EquationPath::RIGHT;
      }
    }

    // Convert the expression to a sum of values.
    auto convertToSumsFn =
        [&](std::function<std::pair<mlir::Value, EquationPath>()> rootFn)
        -> mlir::LogicalResult {
      if (auto root = rootFn(); mlir::failed(
              removeSubtractions(rewriter, root.first.getDefiningOp()))) {
        return mlir::failure();
      }

      if (auto root = rootFn(); mlir::failed(
              distributeMulAndDivOps(rewriter, root.first.getDefiningOp()))) {
        return mlir::failure();
      }

      if (auto root = rootFn(); mlir::failed(
              pushNegateOps(rewriter, root.first.getDefiningOp()))) {
        return mlir::failure();
      }

      return mlir::success();
    };

    llvm::SmallVector<std::pair<mlir::Value, EquationPath>> lhsSummedValues;
    llvm::SmallVector<std::pair<mlir::Value, EquationPath>> rhsSummedValues;

    if (lhsHasAccess) {
      auto rootFn = [&]() -> std::pair<mlir::Value, EquationPath> {
        auto equationSidesOp =
            mlir::cast<EquationSidesOp>(getBody()->getTerminator());

        return std::make_pair(
            equationSidesOp.getLhsValues()[viewElementIndex],
            EquationPath(EquationPath::LEFT, viewElementIndex));
      };

      if (mlir::failed(convertToSumsFn(rootFn))) {
        return mlir::failure();
      }

      if (auto root = rootFn(); mlir::failed(collectSummedValues(
              lhsSummedValues, root.first, root.second))) {
        return mlir::failure();
      }
    }

    if (rhsHasAccess) {
      auto rootFn = [&]() -> std::pair<mlir::Value, EquationPath> {
        auto equationSidesOp =
            mlir::cast<EquationSidesOp>(getBody()->getTerminator());

        return std::make_pair(
            equationSidesOp.getRhsValues()[viewElementIndex],
            EquationPath(EquationPath::RIGHT, viewElementIndex));
      };

      if (mlir::failed(convertToSumsFn(rootFn))) {
        return mlir::failure();
      }

      if (auto root = rootFn(); mlir::failed(collectSummedValues(
              rhsSummedValues, root.first, root.second))) {
        return mlir::failure();
      }
    }

    auto containsAccessFn =
        [&](bool& result,
            mlir::Value value,
            EquationPath path,
            const VariableAccess& access) -> mlir::LogicalResult {
      llvm::SmallVector<VariableAccess> accesses;

      if (mlir::failed(searchAccesses(
              accesses, symbolTableCollection, inductionsPositionMap,
              value, path))) {
        return mlir::failure();
      }

      const AccessFunction& accessFunction = access.getAccessFunction();

      result = llvm::any_of(accesses, [&](const VariableAccess& acc) {
        if (acc.getVariable() != access.getVariable()) {
          return false;
        }

        IndexSet requestedIndices = accessFunction.map(equationIndices);

        const AccessFunction& currentAccessFunction = acc.getAccessFunction();
        auto currentIndices = currentAccessFunction.map(equationIndices);

        assert(requestedIndices == currentIndices ||
               !requestedIndices.overlaps(currentIndices));
        return requestedIndices == currentIndices;
      });

      return mlir::success();
    };

    auto groupFactorsFn = [&](auto beginIt, auto endIt) -> mlir::Value {
      mlir::Value result = rewriter.create<ConstantOp>(
          getOperation()->getLoc(), RealAttr::get(rewriter.getContext(), 0));

      for (auto it = beginIt; it != endIt; ++it) {
        auto factor = getMultiplyingFactor(
            rewriter, symbolTableCollection, inductionsPositionMap,
            equationIndices, it->first,
            requestedAccess.getVariable().getRootReference().getValue(),
            requestedIndices, it->second);

        if (!factor) {
          return nullptr;
        }

        if (!factor->second || factor->first > 1) {
          return nullptr;
        }

        result = rewriter.create<AddOp>(
            it->first.getLoc(),
            getMostGenericType(result.getType(), it->first.getType()),
            result, factor->second);
      }

      return result;
    };

    auto groupRemainingFn = [&](auto beginIt, auto endIt) -> mlir::Value {
      auto zeroConstantOp = rewriter.create<ConstantOp>(
          getOperation()->getLoc(), RealAttr::get(rewriter.getContext(), 0));

      mlir::Value result = zeroConstantOp.getResult();

      for (auto it = beginIt; it != endIt; ++it) {
        mlir::Value value = it->first;

        result = rewriter.create<AddOp>(
            value.getLoc(),
            getMostGenericType(result.getType(), value.getType()),
            result, value);
      }

      return result;
    };

    if (lhsHasAccess && rhsHasAccess) {
      bool error = false;

      auto leftPos = llvm::partition(lhsSummedValues, [&](const auto& value) {
        bool result = false;

        if (mlir::failed(containsAccessFn(
                result, value.first, value.second, requestedAccess))) {
          error = true;
          return false;
        }

        return result;
      });

      auto rightPos = llvm::partition(rhsSummedValues, [&](const auto& value) {
        bool result = false;

        if (mlir::failed(containsAccessFn(
                result, value.first, value.second, requestedAccess))) {
          error = true;
          return false;
        }

        return result;
      });

      if (error) {
        return mlir::failure();
      }

      mlir::Value lhsFactor = groupFactorsFn(lhsSummedValues.begin(), leftPos);
      mlir::Value rhsFactor = groupFactorsFn(rhsSummedValues.begin(), rightPos);

      if (!lhsFactor || !rhsFactor) {
        return mlir::failure();
      }

      mlir::Value lhsRemaining = groupRemainingFn(leftPos, lhsSummedValues.end());
      mlir::Value rhsRemaining = groupRemainingFn(rightPos, rhsSummedValues.end());

      auto rhs = rewriter.create<DivOp>(
          getLoc(), requestedValue.getType(),
          rewriter.create<SubOp>(
              getLoc(),
              getMostGenericType(
                  rhsRemaining.getType(), lhsRemaining.getType()),
              rhsRemaining, lhsRemaining),
          rewriter.create<SubOp>(
              getLoc(),
              getMostGenericType(
                  lhsFactor.getType(), rhsFactor.getType()),
              lhsFactor, rhsFactor));

      // Check if we are dividing by zero.
      foldValue(rewriter, rhs.getRhs(), getBody());

      if (auto divisorOp =
              mlir::dyn_cast<ConstantOp>(rhs.getRhs().getDefiningOp())) {
        std::optional<bool> isZero = isZeroAttr(divisorOp.getValue());

        if (!isZero || *isZero) {
          return mlir::failure();
        }
      }

      auto equationSidesOp =
          mlir::cast<EquationSidesOp>(getBody()->getTerminator());

      auto lhsOp = equationSidesOp.getLhs().getDefiningOp<EquationSideOp>();
      auto rhsOp = equationSidesOp.getRhs().getDefiningOp<EquationSideOp>();

      auto oldLhsValues = lhsOp.getValues();
      llvm::SmallVector<mlir::Value> newLhsValues(
          oldLhsValues.begin(), oldLhsValues.end());

      auto oldRhsValues = rhsOp.getValues();
      llvm::SmallVector<mlir::Value> newRhsValues(
          oldRhsValues.begin(), oldRhsValues.end());

      newLhsValues[viewElementIndex] = requestedValue;
      newRhsValues[viewElementIndex] = rhs.getResult();

      rewriter.setInsertionPoint(lhsOp);
      rewriter.replaceOpWithNewOp<EquationSideOp>(lhsOp, newLhsValues);

      rewriter.setInsertionPoint(rhsOp);
      rewriter.replaceOpWithNewOp<EquationSideOp>(rhsOp, newRhsValues);

      return mlir::success();
    }

    if (lhsHasAccess) {
      bool error = false;

      auto leftPos = llvm::partition(lhsSummedValues, [&](const auto& value) {
        bool result = false;

        if (mlir::failed(containsAccessFn(
                result, value.first, value.second, requestedAccess))) {
          error = true;
          return false;
        }

        return result;
      });

      if (error) {
        return mlir::failure();
      }

      mlir::Value lhsFactor = groupFactorsFn(lhsSummedValues.begin(), leftPos);

      if (!lhsFactor) {
        return mlir::failure();
      }

      mlir::Value lhsRemaining = groupRemainingFn(leftPos, lhsSummedValues.end());

      auto equationSidesOp =
          mlir::cast<EquationSidesOp>(getBody()->getTerminator());

      auto rhs = rewriter.create<DivOp>(
          getLoc(), requestedValue.getType(),
          rewriter.create<SubOp>(
              getLoc(),
              getMostGenericType(
                  equationSidesOp.getRhsValues()[viewElementIndex].getType(),
                  lhsRemaining.getType()),
              equationSidesOp.getRhsValues()[viewElementIndex],
              lhsRemaining),
          lhsFactor);

      // Check if we are dividing by zero.
      foldValue(rewriter, rhs.getRhs(), getBody());

      if (auto divisorOp =
              mlir::dyn_cast<ConstantOp>(rhs.getRhs().getDefiningOp())) {
        std::optional<bool> isZero = isZeroAttr(divisorOp.getValue());

        if (!isZero || *isZero) {
          return mlir::failure();
        }
      }

      auto lhsOp = equationSidesOp.getLhs().getDefiningOp<EquationSideOp>();
      auto rhsOp = equationSidesOp.getRhs().getDefiningOp<EquationSideOp>();

      auto oldLhsValues = lhsOp.getValues();
      llvm::SmallVector<mlir::Value> newLhsValues(
          oldLhsValues.begin(), oldLhsValues.end());

      auto oldRhsValues = rhsOp.getValues();
      llvm::SmallVector<mlir::Value> newRhsValues(
          oldRhsValues.begin(), oldRhsValues.end());

      newLhsValues[viewElementIndex] = requestedValue;
      newRhsValues[viewElementIndex] = rhs.getResult();

      rewriter.setInsertionPoint(lhsOp);
      rewriter.replaceOpWithNewOp<EquationSideOp>(lhsOp, newLhsValues);

      rewriter.setInsertionPoint(rhsOp);
      rewriter.replaceOpWithNewOp<EquationSideOp>(rhsOp, newRhsValues);

      return mlir::success();
    }

    if (rhsHasAccess) {
      bool error = false;

      auto rightPos = llvm::partition(rhsSummedValues, [&](const auto& value) {
        bool result = false;

        if (mlir::failed(containsAccessFn(
                result, value.first, value.second, requestedAccess))) {
          error = true;
          return false;
        }

        return result;
      });

      if (error) {
        return mlir::failure();
      }

      mlir::Value rhsFactor = groupFactorsFn(rhsSummedValues.begin(), rightPos);

      if (!rhsFactor) {
        return mlir::failure();
      }

      mlir::Value rhsRemaining = groupRemainingFn(rightPos, rhsSummedValues.end());

      auto equationSidesOp =
          mlir::cast<EquationSidesOp>(getBody()->getTerminator());

      auto rhs = rewriter.create<DivOp>(
          getLoc(), requestedValue.getType(),
          rewriter.create<SubOp>(
              getLoc(),
              getMostGenericType(
                  equationSidesOp.getLhsValues()[viewElementIndex].getType(),
                  rhsRemaining.getType()),
              equationSidesOp.getLhsValues()[viewElementIndex],
              rhsRemaining),
          rhsFactor);

      // Check if we are dividing by zero.
      foldValue(rewriter, rhs.getRhs(), getBody());

      if (auto divisorOp =
              mlir::dyn_cast<ConstantOp>(rhs.getRhs().getDefiningOp())) {
        std::optional<bool> isZero = isZeroAttr(divisorOp.getValue());

        if (!isZero || *isZero) {
          return mlir::failure();
        }
      }

      auto lhsOp = equationSidesOp.getLhs().getDefiningOp<EquationSideOp>();
      auto rhsOp = equationSidesOp.getRhs().getDefiningOp<EquationSideOp>();

      auto oldLhsValues = lhsOp.getValues();
      llvm::SmallVector<mlir::Value> newLhsValues(
          oldLhsValues.begin(), oldLhsValues.end());

      auto oldRhsValues = rhsOp.getValues();
      llvm::SmallVector<mlir::Value> newRhsValues(
          oldRhsValues.begin(), oldRhsValues.end());

      newLhsValues[viewElementIndex] = requestedValue;
      newRhsValues[viewElementIndex] = rhs.getResult();

      rewriter.setInsertionPoint(lhsOp);
      rewriter.replaceOpWithNewOp<EquationSideOp>(lhsOp, newLhsValues);

      rewriter.setInsertionPoint(rhsOp);
      rewriter.replaceOpWithNewOp<EquationSideOp>(rhsOp, newRhsValues);

      return mlir::success();
    }

    llvm_unreachable("Access not found");
    return mlir::failure();
  }
}

//===---------------------------------------------------------------------===//
// EquationInstanceOp

namespace mlir::modelica
{
  void EquationInstanceOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      EquationTemplateOp equationTemplate,
      bool initial)
  {
    build(builder, state, equationTemplate.getResult(), initial,
          nullptr, nullptr, nullptr);
  }

  mlir::LogicalResult EquationInstanceOp::verify()
  {
    auto indicesRank =
        [&](std::optional<MultidimensionalRangeAttr> ranges) -> size_t {
      if (!ranges) {
        return 0;
      }

      return ranges->getValue().rank();
    };

    // Check the indices for the explicit inductions.
    size_t numOfExplicitInductions = getInductionVariables().size();

    if (size_t explicitIndicesRank = indicesRank(getIndices());
        numOfExplicitInductions != explicitIndicesRank) {
      return emitOpError()
          << "Unexpected rank of iteration indices (expected "
          << numOfExplicitInductions << ", got " << explicitIndicesRank << ")";
    }

    return mlir::success();
  }

  EquationTemplateOp EquationInstanceOp::getTemplate()
  {
    auto result = getBase().getDefiningOp<EquationTemplateOp>();
    assert(result != nullptr);
    return result;
  }

  mlir::ValueRange EquationInstanceOp::getInductionVariables()
  {
    return getTemplate().getInductionVariables();
  }

  IndexSet EquationInstanceOp::getExplicitIterationSpace()
  {
    if (auto indices = getIndices()) {
      return IndexSet(indices->getValue());
    }

    return {};
  }

  uint64_t EquationInstanceOp::getNumOfImplicitInductionVariables()
  {
    if (auto implicitIndices = getImplicitIndices()) {
      return static_cast<uint64_t>(implicitIndices->getValue().rank());
    }

    return 0;
  }

  IndexSet EquationInstanceOp::getImplicitIterationSpace()
  {
    if (auto indices = getImplicitIndices()) {
      return IndexSet(indices->getValue());
    }

    return {};
  }

  IndexSet EquationInstanceOp::getIterationSpace()
  {
    IndexSet explicitIterationSpace = getExplicitIterationSpace();
    IndexSet implicitIterationSpace = getImplicitIterationSpace();

    if (explicitIterationSpace.empty()) {
      return implicitIterationSpace;
    }

    return explicitIterationSpace.append(implicitIterationSpace);
  }

  mlir::LogicalResult EquationInstanceOp::getAccesses(
      llvm::SmallVectorImpl<VariableAccess>& result,
      mlir::SymbolTableCollection& symbolTable)
  {
    return getTemplate().getAccesses(
        result, symbolTable, getViewElementIndex().value_or(0));
  }

  std::optional<VariableAccess> EquationInstanceOp::getAccessAtPath(
      mlir::SymbolTableCollection& symbolTable,
      const EquationPath& path)
  {
    return getTemplate().getAccessAtPath(symbolTable, path);
  }

  mlir::LogicalResult EquationInstanceOp::cloneWithReplacedAccess(
      mlir::RewriterBase& rewriter,
      std::optional<std::reference_wrapper<const IndexSet>> equationIndices,
      const VariableAccess& access,
      EquationTemplateOp replacementEquation,
      const VariableAccess& replacementAccess,
      llvm::SmallVectorImpl<EquationInstanceOp>& results)
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    llvm::SmallVector<std::pair<IndexSet, EquationTemplateOp>> templateResults;

    if (mlir::failed(getTemplate().cloneWithReplacedAccess(
            rewriter, equationIndices, access, replacementEquation,
            replacementAccess, templateResults))) {
      return mlir::failure();
    }

    rewriter.setInsertionPointAfter(getOperation());

    auto temporaryClonedOp = mlir::cast<EquationInstanceOp>(
        rewriter.clone(*getOperation()));

    for (auto& [assignedIndices, equationTemplateOp] : templateResults) {
      if (assignedIndices.empty()) {
        auto clonedOp = mlir::cast<EquationInstanceOp>(
            rewriter.clone(*temporaryClonedOp.getOperation()));

        clonedOp.setOperand(equationTemplateOp.getResult());
        clonedOp.removeIndicesAttr();
        clonedOp.removeImplicitIndicesAttr();
        results.push_back(clonedOp);
      } else {
        for (const MultidimensionalRange& assignedIndicesRange :
             llvm::make_range(assignedIndices.rangesBegin(),
                              assignedIndices.rangesEnd())) {
          auto clonedOp = mlir::cast<EquationInstanceOp>(
              rewriter.clone(*temporaryClonedOp.getOperation()));

          clonedOp.setOperand(equationTemplateOp.getResult());

          if (auto explicitIndices = getIndices()) {
            MultidimensionalRange explicitRange =
                assignedIndicesRange.takeFirstDimensions(
                    explicitIndices->getValue().rank());

            clonedOp.setIndicesAttr(
                MultidimensionalRangeAttr::get(
                    rewriter.getContext(), std::move(explicitRange)));
          }

          if (auto implicitIndices = getImplicitIndices()) {
            MultidimensionalRange implicitRange =
                assignedIndicesRange.takeLastDimensions(
                    implicitIndices->getValue().rank());

            clonedOp.setImplicitIndicesAttr(
                MultidimensionalRangeAttr::get(
                    rewriter.getContext(), std::move(implicitRange)));
          }

          results.push_back(clonedOp);
        }
      }
    }

    rewriter.eraseOp(temporaryClonedOp);
    return mlir::success();
  }
}

//===---------------------------------------------------------------------===//
// MatchedEquationInstanceOp

namespace mlir::modelica
{
  void MatchedEquationInstanceOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      EquationTemplateOp equationTemplate,
      bool initial,
      EquationPathAttr path)
  {
    build(builder, state, equationTemplate.getResult(), initial, nullptr,
          nullptr, path);
  }

  mlir::LogicalResult MatchedEquationInstanceOp::verify()
  {
    auto indicesRank =
        [&](std::optional<MultidimensionalRangeAttr> ranges) -> size_t {
      if (!ranges) {
        return 0;
      }

      return ranges->getValue().rank();
    };

    // Check the indices for the explicit inductions.
    size_t numOfExplicitInductions = getInductionVariables().size();

    if (size_t explicitIndicesRank = indicesRank(getIndices());
        numOfExplicitInductions != explicitIndicesRank) {
      return emitOpError()
          << "Unexpected rank of iteration indices (expected "
          << numOfExplicitInductions << ", got " << explicitIndicesRank << ")";
    }

    // Check the indices for the implicit inductions.
    uint64_t numOfImplicitInductions = getNumOfImplicitInductionVariables();

    if (size_t implicitIndicesRank = indicesRank(getImplicitIndices());
        numOfImplicitInductions !=
        static_cast<uint64_t>(implicitIndicesRank)) {
      return emitOpError()
          << "Unexpected rank of iteration indices (expected "
          << numOfImplicitInductions << ", got " << implicitIndicesRank << ")";
    }

    return mlir::success();
  }

  EquationTemplateOp MatchedEquationInstanceOp::getTemplate()
  {
    auto result = getBase().getDefiningOp<EquationTemplateOp>();
    assert(result != nullptr);
    return result;
  }

  uint64_t MatchedEquationInstanceOp::getViewElementIndex()
  {
    return getPath().getValue()[0];
  }

  mlir::ValueRange MatchedEquationInstanceOp::getInductionVariables()
  {
    return getTemplate().getInductionVariables();
  }

  IndexSet MatchedEquationInstanceOp::getExplicitIterationSpace()
  {
    if (auto indices = getIndices()) {
      return IndexSet(indices->getValue());
    }

    return {};
  }

  uint64_t MatchedEquationInstanceOp::getNumOfImplicitInductionVariables()
  {
    if (auto implicitIndices = getImplicitIndices()) {
      return static_cast<uint64_t>(implicitIndices->getValue().rank());
    }

    return 0;
  }

  IndexSet MatchedEquationInstanceOp::getImplicitIterationSpace()
  {
    if (auto indices = getImplicitIndices()) {
      return IndexSet(indices->getValue());
    }

    return {};
  }

  IndexSet MatchedEquationInstanceOp::getIterationSpace()
  {
    IndexSet explicitIterationSpace = getExplicitIterationSpace();
    IndexSet implicitIterationSpace = getImplicitIterationSpace();

    if (explicitIterationSpace.empty()) {
      return implicitIterationSpace;
    }

    return explicitIterationSpace.append(implicitIterationSpace);
  }

  std::optional<VariableAccess> MatchedEquationInstanceOp::getMatchedAccess(
      mlir::SymbolTableCollection& symbolTableCollection)
  {
    return getAccessAtPath(symbolTableCollection, getPath().getValue());
  }

  mlir::LogicalResult MatchedEquationInstanceOp::getAccesses(
      llvm::SmallVectorImpl<VariableAccess>& result,
      mlir::SymbolTableCollection& symbolTable)
  {
    return getTemplate().getAccesses(
        result, symbolTable, getViewElementIndex());
  }

  mlir::LogicalResult MatchedEquationInstanceOp::getWriteAccesses(
      llvm::SmallVectorImpl<VariableAccess>& result,
      mlir::SymbolTableCollection& symbolTableCollection,
      llvm::ArrayRef<VariableAccess> accesses)
  {
    return getWriteAccesses(
        result, symbolTableCollection, getIterationSpace(), accesses);
  }

  mlir::LogicalResult MatchedEquationInstanceOp::getWriteAccesses(
      llvm::SmallVectorImpl<VariableAccess>& result,
      mlir::SymbolTableCollection& symbolTableCollection,
      const IndexSet& equationIndices,
      llvm::ArrayRef<VariableAccess> accesses)
  {
    std::optional<VariableAccess> matchedAccess =
        getMatchedAccess(symbolTableCollection);

    if (!matchedAccess) {
      return mlir::failure();
    }

    return getTemplate().getWriteAccesses(
        result, equationIndices, accesses, *matchedAccess);
  }

  mlir::LogicalResult MatchedEquationInstanceOp::getReadAccesses(
      llvm::SmallVectorImpl<VariableAccess>& result,
      mlir::SymbolTableCollection& symbolTableCollection,
      llvm::ArrayRef<VariableAccess> accesses)
  {
    return getReadAccesses(
        result, symbolTableCollection, getIterationSpace(), accesses);
  }

  mlir::LogicalResult MatchedEquationInstanceOp::getReadAccesses(
      llvm::SmallVectorImpl<VariableAccess>& result,
      mlir::SymbolTableCollection& symbolTableCollection,
      const IndexSet& equationIndices,
      llvm::ArrayRef<VariableAccess> accesses)
  {
    std::optional<VariableAccess> matchedAccess =
        getMatchedAccess(symbolTableCollection);

    if (!matchedAccess) {
      return mlir::failure();
    }

    return getTemplate().getReadAccesses(
        result, equationIndices, accesses, *matchedAccess);
  }

  std::optional<VariableAccess> MatchedEquationInstanceOp::getAccessAtPath(
      mlir::SymbolTableCollection& symbolTable,
      const EquationPath& path)
  {
    return getTemplate().getAccessAtPath(symbolTable, path);
  }

  mlir::LogicalResult MatchedEquationInstanceOp::explicitate(
      mlir::RewriterBase& rewriter,
      mlir::SymbolTableCollection& symbolTableCollection)
  {
    std::optional<MultidimensionalRange> indices = std::nullopt;
    std::optional<MultidimensionalRange> implicitIndices = std::nullopt;

    if (auto indicesAttr = getIndices()) {
      indices = indicesAttr->getValue();
    }

    if (auto implicitIndicesAttr = getImplicitIndices()) {
      implicitIndices = implicitIndicesAttr->getValue();
    }

    if (mlir::failed(getTemplate().explicitate(
            rewriter, symbolTableCollection,
            indices, implicitIndices, getPath().getValue()))) {
      return mlir::failure();
    }

    setPathAttr(EquationPathAttr::get(
        getContext(),
        EquationPath(EquationPath::LEFT, getViewElementIndex())));

    return mlir::success();
  }

  MatchedEquationInstanceOp MatchedEquationInstanceOp::cloneAndExplicitate(
      mlir::RewriterBase& rewriter,
      mlir::SymbolTableCollection& symbolTableCollection)
  {
    std::optional<MultidimensionalRange> indices = std::nullopt;
    std::optional<MultidimensionalRange> implicitIndices = std::nullopt;

    if (auto indicesAttr = getIndices()) {
      indices = indicesAttr->getValue();
    }

    if (auto implicitIndicesAttr = getImplicitIndices()) {
      implicitIndices = implicitIndicesAttr->getValue();
    }

    EquationTemplateOp clonedTemplate = getTemplate().cloneAndExplicitate(
        rewriter, symbolTableCollection,
        indices, implicitIndices,
        getPath().getValue());

    if (!clonedTemplate) {
      return nullptr;
    }

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(getOperation());

    auto result = rewriter.create<MatchedEquationInstanceOp>(
        getLoc(), clonedTemplate, getInitial(),
        EquationPathAttr::get(
            getContext(),
            EquationPath(EquationPath::LEFT, getViewElementIndex())));

    if (indices) {
      result.setIndicesAttr(MultidimensionalRangeAttr::get(
          getContext(), *indices));
    }

    if (implicitIndices) {
      result.setImplicitIndicesAttr(MultidimensionalRangeAttr::get(
          getContext(), *implicitIndices));
    }

    return result;
  }

  mlir::LogicalResult MatchedEquationInstanceOp::cloneWithReplacedAccess(
      mlir::RewriterBase& rewriter,
      std::optional<std::reference_wrapper<const IndexSet>> equationIndices,
      const VariableAccess& access,
      EquationTemplateOp replacementEquation,
      const VariableAccess& replacementAccess,
      llvm::SmallVectorImpl<MatchedEquationInstanceOp>& results)
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    llvm::SmallVector<std::pair<IndexSet, EquationTemplateOp>> templateResults;

    if (mlir::failed(getTemplate().cloneWithReplacedAccess(
            rewriter, equationIndices, access, replacementEquation,
            replacementAccess, templateResults))) {
      return mlir::failure();
    }

    rewriter.setInsertionPointAfter(getOperation());

    auto temporaryClonedOp = mlir::cast<MatchedEquationInstanceOp>(
        rewriter.clone(*getOperation()));

    for (auto& [assignedIndices, equationTemplateOp] : templateResults) {
      if (assignedIndices.empty()) {
        auto clonedOp = mlir::cast<MatchedEquationInstanceOp>(
            rewriter.clone(*temporaryClonedOp.getOperation()));

        clonedOp.setOperand(equationTemplateOp.getResult());
        clonedOp.removeIndicesAttr();
        clonedOp.removeImplicitIndicesAttr();
        results.push_back(clonedOp);
      } else {
        for (const MultidimensionalRange& assignedIndicesRange :
             llvm::make_range(assignedIndices.rangesBegin(),
                              assignedIndices.rangesEnd())) {
          auto clonedOp = mlir::cast<MatchedEquationInstanceOp>(
              rewriter.clone(*temporaryClonedOp.getOperation()));

          clonedOp.setOperand(equationTemplateOp.getResult());

          if (auto explicitIndices = getIndices()) {
            MultidimensionalRange explicitRange =
                assignedIndicesRange.takeFirstDimensions(
                    explicitIndices->getValue().rank());

            clonedOp.setIndicesAttr(
                MultidimensionalRangeAttr::get(
                    rewriter.getContext(), std::move(explicitRange)));
          }

          if (auto implicitIndices = getImplicitIndices()) {
            MultidimensionalRange implicitRange =
                assignedIndicesRange.takeLastDimensions(
                    implicitIndices->getValue().rank());

            clonedOp.setImplicitIndicesAttr(
                MultidimensionalRangeAttr::get(
                    rewriter.getContext(), std::move(implicitRange)));
          }

          results.push_back(clonedOp);
        }
      }
    }

    rewriter.eraseOp(temporaryClonedOp);
    return mlir::success();
  }
}

//===---------------------------------------------------------------------===//
// SCCOp

namespace mlir::modelica
{
  mlir::RegionKind SCCOp::getRegionKind(unsigned index)
  {
    return mlir::RegionKind::Graph;
  }

  void SCCOp::collectEquations(
      llvm::SmallVectorImpl<ScheduledEquationInstanceOp>& equations)
  {
    for (ScheduledEquationInstanceOp equationOp
         : getOps<ScheduledEquationInstanceOp>()) {
      equations.push_back(equationOp);
    }
  }
}

//===---------------------------------------------------------------------===//
// ScheduledEquationInstanceOp

namespace mlir::modelica
{
  void ScheduledEquationInstanceOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      EquationTemplateOp equationTemplate,
      EquationPathAttr path,
      mlir::ArrayAttr iterationDirections)
  {
    build(builder, state, equationTemplate.getResult(), nullptr,
          nullptr, path, iterationDirections);
  }

  mlir::LogicalResult ScheduledEquationInstanceOp::verify()
  {
    auto indicesRank =
        [&](std::optional<MultidimensionalRangeAttr> ranges) -> size_t {
      if (!ranges) {
        return 0;
      }

      return ranges->getValue().rank();
    };

    // Check the indices for the explicit inductions.
    size_t numOfExplicitInductions = getInductionVariables().size();

    if (size_t explicitIndicesRank = indicesRank(getIndices());
        numOfExplicitInductions != explicitIndicesRank) {
      return emitOpError()
          << "Unexpected rank of iteration indices (expected "
          << numOfExplicitInductions << ", got " << explicitIndicesRank << ")";
    }

    // Check the indices for the implicit inductions.
    uint64_t numOfImplicitInductions = getNumOfImplicitInductionVariables();

    if (size_t implicitIndicesRank = indicesRank(getImplicitIndices());
        numOfImplicitInductions !=
        static_cast<uint64_t>(implicitIndicesRank)) {
      return emitOpError()
          << "Unexpected rank of iteration indices (expected "
          << numOfImplicitInductions << ", got " << implicitIndicesRank << ")";
    }

    // Check the iteration directions.
    uint64_t totalNumberOfInductions =
        static_cast<uint64_t>(numOfExplicitInductions) +
        numOfImplicitInductions;

    if (size_t numOfIterationDirections = getIterationDirections().size();
        totalNumberOfInductions !=
        static_cast<uint64_t>(numOfIterationDirections)) {
      return emitOpError()
          << "Unexpected number of iteration directions (expected "
          << totalNumberOfInductions << ", got " << numOfIterationDirections
          << ")";
    }

    return mlir::success();
  }

  EquationTemplateOp ScheduledEquationInstanceOp::getTemplate()
  {
    auto result = getBase().getDefiningOp<EquationTemplateOp>();
    assert(result != nullptr);
    return result;
  }

  uint64_t ScheduledEquationInstanceOp::getViewElementIndex()
  {
    return getPath().getValue()[0];
  }

  mlir::ValueRange ScheduledEquationInstanceOp::getInductionVariables()
  {
    return getTemplate().getInductionVariables();
  }

  mlir::LogicalResult ScheduledEquationInstanceOp::getAccesses(
      llvm::SmallVectorImpl<VariableAccess>& result,
      mlir::SymbolTableCollection& symbolTable)
  {
    return getTemplate().getAccesses(
        result, symbolTable, getViewElementIndex());
  }

  std::optional<VariableAccess> ScheduledEquationInstanceOp::getAccessAtPath(
      mlir::SymbolTableCollection& symbolTable,
      const EquationPath& path)
  {
    return getTemplate().getAccessAtPath(symbolTable, path);
  }

  IndexSet ScheduledEquationInstanceOp::getExplicitIterationSpace()
  {
    if (auto indices = getIndices()) {
      return IndexSet(indices->getValue());
    }

    return {};
  }

  uint64_t ScheduledEquationInstanceOp::getNumOfImplicitInductionVariables()
  {
    if (auto implicitIndices = getImplicitIndices()) {
      return static_cast<uint64_t>(implicitIndices->getValue().rank());
    }

    return 0;
  }

  IndexSet ScheduledEquationInstanceOp::getImplicitIterationSpace()
  {
    if (auto indices = getImplicitIndices()) {
      return IndexSet(indices->getValue());
    }

    return {};
  }

  IndexSet ScheduledEquationInstanceOp::getIterationSpace()
  {
    IndexSet explicitIterationSpace = getExplicitIterationSpace();
    IndexSet implicitIterationSpace = getImplicitIterationSpace();

    if (explicitIterationSpace.empty()) {
      return implicitIterationSpace;
    }

    return explicitIterationSpace.append(implicitIterationSpace);
  }

  std::optional<VariableAccess> ScheduledEquationInstanceOp::getMatchedAccess(
      mlir::SymbolTableCollection& symbolTableCollection)
  {
    return getAccessAtPath(symbolTableCollection, getPath().getValue());
  }

  mlir::LogicalResult ScheduledEquationInstanceOp::getWriteAccesses(
      llvm::SmallVectorImpl<VariableAccess>& result,
      mlir::SymbolTableCollection& symbolTableCollection,
      llvm::ArrayRef<VariableAccess> accesses)
  {
    return getWriteAccesses(
        result, symbolTableCollection, getIterationSpace(), accesses);
  }

  mlir::LogicalResult ScheduledEquationInstanceOp::getWriteAccesses(
      llvm::SmallVectorImpl<VariableAccess>& result,
      mlir::SymbolTableCollection& symbolTableCollection,
      const IndexSet& equationIndices,
      llvm::ArrayRef<VariableAccess> accesses)
  {
    std::optional<VariableAccess> matchedAccess =
        getMatchedAccess(symbolTableCollection);

    if (!matchedAccess) {
      return mlir::failure();
    }

    return getTemplate().getWriteAccesses(
        result, equationIndices, accesses, *matchedAccess);
  }

  mlir::LogicalResult ScheduledEquationInstanceOp::getReadAccesses(
      llvm::SmallVectorImpl<VariableAccess>& result,
      mlir::SymbolTableCollection& symbolTableCollection,
      llvm::ArrayRef<VariableAccess> accesses)
  {
    return getReadAccesses(
        result, symbolTableCollection, getIterationSpace(), accesses);
  }

  mlir::LogicalResult ScheduledEquationInstanceOp::getReadAccesses(
      llvm::SmallVectorImpl<VariableAccess>& result,
      mlir::SymbolTableCollection& symbolTableCollection,
      const IndexSet& equationIndices,
      llvm::ArrayRef<VariableAccess> accesses)
  {
    std::optional<VariableAccess> matchedAccess =
        getMatchedAccess(symbolTableCollection);

    if (!matchedAccess) {
      return mlir::failure();
    }

    return getTemplate().getReadAccesses(
        result, equationIndices, accesses, *matchedAccess);
  }

  mlir::LogicalResult ScheduledEquationInstanceOp::explicitate(
      mlir::RewriterBase& rewriter,
      mlir::SymbolTableCollection& symbolTableCollection)
  {
    std::optional<MultidimensionalRange> indices = std::nullopt;
    std::optional<MultidimensionalRange> implicitIndices = std::nullopt;

    if (auto indicesAttr = getIndices()) {
      indices = indicesAttr->getValue();
    }

    if (auto implicitIndicesAttr = getImplicitIndices()) {
      implicitIndices = implicitIndicesAttr->getValue();
    }

    if (mlir::failed(getTemplate().explicitate(
            rewriter, symbolTableCollection,
            indices, implicitIndices, getPath().getValue()))) {
      return mlir::failure();
    }

    setPathAttr(EquationPathAttr::get(
        getContext(),
        EquationPath(EquationPath::LEFT, getViewElementIndex())));

    return mlir::success();
  }

  ScheduledEquationInstanceOp ScheduledEquationInstanceOp::cloneAndExplicitate(
      mlir::RewriterBase& rewriter,
      mlir::SymbolTableCollection& symbolTableCollection)
  {
    std::optional<MultidimensionalRange> indices = std::nullopt;
    std::optional<MultidimensionalRange> implicitIndices = std::nullopt;

    if (auto indicesAttr = getIndices()) {
      indices = indicesAttr->getValue();
    }

    if (auto implicitIndicesAttr = getImplicitIndices()) {
      implicitIndices = implicitIndicesAttr->getValue();
    }

    EquationTemplateOp clonedTemplate = getTemplate().cloneAndExplicitate(
        rewriter, symbolTableCollection,
        indices, implicitIndices,
        getPath().getValue());

    if (!clonedTemplate) {
      return nullptr;
    }

    mlir::OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(getOperation());

    auto result = rewriter.create<ScheduledEquationInstanceOp>(
        getLoc(), clonedTemplate,
        EquationPathAttr::get(
            getContext(),
            EquationPath(EquationPath::LEFT, getViewElementIndex())),
        getIterationDirections());

    if (indices) {
      result.setIndicesAttr(MultidimensionalRangeAttr::get(
          getContext(), *indices));
    }

    if (implicitIndices) {
      result.setImplicitIndicesAttr(MultidimensionalRangeAttr::get(
          getContext(), *implicitIndices));
    }

    return result;
  }

  mlir::LogicalResult ScheduledEquationInstanceOp::cloneWithReplacedAccess(
      mlir::RewriterBase& rewriter,
      std::optional<std::reference_wrapper<const IndexSet>> equationIndices,
      const VariableAccess& access,
      EquationTemplateOp replacementEquation,
      const VariableAccess& replacementAccess,
      llvm::SmallVectorImpl<ScheduledEquationInstanceOp>& results)
  {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    llvm::SmallVector<std::pair<IndexSet, EquationTemplateOp>> templateResults;

    if (mlir::failed(getTemplate().cloneWithReplacedAccess(
            rewriter, equationIndices, access, replacementEquation,
            replacementAccess, templateResults))) {
      return mlir::failure();
    }

    rewriter.setInsertionPointAfter(getOperation());

    auto temporaryClonedOp = mlir::cast<ScheduledEquationInstanceOp>(
        rewriter.clone(*getOperation()));

    for (auto& [assignedIndices, equationTemplateOp] : templateResults) {
      if (assignedIndices.empty()) {
        auto clonedOp = mlir::cast<ScheduledEquationInstanceOp>(
            rewriter.clone(*temporaryClonedOp.getOperation()));

        clonedOp.setOperand(equationTemplateOp.getResult());
        clonedOp.removeIndicesAttr();
        clonedOp.removeImplicitIndicesAttr();
        results.push_back(clonedOp);
      } else {
        for (const MultidimensionalRange& assignedIndicesRange :
             llvm::make_range(assignedIndices.rangesBegin(),
                              assignedIndices.rangesEnd())) {
          auto clonedOp = mlir::cast<ScheduledEquationInstanceOp>(
              rewriter.clone(*temporaryClonedOp.getOperation()));

          clonedOp.setOperand(equationTemplateOp.getResult());

          if (auto explicitIndices = getIndices()) {
            MultidimensionalRange explicitRange =
                assignedIndicesRange.takeFirstDimensions(
                    explicitIndices->getValue().rank());

            clonedOp.setIndicesAttr(
                MultidimensionalRangeAttr::get(
                    rewriter.getContext(), std::move(explicitRange)));
          }

          if (auto implicitIndices = getImplicitIndices()) {
            MultidimensionalRange implicitRange =
                assignedIndicesRange.takeLastDimensions(
                    implicitIndices->getValue().rank());

            clonedOp.setImplicitIndicesAttr(
                MultidimensionalRangeAttr::get(
                    rewriter.getContext(), std::move(implicitRange)));
          }

          results.push_back(clonedOp);
        }
      }
    }

    rewriter.eraseOp(temporaryClonedOp);
    return mlir::success();
  }
}

//===---------------------------------------------------------------------===//
// EquationSideOp

namespace mlir::modelica
{
  mlir::ParseResult EquationSideOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 1> values;
    mlir::Type resultType;
    auto loc = parser.getCurrentLocation();

    if (parser.parseOperandList(values) ||
        parser.parseColon() ||
        parser.parseType(resultType)) {
      return mlir::failure();
    }

    assert(resultType.isa<mlir::TupleType>());
    auto tupleType = resultType.cast<mlir::TupleType>();

    llvm::SmallVector<mlir::Type, 1> types(tupleType.begin(), tupleType.end());
    assert(types.size() == values.size());

    if (parser.resolveOperands(values, types, loc, result.operands)) {
      return mlir::failure();
    }

    result.addTypes(resultType);
    return mlir::success();
  }

  void EquationSideOp::print(mlir::OpAsmPrinter& printer)
  {
    printer.printOptionalAttrDict(getOperation()->getAttrs());
    printer << " " << getValues() << " : " << getResult().getType();
  }
}

//===---------------------------------------------------------------------===//
// AssignmentOp

namespace mlir::modelica
{
  void AssignmentOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    if (getValue().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getValue(),
          mlir::SideEffects::DefaultResource::get());
    }

    if (getDestination().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Write::get(),
          getDestination(),
          mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange AssignmentOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    mlir::Location loc = getLoc();

    mlir::Value derivedSource = derivatives.lookup(getValue());
    mlir::Value derivedDestination = derivatives.lookup(getDestination());

    builder.create<AssignmentOp>(loc, derivedDestination, derivedSource);
    return std::nullopt;
  }

  void AssignmentOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getDestination());
    toBeDerived.push_back(getValue());
  }

  void AssignmentOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// FunctionOp

namespace mlir::modelica
{
  void FunctionOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      llvm::StringRef name)
  {
    state.addRegion()->emplaceBlock();

    state.attributes.push_back(builder.getNamedAttr(
        mlir::SymbolTable::getSymbolAttrName(),
        builder.getStringAttr(name)));
  }

  mlir::ParseResult FunctionOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::StringAttr nameAttr;

    if (parser.parseSymbolName(
            nameAttr,
            mlir::SymbolTable::getSymbolAttrName(),
            result.attributes)) {
      return mlir::failure();
    }

    if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
      return mlir::failure();
    }

    mlir::Region* bodyRegion = result.addRegion();
    
    if (parser.parseRegion(*bodyRegion)) {
      return mlir::failure();
    }

    if (bodyRegion->empty()) {
      bodyRegion->emplaceBlock();
    }

    return mlir::success();
  }

  void FunctionOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " ";
    printer.printSymbolName(getSymName());

    llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
    elidedAttrs.push_back(mlir::SymbolTable::getSymbolAttrName());

    printer.printOptionalAttrDictWithKeyword(
        getOperation()->getAttrs(), elidedAttrs);

    printer << " ";
    printer.printRegion(getBody());
  }

  mlir::Block* FunctionOp::bodyBlock()
  {
    assert(getBody().hasOneBlock());
    return &getBody().front();
  }

  llvm::SmallVector<mlir::Type> FunctionOp::getArgumentTypes()
  {
    llvm::SmallVector<mlir::Type> types;

    for (VariableOp variableOp : getVariables()) {
      VariableType variableType = variableOp.getVariableType();

      if (variableType.isInput()) {
        types.push_back(variableType.unwrap());
      }
    }

    return types;
  }

  llvm::SmallVector<mlir::Type> FunctionOp::getResultTypes()
  {
    llvm::SmallVector<mlir::Type> types;

    for (VariableOp variableOp : getVariables()) {
      VariableType variableType = variableOp.getVariableType();

      if (variableType.isOutput()) {
        types.push_back(variableType.unwrap());
      }
    }

    return types;
  }

  mlir::FunctionType FunctionOp::getFunctionType()
  {
    llvm::SmallVector<mlir::Type> argTypes;
    llvm::SmallVector<mlir::Type> resultTypes;

    for (VariableOp variableOp : getVariables()) {
      VariableType variableType = variableOp.getVariableType();

      if (variableType.isInput()) {
        argTypes.push_back(variableType.unwrap());
      } else if (variableType.isOutput()) {
        resultTypes.push_back(variableType.unwrap());
      }
    }

    return mlir::FunctionType::get(getContext(), argTypes, resultTypes);
  }

  bool FunctionOp::shouldBeInlined()
  {
    if (!getOperation()->hasAttrOfType<mlir::BoolAttr>("inline")) {
      return false;
    }

    auto inlineAttribute =
        getOperation()->getAttrOfType<mlir::BoolAttr>("inline");

    return inlineAttribute.getValue();
  }
}

//===---------------------------------------------------------------------===//
// DerFunctionOp

namespace mlir::modelica
{
  mlir::ParseResult DerFunctionOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::StringAttr nameAttr;

    if (parser.parseSymbolName(
            nameAttr,
            mlir::SymbolTable::getSymbolAttrName(),
            result.attributes) ||
        parser.parseOptionalAttrDict(result.attributes)) {
      return mlir::failure();
    }

    return mlir::success();
  }

  void DerFunctionOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " ";
    printer.printSymbolName(getSymName());

    llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
    elidedAttrs.push_back(mlir::SymbolTable::getSymbolAttrName());

    printer.printOptionalAttrDict(getOperation()->getAttrs(), elidedAttrs);
  }
}

//===---------------------------------------------------------------------===//
// RawFunctionOp

namespace mlir::modelica
{
  RawFunctionOp RawFunctionOp::create(
      mlir::Location location,
      llvm::StringRef name,
      mlir::FunctionType type,
      llvm::ArrayRef<mlir::NamedAttribute> attrs)
  {
    mlir::OpBuilder builder(location->getContext());
    mlir::OperationState state(location, getOperationName());
    RawFunctionOp::build(builder, state, name, type, attrs);
    return mlir::cast<RawFunctionOp>(mlir::Operation::create(state));
  }

  RawFunctionOp RawFunctionOp::create(
      mlir::Location location,
      llvm::StringRef name,
      mlir::FunctionType type,
      mlir::Operation::dialect_attr_range attrs)
  {
    llvm::SmallVector<mlir::NamedAttribute, 8> attrRef(attrs);
    return create(location, name, type, llvm::ArrayRef(attrRef));
  }

  RawFunctionOp RawFunctionOp::create(
      mlir::Location location,
      llvm::StringRef name,
      mlir::FunctionType type,
      llvm::ArrayRef<mlir::NamedAttribute> attrs,
      llvm::ArrayRef<mlir::DictionaryAttr> argAttrs)
  {
    RawFunctionOp func = create(location, name, type, attrs);
    func.setAllArgAttrs(argAttrs);
    return func;
  }

  void RawFunctionOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      llvm::StringRef name,
      mlir::FunctionType type,
      llvm::ArrayRef<mlir::NamedAttribute> attrs,
      llvm::ArrayRef<mlir::DictionaryAttr> argAttrs)
  {
    state.addAttribute(
        mlir::SymbolTable::getSymbolAttrName(),
        builder.getStringAttr(name));

    state.addAttribute(
        getFunctionTypeAttrName(state.name),
        mlir::TypeAttr::get(type));

    state.attributes.append(attrs.begin(), attrs.end());
    state.addRegion();

    if (argAttrs.empty()) {
      return;
    }

    assert(type.getNumInputs() == argAttrs.size());

    function_interface_impl::addArgAndResultAttrs(
        builder, state, argAttrs, std::nullopt,
        getArgAttrsAttrName(state.name), getResAttrsAttrName(state.name));
  }

  mlir::ParseResult RawFunctionOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    auto buildFuncType =
        [](mlir::Builder& builder,
           llvm::ArrayRef<mlir::Type> argTypes,
           llvm::ArrayRef<mlir::Type> results,
           mlir::function_interface_impl::VariadicFlag,
           std::string&) {
          return builder.getFunctionType(argTypes, results);
        };

    return mlir::function_interface_impl::parseFunctionOp(
        parser, result, false,
        getFunctionTypeAttrName(result.name), buildFuncType,
        getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
  }

  void RawFunctionOp::print(OpAsmPrinter& printer)
  {
    mlir::function_interface_impl::printFunctionOp(
        printer, *this, false, getFunctionTypeAttrName(),
        getArgAttrsAttrName(), getResAttrsAttrName());
  }

  bool RawFunctionOp::shouldBeInlined()
  {
    if (!getOperation()->hasAttrOfType<mlir::BoolAttr>("inline")) {
      return false;
    }

    auto inlineAttribute =
        getOperation()->getAttrOfType<mlir::BoolAttr>("inline");

    return inlineAttribute.getValue();
  }

  /// Clone the internal blocks from this function into dest and all attributes
  /// from this function to dest.
  void RawFunctionOp::cloneInto(
      RawFunctionOp dest, mlir::IRMapping& mapper)
  {
    // Add the attributes of this function to dest.
    llvm::MapVector<mlir::StringAttr, mlir::Attribute> newAttrMap;

    for (const auto &attr : dest->getAttrs()) {
      newAttrMap.insert({attr.getName(), attr.getValue()});
    }

    for (const auto &attr : (*this)->getAttrs()) {
      newAttrMap.insert({attr.getName(), attr.getValue()});
    }

    auto newAttrs = llvm::to_vector(llvm::map_range(
        newAttrMap, [](std::pair<StringAttr, Attribute> attrPair) {
          return NamedAttribute(attrPair.first, attrPair.second);
        }));

    dest->setAttrs(mlir::DictionaryAttr::get(getContext(), newAttrs));

    // Clone the body.
    getBody().cloneInto(&dest.getBody(), mapper);
  }

  /// Create a deep copy of this function and all of its blocks, remapping
  /// any operands that use values outside of the function using the map that is
  /// provided (leaving them alone if no entry is present). Replaces references
  /// to cloned sub-values with the corresponding value that is copied, and adds
  /// those mappings to the mapper.
  RawFunctionOp RawFunctionOp::clone(mlir::IRMapping& mapper)
  {
    // Create the new function.
    RawFunctionOp newFunc = cast<RawFunctionOp>(
        getOperation()->cloneWithoutRegions());

    // If the function has a body, then the user might be deleting arguments to
    // the function by specifying them in the mapper. If so, we don't add the
    // argument to the input type vector.
    if (!isExternal()) {
      mlir::FunctionType oldType = getFunctionType();

      unsigned oldNumArgs = oldType.getNumInputs();
      llvm::SmallVector<Type, 4> newInputs;
      newInputs.reserve(oldNumArgs);

      for (unsigned i = 0; i != oldNumArgs; ++i) {
        if (!mapper.contains(getArgument(i))) {
          newInputs.push_back(oldType.getInput(i));
        }
      }

      // If any of the arguments were dropped, update the type and drop any
      // necessary argument attributes.
      if (newInputs.size() != oldNumArgs) {
        newFunc.setType(mlir::FunctionType::get(
            oldType.getContext(), newInputs, oldType.getResults()));

        if (mlir::ArrayAttr argAttrs = getAllArgAttrs()) {
          SmallVector<Attribute> newArgAttrs;
          newArgAttrs.reserve(newInputs.size());

          for (unsigned i = 0; i != oldNumArgs; ++i) {
            if (!mapper.contains(getArgument(i))) {
              newArgAttrs.push_back(argAttrs[i]);
            }
          }

          newFunc.setAllArgAttrs(newArgAttrs);
        }
      }
    }

    // Clone the current function into the new one and return it.
    cloneInto(newFunc, mapper);
    return newFunc;
  }

  RawFunctionOp RawFunctionOp::clone()
  {
    mlir::IRMapping mapper;
    return clone(mapper);
  }
}

//===---------------------------------------------------------------------===//
// RawReturnOp

namespace mlir::modelica
{
  mlir::LogicalResult RawReturnOp::verify() {
    auto function = cast<RawFunctionOp>((*this)->getParentOp());

    // The operand number and types must match the function signature
    const auto& results = function.getFunctionType().getResults();

    if (getNumOperands() != results.size()) {
      return emitOpError("has ")
          << getNumOperands() << " operands, but enclosing function (@"
          << function.getName() << ") returns " << results.size();
    }

    for (unsigned i = 0, e = results.size(); i != e; ++i) {
      if (getOperand(i).getType() != results[i]) {
        return emitOpError()
            << "type of return operand " << i << " ("
            << getOperand(i).getType()
            << ") doesn't match function result type (" << results[i] << ")"
            << " in function @" << function.getName();
      }
    }

    return mlir::success();
  }
}

//===---------------------------------------------------------------------===//
// RawVariableOp

namespace mlir::modelica
{
  mlir::ParseResult RawVariableOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    auto& builder = parser.getBuilder();

    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> dynamicSizes;
    mlir::Type variableType;

    if (parser.parseOperandList(dynamicSizes) ||
        parser.resolveOperands(
            dynamicSizes, builder.getIndexType(), result.operands) ||
        parser.parseColon() ||
        parser.parseType(variableType)) {
      return mlir::failure();
    }

    result.addTypes(variableType);

    // Dimensions constraints.
    llvm::SmallVector<llvm::StringRef> dimensionsConstraints;

    if (mlir::succeeded(parser.parseOptionalLSquare())) {
      do {
        if (mlir::succeeded(
                parser.parseOptionalKeyword(kDimensionConstraintUnbounded))) {
          dimensionsConstraints.push_back(kDimensionConstraintUnbounded);
        } else {
          if (parser.parseKeyword(kDimensionConstraintFixed)) {
            return mlir::failure();
          }

          dimensionsConstraints.push_back(kDimensionConstraintFixed);
        }
      } while (mlir::succeeded(parser.parseOptionalComma()));

      if (parser.parseRSquare()) {
        return mlir::failure();
      }
    }

    result.attributes.append(
        getDimensionsConstraintsAttrName(result.name),
        builder.getStrArrayAttr(dimensionsConstraints));

    // Attributes.
    if (parser.parseOptionalAttrDict(result.attributes)) {
      return mlir::failure();
    }

    return mlir::success();
  }

  void RawVariableOp::print(mlir::OpAsmPrinter& printer)
  {
    if (auto dynamicSizes = getDynamicSizes(); !dynamicSizes.empty()) {
      printer << " " << dynamicSizes;
    }

    printer << " : " << getVariableType();

    auto dimConstraints =
        getDimensionsConstraints().getAsRange<mlir::StringAttr>();

    if (llvm::any_of(dimConstraints, [](mlir::StringAttr constraint) {
          return constraint == kDimensionConstraintFixed;
        })) {
      printer << " [";

      for (const auto& constraint : llvm::enumerate(dimConstraints)) {
        if (constraint.index() != 0) {
          printer << ", ";
        }

        printer << constraint.value().getValue();
      }

      printer << "] ";
    }

    llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
    elidedAttrs.push_back(getDimensionsConstraintsAttrName());

    printer.printOptionalAttrDict(getOperation()->getAttrs(), elidedAttrs);
  }
}

//===---------------------------------------------------------------------===//
// RawVariableGetOp

namespace mlir::modelica
{
  mlir::ParseResult RawVariableGetOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::OpAsmParser::UnresolvedOperand variable;
    mlir::Type variableType;

    if (parser.parseOperand(variable) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColonType(variableType) ||
        parser.resolveOperand(variable, variableType, result.operands)) {
      return mlir::failure();
    }

    result.addTypes(variableType.cast<VariableType>().unwrap());
    return mlir::success();
  }

  void RawVariableGetOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " ";
    printer << getVariable();
    printer.printOptionalAttrDict(getOperation()->getAttrs());
    printer << " : " << getVariable().getType();
  }
}

//===---------------------------------------------------------------------===//
// RawVariableSetOp

namespace mlir::modelica
{
  mlir::ParseResult RawVariableSetOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::OpAsmParser::UnresolvedOperand variable;
    mlir::OpAsmParser::UnresolvedOperand value;
    mlir::Type variableType;
    mlir::Type valueType;

    if (parser.parseOperand(variable) ||
        parser.parseComma() ||
        parser.parseOperand(value) ||
        parser.parseOptionalAttrDict(result.attributes) ||
        parser.parseColon() ||
        parser.parseType(variableType) ||
        parser.parseComma() ||
        parser.parseType(valueType) ||
        parser.resolveOperand(variable, variableType, result.operands) ||
        parser.resolveOperand(value, valueType, result.operands)) {
      return mlir::failure();
    }

    return mlir::success();
  }

  void RawVariableSetOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " ";
    printer << getVariable() << ", " << getValue();
    printer.printOptionalAttrDict(getOperation()->getAttrs());
    printer << " : " << getVariable().getType() << ", " << getValue().getType();
  }
}

//===---------------------------------------------------------------------===//
// CallOp

namespace mlir::modelica
{
  void CallOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      FunctionOp callee,
      mlir::ValueRange args,
      std::optional<mlir::ArrayAttr> argNames)
  {
    mlir::SymbolRefAttr symbol = getSymbolRefFromRoot(callee);
    build(builder, state, symbol, callee.getResultTypes(), args, argNames);
  }

  void CallOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      RawFunctionOp callee,
      mlir::ValueRange args)
  {
    mlir::SymbolRefAttr symbol = getSymbolRefFromRoot(callee);
    build(builder, state, symbol, callee.getResultTypes(), args);
  }

  void CallOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      mlir::SymbolRefAttr callee,
      mlir::TypeRange resultTypes,
      mlir::ValueRange args,
      std::optional<mlir::ArrayAttr> argNames)
  {
    state.addOperands(args);
    state.addAttribute(getCalleeAttrName(state.name), callee);

    if (argNames) {
      state.addAttribute(getArgNamesAttrName(state.name), *argNames);
    }

    state.addTypes(resultTypes);
  }

  mlir::LogicalResult CallOp::verifySymbolUses(
      mlir::SymbolTableCollection& symbolTable)
  {
    auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
    mlir::Operation* callee = getFunction(moduleOp, symbolTable);

    if (!callee) {
      // TODO
      // At the moment verification would fail for derivatives of functions,
      // because they are declared through an attribute. We should look into
      // turning that attribute into an operation, so that the symbol becomes
      // declared within the module, and thus obtainable.
      return mlir::success();
    }

    if (mlir::isa<DerFunctionOp>(callee)) {
      // TODO implement proper verification of DerFunctionOp function type.
      return mlir::success();
    }

    // Verify that the operand and result types match the callee.
    if (auto functionOp = mlir::dyn_cast<FunctionOp>(callee)) {
      mlir::FunctionType functionType = functionOp.getFunctionType();

      llvm::SmallVector<mlir::StringAttr> inputVariables;
      llvm::DenseSet<mlir::StringAttr> inputVariablesSet;

      for (VariableOp variableOp : functionOp.getVariables()) {
        mlir::StringAttr variableName = variableOp.getSymNameAttr();

        if (variableOp.isInput()) {
          inputVariables.push_back(variableName);
          inputVariablesSet.insert(variableName);
        }
      }

      llvm::DenseSet<mlir::StringAttr> variablesWithDefaultValue;

      for (DefaultOp defaultOp : functionOp.getDefaultValues()) {
        variablesWithDefaultValue.insert(defaultOp.getVariableAttr());
      }

      auto args = getArgs();

      if (auto argNames = getArgNames()) {
        if (argNames->size() != args.size()) {
          return emitOpError()
              << "the number of arguments (" << args.size()
              << ") does not match the number of argument names ("
              << argNames->size() << ")";
        }

        llvm::DenseSet<mlir::StringAttr> specifiedInputs;

        for (mlir::FlatSymbolRefAttr argName :
             argNames->getAsRange<mlir::FlatSymbolRefAttr>()) {
          if (!inputVariablesSet.contains(argName.getAttr())) {
            return emitOpError() << "unknown argument '"
                                 << argName.getValue() << "'";
          }

          if (specifiedInputs.contains(argName.getAttr())) {
            return emitOpError() << "multiple values for argument '"
                                 << argName.getValue() << "'";
          }

          specifiedInputs.insert(argName.getAttr());
        }

        for (mlir::StringAttr variableName : inputVariables) {
          if (!variablesWithDefaultValue.contains(variableName) &&
              !specifiedInputs.contains(variableName)) {
            return emitOpError() << "missing value for argument '"
                                 << variableName.getValue() << "'";
          }
        }
      } else {
        if (args.size() > inputVariables.size()) {
          return emitOpError() << "too many arguments specified (expected "
              << inputVariables.size() << ", got " << args.size() << ")";
        }

        for (mlir::StringAttr variableName :
             llvm::ArrayRef(inputVariables).drop_front(args.size())) {
          if (!variablesWithDefaultValue.contains(variableName)) {
            return emitOpError() << "missing value for argument '"
                                 << variableName.getValue() << "'";
          }
        }
      }

      unsigned int expectedResults = functionType.getNumResults();
      unsigned int actualResults = getNumResults();

      if (expectedResults != actualResults) {
        return emitOpError()
            << "incorrect number of results for callee (expected "
            << expectedResults << ", got " << actualResults << ")";

        return mlir::failure();
      }

      return mlir::success();
    }

    if (auto rawFunctionOp = mlir::dyn_cast<RawFunctionOp>(callee)) {
      mlir::FunctionType functionType = rawFunctionOp.getFunctionType();

      unsigned int expectedInputs = functionType.getNumInputs();
      unsigned int actualInputs = getNumOperands();

      if (expectedInputs != actualInputs) {
        return emitOpError()
            << "incorrect number of operands for callee (expected "
            << expectedInputs << ", got "
            << actualInputs << ")";
      }

      unsigned int expectedResults = functionType.getNumResults();
      unsigned int actualResults = getNumResults();

      if (expectedResults != actualResults) {
        return emitOpError()
            << "incorrect number of results for callee (expected "
            << expectedResults << ", got "
            << actualResults << ")";
      }

      return mlir::success();
    }

    return emitOpError() << "'" << getCallee()
                         << "' does not reference a valid function";

    return mlir::failure();
  }

  void CallOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    // The callee may have no arguments and no results, but still have side
    // effects (i.e. an external function writing elsewhere). Thus we need to
    // consider the call itself as if it is has side effects and prevent the
    // CSE pass to erase it.
    effects.emplace_back(
        mlir::MemoryEffects::Write::get(),
        mlir::SideEffects::DefaultResource::get());

    for (mlir::Value result : getResults()) {
      if (auto arrayType = result.getType().dyn_cast<ArrayType>()) {
        effects.emplace_back(
            mlir::MemoryEffects::Allocate::get(),
            result,
            mlir::SideEffects::DefaultResource::get());

        effects.emplace_back(
            mlir::MemoryEffects::Write::get(),
            result,
            mlir::SideEffects::DefaultResource::get());
      }
    }
  }

  unsigned int CallOp::getArgExpectedRank(
      unsigned int argIndex, mlir::SymbolTableCollection& symbolTable)
  {
    auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto calleeOp = resolveSymbol(moduleOp, symbolTable, getCallee());

    if (calleeOp == nullptr) {
      // If the function is not declared, then assume that the arguments types
      // already match its hypothetical signature.

      mlir::Type argType = getArgs()[argIndex].getType();

      if (auto arrayType = argType.dyn_cast<ArrayType>())
        return arrayType.getRank();

      return 0;
    }

    mlir::Type argType = mlir::cast<FunctionOp>(calleeOp).getArgumentTypes()[argIndex];

    if (auto arrayType = argType.dyn_cast<ArrayType>()) {
      return arrayType.getRank();
    }

    return 0;
  }

  mlir::ValueRange CallOp::scalarize(
      mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    llvm::SmallVector<mlir::Type, 3> newResultsTypes;

    for (mlir::Type type : getResultTypes()) {
      mlir::Type newResultType =
          type.cast<ArrayType>().slice(indexes.size());

      if (auto arrayType = newResultType.dyn_cast<ArrayType>();
          arrayType.getRank() == 0) {
        newResultType = arrayType.getElementType();
      }

      newResultsTypes.push_back(newResultType);
    }

    llvm::SmallVector<mlir::Value, 3> newArgs;

    for (mlir::Value arg : getArgs()) {
      assert(arg.getType().isa<ArrayType>());

      mlir::Value newArg = builder.create<SubscriptionOp>(
          getLoc(), arg, indexes);

      if (auto arrayType = newArg.getType().dyn_cast<ArrayType>();
          arrayType.getRank() == 0) {
        newArg = builder.create<LoadOp>(getLoc(), newArg);
      }

      newArgs.push_back(newArg);
    }

    auto op = builder.create<CallOp>(
        getLoc(), getCallee(), newResultsTypes, newArgs);

    return op->getResults();
  }

  mlir::Value CallOp::inverse(
      mlir::OpBuilder& builder,
      unsigned int argumentIndex,
      mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (getNumResults() != 1) {
      emitOpError() << "The callee must have one and only one result.";
      return nullptr;
    }

    if (argumentIndex >= getArgs().size()) {
      emitOpError() << "Index out of bounds: " << argumentIndex << ".";
      return nullptr;
    }

    if (size_t size = currentResult.size(); size != 1) {
      emitOpError() << "Invalid amount of values to be nested: " << size
                    << " (expected 1).";

      return nullptr;
    }

    mlir::Value toNest = currentResult[0];

    auto module = getOperation()->getParentOfType<mlir::ModuleOp>();
    auto callee = module.lookupSymbol<FunctionOp>(this->getCallee());

    if (!callee->hasAttr("inverse")) {
      emitOpError() << "Function " << callee->getName().getStringRef()
                    << " is not invertible.";

      return nullptr;
    }

    auto inverseAnnotation =
        callee->getAttrOfType<InverseFunctionsAttr>("inverse");

    if (!inverseAnnotation.isInvertible(argumentIndex)) {
      emitOpError() << "Function " << callee->getName().getStringRef()
                    << " is not invertible for argument " << argumentIndex
                    << ".";

      return nullptr;
    }

    size_t argsSize = getArgs().size();
    llvm::SmallVector<mlir::Value, 3> args;

    for (auto arg : inverseAnnotation.getArgumentsIndexes(argumentIndex)) {
      if (arg < argsSize) {
        args.push_back(this->getArgs()[arg]);
      } else {
        assert(arg == argsSize);
        args.push_back(toNest);
      }
    }

    auto invertedCall = builder.create<CallOp>(
        getLoc(),
        mlir::SymbolRefAttr::get(builder.getStringAttr(
            inverseAnnotation.getFunction(argumentIndex))),
        this->getArgs()[argumentIndex].getType(),
        args);

    return invertedCall.getResult(0);
  }

  mlir::Operation* CallOp::getFunction(
      mlir::ModuleOp moduleOp, mlir::SymbolTableCollection& symbolTable)
  {
    mlir::SymbolRefAttr callee = getCallee();

    mlir::Operation* result = symbolTable.lookupSymbolIn(
        moduleOp, callee.getRootReference());

    for (mlir::FlatSymbolRefAttr flatSymbolRef : callee.getNestedReferences()) {
      if (result == nullptr) {
        return nullptr;
      }

      result = symbolTable.lookupSymbolIn(result, flatSymbolRef.getAttr());
    }

    return result;
  }
}

//===---------------------------------------------------------------------===//
// Control flow operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// ForOp

namespace mlir::modelica
{
  mlir::Block* ForOp::conditionBlock()
  {
    assert(!getConditionRegion().empty());
    return &getConditionRegion().front();
  }

  mlir::Block* ForOp::bodyBlock()
  {
    assert(!getBodyRegion().empty());
    return &getBodyRegion().front();
  }

  mlir::Block* ForOp::stepBlock()
  {
    assert(!getStepRegion().empty());
    return &getStepRegion().front();
  }

  mlir::ParseResult ForOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::Region* conditionRegion = result.addRegion();

    if (mlir::succeeded(parser.parseOptionalLParen())) {
      if (mlir::failed(parser.parseOptionalRParen())) {
        do {
          mlir::OpAsmParser::UnresolvedOperand arg;
          mlir::Type argType;

          if (parser.parseOperand(arg) ||
              parser.parseColonType(argType) ||
              parser.resolveOperand(arg, argType, result.operands))
            return mlir::failure();
        } while (mlir::succeeded(parser.parseOptionalComma()));
      }

      if (parser.parseRParen()) {
        return mlir::failure();
      }
    }

    if (parser.parseKeyword("condition")) {
      return mlir::failure();
    }

    if (parser.parseRegion(*conditionRegion)) {
      return mlir::failure();
    }

    if (parser.parseKeyword("body")) {
      return mlir::failure();
    }

    mlir::Region* bodyRegion = result.addRegion();

    if (parser.parseRegion(*bodyRegion)) {
      return mlir::failure();
    }

    if (parser.parseKeyword("step")) {
      return mlir::failure();
    }

    mlir::Region* stepRegion = result.addRegion();

    if (parser.parseRegion(*stepRegion)) {
      return mlir::failure();
    }

    if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
      return mlir::failure();
    }

    return mlir::success();
  }

  void ForOp::print(mlir::OpAsmPrinter& printer)
  {
    if (auto values = getArgs(); !values.empty()) {
      printer << "(";

      for (auto arg : llvm::enumerate(values)) {
        if (arg.index() != 0) {
          printer << ", ";
        }

        printer << arg.value() << " : " << arg.value().getType();
      }

      printer << ")";
    }

    printer << " condition ";
    printer.printRegion(getConditionRegion(), true);
    printer << " body ";
    printer.printRegion(getBodyRegion(), true);
    printer << " step ";
    printer.printRegion(getStepRegion(), true);
    printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
  }

  mlir::ValueRange ForOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    return std::nullopt;
  }

  void ForOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void ForOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    regions.push_back(&getBodyRegion());
  }
}

//===---------------------------------------------------------------------===//
// IfOp

namespace mlir::modelica
{
  void IfOp::build(
      mlir::OpBuilder& builder,
      mlir::OperationState& state,
      mlir::Value condition,
      bool withElseRegion)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    state.addOperands(condition);

    // Create the "then" region.
    mlir::Region* thenRegion = state.addRegion();
    builder.createBlock(thenRegion);

    // Create the "else" region.
    mlir::Region* elseRegion = state.addRegion();

    if (withElseRegion) {
      builder.createBlock(elseRegion);
    }
  }

  mlir::Block* IfOp::thenBlock()
  {
    return &getThenRegion().front();
  }

  mlir::Block* IfOp::elseBlock()
  {
    return &getElseRegion().front();
  }

  mlir::ParseResult IfOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::OpAsmParser::UnresolvedOperand condition;
    mlir::Type conditionType;

    if (parser.parseLParen() ||
        parser.parseOperand(condition) ||
        parser.parseColonType(conditionType) ||
        parser.parseRParen() ||
        parser.resolveOperand(condition, conditionType, result.operands)) {
      return mlir::failure();
    }

    if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
      return mlir::failure();
    }

    mlir::Region* thenRegion = result.addRegion();

    if (parser.parseRegion(*thenRegion)) {
      return mlir::failure();
    }

    mlir::Region* elseRegion = result.addRegion();

    if (mlir::succeeded(parser.parseOptionalKeyword("else"))) {
      if (parser.parseRegion(*elseRegion)) {
        return mlir::failure();
      }
    }

    return mlir::success();
  }

  void IfOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " (" << getCondition() << " : "
            << getCondition().getType() << ") ";

    printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
    printer.printRegion(getThenRegion());

    if (!getElseRegion().empty()) {
      printer << " else ";
      printer.printRegion(getElseRegion());
    }
  }

  mlir::ValueRange IfOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    return std::nullopt;
  }

  void IfOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void IfOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    regions.push_back(&getThenRegion());
    regions.push_back(&getElseRegion());
  }
}

//===---------------------------------------------------------------------===//
// WhileOp

namespace mlir::modelica
{
  mlir::ParseResult WhileOp::parse(
      mlir::OpAsmParser& parser, mlir::OperationState& result)
  {
    mlir::Region* conditionRegion = result.addRegion();
    mlir::Region* bodyRegion = result.addRegion();

    if (parser.parseRegion(*conditionRegion) ||
        parser.parseKeyword("do") ||
        parser.parseRegion(*bodyRegion)) {
      return mlir::failure();
    }

    if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
      return mlir::failure();
    }

    if (conditionRegion->empty()) {
      conditionRegion->emplaceBlock();
    }

    if (bodyRegion->empty()) {
      bodyRegion->emplaceBlock();
    }

    return mlir::success();
  }

  void WhileOp::print(mlir::OpAsmPrinter& printer)
  {
    printer << " ";
    printer.printRegion(getConditionRegion(), false);
    printer << " do ";
    printer.printRegion(getBodyRegion(), false);
    printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
  }

  mlir::ValueRange WhileOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    return std::nullopt;
  }

  void WhileOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void WhileOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    regions.push_back(&getBodyRegion());
  }
}

//===---------------------------------------------------------------------===//
// Utility operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// CastOp

namespace mlir::modelica
{
  mlir::ValueRange CastOp::derive(
      mlir::OpBuilder& builder,
      const llvm::DenseMap<
          mlir::StringAttr, mlir::StringAttr>& symbolDerivatives,
      mlir::IRMapping& derivatives)
  {
    auto derivedOp = builder.create<CastOp>(
        getLoc(), getResult().getType(), derivatives.lookup(getValue()));

    return derivedOp->getResults();
  }

  void CastOp::getOperandsToBeDerived(
      llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(getValue());
  }

  void CastOp::getDerivableRegions(
      llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    // No regions to be derived.
  }
}

//===---------------------------------------------------------------------===//
// PrintOp

namespace mlir::modelica
{
  void PrintOp::getEffects(
      mlir::SmallVectorImpl<
          mlir::SideEffects::EffectInstance<
              mlir::MemoryEffects::Effect>>& effects)
  {
    // Ensure the operation doesn't get erased, no matter what.
    effects.emplace_back(
        mlir::MemoryEffects::Write::get(),
        mlir::SideEffects::DefaultResource::get());

    if (getValue().getType().isa<ArrayType>()) {
      effects.emplace_back(
          mlir::MemoryEffects::Read::get(),
          getValue(),
          mlir::SideEffects::DefaultResource::get());
    }
  }
}
