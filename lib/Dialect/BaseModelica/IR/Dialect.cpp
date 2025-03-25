#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Interfaces/FoldInterfaces.h"
#include "mlir/Transforms/InliningUtils.h"

using namespace ::mlir::bmodelica;

#include "marco/Dialect/BaseModelica/IR/BaseModelica.cpp.inc"

namespace {
struct BaseModelicaOpAsmDialectInterface : public mlir::OpAsmDialectInterface {
  explicit BaseModelicaOpAsmDialectInterface(mlir::Dialect *dialect)
      : OpAsmDialectInterface(dialect) {}

  AliasResult getAlias(mlir::Attribute attr,
                       llvm::raw_ostream &os) const override {
    if (mlir::isa<EquationPathAttr>(attr)) {
      os << "equation_path";
      return AliasResult::OverridableAlias;
    }

    return AliasResult::NoAlias;
  }

  AliasResult getAlias(mlir::Type type, llvm::raw_ostream &os) const final {
    return AliasResult::NoAlias;
  }
};

struct ModelicaFoldInterface : public mlir::DialectFoldInterface {
  using DialectFoldInterface::DialectFoldInterface;

  bool shouldMaterializeInto(mlir::Region *region) const final {
    return mlir::isa<AlgorithmOp, BindingEquationOp, DefaultOp, EquationOp,
                     EquationTemplateOp, ReductionOp, StartOp, VariableOp>(
        region->getParentOp());
  }
};

/// This class defines the interface for handling inlining with Modelica
/// operations.
struct ModelicaInlinerInterface : public mlir::DialectInlinerInterface {
  using mlir::DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(mlir::Operation *call, mlir::Operation *callable,
                       bool wouldBeCloned) const final {
    if (auto rawFunctionOp = mlir::dyn_cast<RawFunctionOp>(callable)) {
      return rawFunctionOp.shouldBeInlined();
    }

    return true;
  }

  bool isLegalToInline(mlir::Operation *op, mlir::Region *dest,
                       bool wouldBeCloned,
                       mlir::IRMapping &valueMapping) const final {
    return true;
  }

  bool isLegalToInline(mlir::Region *dest, mlir::Region *src,
                       bool wouldBeCloned,
                       mlir::IRMapping &valueMapping) const final {
    return true;
  }

  void handleTerminator(mlir::Operation *op,
                        mlir::ValueRange valuesToReplace) const final {
    // Only "modelica.raw_return" needs to be handled here.
    auto returnOp = mlir::cast<RawReturnOp>(op);

    // Replace the values directly with the return operands.
    assert(returnOp.getNumOperands() == valuesToReplace.size());

    for (const auto &operand : llvm::enumerate(returnOp.getOperands())) {
      valuesToReplace[operand.index()].replaceAllUsesWith(operand.value());
    }
  }
};
} // namespace

namespace mlir::bmodelica {
//===-------------------------------------------------------------------===//
// Base Modelica dialect
//===-------------------------------------------------------------------===//

void BaseModelicaDialect::initialize() {
  registerTypes();
  registerAttributes();
  registerOperations();

  addInterfaces<
      // ModelicaOpAsmDialectInterface,
      ModelicaFoldInterface, ModelicaInlinerInterface>();

  // Mark the built-in types as elementary.
  mlir::IndexType::attachInterface<ElementaryType>(*getContext());
  mlir::IntegerType::attachInterface<ElementaryType>(*getContext());
  mlir::Float16Type::attachInterface<ElementaryType>(*getContext());
  mlir::Float32Type::attachInterface<ElementaryType>(*getContext());
  mlir::Float64Type::attachInterface<ElementaryType>(*getContext());
  mlir::Float80Type::attachInterface<ElementaryType>(*getContext());
  mlir::Float128Type::attachInterface<ElementaryType>(*getContext());
}

Operation *BaseModelicaDialect::materializeConstant(mlir::OpBuilder &builder,
                                                    mlir::Attribute value,
                                                    mlir::Type type,
                                                    mlir::Location loc) {
  return builder.create<ConstantOp>(loc, type,
                                    mlir::cast<mlir::TypedAttr>(value));
}
} // namespace mlir::bmodelica

namespace mlir::bmodelica {
mlir::SymbolRefAttr getSymbolRefFromRoot(mlir::Operation *symbol) {
  llvm::SmallVector<mlir::FlatSymbolRefAttr> flatSymbolAttrs;

  flatSymbolAttrs.push_back(mlir::FlatSymbolRefAttr::get(
      symbol->getContext(),
      mlir::cast<mlir::SymbolOpInterface>(symbol).getName()));

  mlir::Operation *parent = symbol->getParentOp();

  while (parent != nullptr) {
    if (auto classInterface = mlir::dyn_cast<ClassInterface>(parent)) {
      flatSymbolAttrs.push_back(mlir::FlatSymbolRefAttr::get(
          symbol->getContext(),
          mlir::cast<mlir::SymbolOpInterface>(classInterface.getOperation())
              .getName()));
    }

    parent = parent->getParentOp();
  }

  std::reverse(flatSymbolAttrs.begin(), flatSymbolAttrs.end());

  return mlir::SymbolRefAttr::get(symbol->getContext(),
                                  flatSymbolAttrs[0].getValue(),
                                  llvm::ArrayRef(flatSymbolAttrs).drop_front());
}

mlir::Operation *
resolveSymbol(mlir::ModuleOp moduleOp,
              mlir::SymbolTableCollection &symbolTableCollection,
              mlir::SymbolRefAttr symbol) {
  mlir::Operation *result =
      symbolTableCollection.lookupSymbolIn(moduleOp, symbol.getRootReference());

  for (mlir::FlatSymbolRefAttr nestedRef : symbol.getNestedReferences()) {
    if (result == nullptr) {
      return nullptr;
    }

    result = symbolTableCollection.lookupSymbolIn(result, nestedRef.getAttr());
  }

  return result;
}

void walkClasses(mlir::Operation *root,
                 llvm::function_ref<void(mlir::Operation *)> callback) {
  llvm::SmallVector<mlir::Operation *, 10> stack;
  stack.push_back(root);

  while (!stack.empty()) {
    mlir::Operation *op = stack.pop_back_val();

    if (op->hasTrait<ClassInterface::Trait>()) {
      callback(op);
    }

    for (auto &region : op->getRegions()) {
      for (auto &nestedOp : region.getOps()) {
        if (nestedOp.hasTrait<ClassInterface::Trait>()) {
          stack.push_back(&nestedOp);
        }
      }
    }
  }
}

mlir::Type getMostGenericScalarType(mlir::Value x, mlir::Value y) {
  assert(x != nullptr && y != nullptr);
  return getMostGenericScalarType(x.getType(), y.getType());
}

// clang-format off
/// Get the most generic scalar type among two.
///
///                 |  Boolean  |  Integer  |  Real  |  MLIR Index  |  MLIR Integer  |  MLIR Float
/// Boolean         |  Boolean  |  Integer  |  Real  |  MLIR Index  |  MLIR Integer  |  MLIR Float
/// Integer         |     -     |  Integer  |  Real  |  Integer     |     Integer    |  MLIR Float
/// Real            |     -     |     -     |  Real  |     Real     |      Real      |     Real
/// MLIR Index      |     -     |     -     |   -    |  MLIR Index  |  MLIR Integer  |  MLIR Float
/// MLIR Integer    |     -     |     -     |   -    |       -      |  MLIR Integer  |  MLIR Float
/// MLIR Float      |     -     |     -     |   -    |       -      |       -        |  MLIR Float
// clang-format on
mlir::Type getMostGenericScalarType(mlir::Type first, mlir::Type second) {
  assert(isScalar(first) && isScalar(second));

  if (mlir::isa<BooleanType>(first)) {
    return second;
  }

  if (mlir::isa<IntegerType>(first)) {
    if (mlir::isa<BooleanType>(second)) {
      return first;
    }

    return second;
  }

  if (mlir::isa<RealType>(first)) {
    return first;
  }

  if (mlir::isa<mlir::IndexType>(first)) {
    if (mlir::isa<BooleanType, mlir::IndexType>(second)) {
      return first;
    }

    if (mlir::isa<IntegerType, RealType, mlir::IntegerType, mlir::FloatType>(
            second)) {
      return second;
    }
  }

  if (mlir::isa<mlir::IntegerType>(first)) {
    if (mlir::isa<BooleanType, mlir::IndexType>(second)) {
      return first;
    }

    if (mlir::isa<mlir::IntegerType>(second)) {
      if (first.getIntOrFloatBitWidth() >= second.getIntOrFloatBitWidth()) {
        return first;
      }

      return second;
    }

    if (mlir::isa<IntegerType, RealType, mlir::FloatType>(second)) {
      return second;
    }
  }

  if (mlir::isa<mlir::FloatType>(first)) {
    if (mlir::isa<BooleanType, IntegerType, mlir::IndexType, mlir::IntegerType>(
            second)) {
      return first;
    }

    if (mlir::isa<mlir::FloatType>(second)) {
      if (first.getIntOrFloatBitWidth() >= second.getIntOrFloatBitWidth()) {
        return first;
      }

      return second;
    }

    if (mlir::isa<RealType>(second)) {
      return second;
    }
  }

  llvm_unreachable("Can't compare types");
  return first;
}

bool areScalarTypesCompatible(mlir::Type first, mlir::Type second) {
  return isScalar(first) && isScalar(second);
}

bool areTypesCompatible(mlir::Type first, mlir::Type second) {
  if (isScalar(first) && isScalar(second)) {
    return areScalarTypesCompatible(first, second);
  }

  auto firstShapedType = mlir::dyn_cast<mlir::ShapedType>(first);
  auto secondShapedType = mlir::dyn_cast<mlir::ShapedType>(second);

  if (firstShapedType && secondShapedType) {
    if (mlir::failed(verifyCompatibleShape(firstShapedType.getShape(),
                                           secondShapedType.getShape()))) {
      return false;
    }

    return areTypesCompatible(firstShapedType.getElementType(),
                              secondShapedType.getElementType());
  }

  auto firstRangeType = mlir::dyn_cast<RangeType>(first);
  auto secondRangeType = mlir::dyn_cast<RangeType>(second);

  if (firstRangeType && secondRangeType) {
    return areTypesCompatible(firstRangeType.getInductionType(),
                              secondRangeType.getInductionType());
  }

  return false;
}

bool isScalar(mlir::Type type) {
  if (!type) {
    return false;
  }

  return mlir::isa<BooleanType, IntegerType, RealType, mlir::IndexType,
                   mlir::IntegerType, mlir::FloatType>(type);
}

bool isScalar(mlir::Attribute attribute) {
  if (!attribute) {
    return false;
  }

  if (auto typedAttr = mlir::dyn_cast<mlir::TypedAttr>(attribute)) {
    return isScalar(typedAttr.getType());
  }

  return false;
}

bool isScalarIntegerLike(mlir::Type type) {
  if (!isScalar(type)) {
    return false;
  }

  return mlir::isa<BooleanType, IntegerType, mlir::IndexType,
                   mlir::IntegerType>(type);
}

bool isScalarIntegerLike(mlir::Attribute attribute) {
  if (!attribute) {
    return false;
  }

  if (auto typedAttr = mlir::dyn_cast<mlir::TypedAttr>(attribute)) {
    return isScalarIntegerLike(typedAttr.getType());
  }

  return false;
}

bool isScalarFloatLike(mlir::Type type) {
  if (!isScalar(type)) {
    return false;
  }

  return mlir::isa<RealType, mlir::FloatType>(type);
}

bool isScalarFloatLike(mlir::Attribute attribute) {
  if (!attribute) {
    return false;
  }

  if (auto typedAttr = mlir::dyn_cast<mlir::TypedAttr>(attribute)) {
    return isScalarFloatLike(typedAttr.getType());
  }

  return false;
}

int64_t getScalarIntegerLikeValue(mlir::Attribute attribute) {
  assert(isScalarIntegerLike(attribute));

  if (auto booleanAttr = mlir::dyn_cast<BooleanAttr>(attribute)) {
    return booleanAttr.getValue();
  }

  if (auto integerAttr = mlir::dyn_cast<IntegerAttr>(attribute)) {
    return integerAttr.getValue().getSExtValue();
  }

  return mlir::dyn_cast<mlir::IntegerAttr>(attribute).getValue().getSExtValue();
}

double getScalarFloatLikeValue(mlir::Attribute attribute) {
  assert(isScalarFloatLike(attribute));

  if (auto realAttr = mlir::dyn_cast<RealAttr>(attribute)) {
    return realAttr.getValue().convertToDouble();
  }

  return mlir::dyn_cast<mlir::FloatAttr>(attribute).getValueAsDouble();
}

int64_t getIntegerFromAttribute(mlir::Attribute attribute) {
  if (isScalarIntegerLike(attribute)) {
    return getScalarIntegerLikeValue(attribute);
  }

  if (isScalarFloatLike(attribute)) {
    return static_cast<int64_t>(getScalarFloatLikeValue(attribute));
  }

  llvm_unreachable("Unknown attribute type");
  return 0;
}

std::unique_ptr<DimensionAccess>
getDimensionAccess(const llvm::DenseMap<mlir::Value, unsigned int>
                       &explicitInductionsPositionMap,
                   const AdditionalInductions &additionalInductions,
                   mlir::Value value) {
  if (auto definingOp = value.getDefiningOp()) {
    if (auto op = mlir::dyn_cast<ConstantOp>(definingOp)) {
      auto attr = mlir::cast<mlir::Attribute>(op.getValue());

      if (auto rangeAttr = mlir::dyn_cast<IntegerRangeAttr>(attr)) {
        assert(rangeAttr.getStep() == 1);

        auto lowerBound =
            static_cast<Range::data_type>(rangeAttr.getLowerBound());

        auto upperBound =
            static_cast<Range::data_type>(rangeAttr.getUpperBound());

        return std::make_unique<DimensionAccessRange>(
            value.getContext(), Range(lowerBound, upperBound));
      }

      if (auto rangeAttr = mlir::dyn_cast<RealRangeAttr>(attr)) {
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

    if (auto op = mlir::dyn_cast<NegateOp>(definingOp)) {
      auto operand = getDimensionAccess(explicitInductionsPositionMap,
                                        additionalInductions, op.getOperand());

      if (!operand) {
        return nullptr;
      }

      auto minusOne =
          std::make_unique<DimensionAccessConstant>(value.getContext(), -1);

      return *minusOne * *operand;
    }

    if (auto op = mlir::dyn_cast<AddOp>(definingOp)) {
      auto lhs = getDimensionAccess(explicitInductionsPositionMap,
                                    additionalInductions, op.getLhs());

      auto rhs = getDimensionAccess(explicitInductionsPositionMap,
                                    additionalInductions, op.getRhs());

      if (!lhs || !rhs) {
        return nullptr;
      }

      return *lhs + *rhs;
    }

    if (auto op = mlir::dyn_cast<SubOp>(definingOp)) {
      auto lhs = getDimensionAccess(explicitInductionsPositionMap,
                                    additionalInductions, op.getLhs());

      auto rhs = getDimensionAccess(explicitInductionsPositionMap,
                                    additionalInductions, op.getRhs());

      if (!lhs || !rhs) {
        return nullptr;
      }

      return *lhs - *rhs;
    }

    if (auto op = mlir::dyn_cast<MulOp>(definingOp)) {
      auto lhs = getDimensionAccess(explicitInductionsPositionMap,
                                    additionalInductions, op.getLhs());

      auto rhs = getDimensionAccess(explicitInductionsPositionMap,
                                    additionalInductions, op.getRhs());

      if (!lhs || !rhs) {
        return nullptr;
      }

      return *lhs * *rhs;
    }

    if (auto op = mlir::dyn_cast<DivOp>(definingOp)) {
      auto lhs = getDimensionAccess(explicitInductionsPositionMap,
                                    additionalInductions, op.getLhs());

      auto rhs = getDimensionAccess(explicitInductionsPositionMap,
                                    additionalInductions, op.getRhs());

      if (!lhs || !rhs) {
        return nullptr;
      }

      return *lhs / *rhs;
    }
  }

  if (auto it = explicitInductionsPositionMap.find(value);
      it != explicitInductionsPositionMap.end()) {
    return std::make_unique<DimensionAccessDimension>(value.getContext(),
                                                      it->getSecond());
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

mlir::LogicalResult
materializeAffineMap(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::AffineMap affineMap, mlir::ValueRange dimensions,
                     llvm::SmallVectorImpl<mlir::Value> &results) {
  for (size_t i = 0, e = affineMap.getNumResults(); i < e; ++i) {
    mlir::Value result =
        materializeAffineExpr(builder, loc, affineMap.getResult(i), dimensions);

    if (!result) {
      return mlir::failure();
    }

    results.push_back(result);
  }

  return mlir::success();
}

mlir::Value materializeAffineExpr(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::AffineExpr expression,
                                  mlir::ValueRange dimensions) {
  if (auto constantExpr =
          mlir::dyn_cast<mlir::AffineConstantExpr>(expression)) {
    return builder.create<ConstantOp>(
        loc, builder.getIndexAttr(constantExpr.getValue()));
  }

  if (auto dimExpr = mlir::dyn_cast<mlir::AffineDimExpr>(expression)) {
    assert(dimExpr.getPosition() < dimensions.size());
    return dimensions[dimExpr.getPosition()];
  }

  if (auto binaryExpr = mlir::dyn_cast<mlir::AffineBinaryOpExpr>(expression)) {
    if (binaryExpr.getKind() == mlir::AffineExprKind::Add) {
      mlir::Value lhs =
          materializeAffineExpr(builder, loc, binaryExpr.getLHS(), dimensions);

      mlir::Value rhs =
          materializeAffineExpr(builder, loc, binaryExpr.getRHS(), dimensions);

      if (!lhs || !rhs) {
        return nullptr;
      }

      return builder.create<AddOp>(loc, builder.getIndexType(), lhs, rhs);
    }

    if (binaryExpr.getKind() == mlir::AffineExprKind::Mul) {
      mlir::Value lhs =
          materializeAffineExpr(builder, loc, binaryExpr.getLHS(), dimensions);

      mlir::Value rhs =
          materializeAffineExpr(builder, loc, binaryExpr.getRHS(), dimensions);

      if (!lhs || !rhs) {
        return nullptr;
      }

      return builder.create<MulOp>(loc, builder.getIndexType(), lhs, rhs);
    }

    if (binaryExpr.getKind() == mlir::AffineExprKind::FloorDiv) {
      mlir::Value lhs =
          materializeAffineExpr(builder, loc, binaryExpr.getLHS(), dimensions);

      mlir::Value rhs =
          materializeAffineExpr(builder, loc, binaryExpr.getRHS(), dimensions);

      if (!lhs || !rhs) {
        return nullptr;
      }

      return builder.create<DivOp>(loc, builder.getIndexType(), lhs, rhs);
    }
  }

  return nullptr;
}
} // namespace mlir::bmodelica

namespace {
/// Get a sorted list of the variables appearing in a writes map.
/// The function is intended to be used only for debugging purposes.
template <typename Equation>
llvm::SmallVector<VariableOp>
getWritesMapVariables(const WritesMap<VariableOp, Equation> &writesMap) {
  llvm::DenseSet<VariableOp> uniqueVariables;
  llvm::SmallVector<VariableOp> variables;

  for (const auto &entry : writesMap) {
    VariableOp variable = entry.first;
    assert(variable != nullptr);

    if (!uniqueVariables.contains(variable)) {
      uniqueVariables.insert(variable);
      variables.push_back(variable);
    }
  }

  return variables;
}

template <typename Equation>
llvm::raw_ostream &printWritesMap(llvm::raw_ostream &os,
                                  const WritesMap<VariableOp, Equation> &obj) {
  for (VariableOp variable : getWritesMapVariables(obj)) {
    os << "Variable: " << variable.getSymName() << "\n";

    for (const auto &entry : llvm::make_range(obj.equal_range(variable))) {
      const IndexSet &indices = entry.second.first;
      Equation equation = entry.second.second;

      os << "  - Indices: " << indices << "\n";
      os << "    Equation: ";
      equation.printInline(os);
      os << "\n";
    }
  }

  return os;
}
} // namespace

namespace mlir::bmodelica {
llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           const WritesMap<VariableOp, StartEquationInstanceOp> &obj) {
  return ::printWritesMap(os, obj);
}

llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           const WritesMap<VariableOp, EquationInstanceOp> &obj) {
  return ::printWritesMap(os, obj);
}
} // namespace mlir::bmodelica

namespace {
template <typename Equation>
mlir::LogicalResult
getWritesMap(WritesMap<VariableOp, Equation> &writesMap, ModelOp modelOp,
             llvm::ArrayRef<Equation> equations,
             mlir::SymbolTableCollection &symbolTableCollection) {
  for (Equation equationOp : equations) {
    IndexSet equationIndices = equationOp.getIterationSpace();
    llvm::SmallVector<VariableAccess> accesses;

    if (mlir::failed(equationOp.getAccesses(accesses, symbolTableCollection))) {
      return mlir::failure();
    }

    llvm::SmallVector<VariableAccess> writeAccesses;

    if (mlir::failed(equationOp.getWriteAccesses(
            writeAccesses, symbolTableCollection, accesses))) {
      return mlir::failure();
    }

    assert(!writeAccesses.empty());

    auto writtenVariableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
        modelOp, writeAccesses[0].getVariable());

    assert(writtenVariableOp != nullptr);

    IndexSet writtenVariableIndices;

    for (const VariableAccess &writeAccess : writeAccesses) {
      const AccessFunction &accessFunction = writeAccess.getAccessFunction();
      writtenVariableIndices += accessFunction.map(equationIndices);
    }

    writesMap.emplace(
        writtenVariableOp,
        std::make_pair(std::move(writtenVariableIndices), equationOp));
  }

  return mlir::success();
}
} // namespace

namespace mlir::bmodelica {
mlir::LogicalResult
getWritesMap(WritesMap<VariableOp, StartEquationInstanceOp> &writesMap,
             ModelOp modelOp, llvm::ArrayRef<StartEquationInstanceOp> equations,
             mlir::SymbolTableCollection &symbolTableCollection) {
  for (StartEquationInstanceOp equation : equations) {
    IndexSet equationIndices = equation.getIterationSpace();
    llvm::SmallVector<VariableAccess> accesses;

    if (mlir::failed(equation.getAccesses(accesses, symbolTableCollection))) {
      return mlir::failure();
    }

    auto writeAccess = equation.getWriteAccess(symbolTableCollection);

    auto writtenVariableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
        modelOp, writeAccess->getVariable());

    assert(writtenVariableOp != nullptr);

    IndexSet writtenVariableIndices =
        writeAccess->getAccessFunction().map(equationIndices);

    writesMap.emplace(
        writtenVariableOp,
        std::make_pair(std::move(writtenVariableIndices), equation));
  }

  return mlir::success();
}

mlir::LogicalResult
getWritesMap(WritesMap<VariableOp, EquationInstanceOp> &writesMap,
             ModelOp modelOp, llvm::ArrayRef<EquationInstanceOp> equations,
             mlir::SymbolTableCollection &symbolTableCollection) {
  return ::getWritesMap<EquationInstanceOp>(writesMap, modelOp, equations,
                                            symbolTableCollection);
}

mlir::LogicalResult
getWritesMap(WritesMap<VariableOp, EquationInstanceOp> &writesMap,
             ModelOp modelOp, llvm::ArrayRef<SCCOp> SCCs,
             mlir::SymbolTableCollection &symbolTableCollection) {
  llvm::SmallVector<EquationInstanceOp> equations;

  for (SCCOp scc : SCCs) {
    scc.collectEquations(equations);
  }

  return getWritesMap(writesMap, modelOp, equations, symbolTableCollection);
}

template <>
mlir::LogicalResult getWritesMap<EquationInstanceOp>(
    WritesMap<VariableOp, SCCOp> &writesMap, ModelOp modelOp,
    llvm::ArrayRef<SCCOp> SCCs,
    mlir::SymbolTableCollection &symbolTableCollection) {
  llvm::SmallVector<EquationInstanceOp> equations;

  for (SCCOp scc : SCCs) {
    for (EquationInstanceOp equation : scc.getOps<EquationInstanceOp>()) {
      equations.push_back(equation);
    }
  }

  WritesMap<VariableOp, EquationInstanceOp> equationsWritesMap;

  if (mlir::failed(getWritesMap(equationsWritesMap, modelOp, equations,
                                symbolTableCollection))) {
    return mlir::failure();
  }

  for (const auto &entry : equationsWritesMap) {
    auto parentSCC = entry.second.second->getParentOfType<SCCOp>();
    assert(parentSCC != nullptr);

    writesMap.emplace(entry.first,
                      std::make_pair(entry.second.first, parentSCC));
  }

  return mlir::success();
}

mlir::LogicalResult
getWritesMap(WritesMap<VariableOp, ScheduleBlockOp> &writesMap, ModelOp modelOp,
             llvm::ArrayRef<ScheduleBlockOp> scheduleBlocks,
             mlir::SymbolTableCollection &symbolTableCollection) {
  for (ScheduleBlockOp block : scheduleBlocks) {
    for (const Variable &writtenVariable :
         block.getProperties().writtenVariables) {
      auto variableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
          modelOp, writtenVariable.name);

      if (!variableOp) {
        return mlir::failure();
      }

      writesMap.emplace(variableOp,
                        std::make_pair(writtenVariable.indices, block));
    }
  }

  return mlir::success();
}

bool isReservedVariable(llvm::StringRef name) { return name.starts_with("__"); }

std::string getReservedVariableName(llvm::StringRef name) {
  return "__" + name.str();
}
} // namespace mlir::bmodelica
