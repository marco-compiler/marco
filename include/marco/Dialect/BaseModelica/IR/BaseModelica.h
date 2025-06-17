#ifndef MARCO_DIALECT_BASEMODELICA_IR_BASEMODELICA_H
#define MARCO_DIALECT_BASEMODELICA_IR_BASEMODELICA_H

#include "marco/Dialect/BaseModelica/IR/Attributes.h"
#include "marco/Dialect/BaseModelica/IR/Common.h"
#include "marco/Dialect/BaseModelica/IR/Enums.h"
#include "marco/Dialect/BaseModelica/IR/OpInterfaces.h"
#include "marco/Dialect/BaseModelica/IR/Ops.h"
#include "marco/Dialect/BaseModelica/IR/Types.h"
#include "marco/Modeling/RTree.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "marco/Dialect/BaseModelica/IR/BaseModelica.h.inc"

namespace mlir::bmodelica {
using DimensionAccess = ::marco::modeling::DimensionAccess;
using DimensionAccessConstant = ::marco::modeling::DimensionAccessConstant;
using DimensionAccessDimension = ::marco::modeling::DimensionAccessDimension;
using DimensionAccessAdd = ::marco::modeling::DimensionAccessAdd;
using DimensionAccessSub = ::marco::modeling::DimensionAccessSub;
using DimensionAccessMul = ::marco::modeling::DimensionAccessMul;
using DimensionAccessDiv = ::marco::modeling::DimensionAccessDiv;
using DimensionAccessRange = ::marco::modeling::DimensionAccessRange;
using DimensionAccessIndices = ::marco::modeling::DimensionAccessIndices;

mlir::SymbolRefAttr getSymbolRefFromRoot(mlir::Operation *symbol);

mlir::Operation *
resolveSymbol(mlir::ModuleOp moduleOp,
              mlir::SymbolTableCollection &symbolTableCollection,
              mlir::SymbolRefAttr symbol);

template <typename Op>
Op resolveSymbol(mlir::ModuleOp moduleOp,
                 mlir::SymbolTableCollection &symbolTableCollection,
                 mlir::SymbolRefAttr symbol) {
  return mlir::dyn_cast_if_present<Op>(
      resolveSymbol(moduleOp, symbolTableCollection, symbol));
}

void walkClasses(mlir::Operation *root,
                 llvm::function_ref<void(mlir::Operation *)> callback);

mlir::Type getMostGenericScalarType(mlir::Value first, mlir::Value second);

mlir::Type getMostGenericScalarType(mlir::Type first, mlir::Type second);

bool areScalarTypesCompatible(mlir::Type first, mlir::Type second);

bool areTypesCompatible(mlir::Type first, mlir::Type second);

bool isScalar(mlir::Type type);

bool isScalar(mlir::Attribute attribute);

bool isScalarIntegerLike(mlir::Type type);

bool isScalarIntegerLike(mlir::Attribute attribute);

bool isScalarFloatLike(mlir::Type type);

bool isScalarFloatLike(mlir::Attribute attribute);

int64_t getScalarIntegerLikeValue(mlir::Attribute attribute);

double getScalarFloatLikeValue(mlir::Attribute attribute);

int64_t getIntegerFromAttribute(mlir::Attribute attribute);

std::unique_ptr<DimensionAccess>
getDimensionAccess(const llvm::DenseMap<mlir::Value, unsigned int>
                       &explicitInductionsPositionMap,
                   const AdditionalInductions &additionalInductions,
                   mlir::Value value);

mlir::LogicalResult
materializeAffineMap(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::AffineMap affineMap, mlir::ValueRange dimensions,
                     llvm::SmallVectorImpl<mlir::Value> &results);

mlir::Value materializeAffineExpr(mlir::OpBuilder &builder, mlir::Location loc,
                                  mlir::AffineExpr expression,
                                  mlir::ValueRange dimensions);

template <typename WritingEntity>
struct WriteInfo {
  IndexSet writtenVariableIndices;
  WritingEntity writingEntity;
};

/// An entry of the writes map.
template <typename WritingEntity>
struct WritesMapRTreeEntry {
  MultidimensionalRange writtenVariableIndices;
  WritingEntity writingEntity;
};

template <typename Variable, typename WritingEntity>
class WritesMap {
  using ScalarMap = llvm::DenseMap<VariableOp, WritingEntity>;
  using RTreeObject = WritesMapRTreeEntry<WritingEntity>;

  using ArrayMap =
      llvm::DenseMap<Variable, marco::modeling::RTree<RTreeObject>>;

  ScalarMap scalarMap;
  ArrayMap arrayMap;

public:
  llvm::SmallVector<Variable> getVariables() const {
    llvm::SmallVector<Variable> result;
    llvm::DenseSet<Variable> uniqueVariables;

    for (const auto &entry : scalarMap) {
      if (!uniqueVariables.contains(entry.first)) {
        result.push_back(entry.first);
      }
    }

    for (const auto &entry : arrayMap) {
      if (!uniqueVariables.contains(entry.first)) {
        result.push_back(entry.first);
      }
    }

    return result;
  }

  /// Get the equation writing to a set of indices that overlap a given one.
  llvm::SmallVector<WriteInfo<WritingEntity>>
  getWrites(const Variable &variable, const IndexSet &variableIndices) const {
    llvm::MapVector<WritingEntity, IndexSet> uniqueEntities;

    if (variableIndices.empty()) {
      auto it = scalarMap.find(variable);

      if (it != scalarMap.end()) {
        uniqueEntities[it->second] = {};
      }
    } else {
      auto it = arrayMap.find(variable);

      if (it != arrayMap.end()) {
        for (const MultidimensionalRange &variableRange : llvm::make_range(
                 variableIndices.rangesBegin(), variableIndices.rangesEnd())) {
          it->second.walkOverlappingObjects(
              variableRange, [&](const RTreeObject &entry) {
                uniqueEntities[entry.writingEntity] +=
                    entry.writtenVariableIndices;
              });
        }
      }
    }

    llvm::SmallVector<WriteInfo<WritingEntity>> result;
    result.reserve(uniqueEntities.size());

    for (auto &entity : uniqueEntities.takeVector()) {
      result.push_back({std::move(entity.second), std::move(entity.first)});
    }

    return result;
  }

  void addWrite(const Variable &variable, const IndexSet &variableIndices,
                WritingEntity writingEntity) {
    if (variableIndices.empty()) {
      scalarMap[variable] = writingEntity;
    } else {
      for (const MultidimensionalRange &variableRange : llvm::make_range(
               variableIndices.rangesBegin(), variableIndices.rangesEnd())) {
        arrayMap[variable].insert(RTreeObject{variableRange, writingEntity});
      }
    }
  }

  bool isWritten(const Variable &variable) const {
    return scalarMap.contains(variable) || arrayMap.contains(variable);
  }

  std::optional<IndexSet> getWrittenIndices(const Variable &variable) const {
    IndexSet result;

    if (!scalarMap.contains(variable) && !arrayMap.contains(variable)) {
      return std::nullopt;
    }

    auto arrayMapIt = arrayMap.find(variable);

    if (arrayMapIt != arrayMap.end()) {
      for (const auto &entry : arrayMapIt->second) {
        result += entry.writtenVariableIndices;
      }
    }

    return result;
  }
};

llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           const WritesMap<VariableOp, StartEquationInstanceOp> &obj);

llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           const WritesMap<VariableOp, EquationInstanceOp> &obj);

mlir::LogicalResult
getWritesMap(WritesMap<VariableOp, StartEquationInstanceOp> &writesMap,
             ModelOp modelOp, llvm::ArrayRef<StartEquationInstanceOp> equations,
             mlir::SymbolTableCollection &symbolTableCollection);

mlir::LogicalResult
getWritesMap(WritesMap<VariableOp, EquationInstanceOp> &writesMap,
             ModelOp modelOp, llvm::ArrayRef<EquationInstanceOp> equations,
             mlir::SymbolTableCollection &symbolTableCollection);

mlir::LogicalResult
getWritesMap(WritesMap<VariableOp, EquationInstanceOp> &writesMap,
             ModelOp modelOp, llvm::ArrayRef<SCCOp> SCCs,
             mlir::SymbolTableCollection &symbolTableCollection);

template <typename Equation>
mlir::LogicalResult
getWritesMap(WritesMap<VariableOp, SCCOp> &writesMap, ModelOp modelOp,
             llvm::ArrayRef<SCCOp> SCCs,
             mlir::SymbolTableCollection &symbolTableCollection);

template <>
mlir::LogicalResult getWritesMap<EquationInstanceOp>(
    WritesMap<VariableOp, SCCOp> &writesMap, ModelOp modelOp,
    llvm::ArrayRef<SCCOp> SCCs,
    mlir::SymbolTableCollection &symbolTableCollection);

mlir::LogicalResult
getWritesMap(WritesMap<VariableOp, ScheduleBlockOp> &writesMap, ModelOp modelOp,
             llvm::ArrayRef<ScheduleBlockOp> scheduleBlocks,
             mlir::SymbolTableCollection &symbolTableCollection);

bool isReservedVariable(llvm::StringRef name);

std::string getReservedVariableName(llvm::StringRef name);
} // namespace mlir::bmodelica

namespace marco::modeling {
template <typename WritingEntity>
struct RTreeInfo<mlir::bmodelica::WritesMapRTreeEntry<WritingEntity>> {
  using Obj = mlir::bmodelica::WritesMapRTreeEntry<WritingEntity>;

  static const MultidimensionalRange &getShape(const Obj &obj) {
    return obj.writtenVariableIndices;
  }

  static bool isEqual(const Obj &first, const Obj &second) {
    return first.writingEntity == second.writingEntity &&
           first.writtenVariableIndices == second.writtenVariableIndices;
  }
};
} // namespace marco::modeling

#endif // MARCO_DIALECT_BASEMODELICA_IR_BASEMODELICA_H
