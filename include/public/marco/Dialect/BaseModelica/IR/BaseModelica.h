#ifndef MARCO_DIALECT_BASEMODELICA_IR_BASEMODELICA_H
#define MARCO_DIALECT_BASEMODELICA_IR_BASEMODELICA_H

#include "marco/Dialect/BaseModelica/IR/Attributes.h"
#include "marco/Dialect/BaseModelica/IR/Common.h"
#include "marco/Dialect/BaseModelica/IR/OpInterfaces.h"
#include "marco/Dialect/BaseModelica/IR/Ops.h"
#include "marco/Dialect/BaseModelica/IR/Types.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "marco/Dialect/BaseModelica/IR/BaseModelica.h.inc"

namespace mlir::bmodelica
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

  mlir::SymbolRefAttr getSymbolRefFromRoot(mlir::Operation* symbol);

  mlir::Operation* resolveSymbol(
      mlir::ModuleOp moduleOp,
      mlir::SymbolTableCollection& symbolTableCollection,
      mlir::SymbolRefAttr symbol);

  template<typename Op>
  Op resolveSymbol(
      mlir::ModuleOp moduleOp,
      mlir::SymbolTableCollection& symbolTableCollection,
      mlir::SymbolRefAttr symbol)
  {
    return mlir::dyn_cast_if_present<Op>(
        resolveSymbol(moduleOp, symbolTableCollection, symbol));
  }

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

  std::unique_ptr<DimensionAccess> getDimensionAccess(
      const llvm::DenseMap<mlir::Value, unsigned int>& explicitInductionsPositionMap,
      const AdditionalInductions& additionalInductions,
      mlir::Value value);

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

  // Map between variables and the equations writing to them.
  // The indices are referred to the written indices of the variable
  // (and not to the indices of the equation).
  template<typename Variable, typename WritingEntity>
  using WritesMap =
      std::multimap<Variable, std::pair<IndexSet, WritingEntity>>;

  llvm::raw_ostream& operator<<(
      llvm::raw_ostream& os,
      const WritesMap<VariableOp, StartEquationInstanceOp>& obj);

  llvm::raw_ostream& operator<<(
      llvm::raw_ostream& os,
      const WritesMap<VariableOp, MatchedEquationInstanceOp>& obj);

  llvm::raw_ostream& operator<<(
      llvm::raw_ostream& os,
      const WritesMap<VariableOp, ScheduledEquationInstanceOp>& obj);

  mlir::LogicalResult getWritesMap(
      WritesMap<VariableOp, StartEquationInstanceOp>& writesMap,
      ModelOp modelOp,
      llvm::ArrayRef<StartEquationInstanceOp> equations,
      mlir::SymbolTableCollection& symbolTableCollection);

  mlir::LogicalResult getWritesMap(
      WritesMap<VariableOp, MatchedEquationInstanceOp>& writesMap,
      ModelOp modelOp,
      llvm::ArrayRef<MatchedEquationInstanceOp> equations,
      mlir::SymbolTableCollection& symbolTableCollection);

  mlir::LogicalResult getWritesMap(
      WritesMap<VariableOp, ScheduledEquationInstanceOp>& writesMap,
      ModelOp modelOp,
      llvm::ArrayRef<ScheduledEquationInstanceOp> equations,
      mlir::SymbolTableCollection& symbolTableCollection);

  mlir::LogicalResult getWritesMap(
      WritesMap<VariableOp, MatchedEquationInstanceOp>& writesMap,
      ModelOp modelOp,
      llvm::ArrayRef<SCCOp> SCCs,
      mlir::SymbolTableCollection& symbolTableCollection);

  template<typename Equation>
  mlir::LogicalResult getWritesMap(
      WritesMap<VariableOp, SCCOp>& writesMap,
      ModelOp modelOp,
      llvm::ArrayRef<SCCOp> SCCs,
      mlir::SymbolTableCollection& symbolTableCollection);

  template<>
  mlir::LogicalResult getWritesMap<MatchedEquationInstanceOp>(
      WritesMap<VariableOp, SCCOp>& writesMap,
      ModelOp modelOp,
      llvm::ArrayRef<SCCOp> SCCs,
      mlir::SymbolTableCollection& symbolTableCollection);

  template<>
  mlir::LogicalResult getWritesMap<ScheduledEquationInstanceOp>(
      WritesMap<VariableOp, SCCOp>& writesMap,
      ModelOp modelOp,
      llvm::ArrayRef<SCCOp> SCCs,
      mlir::SymbolTableCollection& symbolTableCollection);

  mlir::LogicalResult getWritesMap(
      WritesMap<VariableOp, ScheduleBlockOp>& writesMap,
      ModelOp modelOp,
      llvm::ArrayRef<ScheduleBlockOp> scheduleBlocks,
      mlir::SymbolTableCollection& symbolTableCollection);
}

#endif // MARCO_DIALECT_BASEMODELICA_IR_BASEMODELICA_H
