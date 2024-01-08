#ifndef MARCO_DIALECTS_MODELICA_MODELICADIALECT_H
#define MARCO_DIALECTS_MODELICA_MODELICADIALECT_H

#include "marco/Dialect/Modelica/Attributes.h"
#include "marco/Dialect/Modelica/Common.h"
#include "marco/Dialect/Modelica/OpInterfaces.h"
#include "marco/Dialect/Modelica/Ops.h"
#include "marco/Dialect/Modelica/Types.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "marco/Dialect/Modelica/ModelicaDialect.h.inc"

namespace mlir::modelica
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

  mlir::Type getMostGenericType(mlir::Value x, mlir::Value y);

  mlir::Type getMostGenericType(mlir::Type x, mlir::Type y);

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
  template<typename Variable, typename Equation>
  using WritesMap = std::multimap<Variable, std::pair<IndexSet, Equation>>;

  mlir::LogicalResult getWritesMap(
      WritesMap<VariableOp, MatchedEquationInstanceOp>& writesMap,
      ModelOp modelOp,
      llvm::ArrayRef<MatchedEquationInstanceOp> equations,
      mlir::SymbolTableCollection& symbolTableCollection);

  mlir::LogicalResult getWritesMap(
      WritesMap<VariableOp, MatchedEquationInstanceOp>& writesMap,
      ModelOp modelOp,
      llvm::ArrayRef<SCCOp> SCCs,
      mlir::SymbolTableCollection& symbolTableCollection);

  mlir::LogicalResult getWritesMap(
      WritesMap<SimulationVariableOp, MatchedEquationInstanceOp>& writesMap,
      mlir::ModuleOp moduleOp,
      ScheduleOp scheduleOp,
      mlir::SymbolTableCollection& symbolTableCollection);
}

#endif // MARCO_DIALECTS_MODELICA_MODELICADIALECT_H
