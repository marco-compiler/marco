#include "marco/Dialect/IDA/IDADialect.h"
#include "marco/Dialect/IDA/Ops.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"

using namespace ::mlir;
using namespace ::mlir::ida;

#define GET_OP_CLASSES
#include "marco/Dialect/IDA/IDA.cpp.inc"

namespace mlir::ida
{
  //===----------------------------------------------------------------------===//
  // CreateOp
  //===----------------------------------------------------------------------===//

  void CreateOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(), mlir::SideEffects::AutomaticAllocationScopeResource::get());
  }

  //===----------------------------------------------------------------------===//
  // SetStartTimeOp
  //===----------------------------------------------------------------------===//

  void SetStartTimeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getInstance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // SetEndTimeOp
  //===----------------------------------------------------------------------===//

  void SetEndTimeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getInstance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // GetCurrentTimeOp
  //===----------------------------------------------------------------------===//

  void GetCurrentTimeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), getInstance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // InitOp
  //===----------------------------------------------------------------------===//

  void InitOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), getInstance(), mlir::SideEffects::DefaultResource::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getInstance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // FreeOp
  //===----------------------------------------------------------------------===//

  void FreeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Free::get(), getInstance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // CalcICOp
  //===----------------------------------------------------------------------===//

  void CalcICOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), getInstance(), mlir::SideEffects::DefaultResource::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getInstance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // StepOp
  //===----------------------------------------------------------------------===//

  void StepOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), getInstance(), mlir::SideEffects::DefaultResource::get());
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getInstance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // AddEquationOp
  //===----------------------------------------------------------------------===//

  void AddEquationOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getInstance(), mlir::SideEffects::DefaultResource::get());
    effects.emplace_back(mlir::MemoryEffects::Read::get(), getEquationRanges(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // AddResidualOp
  //===----------------------------------------------------------------------===//

  void AddResidualOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getInstance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // AddJacobianOp
  //===----------------------------------------------------------------------===//

  void AddJacobianOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getInstance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // AddAlgebraicVariableOp
  //===----------------------------------------------------------------------===//

  void AddAlgebraicVariableOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getInstance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // AddStateVariableOp
  //===----------------------------------------------------------------------===//

  void AddStateVariableOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getInstance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // SetDerivativeOp
  //===----------------------------------------------------------------------===//

  void SetDerivativeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getInstance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // AddParametricVariableOp
  //===----------------------------------------------------------------------===//

  void AddParametricVariableOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getInstance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // VariableGetterOp
  //===----------------------------------------------------------------------===//

  mlir::BlockArgument VariableGetterOp::getVariable()
  {
    return getBodyRegion().getArgument(0);
  }

  llvm::ArrayRef<BlockArgument> VariableGetterOp::getVariableIndices()
  {
    return getBodyRegion().getArguments().slice(1);
  }

  //===----------------------------------------------------------------------===//
  // VariableSetterOp
  //===----------------------------------------------------------------------===//

  mlir::BlockArgument VariableSetterOp::getVariable()
  {
    return getBodyRegion().getArgument(0);
  }

  BlockArgument VariableSetterOp::getValue()
  {
    return getBodyRegion().getArgument(1);
  }

  llvm::ArrayRef<BlockArgument> VariableSetterOp::getVariableIndices()
  {
    return getBodyRegion().getArguments().slice(2);
  }

  //===----------------------------------------------------------------------===//
  // AddVariableAccessOp
  //===----------------------------------------------------------------------===//

  void AddVariableAccessOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getInstance(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // ResidualFunctionOp
  //===----------------------------------------------------------------------===//

  mlir::BlockArgument ResidualFunctionOp::getTime()
  {
    return getBodyRegion().getArgument(0);
  }

  llvm::ArrayRef<BlockArgument> ResidualFunctionOp::getVariables()
  {
    size_t numVariables = getBodyRegion().getNumArguments() - 1 - getEquationRank().getSExtValue();
    return getBodyRegion().getArguments().slice(1, numVariables);
  }

  llvm::ArrayRef<BlockArgument> ResidualFunctionOp::getEquationIndices()
  {
    return getBodyRegion().getArguments().take_back(getEquationRank().getSExtValue());
  }

  //===----------------------------------------------------------------------===//
  // JacobianFunctionOp
  //===----------------------------------------------------------------------===//

  mlir::BlockArgument JacobianFunctionOp::getTime()
  {
    return getBodyRegion().getArgument(0);
  }

  llvm::ArrayRef<BlockArgument> JacobianFunctionOp::getVariables()
  {
    size_t numVariables = getBodyRegion().getNumArguments() - 1 - getEquationRank().getSExtValue() - getVariableRank().getSExtValue() - 1;
    return getBodyRegion().getArguments().slice(1, numVariables);
  }

  llvm::ArrayRef<BlockArgument> JacobianFunctionOp::getEquationIndices()
  {
    size_t offset = getBodyRegion().getNumArguments() - getEquationRank().getSExtValue() - getVariableRank().getSExtValue() - 1;
    return getBodyRegion().getArguments().slice(offset, getEquationRank().getSExtValue());
  }

  llvm::ArrayRef<BlockArgument> JacobianFunctionOp::getVariableIndices()
  {
    size_t offset = getBodyRegion().getNumArguments() - getVariableRank().getSExtValue() - 1;
    return getBodyRegion().getArguments().slice(offset, getVariableRank().getSExtValue());
  }

  BlockArgument JacobianFunctionOp::getAlpha()
  {
    return getBodyRegion().getArguments().back();
  }

  //===----------------------------------------------------------------------===//
  // PrintStatisticsOp
  //===----------------------------------------------------------------------===//

  void PrintStatisticsOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), getInstance(), mlir::SideEffects::DefaultResource::get());
  }
}
