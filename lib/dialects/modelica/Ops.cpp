#ifdef MSVC_BUILD
// Allows to compile functions that do not return.
#pragma warning(disable : 4716)
#endif

#include "marco/dialects/modelica/ModelicaDialect.h"
#include "marco/dialects/modelica/Ops.h"
#include "mlir/IR/OpImplementation.h"

using namespace ::mlir;
using namespace ::mlir::modelica;

static mlir::Value readValue(mlir::OpBuilder& builder, mlir::Value operand)
{
  /*
  if (auto arrayType = operand.getType().dyn_cast<ArrayType>(); arrayType && arrayType.getRank() == 0)
    return builder.create<LoadOp>(operand.getLoc(), operand);

  return operand;
   */
  return nullptr;
}

static mlir::LogicalResult verify(AbsOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(AcosOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(AddOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(AddEWOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(AndOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(ArrayCastOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(ArrayCloneOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(AsinOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(AtanOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(Atan2Op op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(ConditionOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(CosOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(CoshOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(DerOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(DiagonalOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(DimOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(DivOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(DivEWOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(EqOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(ExpOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(FillOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(FreeOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(GtOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(GteOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(IdentityOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(LinspaceOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(LoadOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(LogOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(Log10Op op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(LtOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(LteOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(MulOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(MulEWOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(NDimsOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(NegateOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(NotOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(NotEqOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(OnesOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(OrOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(PowOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(PowEWOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(PrintOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(ProductOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(SignOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(SinOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(SinhOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(SizeOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(SqrtOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(StoreOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(SubOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(SubEWOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(SumOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(SubscriptionOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(SymmetricOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(TanOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(TanhOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(TransposeOp op)
{
  return mlir::success();
}

static mlir::LogicalResult verify(ZerosOp op)
{
  return mlir::success();
}

#define GET_OP_CLASSES
#include "marco/dialects/modelica/Modelica.cpp.inc"

namespace mlir::modelica
{
  //===----------------------------------------------------------------------===//
  // AbsOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange AbsOp::getArgs()
  {

  }

  unsigned int AbsOp::getArgExpectedRank(unsigned int argIndex)
  {

  }

  mlir::ValueRange AbsOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {

  }

  //===----------------------------------------------------------------------===//
  // AcosOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange AcosOp::getArgs()
  {

  }

  unsigned int AcosOp::getArgExpectedRank(unsigned int argIndex)
  {

  }

  mlir::ValueRange AcosOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {

  }

  mlir::ValueRange AcosOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void AcosOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void AcosOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // AddEWOp
  //===----------------------------------------------------------------------===//

  void AddEWOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }

  mlir::LogicalResult AddEWOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {

  }

  mlir::Value AddEWOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {

  }

  mlir::Value AddEWOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {

  }

  mlir::Value AddEWOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {

  }

  mlir::ValueRange AddEWOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void AddEWOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void AddEWOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // AddOp
  //===----------------------------------------------------------------------===//

  void AddOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }

  mlir::LogicalResult AddOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {

  }

  mlir::Value AddOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {

  }

  mlir::Value AddOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {

  }

  mlir::Value AddOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {

  }

  mlir::ValueRange AddOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void AddOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void AddOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // AndOp
  //===----------------------------------------------------------------------===//

  void AndOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }

  //===----------------------------------------------------------------------===//
  // ArrayCastOp
  //===----------------------------------------------------------------------===//

  mlir::Value ArrayCastOp::getViewSource()
  {

  }

  //===----------------------------------------------------------------------===//
  // ArrayCloneOp
  //===----------------------------------------------------------------------===//

  void ArrayCloneOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }

  /*
  bool ArrayCloneOp::canSourceBeForwarded()
  {
    return Adaptor(*this).canSourceBeForwarded();
  }
   */

  //===----------------------------------------------------------------------===//
  // AsinOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange AsinOp::getArgs()
  {

  }

  unsigned int AsinOp::getArgExpectedRank(unsigned int argIndex)
  {

  }

  mlir::ValueRange AsinOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {

  }

  mlir::ValueRange AsinOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void AsinOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void AsinOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // Atan2Op
  //===----------------------------------------------------------------------===//

  mlir::ValueRange Atan2Op::getArgs()
  {

  }

  unsigned int Atan2Op::getArgExpectedRank(unsigned int argIndex)
  {

  }

  mlir::ValueRange Atan2Op::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {

  }

  //===----------------------------------------------------------------------===//
  // AtanOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange AtanOp::getArgs()
  {

  }

  unsigned int AtanOp::getArgExpectedRank(unsigned int argIndex)
  {

  }

  mlir::ValueRange AtanOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {

  }

  mlir::ValueRange AtanOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void AtanOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void AtanOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // CosOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange CosOp::getArgs()
  {

  }

  unsigned int CosOp::getArgExpectedRank(unsigned int argIndex)
  {

  }

  mlir::ValueRange CosOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {

  }

  mlir::ValueRange CosOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void CosOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void CosOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // CoshOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange CoshOp::getArgs()
  {

  }

  unsigned int CoshOp::getArgExpectedRank(unsigned int argIndex)
  {

  }

  mlir::ValueRange CoshOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {

  }

  mlir::ValueRange CoshOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void CoshOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void CoshOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // DiagonalOp
  //===----------------------------------------------------------------------===//

  void DiagonalOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }

  //===----------------------------------------------------------------------===//
  // DivEWOp
  //===----------------------------------------------------------------------===//

  void DivEWOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }

  mlir::LogicalResult DivEWOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {

  }

  mlir::Value DivEWOp::distribute(mlir::OpBuilder& builder)
  {

  }

  mlir::Value DivEWOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {

  }

  mlir::Value DivEWOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {

  }

  mlir::Value DivEWOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {

  }

  mlir::ValueRange DivEWOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void DivEWOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void DivEWOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // DivOp
  //===----------------------------------------------------------------------===//

  void DivOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }

  mlir::LogicalResult DivOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {

  }

  mlir::Value DivOp::distribute(mlir::OpBuilder& builder)
  {

  }

  mlir::Value DivOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {

  }

  mlir::Value DivOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {

  }

  mlir::Value DivOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {

  }

  mlir::ValueRange DivOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void DivOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void DivOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // ExpOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange ExpOp::getArgs()
  {

  }

  unsigned int ExpOp::getArgExpectedRank(unsigned int argIndex)
  {

  }

  mlir::ValueRange ExpOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {

  }

  mlir::ValueRange ExpOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void ExpOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void ExpOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // ForOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange ForOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void ForOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void ForOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // FreeOp
  //===----------------------------------------------------------------------===//

  void FreeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }

  //===----------------------------------------------------------------------===//
  // IdentityOp
  //===----------------------------------------------------------------------===//

  void IdentityOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }

  //===----------------------------------------------------------------------===//
  // IfOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange IfOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void IfOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void IfOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // LinspaceOp
  //===----------------------------------------------------------------------===//

  void LinspaceOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }

  //===----------------------------------------------------------------------===//
  // LoadOp
  //===----------------------------------------------------------------------===//

  void LoadOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }

  mlir::ValueRange LoadOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void LoadOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void LoadOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // Log10Op
  //===----------------------------------------------------------------------===//

  mlir::ValueRange Log10Op::getArgs()
  {

  }

  unsigned int Log10Op::getArgExpectedRank(unsigned int argIndex)
  {

  }

  mlir::ValueRange Log10Op::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {

  }

  mlir::ValueRange Log10Op::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void Log10Op::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void Log10Op::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // LogOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange LogOp::getArgs()
  {

  }

  unsigned int LogOp::getArgExpectedRank(unsigned int argIndex)
  {

  }

  mlir::ValueRange LogOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {

  }

  mlir::ValueRange LogOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void LogOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void LogOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // MulEWOp
  //===----------------------------------------------------------------------===//

  void MulEWOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }

  mlir::LogicalResult MulEWOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {

  }

  mlir::Value MulEWOp::distribute(mlir::OpBuilder& builder)
  {

  }

  mlir::Value MulEWOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {

  }

  mlir::Value MulEWOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {

  }

  mlir::Value MulEWOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {

  }

  mlir::ValueRange MulEWOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void MulEWOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void MulEWOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // MulOp
  //===----------------------------------------------------------------------===//

  void MulOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }

  mlir::LogicalResult MulOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {

  }

  mlir::Value MulOp::distribute(mlir::OpBuilder& builder)
  {

  }

  mlir::Value MulOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {

  }

  mlir::Value MulOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {

  }

  mlir::Value MulOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {

  }

  mlir::ValueRange MulOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void MulOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void MulOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // NegateOp
  //===----------------------------------------------------------------------===//

  void NegateOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }

  mlir::LogicalResult NegateOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {

  }

  mlir::Value NegateOp::distribute(mlir::OpBuilder& builder)
  {

  }

  mlir::Value NegateOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {

  }

  mlir::Value NegateOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {

  }

  mlir::Value NegateOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {

  }

  mlir::ValueRange NegateOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void NegateOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void NegateOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // NotOp
  //===----------------------------------------------------------------------===//

  void NotOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }

  //===----------------------------------------------------------------------===//
  // OnesOp
  //===----------------------------------------------------------------------===//

  void OnesOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }

  //===----------------------------------------------------------------------===//
  // OrOp
  //===----------------------------------------------------------------------===//

  void OrOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }

  //===----------------------------------------------------------------------===//
  // PowEWOp
  //===----------------------------------------------------------------------===//

  void PowEWOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }

  mlir::ValueRange PowEWOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void PowEWOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void PowEWOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // PowOp
  //===----------------------------------------------------------------------===//

  void PowOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }

  mlir::ValueRange PowOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void PowOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void PowOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // SignOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange SignOp::getArgs()
  {

  }

  unsigned int SignOp::getArgExpectedRank(unsigned int argIndex)
  {

  }

  mlir::ValueRange SignOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {

  }

  //===----------------------------------------------------------------------===//
  // SinOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange SinOp::getArgs()
  {

  }

  unsigned int SinOp::getArgExpectedRank(unsigned int argIndex)
  {

  }

  mlir::ValueRange SinOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {

  }

  mlir::ValueRange SinOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void SinOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void SinOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // SinhOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange SinhOp::getArgs()
  {

  }

  unsigned int SinhOp::getArgExpectedRank(unsigned int argIndex)
  {

  }

  mlir::ValueRange SinhOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {

  }

  mlir::ValueRange SinhOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void SinhOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void SinhOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // SizeOp
  //===----------------------------------------------------------------------===//

  void SizeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }

  //===----------------------------------------------------------------------===//
  // SqrtOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange SqrtOp::getArgs()
  {

  }

  unsigned int SqrtOp::getArgExpectedRank(unsigned int argIndex)
  {

  }

  mlir::ValueRange SqrtOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {

  }

  //===----------------------------------------------------------------------===//
  // StoreOp
  //===----------------------------------------------------------------------===//

  void StoreOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }

  mlir::ValueRange StoreOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void StoreOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void StoreOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // SubEWOp
  //===----------------------------------------------------------------------===//

  void SubEWOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }

  mlir::LogicalResult SubEWOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {

  }

  mlir::Value SubEWOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {

  }

  mlir::Value SubEWOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {

  }

  mlir::Value SubEWOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {

  }

  mlir::ValueRange SubEWOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void SubEWOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void SubEWOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // SubOp
  //===----------------------------------------------------------------------===//

  void SubOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }

  mlir::LogicalResult SubOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {

  }

  mlir::Value SubOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {

  }

  mlir::Value SubOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {

  }

  mlir::Value SubOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {

  }

  mlir::ValueRange SubOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void SubOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void SubOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // SubscriptionOp
  //===----------------------------------------------------------------------===//

  mlir::Value SubscriptionOp::getViewSource()
  {

  }

  mlir::ValueRange SubscriptionOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void SubscriptionOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void SubscriptionOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // SymmetricOp
  //===----------------------------------------------------------------------===//

  void SymmetricOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }

  //===----------------------------------------------------------------------===//
  // TanOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange TanOp::getArgs()
  {

  }

  unsigned int TanOp::getArgExpectedRank(unsigned int argIndex)
  {

  }

  mlir::ValueRange TanOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {

  }

  mlir::ValueRange TanOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void TanOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void TanOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // TanhOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange TanhOp::getArgs()
  {

  }

  unsigned int TanhOp::getArgExpectedRank(unsigned int argIndex)
  {

  }

  mlir::ValueRange TanhOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {

  }

  mlir::ValueRange TanhOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void TanhOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void TanhOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // WhileOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange WhileOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {

  }

  void WhileOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void WhileOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // ZerosOp
  //===----------------------------------------------------------------------===//

  void ZerosOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {

  }
}
