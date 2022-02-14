#include "marco/dialects/modelica/ModelicaDialect.h"
#include "marco/dialects/modelica/Ops.h"
#include "mlir/IR/OpImplementation.h"

using namespace ::mlir;
using namespace ::mlir::modelica;

static void populateAllocationEffects(
    mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects,
    mlir::Value value,
    bool isManuallyDeallocated = false)
{
  if (auto arrayType = value.getType().dyn_cast<ArrayType>()) {
    auto allocationScope = arrayType.getAllocationScope();
    assert(allocationScope == ArrayAllocationScope::stack || allocationScope == ArrayAllocationScope::heap);

    if (allocationScope == ArrayAllocationScope::stack) {
      // Stack-allocated arrays are automatically deallocated when the
      // surrounding function ends.

      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), value, mlir::SideEffects::AutomaticAllocationScopeResource::get());
    } else if (allocationScope == ArrayAllocationScope::heap) {
      // We need to check if there exists a clone operation with forwarding
      // enabled and whose result is manually deallocated. If that is the
      // case, then also the original buffer must not be deallocated, or a
      // double free would happen.

      bool isForwardedAsManuallyDeallocated = llvm::any_of(value.getUsers(), [](const auto& op) -> bool {
        if (auto cloneOp = mlir::dyn_cast<ArrayCloneOp>(op)) {
          return cloneOp.canSourceBeForwarded() &&
            !mlir::cast<HeapAllocator>(cloneOp.getOperation()).shouldBeDeallocated();
        }

        return false;
      });

      if (!isManuallyDeallocated && !isForwardedAsManuallyDeallocated) {
        // Mark the value as heap-allocated so that the deallocation pass can
        // place the deallocation instruction.

        effects.emplace_back(mlir::MemoryEffects::Allocate::get(), value, mlir::SideEffects::DefaultResource::get());
      } else {
        // If the buffer is marked as manually deallocated, then we need to
        // set the operation to have a generic side effect, or the CSE pass
        // would otherwise consider all the allocations with the same
        // structure as equal, and thus would replace all the subsequent
        // buffers with the first allocated one.

        effects.emplace_back(mlir::MemoryEffects::Write::get(), mlir::SideEffects::DefaultResource::get());
      }
    }
  }
}

static mlir::Value readValue(mlir::OpBuilder& builder, mlir::Value operand)
{
  if (auto arrayType = operand.getType().dyn_cast<ArrayType>(); arrayType && arrayType.getRank() == 0) {
    return builder.create<LoadOp>(operand.getLoc(), operand);
  }

  return operand;
}

static mlir::Type convertToRealType(mlir::Type type)
{
  if (auto arrayType = type.dyn_cast<ArrayType>())
    return arrayType.toElementType(RealType::get(type.getContext()));

  return RealType::get(type.getContext());
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
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int AbsOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange AbsOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<AbsOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  //===----------------------------------------------------------------------===//
  // AcosOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange AcosOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int AcosOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange AcosOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<AcosOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange AcosOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[acos(x)] = -x' / sqrt(1 - x^2)

    mlir::Location loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(operand());
    mlir::Type type = convertToRealType(result().getType());

    mlir::Value one = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 1));
    mlir::Value two = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 2));
    mlir::Value argSquared = builder.create<PowEWOp>(loc, type, operand(), two);
    mlir::Value sub = builder.create<SubEWOp>(loc, type, one, argSquared);
    mlir::Value denominator = builder.create<SqrtOp>(loc, type, sub);
    mlir::Value div = builder.create<DivEWOp>(loc, type, derivedOperand, denominator);
    auto derivedOp = builder.create<NegateOp>(loc, type, div);

    return derivedOp->getResults();
  }

  void AcosOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(operand());
  }

  void AcosOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // AddOp
  //===----------------------------------------------------------------------===//

  void AddOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (lhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (rhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());
    }

    populateAllocationEffects(effects, getResult());

    if (result().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::LogicalResult AddOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (auto size = currentResult.size(); size != 1) {
      return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");
    }

    mlir::Value toNest = currentResult[0];

    if (argumentIndex == 0) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<SubOp>(getLoc(), nestedOperand.getType(), nestedOperand, rhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(lhs());
      erase();

      return mlir::success();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<SubOp>(getLoc(), nestedOperand.getType(), nestedOperand, lhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(rhs());
      erase();

      return mlir::success();
    }

    return emitError("Can't invert the operand #" + std::to_string(argumentIndex) + ". The operation has 2 operands.");
  }

  mlir::Value AddOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeNegateOp(builder, resultType);
      }

      return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = distributeFn(this->rhs());

    return builder.create<AddOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value AddOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeMulOp(builder, resultType, value);
      }

      return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = distributeFn(this->rhs());

    return builder.create<AddOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value AddOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeMulOp(builder, resultType, value);
      }

      return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = distributeFn(this->rhs());

    return builder.create<AddOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::ValueRange AddOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    mlir::Location loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(lhs());
    mlir::Value derivedRhs = derivatives.lookup(rhs());

    auto derivedOp = builder.create<AddOp>(
        loc, convertToRealType(result().getType()), derivedLhs, derivedRhs);

    return derivedOp->getResults();
  }

  void AddOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(lhs());
    toBeDerived.push_back(rhs());
  }

  void AddOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // AddEWOp
  //===----------------------------------------------------------------------===//

  void AddEWOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (lhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (rhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());
    }

    populateAllocationEffects(effects, getResult());

    if (result().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::LogicalResult AddEWOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (auto size = currentResult.size(); size != 1) {
      return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");
    }

    mlir::Value toNest = currentResult[0];

    if (argumentIndex == 0) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<SubEWOp>(getLoc(), nestedOperand.getType(), nestedOperand, rhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(lhs());
      erase();

      return mlir::success();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<SubEWOp>(getLoc(), nestedOperand.getType(), nestedOperand, lhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(rhs());
      erase();

      return mlir::success();
    }

    return emitError("Can't invert the operand #" + std::to_string(argumentIndex) + ". The operation has 2 operands.");
  }

  mlir::Value AddEWOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(child.getDefiningOp()))
        return casted.distributeNegateOp(builder, resultType);

      return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = distributeFn(this->rhs());

    return builder.create<AddEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value AddEWOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp()))
        return casted.distributeMulOp(builder, resultType, value);

      return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = distributeFn(this->rhs());

    return builder.create<AddEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value AddEWOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeMulOp(builder, resultType, value);
      }

      return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = distributeFn(this->rhs());

    return builder.create<AddEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::ValueRange AddEWOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    mlir::Location loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(lhs());
    mlir::Value derivedRhs = derivatives.lookup(rhs());

    auto derivedOp = builder.create<AddEWOp>(
        loc, convertToRealType(result().getType()), derivedLhs, derivedRhs);

    return derivedOp->getResults();
  }

  void AddEWOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(lhs());
    toBeDerived.push_back(rhs());
  }

  void AddEWOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // AndOp
  //===----------------------------------------------------------------------===//

  void AndOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (lhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (rhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());
    }

    populateAllocationEffects(effects, getResult());

    if (result().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  //===----------------------------------------------------------------------===//
  // ArrayCastOp
  //===----------------------------------------------------------------------===//

  mlir::Value ArrayCastOp::getViewSource()
  {
    return array();
  }

  //===----------------------------------------------------------------------===//
  // ArrayCloneOp
  //===----------------------------------------------------------------------===//

  bool ArrayCloneOp::canSourceBeForwarded() const
  {
    return false;
  }

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
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int AsinOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange AsinOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<AsinOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange AsinOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[arcsin(x)] = x' / sqrt(1 - x^2)

    mlir::Location loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(operand());
    mlir::Type type = convertToRealType(result().getType());

    mlir::Value one = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 1));
    mlir::Value two = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 2));
    mlir::Value argSquared = builder.create<PowEWOp>(loc, type, operand(), two);
    mlir::Value sub = builder.create<SubEWOp>(loc, type, one, argSquared);
    mlir::Value denominator = builder.create<SqrtOp>(loc, type, sub);
    auto derivedOp = builder.create<DivEWOp>(loc, type, derivedOperand, denominator);

    return derivedOp->getResults();
  }

  void AsinOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(operand());
  }

  void AsinOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // AtanOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange AtanOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int AtanOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange AtanOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<AtanOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange AtanOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[atan(x)] = x' / (1 + x^2)

    mlir::Location loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(operand());
    mlir::Type type = convertToRealType(result().getType());

    mlir::Value one = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 1));
    mlir::Value two = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 2));
    mlir::Value argSquared = builder.create<PowEWOp>(loc, type, operand(), two);
    mlir::Value denominator = builder.create<AddEWOp>(loc, type, one, argSquared);
    auto derivedOp = builder.create<DivEWOp>(loc, type, derivedOperand, denominator);

    return derivedOp->getResults();
  }

  void AtanOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(operand());
  }

  void AtanOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // Atan2Op
  //===----------------------------------------------------------------------===//

  mlir::ValueRange Atan2Op::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int Atan2Op::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange Atan2Op::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newY = builder.create<SubscriptionOp>(getLoc(), y(), indexes);

    if (auto arrayType = newY.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newY = builder.create<LoadOp>(getLoc(), newY);
    }

    mlir::Value newX = builder.create<SubscriptionOp>(getLoc(), x(), indexes);

    if (auto arrayType = newX.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newX = builder.create<LoadOp>(getLoc(), newX);
    }

    auto op = builder.create<Atan2Op>(getLoc(), newResultType, newY, newX);
    return op->getResults();
  }

  //===----------------------------------------------------------------------===//
  // CosOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange CosOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int CosOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange CosOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<CosOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange CosOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[cos(x)] = -x' * sin(x)

    mlir::Location loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(operand());
    mlir::Type type = convertToRealType(result().getType());
    bool elementWise = derivedOperand.getType().isa<ArrayType>();

    mlir::Value sin = builder.create<SinOp>(loc, type, operand());
    mlir::Value negatedSin = builder.create<NegateOp>(loc, type, sin);
    auto derivedOp = builder.create<MulEWOp>(loc, type, negatedSin, derivedOperand);

    return derivedOp->getResults();
  }

  void CosOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(operand());
  }

  void CosOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // CoshOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange CoshOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int CoshOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange CoshOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<CoshOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange CoshOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[cosh(x)] = x' * sinh(x)

    mlir::Location loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(operand());
    mlir::Type type = convertToRealType(result().getType());

    mlir::Value sinh = builder.create<SinhOp>(loc, type, operand());
    auto derivedOp = builder.create<MulEWOp>(loc, type, sinh, derivedOperand);

    return derivedOp->getResults();
  }

  void CoshOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(operand());
  }

  void CoshOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // DiagonalOp
  //===----------------------------------------------------------------------===//

  void DiagonalOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), values(), mlir::SideEffects::DefaultResource::get());
    populateAllocationEffects(effects, getResult());
    effects.emplace_back(mlir::MemoryEffects::Write::get(), result(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // DivOp
  //===----------------------------------------------------------------------===//

  void DivOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (lhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (rhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());
    }

    populateAllocationEffects(effects, getResult());

    if (result().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::LogicalResult DivOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (auto size = currentResult.size(); size != 1) {
      return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");
    }

    mlir::Value toNest = currentResult[0];

    if (argumentIndex == 0) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<MulOp>(getLoc(), nestedOperand.getType(), nestedOperand, rhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      getResult().replaceAllUsesWith(lhs());
      erase();

      return mlir::success();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<DivOp>(getLoc(), nestedOperand.getType(), lhs(), nestedOperand);

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      getResult().replaceAllUsesWith(rhs());
      erase();

      return mlir::success();
    }

    return emitError("Index out of bounds: " + std::to_string(argumentIndex));
  }

  mlir::Value DivOp::distribute(mlir::OpBuilder& builder)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(*this);

    if (!mlir::isa<DivOpDistributionInterface>(lhs().getDefiningOp()) &&
        !mlir::isa<DivOpDistributionInterface>(rhs().getDefiningOp())) {
      // The operation can't be propagated because none of the children
      // know how to distribute the multiplication to their children.
      return getResult();
    }

    DivOpDistributionInterface childOp = mlir::isa<DivOpDistributionInterface>(lhs().getDefiningOp()) ?
                                         mlir::cast<DivOpDistributionInterface>(lhs().getDefiningOp()) :
                                         mlir::cast<DivOpDistributionInterface>(rhs().getDefiningOp());

    mlir::Value toDistribute = mlir::isa<DivOpDistributionInterface>(lhs().getDefiningOp()) ? rhs() : lhs();

    return childOp.distributeDivOp(builder, result().getType(), toDistribute);
  }

  mlir::Value DivOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeNegateOp(builder, resultType);
      }

      return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = this->rhs();

    return builder.create<MulOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value DivOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeMulOp(builder, resultType, value);
      }

      return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = this->rhs();

    return builder.create<DivOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value DivOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeMulOp(builder, resultType, value);
      }

      return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = this->rhs();

    return builder.create<DivOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::ValueRange DivOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    mlir::Location loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(lhs());
    mlir::Value derivedRhs = derivatives.lookup(rhs());

    mlir::Type type = convertToRealType(result().getType());

    mlir::Value firstMul = builder.create<MulOp>(loc, type, derivedLhs, rhs());
    mlir::Value secondMul = builder.create<MulOp>(loc, type, lhs(), derivedRhs);
    mlir::Value numerator = builder.create<SubOp>(loc, type, firstMul, secondMul);
    mlir::Value two = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 2));
    mlir::Value denominator = builder.create<PowOp>(loc, convertToRealType(rhs().getType()), rhs(), two);
    auto derivedOp = builder.create<DivOp>(loc, type, numerator, denominator);

    return derivedOp->getResults();
  }

  void DivOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(lhs());
    toBeDerived.push_back(rhs());
  }

  void DivOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // DivEWOp
  //===----------------------------------------------------------------------===//

  void DivEWOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (lhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (rhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());
    }

    populateAllocationEffects(effects, getResult());

    if (result().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::LogicalResult DivEWOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (auto size = currentResult.size(); size != 1) {
      return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");
    }

    mlir::Value toNest = currentResult[0];

    if (argumentIndex == 0) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<MulEWOp>(getLoc(), nestedOperand.getType(), nestedOperand, rhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      getResult().replaceAllUsesWith(lhs());
      erase();

      return mlir::success();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<DivEWOp>(getLoc(), nestedOperand.getType(), lhs(), nestedOperand);

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      getResult().replaceAllUsesWith(rhs());
      erase();

      return mlir::success();
    }

    return emitError("Index out of bounds: " + std::to_string(argumentIndex));
  }

  mlir::Value DivEWOp::distribute(mlir::OpBuilder& builder)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(*this);

    if (!mlir::isa<DivOpDistributionInterface>(lhs().getDefiningOp()) &&
        !mlir::isa<DivOpDistributionInterface>(rhs().getDefiningOp())) {
      // The operation can't be propagated because none of the children
      // know how to distribute the multiplication to their children.
      return getResult();
    }

    DivOpDistributionInterface childOp = mlir::isa<DivOpDistributionInterface>(lhs().getDefiningOp()) ?
                                         mlir::cast<DivOpDistributionInterface>(lhs().getDefiningOp()) :
                                         mlir::cast<DivOpDistributionInterface>(rhs().getDefiningOp());

    mlir::Value toDistribute = mlir::isa<DivOpDistributionInterface>(lhs().getDefiningOp()) ? rhs() : lhs();

    return childOp.distributeDivOp(builder, result().getType(), toDistribute);
  }

  mlir::Value DivEWOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeNegateOp(builder, resultType);
      }

      return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = this->rhs();

    return builder.create<MulEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value DivEWOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeMulOp(builder, resultType, value);
      }

      return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = this->rhs();

    return builder.create<DivEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value DivEWOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeMulOp(builder, resultType, value);
      }

      return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = this->rhs();

    return builder.create<DivEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::ValueRange DivEWOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    mlir::Location loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(lhs());
    mlir::Value derivedRhs = derivatives.lookup(rhs());

    mlir::Type type = convertToRealType(result().getType());

    mlir::Value firstMul = builder.create<MulEWOp>(loc, type, derivedLhs, rhs());
    mlir::Value secondMul = builder.create<MulEWOp>(loc, type, lhs(), derivedRhs);
    mlir::Value numerator = builder.create<SubEWOp>(loc, type, firstMul, secondMul);
    mlir::Value two = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 2));
    mlir::Value denominator = builder.create<PowEWOp>(loc, convertToRealType(rhs().getType()), rhs(), two);
    auto derivedOp = builder.create<DivEWOp>(loc, type, numerator, denominator);

    return derivedOp->getResults();
  }

  void DivEWOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(lhs());
    toBeDerived.push_back(rhs());
  }

  void DivEWOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // ExpOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange ExpOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int ExpOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange ExpOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), exponent(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<ExpOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange ExpOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[e^x] = x' * e^x

    mlir::Location loc = getLoc();
    mlir::Value derivedExponent = derivatives.lookup(exponent());
    mlir::Type type = convertToRealType(result().getType());

    mlir::Value pow = builder.create<ExpOp>(loc, type, exponent());
    auto derivedOp = builder.create<MulEWOp>(loc, type, pow, derivedExponent);

    return derivedOp->getResults();
  }

  void ExpOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(exponent());
  }

  void ExpOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // ForOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange ForOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    return llvm::None;
  }

  void ForOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void ForOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    regions.push_back(&bodyRegion());
  }

  //===----------------------------------------------------------------------===//
  // FreeOp
  //===----------------------------------------------------------------------===//

  void FreeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Free::get(), array(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // IdentityOp
  //===----------------------------------------------------------------------===//

  void IdentityOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    populateAllocationEffects(effects, result());
    effects.emplace_back(mlir::MemoryEffects::Write::get(), result(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // IfOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange IfOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    return llvm::None;
  }

  void IfOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void IfOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    regions.push_back(&thenRegion());
    regions.push_back(&elseRegion());
  }

  //===----------------------------------------------------------------------===//
  // LinspaceOp
  //===----------------------------------------------------------------------===//

  void LinspaceOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    populateAllocationEffects(effects, getResult());
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // LoadOp
  //===----------------------------------------------------------------------===//

  void LoadOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), array(), mlir::SideEffects::DefaultResource::get());
  }

  mlir::ValueRange LoadOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto derivedOp = builder.create<LoadOp>(getLoc(), derivatives.lookup(array()), indexes());
    return derivedOp->getResults();
  }

  void LoadOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(array());
  }

  void LoadOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // LogOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange LogOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int LogOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange LogOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<LogOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange LogOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[ln(x)] = x' / x

    mlir::Value derivedOperand = derivatives.lookup(operand());

    auto derivedOp = builder.create<DivEWOp>(
        getLoc(), convertToRealType(result().getType()), derivedOperand, operand());

    return derivedOp->getResults();
  }

  void LogOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(operand());
  }

  void LogOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // Log10Op
  //===----------------------------------------------------------------------===//

  mlir::ValueRange Log10Op::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int Log10Op::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange Log10Op::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<Log10Op>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange Log10Op::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[log10(x)] = x' / (x * ln(10))

    mlir::Location loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(operand());
    mlir::Type type = convertToRealType(result().getType());

    mlir::Value ten = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 10));
    mlir::Value log = builder.create<LogOp>(loc, RealType::get(getContext()), ten);
    mlir::Value mul = builder.create<MulEWOp>(loc, type, operand(), log);
    auto derivedOp = builder.create<DivEWOp>(loc, result().getType(), derivedOperand, mul);

    return derivedOp->getResults();
  }

  void Log10Op::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(operand());
  }

  void Log10Op::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // MulOp
  //===----------------------------------------------------------------------===//

  void MulOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (lhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (rhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());
    }

    populateAllocationEffects(effects, getResult());

    if (result().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Write::get(), result(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::LogicalResult MulOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (auto size = currentResult.size(); size != 1) {
      return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");
    }

    mlir::Value toNest = currentResult[0];

    if (argumentIndex == 0) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<DivOp>(getLoc(), nestedOperand.getType(), nestedOperand, rhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
          use.set(right.getResult());
      }

      replaceAllUsesWith(lhs());
      erase();

      return mlir::success();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<DivOp>(getLoc(), nestedOperand.getType(), nestedOperand, lhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right))
          use.set(right.getResult());
      }

      getResult().replaceAllUsesWith(rhs());
      erase();

      return mlir::success();
    }

    return emitError("Index out of bounds: " + std::to_string(argumentIndex));
  }

  mlir::Value MulOp::distribute(mlir::OpBuilder& builder)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(*this);

    if (!mlir::isa<MulOpDistributionInterface>(lhs().getDefiningOp()) &&
        !mlir::isa<MulOpDistributionInterface>(rhs().getDefiningOp())) {
      // The operation can't be propagated because none of the children
      // know how to distribute the multiplication to their children.
      return getResult();
    }

    MulOpDistributionInterface childOp = mlir::isa<MulOpDistributionInterface>(lhs().getDefiningOp()) ?
                                         mlir::cast<MulOpDistributionInterface>(lhs().getDefiningOp()) :
                                         mlir::cast<MulOpDistributionInterface>(rhs().getDefiningOp());

    mlir::Value toDistribute = mlir::isa<MulOpDistributionInterface>(lhs().getDefiningOp()) ? rhs() : lhs();

    return childOp.distributeMulOp(builder, result().getType(), toDistribute);
  }

  mlir::Value MulOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeNegateOp(builder, resultType);
      }

      return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = this->rhs();

    return builder.create<MulOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value MulOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeMulOp(builder, resultType, value);
      }

      return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = this->rhs();

    return builder.create<MulOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value MulOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeMulOp(builder, resultType, value);
      }

      return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = this->rhs();

    return builder.create<MulOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::ValueRange MulOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    mlir::Location loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(lhs());
    mlir::Value derivedRhs = derivatives.lookup(rhs());

    mlir::Type type = convertToRealType(result().getType());

    mlir::Value firstMul = builder.create<MulOp>(loc, type, derivedLhs, rhs());
    mlir::Value secondMul = builder.create<MulOp>(loc, type, lhs(), derivedRhs);
    auto derivedOp = builder.create<AddOp>(loc, type, firstMul, secondMul);

    return derivedOp->getResults();
  }

  void MulOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(lhs());
    toBeDerived.push_back(rhs());
  }

  void MulOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // MulEWOp
  //===----------------------------------------------------------------------===//

  void MulEWOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (lhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (rhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());
    }

    populateAllocationEffects(effects, getResult());

    if (result().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::LogicalResult MulEWOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (auto size = currentResult.size(); size != 1) {
      return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");
    }

    mlir::Value toNest = currentResult[0];

    if (argumentIndex == 0) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<DivEWOp>(getLoc(), nestedOperand.getType(), nestedOperand, rhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(lhs());
      erase();

      return mlir::success();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<DivEWOp>(getLoc(), nestedOperand.getType(), nestedOperand, lhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      getResult().replaceAllUsesWith(rhs());
      erase();

      return mlir::success();
    }

    return emitError("Index out of bounds: " + std::to_string(argumentIndex));
  }

  mlir::Value MulEWOp::distribute(mlir::OpBuilder& builder)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(*this);

    if (!mlir::isa<MulOpDistributionInterface>(lhs().getDefiningOp()) &&
        !mlir::isa<MulOpDistributionInterface>(rhs().getDefiningOp())) {
      // The operation can't be propagated because none of the children
      // know how to distribute the multiplication to their children.
      return getResult();
    }

    MulOpDistributionInterface childOp = mlir::isa<MulOpDistributionInterface>(lhs().getDefiningOp()) ?
                                         mlir::cast<MulOpDistributionInterface>(lhs().getDefiningOp()) :
                                         mlir::cast<MulOpDistributionInterface>(rhs().getDefiningOp());

    mlir::Value toDistribute = mlir::isa<MulOpDistributionInterface>(lhs().getDefiningOp()) ? rhs() : lhs();

    return childOp.distributeMulOp(builder, result().getType(), toDistribute);
  }

  mlir::Value MulEWOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeNegateOp(builder, resultType);
      }

      return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = this->rhs();

    return builder.create<MulEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value MulEWOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeMulOp(builder, resultType, value);
      }

      return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = this->rhs();

    return builder.create<MulEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value MulEWOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeMulOp(builder, resultType, value);
      }

      return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = this->rhs();

    return builder.create<MulEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::ValueRange MulEWOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    mlir::Location loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(lhs());
    mlir::Value derivedRhs = derivatives.lookup(rhs());

    mlir::Type type = convertToRealType(result().getType());

    mlir::Value firstMul = builder.create<MulEWOp>(loc, type, derivedLhs, rhs());
    mlir::Value secondMul = builder.create<MulEWOp>(loc, type, lhs(), derivedRhs);
    auto derivedOp = builder.create<AddEWOp>(loc, type, firstMul, secondMul);

    return derivedOp->getResults();
  }

  void MulEWOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(lhs());
    toBeDerived.push_back(rhs());
  }

  void MulEWOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // NegateOp
  //===----------------------------------------------------------------------===//

  void NegateOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (operand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), operand(), mlir::SideEffects::DefaultResource::get());
    }

    populateAllocationEffects(effects, getResult());

    if (result().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::LogicalResult NegateOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (argumentIndex > 0) {
      return emitError("Index out of bounds: " + std::to_string(argumentIndex));
    }

    if (auto size = currentResult.size(); size != 1) {
      return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");
    }

    mlir::Value toNest = currentResult[0];

    mlir::Value nestedOperand = readValue(builder, toNest);
    auto right = builder.create<NegateOp>(getLoc(), nestedOperand.getType(), nestedOperand);

    for (auto& use : toNest.getUses()) {
      if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
        use.set(right.getResult());
      }
    }

    replaceAllUsesWith(operand());
    erase();

    return mlir::success();
  }

  mlir::Value NegateOp::distribute(mlir::OpBuilder& builder)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(*this);

    if (auto childOp = mlir::dyn_cast<NegateOpDistributionInterface>(operand().getDefiningOp())) {
      return childOp.distributeNegateOp(builder, result().getType());
    }

    // The operation can't be propagated because the child doesn't
    // know how to distribute the multiplication to its children.
    return getResult();
  }

  mlir::Value NegateOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeNegateOp(builder, resultType);
      }

      return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
    };

    mlir::Value operand = distributeFn(this->operand());

    return builder.create<NegateOp>(getLoc(), resultType, operand);
  }

  mlir::Value NegateOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeMulOp(builder, resultType, value);
      }

      return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value operand = distributeFn(this->operand());

    return builder.create<NegateOp>(getLoc(), resultType, operand);
  }

  mlir::Value NegateOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeMulOp(builder, resultType, value);
      }

      return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value operand = distributeFn(this->operand());

    return builder.create<NegateOp>(getLoc(), resultType, operand);
  }

  mlir::ValueRange NegateOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    mlir::Value derivedOperand = derivatives.lookup(operand());
    auto derivedOp = builder.create<NegateOp>(getLoc(), convertToRealType(result().getType()), derivedOperand);
    return derivedOp->getResults();
  }

  void NegateOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(operand());
  }

  void NegateOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // NotOp
  //===----------------------------------------------------------------------===//

  void NotOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (operand().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), operand(), mlir::SideEffects::DefaultResource::get());
    }

    populateAllocationEffects(effects, getResult());

    if (result().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Write::get(), result(), mlir::SideEffects::DefaultResource::get());
    }
  }

  //===----------------------------------------------------------------------===//
  // OnesOp
  //===----------------------------------------------------------------------===//

  void OnesOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    populateAllocationEffects(effects, getResult());
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // OrOp
  //===----------------------------------------------------------------------===//

  void OrOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (lhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (rhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());
    }

    populateAllocationEffects(effects, getResult());

    if (result().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  //===----------------------------------------------------------------------===//
  // PowOp
  //===----------------------------------------------------------------------===//

  void PowOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (base().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), base(), mlir::SideEffects::DefaultResource::get());
    }

    if (exponent().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), exponent(), mlir::SideEffects::DefaultResource::get());
    }

    populateAllocationEffects(effects, getResult());

    if (result().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange PowOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[x ^ y] = (x ^ y) * (y' * ln(x) + (y * x') / x)

    mlir::Location loc = getLoc();

    mlir::Value derivedBase = derivatives.lookup(base());
    mlir::Value derivedExponent = derivatives.lookup(exponent());

    mlir::Type type = convertToRealType(result().getType());

    mlir::Value pow = builder.create<PowOp>(loc, type, base(), exponent());
    mlir::Value ln = builder.create<LogOp>(loc, type, base());
    mlir::Value firstOperand = builder.create<MulOp>(loc, type, derivedExponent, ln);
    mlir::Value numerator = builder.create<MulOp>(loc, type, exponent(), derivedBase);
    mlir::Value secondOperand = builder.create<DivOp>(loc, type, numerator, base());
    mlir::Value sum = builder.create<AddOp>(loc, type, firstOperand, secondOperand);
    auto derivedOp = builder.create<MulOp>(loc, type, pow, sum);

    return derivedOp->getResults();
  }

  void PowOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(base());
    toBeDerived.push_back(exponent());
  }

  void PowOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // PowEWOp
  //===----------------------------------------------------------------------===//

  void PowEWOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (base().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), base(), mlir::SideEffects::DefaultResource::get());
    }

    if (exponent().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), exponent(), mlir::SideEffects::DefaultResource::get());
    }

    populateAllocationEffects(effects, getResult());

    if (result().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::ValueRange PowEWOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[x ^ y] = (x ^ y) * (y' * ln(x) + (y * x') / x)

    mlir::Location loc = getLoc();

    mlir::Value derivedBase = derivatives.lookup(base());
    mlir::Value derivedExponent = derivatives.lookup(exponent());

    mlir::Type type = convertToRealType(result().getType());

    mlir::Value pow = builder.create<PowEWOp>(loc, type, base(), exponent());
    mlir::Value ln = builder.create<LogOp>(loc, type, base());
    mlir::Value firstOperand = builder.create<MulEWOp>(loc, type, derivedExponent, ln);
    mlir::Value numerator = builder.create<MulEWOp>(loc, type, exponent(), derivedBase);
    mlir::Value secondOperand = builder.create<DivEWOp>(loc, type, numerator, base());
    mlir::Value sum = builder.create<AddEWOp>(loc, type, firstOperand, secondOperand);
    auto derivedOp = builder.create<MulEWOp>(loc, type, pow, sum);

    return derivedOp->getResults();
  }

  void PowEWOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(base());
    toBeDerived.push_back(exponent());
  }

  void PowEWOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // SignOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange SignOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int SignOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange SignOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0)
      newResultType = arrayType.getElementType();

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);

    auto op = builder.create<SignOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  //===----------------------------------------------------------------------===//
  // SinOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange SinOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int SinOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange SinOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0)
      newResultType = arrayType.getElementType();

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0)
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);

    auto op = builder.create<SinOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange SinOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[sin(x)] = x' * cos(x)

    mlir::Location loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(operand());
    mlir::Type type = convertToRealType(result().getType());

    mlir::Value cos = builder.create<CosOp>(loc, type, operand());
    auto derivedOp = builder.create<MulEWOp>(loc, type, cos, derivedOperand);

    return derivedOp->getResults();
  }

  void SinOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(operand());
  }

  void SinOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // SinhOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange SinhOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int SinhOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange SinhOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<SinhOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange SinhOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[sinh(x)] = x' * cosh(x)

    mlir::Location loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(operand());
    mlir::Type type = convertToRealType(result().getType());

    mlir::Value cosh = builder.create<CoshOp>(loc, type, operand());
    auto derivedOp = builder.create<MulEWOp>(loc, type, cosh, derivedOperand);

    return derivedOp->getResults();
  }

  void SinhOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(operand());
  }

  void SinhOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // SizeOp
  //===----------------------------------------------------------------------===//

  void SizeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Read::get(), array(), mlir::SideEffects::DefaultResource::get());
    populateAllocationEffects(effects, result());

    if (result().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Write::get(), result(), mlir::SideEffects::DefaultResource::get());
    }
  }

  //===----------------------------------------------------------------------===//
  // SqrtOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange SqrtOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int SqrtOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange SqrtOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<SqrtOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  //===----------------------------------------------------------------------===//
  // StoreOp
  //===----------------------------------------------------------------------===//

  void StoreOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    effects.emplace_back(mlir::MemoryEffects::Write::get(), array(), mlir::SideEffects::DefaultResource::get());
  }

  mlir::ValueRange StoreOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    auto derivedOp = builder.create<StoreOp>(
        getLoc(), derivatives.lookup(value()), derivatives.lookup(array()), indexes());

    return derivedOp->getResults();
  }

  void StoreOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(array());
    toBeDerived.push_back(value());
  }

  void StoreOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // SubOp
  //===----------------------------------------------------------------------===//

  void SubOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (lhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (rhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());
    }

    populateAllocationEffects(effects, getResult());

    if (result().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::LogicalResult SubOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (auto size = currentResult.size(); size != 1) {
      return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");
    }

    mlir::Value toNest = currentResult[0];

    if (argumentIndex == 0) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<AddOp>(getLoc(), nestedOperand.getType(), nestedOperand, rhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(lhs());
      erase();

      return mlir::success();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<SubOp>(getLoc(), nestedOperand.getType(), lhs(), nestedOperand);

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(rhs());
      erase();

      return mlir::success();
    }

    return emitError("Can't invert the operand #" + std::to_string(argumentIndex) + ". The operation has 2 operands.");
  }

  mlir::Value SubOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeNegateOp(builder, resultType);
      }

      return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = distributeFn(this->rhs());

    return builder.create<AddOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value SubOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeMulOp(builder, resultType, value);
      }

      return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = distributeFn(this->rhs());

    return builder.create<SubOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value SubOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeMulOp(builder, resultType, value);
      }

      return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = distributeFn(this->rhs());

    return builder.create<SubOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::ValueRange SubOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    mlir::Location loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(lhs());
    mlir::Value derivedRhs = derivatives.lookup(rhs());

    auto derivedOp = builder.create<SubOp>(
        loc, convertToRealType(result().getType()), derivedLhs, derivedRhs);

    return derivedOp->getResults();
  }

  void SubOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(lhs());
    toBeDerived.push_back(rhs());
  }

  void SubOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // SubEWOp
  //===----------------------------------------------------------------------===//

  void SubEWOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    if (lhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), lhs(), mlir::SideEffects::DefaultResource::get());
    }

    if (rhs().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Read::get(), rhs(), mlir::SideEffects::DefaultResource::get());
    }

    populateAllocationEffects(effects, getResult());

    if (result().getType().isa<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
    }
  }

  mlir::LogicalResult SubEWOp::invert(mlir::OpBuilder& builder, unsigned int argumentIndex, mlir::ValueRange currentResult)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (auto size = currentResult.size(); size != 1) {
      return emitError("Invalid amount of values to be nested: " + std::to_string(size) + " (expected 1)");
    }

    mlir::Value toNest = currentResult[0];

    if (argumentIndex == 0) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<AddEWOp>(getLoc(), nestedOperand.getType(), nestedOperand, rhs());

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(lhs());
      erase();

      return mlir::success();
    }

    if (argumentIndex == 1) {
      mlir::Value nestedOperand = readValue(builder, toNest);
      auto right = builder.create<SubEWOp>(getLoc(), nestedOperand.getType(), lhs(), nestedOperand);

      for (auto& use : toNest.getUses()) {
        if (auto* owner = use.getOwner(); owner != right && !owner->isBeforeInBlock(right)) {
          use.set(right.getResult());
        }
      }

      replaceAllUsesWith(rhs());
      erase();

      return mlir::success();
    }

    return emitError("Can't invert the operand #" + std::to_string(argumentIndex) + ". The operation has 2 operands.");
  }

  mlir::Value SubEWOp::distributeNegateOp(mlir::OpBuilder& builder, mlir::Type resultType)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<NegateOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeNegateOp(builder, resultType);
      }

      return builder.create<NegateOp>(child.getLoc(), child.getType(), child);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = distributeFn(this->rhs());

    return builder.create<AddEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value SubEWOp::distributeMulOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeMulOp(builder, resultType, value);
      }

      return builder.create<MulOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = distributeFn(this->rhs());

    return builder.create<SubEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::Value SubEWOp::distributeDivOp(mlir::OpBuilder& builder, mlir::Type resultType, mlir::Value value)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto distributeFn = [&](mlir::Value child) -> mlir::Value {
      if (auto casted = mlir::dyn_cast<MulOpDistributionInterface>(child.getDefiningOp())) {
        return casted.distributeMulOp(builder, resultType, value);
      }

      return builder.create<DivOp>(child.getLoc(), child.getType(), child, value);
    };

    mlir::Value lhs = distributeFn(this->lhs());
    mlir::Value rhs = distributeFn(this->rhs());

    return builder.create<SubEWOp>(getLoc(), resultType, lhs, rhs);
  }

  mlir::ValueRange SubEWOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    mlir::Location loc = getLoc();

    mlir::Value derivedLhs = derivatives.lookup(lhs());
    mlir::Value derivedRhs = derivatives.lookup(rhs());

    auto derivedOp = builder.create<SubEWOp>(
        loc, convertToRealType(result().getType()), derivedLhs, derivedRhs);

    return derivedOp->getResults();
  }

  void SubEWOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(lhs());
    toBeDerived.push_back(rhs());
  }

  void SubEWOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
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
    effects.emplace_back(mlir::MemoryEffects::Read::get(), matrix(), mlir::SideEffects::DefaultResource::get());
    populateAllocationEffects(effects, getResult());
    assert(getResult().getType().isa<ArrayType>());
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
  }

  //===----------------------------------------------------------------------===//
  // TanOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange TanOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int TanOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange TanOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<TanOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange TanOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[tan(x)] = x' / (cos(x))^2

    mlir::Location loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(operand());
    mlir::Type type = convertToRealType(result().getType());
    bool elementWise = derivedOperand.getType().isa<ArrayType>();

    mlir::Value cos = builder.create<CosOp>(loc, type, operand());
    mlir::Value two = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 2));
    mlir::Value denominator = builder.create<PowEWOp>(loc, type, cos, two);
    auto derivedOp = builder.create<DivEWOp>(loc, type, derivedOperand, denominator);

    return derivedOp->getResults();
  }

  void TanOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(operand());
  }

  void TanOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // TanhOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange TanhOp::getArgs()
  {
    return mlir::ValueRange(getOperation()->getOperands());
  }

  unsigned int TanhOp::getArgExpectedRank(unsigned int argIndex)
  {
    return 0;
  }

  mlir::ValueRange TanhOp::scalarize(mlir::OpBuilder& builder, mlir::ValueRange indexes)
  {
    mlir::Type newResultType = result().getType().cast<ArrayType>().slice(indexes.size());

    if (auto arrayType = newResultType.dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newResultType = arrayType.getElementType();
    }

    mlir::Value newOperand = builder.create<SubscriptionOp>(getLoc(), operand(), indexes);

    if (auto arrayType = newOperand.getType().dyn_cast<ArrayType>(); arrayType.getRank() == 0) {
      newOperand = builder.create<LoadOp>(getLoc(), newOperand);
    }

    auto op = builder.create<TanhOp>(getLoc(), newResultType, newOperand);
    return op->getResults();
  }

  mlir::ValueRange TanhOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    // D[tanh(x)] = x' / (cosh(x))^2

    mlir::Location loc = getLoc();
    mlir::Value derivedOperand = derivatives.lookup(operand());
    mlir::Type type = convertToRealType(result().getType());

    mlir::Value cosh = builder.create<CoshOp>(loc, type, operand());
    mlir::Value two = builder.create<ConstantOp>(loc, RealAttr::get(getContext(), 2));
    mlir::Value pow = builder.create<PowEWOp>(loc, type, cosh, two);
    auto derivedOp = builder.create<DivEWOp>(loc, type, derivedOperand, pow);

    return derivedOp->getResults();
  }

  void TanhOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {
    toBeDerived.push_back(operand());
  }

  void TanhOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {

  }

  //===----------------------------------------------------------------------===//
  // WhileOp
  //===----------------------------------------------------------------------===//

  mlir::ValueRange WhileOp::derive(mlir::OpBuilder& builder, mlir::BlockAndValueMapping& derivatives)
  {
    return llvm::None;
  }

  void WhileOp::getOperandsToBeDerived(llvm::SmallVectorImpl<mlir::Value>& toBeDerived)
  {

  }

  void WhileOp::getDerivableRegions(llvm::SmallVectorImpl<mlir::Region*>& regions)
  {
    regions.push_back(&bodyRegion());
  }

  //===----------------------------------------------------------------------===//
  // ZerosOp
  //===----------------------------------------------------------------------===//

  void ZerosOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>& effects)
  {
    populateAllocationEffects(effects, getResult());
    effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(), mlir::SideEffects::DefaultResource::get());
  }
}
