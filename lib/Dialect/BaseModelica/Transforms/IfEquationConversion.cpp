#include "marco/Dialect/BaseModelica/Transforms/IfEquationConversion.h"
#include "marco/Dialect/BaseModelica/IR/Dialect.h"
#include "marco/Dialect/BaseModelica/IR/Ops.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_IFEQUATIONCONVERSIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {

bool areAccessesEquivalent(mlir::Value a, mlir::Value b) {
  if (a == b)
    return true;

  auto aResult = mlir::dyn_cast<mlir::OpResult>(a);
  auto bResult = mlir::dyn_cast<mlir::OpResult>(b);

  // Block arguments (no defining op) are equivalent only when identical, which
  // is already handled above.
  if (!aResult || !bResult)
    return false;

  if (aResult.getResultNumber() != bResult.getResultNumber())
    return false;

  return mlir::OperationEquivalence::isEquivalentTo(
      aResult.getOwner(), bResult.getOwner(),
      [](mlir::Value a, mlir::Value b) {
        return areAccessesEquivalent(a, b) ? mlir::success() : mlir::failure();
      },
      nullptr, mlir::OperationEquivalence::IgnoreLocations);
}

class IfEquationConversionPass
    : public impl::IfEquationConversionPassBase<IfEquationConversionPass> {
public:
  using IfEquationConversionPassBase<
      IfEquationConversionPass>::IfEquationConversionPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult convertIfEquationOp(mlir::IRRewriter &rewriter,
                                          IfEquationOp ifEqOp);
};

void IfEquationConversionPass::runOnOperation() {
  mlir::IRRewriter rewriter(&getContext());
  llvm::SmallVector<IfEquationOp> worklist;

  getOperation()->walk([&](IfEquationOp ifEqOp) {
    if (mlir::isa<DynamicOp, InitialOp>(ifEqOp->getParentOp()))
      worklist.push_back(ifEqOp);
  });

  for (IfEquationOp ifEqOp : worklist) {
    if (mlir::failed(convertIfEquationOp(rewriter, ifEqOp)))
      return signalPassFailure();
  }
}

mlir::LogicalResult
IfEquationConversionPass::convertIfEquationOp(mlir::IRRewriter &rewriter,
                                              IfEquationOp ifEqOp) {
  // --- Assertion 1: each branch must contain exactly one EquationOp ---
  ::mlir::Block *thenBlock = ifEqOp.thenBlock();
  ::mlir::Block *elseBlock = ifEqOp.elseBlock();

  if (std::distance(thenBlock->begin(), thenBlock->end()) != 1)
    return ifEqOp.emitError()
           << "if_equation then-branch must contain exactly one equation "
              "(else-if chains are not yet supported)";
  if (!mlir::isa<EquationOp>(thenBlock->front()))
    return ifEqOp.emitError()
           << "if_equation then-branch must contain an equation op";

  if (std::distance(elseBlock->begin(), elseBlock->end()) != 1)
    return ifEqOp.emitError()
           << "if_equation else-branch must contain exactly one equation "
              "(else-if chains are not yet supported)";
  if (!mlir::isa<EquationOp>(elseBlock->front()))
    return ifEqOp.emitError()
           << "if_equation else-branch must contain an equation op";

  EquationOp thenEq = mlir::cast<EquationOp>(thenBlock->front());
  EquationOp elseEq = mlir::cast<EquationOp>(elseBlock->front());

  EquationSidesOp thenSides = mlir::cast<EquationSidesOp>(
      thenEq.getBodyRegion().front().getTerminator());
  EquationSidesOp elseSides = mlir::cast<EquationSidesOp>(
      elseEq.getBodyRegion().front().getTerminator());

  if (thenSides.getLhsValues().size() != 1 ||
      elseSides.getLhsValues().size() != 1)
    return ifEqOp.emitError(
        "expected exactly one LHS value in each branch equation");

  // --- Assertion 2: both branches must write to the same left-hand side ---
  if (!areAccessesEquivalent(thenSides.getLhsValues()[0],
                             elseSides.getLhsValues()[0]))
    return ifEqOp.emitError()
           << "all branches of an if_equation must write to the same "
              "left-hand side";

  // --- Transformation ---
  mlir::Location loc = ifEqOp.getLoc();

  mlir::Value thenRhsValue = thenSides.getRhsValues()[0];
  mlir::Value elseRhsValue = elseSides.getRhsValues()[0];
  mlir::Value lhsValue = thenSides.getLhsValues()[0];
  mlir::Operation *condYield = ifEqOp.conditionBlock()->getTerminator();
  mlir::Value condValue = condYield->getOperand(0);

  mlir::Operation *thenLhsSideOp = thenSides.getLhs().getDefiningOp();
  mlir::Operation *thenRhsSideOp = thenSides.getRhs().getDefiningOp();
  rewriter.eraseOp(thenSides);
  rewriter.eraseOp(thenLhsSideOp);
  rewriter.eraseOp(thenRhsSideOp);

  mlir::Operation *elseLhsSideOp = elseSides.getLhs().getDefiningOp();
  mlir::Operation *elseRhsSideOp = elseSides.getRhs().getDefiningOp();
  rewriter.eraseOp(elseSides);
  rewriter.eraseOp(elseLhsSideOp);
  rewriter.eraseOp(elseRhsSideOp);

  rewriter.setInsertionPoint(ifEqOp);
  EquationOp newEq = rewriter.create<EquationOp>(loc);
  mlir::Block *eqBody = rewriter.createBlock(&newEq.getBodyRegion());

  ::mlir::Block *condBlock = ifEqOp.conditionBlock();
  while (&condBlock->front() != condYield)
    condBlock->front().moveBefore(eqBody, eqBody->end());
  rewriter.eraseOp(condYield);

  ::mlir::Block *thenEqBody = &thenEq.getBodyRegion().front();
  while (!thenEqBody->empty())
    thenEqBody->front().moveBefore(eqBody, eqBody->end());

  ::mlir::Block *elseEqBody = &elseEq.getBodyRegion().front();
  while (!elseEqBody->empty())
    elseEqBody->front().moveBefore(eqBody, eqBody->end());

  rewriter.setInsertionPointToEnd(eqBody);
  mlir::Value selectedRhs =
      rewriter
          .create<SelectOp>(loc, condValue, mlir::ValueRange{thenRhsValue},
                            mlir::ValueRange{elseRhsValue})
          .getResult(0);

  mlir::Value lhsSide =
      rewriter.create<EquationSideOp>(loc, mlir::ValueRange{lhsValue});
  mlir::Value rhsSide =
      rewriter.create<EquationSideOp>(loc, mlir::ValueRange{selectedRhs});
  rewriter.create<EquationSidesOp>(loc, lhsSide, rhsSide);

  bool erasedDeadOp = true;
  while (erasedDeadOp) {
    erasedDeadOp = false;
    for (mlir::Operation &op :
         llvm::make_early_inc_range(llvm::reverse(*eqBody))) {
      if (mlir::isOpTriviallyDead(&op)) {
        rewriter.eraseOp(&op);
        erasedDeadOp = true;
      }
    }
  }

  rewriter.eraseOp(ifEqOp);
  return mlir::success();
}

} // namespace

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createIfEquationConversionPass() {
  return std::make_unique<IfEquationConversionPass>();
}
} // namespace mlir::bmodelica
