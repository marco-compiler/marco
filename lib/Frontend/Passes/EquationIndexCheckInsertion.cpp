#include "marco/Frontend/Passes/EquationIndexCheckInsertion.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IntegerSet.h"

namespace marco::frontend {
#define GEN_PASS_DEF_EQUATIONINDEXCHECKINSERTIONPASS
#include "marco/Frontend/Passes.h.inc"
} // namespace marco::frontend

namespace {
class EquationIndexCheckInsertionPass
    : public marco::frontend::impl::EquationIndexCheckInsertionPassBase<
          EquationIndexCheckInsertionPass> {
public:
  using EquationIndexCheckInsertionPassBase::
      EquationIndexCheckInsertionPassBase;

  void runOnOperation() override;
};

mlir::IntegerSet getInBoundsCondition(mlir::MLIRContext *context,
                                      unsigned int rank) {
  unsigned int numSymbols = rank * 2;
  llvm::SmallVector<mlir::AffineExpr> expressions;
  llvm::SmallVector<bool> equalityFlags;

  for (unsigned i = 0; i < rank; ++i) {
    mlir::AffineExpr dim = mlir::getAffineDimExpr(i, context);
    mlir::AffineExpr lb = mlir::getAffineSymbolExpr(i * 2, context);
    mlir::AffineExpr ub = mlir::getAffineSymbolExpr(i * 2 + 1, context);
    expressions.push_back(dim - lb);     // d_i >= lb_i
    expressions.push_back(ub + 1 - dim); // d_i <= ub_i + 1
    equalityFlags.push_back(false);
    equalityFlags.push_back(false);
  }

  return mlir::IntegerSet::get(rank, numSymbols, expressions, equalityFlags);
} // namespace

void EquationIndexCheckInsertionPass::runOnOperation() {
  if (!getOperation()->hasAttr(
          mlir::bmodelica::BaseModelicaDialect::kEquationFunctionAttrName)) {
    // Not an equation function.
    return;
  }

  // Collect the lower and upper bounds.
  mlir::func::FuncOp funcOp = getOperation();
  unsigned int numOfArgs = funcOp.getNumArguments();
  unsigned int rank = numOfArgs / 2;

  assert(numOfArgs % 2 == 0 &&
         "Equation function must have even number of arguments.");

  llvm::SmallVector<mlir::Value, 32> lowerBounds;
  llvm::SmallVector<mlir::Value, 32> upperBounds;

  for (unsigned int dim = 0; dim < rank; ++dim) {
    lowerBounds.push_back(funcOp.getArgument(dim * 2));
    upperBounds.push_back(funcOp.getArgument(dim * 2 + 1));
  }

  for (mlir::affine::AffineForOp forOp :
       funcOp.getBody().getOps<mlir::affine::AffineForOp>()) {
    // Collect the perfectly nested loops.
    llvm::SmallVector<mlir::affine::AffineForOp, 3> nestedLoops;
    mlir::affine::getPerfectlyNestedLoops(nestedLoops, forOp);
    mlir::affine::AffineForOp innermostLoop = nestedLoops.back();

    mlir::IRRewriter rewriter(&getContext());
    rewriter.setInsertionPointToStart(innermostLoop.getBody());

    // Create the 'if' operation.
    mlir::IntegerSet condition = getInBoundsCondition(&getContext(), rank);
    llvm::SmallVector<mlir::Value, 3> ifArgs;

    for (mlir::affine::AffineForOp loopOp : nestedLoops) {
      ifArgs.push_back(loopOp.getInductionVar());
    }

    for (unsigned int dim = 0; dim < rank; ++dim) {
      ifArgs.push_back(lowerBounds[dim]);
      ifArgs.push_back(upperBounds[dim]);
    }

    auto ifOp = rewriter.create<mlir::affine::AffineIfOp>(
        innermostLoop.getLoc(), condition, ifArgs, false);

    // Move the body of the innermost loop inside the 'if' operation.
    auto remainingOps = llvm::make_range(std::next(ifOp->getIterator()),
                                         innermostLoop.getBody()->end());

    auto yieldOp = ifOp.getBody()->getTerminator();

    for (auto &nestedOp : llvm::make_early_inc_range(remainingOps)) {
      if (mlir::isa<mlir::affine::AffineYieldOp>(nestedOp)) {
        continue;
      }

      rewriter.moveOpBefore(&nestedOp, yieldOp);
    }
  }
}
} // namespace

namespace marco::frontend {
std::unique_ptr<mlir::Pass> createEquationIndexCheckInsertionPass() {
  return std::make_unique<EquationIndexCheckInsertionPass>();
}
} // namespace marco::frontend
