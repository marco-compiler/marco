#include "marco/Dialect/BaseModelica/Transforms/InliningAttributeInsertion.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_INLININGATTRIBUTEINSERTIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class InliningAttributeInsertionPass
    : public mlir::bmodelica::impl::InliningAttributeInsertionPassBase<
          InliningAttributeInsertionPass> {
public:
  using InliningAttributeInsertionPassBase<
      InliningAttributeInsertionPass>::InliningAttributeInsertionPassBase;

  void runOnOperation() override;
};
} // namespace

namespace {
void addInlineHint(mlir::Operation *op) {
  op->setAttr("inline_hint", mlir::UnitAttr::get(op->getContext()));
}

void addInliningAttribute(CallOp callOp, mlir::ModuleOp moduleOp,
                          mlir::SymbolTableCollection &symbolTables) {
  auto callee = callOp.getFunction(moduleOp, symbolTables);

  if (auto functionOp = mlir::dyn_cast_if_present<FunctionOp>(callee)) {
    if (functionOp.shouldBeInlined()) {
      addInlineHint(callOp);
    }
  }

  if (auto rawFunctionOp = mlir::dyn_cast_if_present<RawFunctionOp>(callee)) {
    if (rawFunctionOp.shouldBeInlined()) {
      addInlineHint(callOp);
    }
  }
}
} // namespace

void InliningAttributeInsertionPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();
  llvm::SmallVector<CallOp> callOps;

  moduleOp.walk([&](CallOp callOp) { callOps.push_back(callOp); });

  mlir::SymbolTableCollection symbolTables;
  mlir::LockedSymbolTableCollection lockedSymbolTables(symbolTables);

  mlir::parallelForEach(&getContext(), callOps, [&](CallOp callOp) {
    addInliningAttribute(callOp, moduleOp, lockedSymbolTables);
  });
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createInliningAttributeInsertionPass() {
  return std::make_unique<InliningAttributeInsertionPass>();
}
} // namespace mlir::bmodelica
