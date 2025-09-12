#include "marco/Dialect/BaseModelica/Transforms/FunctionMangling.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_FUNCTIONMANGLINGPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class FunctionManglingPass
    : public mlir::bmodelica::impl::FunctionManglingPassBase<
          FunctionManglingPass> {
public:
  using FunctionManglingPassBase::FunctionManglingPassBase;

  void runOnOperation() override;
};

std::string getMangled(llvm::StringRef name) { return "_M" + name.str(); }

mlir::SymbolRefAttr getMangled(mlir::SymbolRefAttr name) {
  return mlir::SymbolRefAttr::get(name.getContext(),
                                  getMangled(name.getRootReference()),
                                  name.getNestedReferences());
}
} // namespace

void FunctionManglingPass::runOnOperation() {
  getOperation().walk([](mlir::Operation *op) {
    if (auto callOp = mlir::dyn_cast<CallOp>(op)) {
      callOp.setCalleeAttr(getMangled(callOp.getCallee()));
      return;
    }

    if (auto functionOp = mlir::dyn_cast<FunctionOp>(op)) {
      functionOp.setSymNameAttr(mlir::StringAttr::get(
          functionOp.getContext(), getMangled(functionOp.getSymName())));
    }
  });
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createFunctionManglingPass() {
  return std::make_unique<FunctionManglingPass>();
}
} // namespace mlir::bmodelica
