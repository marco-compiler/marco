#include "marco/Frontend/Passes/GPUVariableDeclaration.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace marco::frontend {
#define GEN_PASS_DEF_GPUVARIABLEDECLARATIONPASS
#include "marco/Frontend/Passes.h.inc"
} // namespace marco::frontend

namespace {
class GPUVariableDeclarationPass
    : public marco::frontend::impl::GPUVariableDeclarationPassBase<
          GPUVariableDeclarationPass> {
public:
  using GPUVariableDeclarationPassBase::GPUVariableDeclarationPassBase;

  void runOnOperation() override;
};
} // namespace

void GPUVariableDeclarationPass::runOnOperation() {
  /*
  if (!getOperation()->hasAttr(
          mlir::bmodelica::BaseModelicaDialect::kEquationFunctionAttrName)) {
    // Not an equation function.
    return;
  }

  mlir::SymbolTableCollection symbolTables;
  mlir::ModuleOp moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();

  // Collect gpu.launch operations and process them.
  llvm::SmallVector<mlir::gpu::LaunchOp> launchOps;

  getOperation()->walk(
      [&](mlir::gpu::LaunchOp launchOp) { launchOps.push_back(launchOp); });

  for (mlir::gpu::LaunchOp launchOp : launchOps) {
    // Resolve the kernel and get the parent module.
    //symbolTables.lookupSymbolIn<mlir::gpu::GPUModuleOp>(moduleOp, launchOp.getModule())

    // Insert memory copy operations before and after the launch operation.
    llvm::SmallVector<mlir::memref::GlobalOp> globalOps;

  }
  */
}

namespace marco::frontend {
std::unique_ptr<mlir::Pass> createGPUVariableDeclarationPass() {
  return std::make_unique<GPUVariableDeclarationPass>();
}
} // namespace marco::frontend
