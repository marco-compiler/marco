#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Frontend/Passes/EquationIndexCheckInsertion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace marco::frontend {
#define GEN_PASS_DEF_GPUMEMORYCOPYINSERTIONPASS
#include "marco/Frontend/Passes.h.inc"
} // namespace marco::frontend

namespace {
class GPUMemoryCopyInsertionPass
    : public marco::frontend::impl::GPUMemoryCopyInsertionPassBase<
          GPUMemoryCopyInsertionPass> {
public:
  using GPUMemoryCopyInsertionPassBase::GPUMemoryCopyInsertionPassBase;

  void runOnOperation() override;
};
} // namespace

void GPUMemoryCopyInsertionPass::runOnOperation() {
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
}

namespace marco::frontend {
std::unique_ptr<mlir::Pass> createGPUMemoryCopyInsertionPass() {
  return std::make_unique<GPUMemoryCopyInsertionPass>();
}
} // namespace marco::frontend
