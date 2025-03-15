#include "marco/JIT/EngineBuilder.h"
#include "marco/Codegen/Conversion/Passes.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Support/TargetSelect.h"

using namespace ::marco::jit;

namespace marco::jit {
EngineBuilder::EngineBuilder(mlir::ModuleOp moduleOp,
                             const mlir::ExecutionEngineOptions &options)
    : moduleOp(moduleOp), options(options) {}

namespace {
std::unique_ptr<mlir::Pass> createMLIROneShotBufferizePass() {
  mlir::bufferization::OneShotBufferizePassOptions options;
  options.bufferizeFunctionBoundaries = true;

  return mlir::bufferization::createOneShotBufferizePass(options);
}

void buildMLIRBufferDeallocationPipeline(mlir::OpPassManager &pm) {
  mlir::bufferization::BufferDeallocationPipelineOptions options;
  mlir::bufferization::buildBufferDeallocationPipeline(pm, options);
}
} // namespace

mlir::ModuleOp EngineBuilder::lowerToLLVM() const {
  mlir::ModuleOp loweredModule = mlir::cast<mlir::ModuleOp>(moduleOp->clone());

  loweredModule.walk([](mlir::bmodelica::ModelOp modelOp) { modelOp.erase(); });

  mlir::PassManager pm(loweredModule->getContext());

  pm.addPass(mlir::createBaseModelicaToCFConversionPass());
  pm.addPass(mlir::createBaseModelicaToMLIRCoreConversionPass());
  pm.addPass(mlir::createRuntimeToFuncConversionPass());
  pm.addPass(mlir::createLinalgGeneralizeNamedOpsPass());
  pm.addPass(createMLIROneShotBufferizePass());
  pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
  buildMLIRBufferDeallocationPipeline(pm);
  pm.addPass(mlir::createConvertBufferizationToMemRefPass());
  pm.addPass(mlir::createBaseModelicaRawVariablesConversionPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createSCFToControlFlowPass());
  pm.addPass(mlir::createBaseModelicaToMLIRCoreConversionPass());
  pm.addNestedPass<mlir::func::FuncOp>(
      mlir::LLVM::createRequestCWrappersPass());
  pm.addPass(mlir::createConvertToLLVMPass());
  pm.addPass(mlir::createBaseModelicaToLLVMConversionPass());
  pm.addPass(mlir::createRuntimeToLLVMConversionPass());

  pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(
      mlir::createReconcileUnrealizedCastsPass());

  pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(mlir::createCanonicalizerPass());
  pm.addPass(mlir::LLVM::createLegalizeForExportPass());

  if (mlir::failed(pm.run(loweredModule))) {
    loweredModule.emitOpError() << "Failed to lower module to LLVM";
    return nullptr;
  }

  return loweredModule;
}

std::unique_ptr<mlir::ExecutionEngine> EngineBuilder::getEngine() const {
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  mlir::ModuleOp loweredModuleOp = lowerToLLVM();

  if (!loweredModuleOp) {
    return nullptr;
  }

  auto executionEngine =
      mlir::ExecutionEngine::create(loweredModuleOp, options);

  if (!executionEngine) {
    llvm::consumeError(executionEngine.takeError());
    return nullptr;
  }

  return std::move(executionEngine.get());
}
} // namespace marco::jit
