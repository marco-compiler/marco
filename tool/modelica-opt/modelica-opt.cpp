#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/IDA/IDADialect.h"
#include "marco/Codegen/Bridge.h"
#include "marco/Codegen/Conversion/Passes.h"
#include "marco/Codegen/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

int main(int argc, char* argv[])
{
  // Register the dialects
	mlir::DialectRegistry registry;

  mlir::registerAllDialects(registry);

	registry.insert<mlir::modelica::ModelicaDialect>();
  registry.insert<mlir::ida::IDADialect>();

  // Register the passes defined by MARCO
  marco::codegen::registerTransformsPasses();
  marco::codegen::registerConversionPasses();

  // Register some useful MLIR built-in transformations
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerConvertArithmeticToLLVMPass();
  mlir::registerConvertFuncToLLVMPass();
  mlir::registerSCFToControlFlowPass();
  mlir::registerConvertControlFlowToLLVMPass();
  mlir::registerReconcileUnrealizedCastsPass();

	auto result = mlir::MlirOptMain(argc, argv, "Modelica optimizer driver\n", registry);
	return mlir::asMainReturnCode(result);
}
