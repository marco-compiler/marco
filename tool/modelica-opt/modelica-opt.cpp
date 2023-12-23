#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Dialect/IDA/IDADialect.h"
#include "marco/Dialect/Simulation/SimulationDialect.h"
#include "marco/Codegen/Bridge.h"
#include "marco/Codegen/Conversion/Passes.h"
#include "marco/Codegen/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char* argv[])
{
  // Register the dialects.
	mlir::DialectRegistry registry;

  mlir::registerAllDialects(registry);

  registry.insert<mlir::modeling::ModelingDialect>();
	registry.insert<mlir::modelica::ModelicaDialect>();
  registry.insert<mlir::ida::IDADialect>();
  registry.insert<mlir::simulation::SimulationDialect>();

  // Register the extensions.
  mlir::func::registerAllExtensions(registry);

  // Register the passes defined by MARCO.
  marco::codegen::registerTransformsPasses();
  marco::codegen::registerConversionPasses();

  // Register some useful MLIR built-in transformations.
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerLoopInvariantCodeMotionPass();
  mlir::registerArithToLLVMConversionPass();
  mlir::registerConvertFuncToLLVMPass();
  mlir::registerFinalizeMemRefToLLVMConversionPass();
  mlir::registerInlinerPass();
  mlir::registerSCFToControlFlowPass();
  mlir::registerConvertControlFlowToLLVMPass();
  mlir::registerConvertVectorToSCFPass();
  mlir::registerConvertVectorToLLVMPass();
  mlir::registerReconcileUnrealizedCastsPass();
  mlir::bufferization::registerBufferLoopHoistingPass();

	auto result = mlir::MlirOptMain(
      argc, argv, "Modelica optimizer driver\n", registry);

	return mlir::asMainReturnCode(result);
}
