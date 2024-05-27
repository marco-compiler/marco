#include "marco/Dialect/IDA/IR/IDA.h"
#include "marco/Dialect/KINSOL/IR/KINSOLDialect.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/Transforms/AllInterfaces.h"
#include "marco/Dialect/BaseModelica/Transforms/Passes.h"
#include "marco/Dialect/Runtime/IR/RuntimeDialect.h"
#include "marco/Dialect/Runtime/Transforms/AllInterfaces.h"
#include "marco/Dialect/Runtime/Transforms/Passes.h"
#include "marco/Dialect/SUNDIALS/IR/SUNDIALSDialect.h"
#include "marco/Codegen/Lowering/Bridge.h"
#include "marco/Codegen/Conversion/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char* argv[])
{
  // Register the dialects.
	mlir::DialectRegistry registry;

  mlir::registerAllDialects(registry);

  registry.insert<mlir::ida::IDADialect>();
  registry.insert<mlir::kinsol::KINSOLDialect>();
	registry.insert<mlir::bmodelica::BaseModelicaDialect>();
  registry.insert<mlir::modeling::ModelingDialect>();
  registry.insert<mlir::runtime::RuntimeDialect>();
  registry.insert<mlir::sundials::SUNDIALSDialect>();

  // Register the extensions.
  mlir::func::registerAllExtensions(registry);

  // Register the external models.
  mlir::bmodelica::registerAllDialectInterfaceImplementations(registry);
  mlir::runtime::registerAllDialectInterfaceImplementations(registry);

  // Register the passes defined by MARCO.
  marco::codegen::registerConversionPasses();

  mlir::bmodelica::registerModelicaPasses();
  mlir::runtime::registerRuntimePasses();

  // Register MLIR built-in transformations.
  mlir::registerAllPasses();

	auto result = mlir::MlirOptMain(
      argc, argv, "Modelica optimizer driver\n", registry);

	return mlir::asMainReturnCode(result);
}
