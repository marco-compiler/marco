#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "marco/Codegen/Bridge.h"
#include "marco/Codegen/Conversion/Passes.h"
#include "marco/Codegen/Transforms/Passes.h"
#include "marco/Codegen/dialects/modelica/ModelicaDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/MlirOptMain.h"

using namespace marco::codegen;

int main(int argc, char* argv[])
{
  registerModelicaTransformationPasses();
	registerModelicaConversionPasses();

	mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

	registry.insert<modelica::ModelicaDialect>();

	auto result = mlir::MlirOptMain(argc, argv, "Modelica optimizer driver\n", registry);
	return mlir::asMainReturnCode(result);
}
