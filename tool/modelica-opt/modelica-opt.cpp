#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "marco/Codegen/Bridge.h"
#include "marco/Codegen/Conversion/Passes.h"
#include "marco/Codegen/Transforms/Passes.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/MlirOptMain.h"

using namespace ::marco::codegen;
using namespace ::mlir::modelica;

int main(int argc, char* argv[])
{
  registerModelicaTransformationPasses();
	registerModelicaConversionPasses();

  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();

	mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

	registry.insert<ModelicaDialect>();

	auto result = mlir::MlirOptMain(argc, argv, "Modelica optimizer driver\n", registry);
	return mlir::asMainReturnCode(result);
}
