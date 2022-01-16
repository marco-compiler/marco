#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/MlirOptMain.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>
#include <marco/codegen/CodeGen.h>
#include <marco/codegen/dialects/modelica/ModelicaDialect.h>

using namespace marco::codegen;

int main(int argc, char* argv[])
{
	registerModelicaPasses();

	mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);

	registry.insert<modelica::ModelicaDialect>();

	auto result = mlir::MlirOptMain(argc, argv, "Modelica optimizer driver\n", registry);
	return mlir::asMainReturnCode(result);
}
