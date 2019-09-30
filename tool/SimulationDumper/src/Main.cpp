#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "modelica/lowerer/Simulation.hpp"

using namespace modelica;
using namespace std;

llvm::cl::opt<string> InputFileName(
		llvm::cl::Positional, llvm::cl::desc("<input-file>"), llvm::cl::init("-"));

llvm::cl::opt<string> outputFile(
		"bc", llvm::cl::desc("<output-file>"), llvm::cl::init("-"));

llvm::ExitOnError exitOnErr;
int main(int argc, char* argv[])
{
	llvm::cl::ParseCommandLineOptions(argc, argv);
	// auto errorOrBuffer = llvm::MemoryBuffer::getFileOrSTDIN(InputFileName);
	// auto buffer = exitOnErr(llvm::errorOrToExpected(move(errorOrBuffer)));

	llvm::LLVMContext context;
	Simulation sim(context);

	if (!sim.addVar("x", Expression(Constant(3))))
		return 1;
	if (!sim.addVar("Y", Expression(Constant(6))))
		return 1;
	if (!sim.addUpdate(
					"x", Expression(Constant(3)) + Expression("Y", BultinTypes::INT)))
		return 1;

	sim.dump();
	sim.lower();

	if (outputFile == "-")
		return 0;

	std::error_code error;
	llvm::raw_fd_ostream OS(outputFile, error, llvm::sys::fs::F_None);
	if (error)
	{
		llvm::errs() << error.message();
		return -1;
	}
	sim.dumpBC(OS);
	OS.flush();

	return 0;
}
