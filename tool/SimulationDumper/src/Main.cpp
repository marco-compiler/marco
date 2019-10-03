#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "modelica/lowerer/Simulation.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;

cl::opt<string> InputFileName(
		cl::Positional, cl::desc("<input-file>"), cl::init("-"));

cl::opt<string> outputFile("bc", cl::desc("<output-file>"), cl::init("-"));

ExitOnError exitOnErr;
int main(int argc, char* argv[])
{
	cl::ParseCommandLineOptions(argc, argv);
	// auto errorOrBuffer = MemoryBuffer::getFileOrSTDIN(InputFileName);
	// auto buffer = exitOnErr(errorOrToExpected(move(errorOrBuffer)));

	LLVMContext context;
	Simulation sim(context);

	if (!sim.addVar("x", SimExp(SimConst(3))))
		return 1;
	if (!sim.addVar("Y", SimExp(SimConst(6))))
		return 1;
	if (!sim.addUpdate(
					"x", SimExp(SimConst(3)) + SimExp("Y", BultinSimTypes::INT)))
		return 1;

	sim.dump();
	sim.lower();

	if (outputFile == "-")
		return 0;

	error_code error;
	raw_fd_ostream OS(outputFile, error, sys::fs::F_None);
	if (error)
	{
		errs() << error.message();
		return -1;
	}
	sim.dumpBC(OS);
	OS.flush();

	return 0;
}
