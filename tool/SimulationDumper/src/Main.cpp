#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "modelica/lowerer/Simulation.hpp"

using namespace modelica;
using namespace std;

llvm::cl::opt<string> InputFileName(
		llvm::cl::Positional, llvm::cl::desc("<input-file>"), llvm::cl::init("-"));

llvm::ExitOnError exitOnErr;
int main(int argc, char* argv[])
{
	llvm::cl::ParseCommandLineOptions(argc, argv);
	auto errorOrBuffer = llvm::MemoryBuffer::getFileOrSTDIN(InputFileName);
	auto buffer = exitOnErr(llvm::errorOrToExpected(move(errorOrBuffer)));

	llvm::LLVMContext context;
	Simulation sim(context);

	if (!sim.addVar(
					"x",
					Expression(Constant(3), modelica::Type(modelica::BultinTypes::INT)) +
							Expression("Y", modelica::Type(modelica::BultinTypes::INT))))
		return 1;

	sim.dump();

	return 0;
}
