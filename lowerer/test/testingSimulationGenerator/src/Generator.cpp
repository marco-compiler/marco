#include "Generator.hpp"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace modelica;
using namespace std;
using namespace llvm;
using namespace cl;

opt<string> outputFile("bc", cl::desc("<output-file>"), cl::init("-"));

int main(int argc, char* argv[])
{
	ParseCommandLineOptions(argc, argv);
	LLVMContext context;
	Simulation sim(context, "Int Test Simulation", "runSimulation");
	sim.setVarsLinkage(GlobalValue::LinkageTypes::ExternalLinkage);
	if (!makeSimulation(sim))
		return -1;

	error_code error;
	raw_fd_ostream OS(outputFile, error, sys::fs::F_None);
	if (error)
	{
		errs() << error.message();
		return -1;
	}
	sim.dump(outs());
	sim.lower();
	sim.dumpBC(OS);

	return 0;
}
