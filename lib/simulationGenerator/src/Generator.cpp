#include "modelica/simulatorGenerator/Generator.hpp"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace modelica;
using namespace std;
using namespace llvm;
using namespace cl;

opt<string> outputFile("bc", cl::desc("<output-file>"), cl::init("-"));
opt<string> headerFile("header", cl::desc("<header-file>"), cl::init("-"));

ExitOnError exitOnErr;
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
	raw_fd_ostream headerOS(headerFile, error, sys::fs::F_None);
	if (error)
	{
		errs() << error.message();
		return -1;
	}
	sim.dump(outs());
	exitOnErr(sim.lower());
	sim.dumpBC(OS);
	sim.dumpHeader(headerOS);

	return 0;
}
