#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "modelica/lowerer/Lowerer.hpp"
#include "modelica/model/ModParser.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;
using namespace cl;

OptionCategory simCCategory("ModC options");
opt<string> InputFileName(
		cl::Positional,
		cl::desc("<input-file>"),
		cl::init("-"),
		cl::cat(simCCategory));
opt<string> outputFile(
		"bc", cl::desc("<output-file>"), cl::init("-"), cl::cat(simCCategory));
opt<string> headerFile(
		"header", cl::desc("<header-file>"), cl::init("-"), cl::cat(simCCategory));

opt<string> entryPointName(
		"entry",
		cl::desc("entry point of the simulation"),
		cl::init("main"),
		cl::cat(simCCategory));

opt<bool> externalLinkage(
		"publicSymbols",
		cl::desc("globals symbols are set as extenal linkage"),
		cl::init(false),
		cl::cat(simCCategory));

opt<bool> dumpModel(
		"dumpModel",
		cl::desc("dump simulation on stdout while running"),
		cl::init(false),
		cl::cat(simCCategory));

opt<int> simulationTime(
		"simTime",
		cl::desc("how many ticks the simulation must perform"),
		cl::init(10),
		cl::cat(simCCategory));

ExitOnError exitOnErr;
int main(int argc, char* argv[])
{
	cl::ParseCommandLineOptions(argc, argv);
	auto errorOrBuffer = MemoryBuffer::getFileOrSTDIN(InputFileName);
	auto buffer = exitOnErr(errorOrToExpected(move(errorOrBuffer)));
	ModParser parser(buffer->getBufferStart());
	auto [init, update] = exitOnErr(parser.simulation());

	LLVMContext context;
	Lowerer sim(
			context,
			move(init),
			move(update),
			"Modulation",
			entryPointName,
			simulationTime);
	if (externalLinkage)
		sim.setVarsLinkage(GlobalValue::LinkageTypes::ExternalLinkage);

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
	if (dumpModel)
		sim.dump(outs());

	exitOnErr(sim.lower());
	sim.dumpBC(OS);
	sim.dumpHeader(headerOS);

	return 0;
}
