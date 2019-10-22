#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "modelica/Parser.hpp"
#include "modelica/lowerer/Lowerer.hpp"
#include "modelica/model/AssignModel.hpp"
#include "modelica/omcToModel/OmcToModelPass.hpp"
#include "modelica/passes/SolveDerivatives.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;
using namespace cl;

cl::OptionCategory omcCCat("OmcC options");
cl::opt<string> InputFileName(
		cl::Positional, cl::desc("<input-file>"), cl::init("-"), cl::cat(omcCCat));

opt<int> simulationTime(
		"simTime",
		cl::desc("how many ticks the simulation must perform"),
		cl::init(10),
		cl::cat(omcCCat));

opt<float> timeStep(
		"timeStep",
		cl::desc("how long in seconds a ticks in simulation lasts"),
		cl::init(0.1f),
		cl::cat(omcCCat));

opt<bool> dumpModel(
		"d", cl::desc("dump model"), cl::init(false), cl::cat(omcCCat));

opt<bool> dumpLowered(
		"l",
		cl::desc("dump lowered model and exit"),
		cl::init(false),
		cl::cat(omcCCat));

opt<string> outputFile(
		"bc", cl::desc("<output-file>"), cl::init("-"), cl::cat(omcCCat));

ExitOnError exitOnErr;
int main(int argc, char* argv[])
{
	cl::ParseCommandLineOptions(argc, argv);
	auto errorOrBuffer = MemoryBuffer::getFileOrSTDIN(InputFileName);
	auto buffer = exitOnErr(errorOrToExpected(move(errorOrBuffer)));
	Parser parser(buffer->getBufferStart());
	UniqueDecl ast = exitOnErr(parser.classDefinition());
	EntryModel model;
	OmcToModelPass pass(model);
	ast = topDownVisit(move(ast), pass);

	if (dumpModel)
	{
		model.dump(outs());
		return 0;
	}

	auto assModel = exitOnErr(solveDer(move(model)));
	LLVMContext context;
	Lowerer sim(
			context,
			move(assModel.getVars()),
			move(assModel.getUpdates()),
			"Modelica Model",
			"main",
			simulationTime);
	if (!sim.addVar("deltaTime", ModExp::constExp<float>(timeStep)))
	{
		outs() << "DeltaTime was already defined\n";
		return -1;
	}

	if (dumpLowered)
	{
		sim.dump(outs());
		return 0;
	}
	exitOnErr(sim.lower());
	error_code error;
	raw_fd_ostream OS(outputFile, error, sys::fs::F_None);
	if (error)
	{
		errs() << error.message();
		return -1;
	}
	sim.dumpBC(OS);

	return 0;
}
