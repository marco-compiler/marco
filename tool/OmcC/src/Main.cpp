#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "modelica/Parser.hpp"
#include "modelica/lowerer/Lowerer.hpp"
#include "modelica/model/AssignModel.hpp"
#include "modelica/model/ModVariable.hpp"
#include "modelica/omcToModel/OmcToModelPass.hpp"
#include "modelica/passes/ConstantFold.hpp"
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
		cl::init(0.1F),
		cl::cat(omcCCat));

opt<bool> dumpModel(
		"d", cl::desc("dump model"), cl::init(false), cl::cat(omcCCat));

opt<bool> dumpSolvedModel(
		"ds",
		cl::desc("dump solved derivatives model"),
		cl::init(false),
		cl::cat(omcCCat));

opt<bool> dumpLowered(
		"l",
		cl::desc("dump lowered model and exit"),
		cl::init(false),
		cl::cat(omcCCat));

opt<string> outputFile(
		"o", cl::desc("<output-file>"), cl::init("-"), cl::cat(omcCCat));

SmallVector<Assigment, 2> toAssign(SmallVector<ModEquation, 2>&& equs)
{
	SmallVector<Assigment, 2> assign;

	for (ModEquation& eq : equs)
	{
		assert(eq.getLeft().isReference() || eq.getLeft().isReferenceAccess());
		assign.emplace_back(
				move(eq.getLeft()), move(eq.getRight()), move(eq.getInductions()));
	}

	return assign;
}

ExitOnError exitOnErr;
int main(int argc, char* argv[])
{
	cl::ParseCommandLineOptions(argc, argv);
	auto errorOrBuffer = MemoryBuffer::getFileOrSTDIN(InputFileName);
	error_code error;
	raw_fd_ostream OS(outputFile, error, sys::fs::F_None);
	if (error)
	{
		errs() << error.message();
		return -1;
	}

	auto buffer = exitOnErr(errorOrToExpected(move(errorOrBuffer)));
	Parser parser(buffer->getBufferStart());
	UniqueDecl ast = exitOnErr(parser.classDefinition());
	EntryModel model;
	OmcToModelPass pass(model);
	ast = topDownVisit(move(ast), pass);

	if (dumpModel)
	{
		model.dump(OS);
		return 0;
	}

	auto foldedModel = exitOnErr(constantFold(move(model)));
	auto assModel = exitOnErr(solveDer(move(foldedModel), timeStep));

	if (dumpSolvedModel)
	{
		assModel.dump(OS);
		return 0;
	}
	LLVMContext context;
	Lowerer sim(
			context,
			move(assModel.getVars()),
			move(assModel.getUpdates()),
			"Modelica Model",
			"main",
			simulationTime);

	if (dumpLowered)
	{
		sim.dump(OS);
		return 0;
	}
	exitOnErr(sim.lower());
	sim.dumpBC(OS);

	return 0;
}
