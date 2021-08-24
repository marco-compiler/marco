#include <limits>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "marco/matching/Flow.hpp"
#include "marco/matching/SVarDependencyGraph.hpp"
#include "marco/matching/SccCollapsing.hpp"
#include "marco/matching/SccLookup.hpp"
#include "marco/matching/Schedule.hpp"
#include "marco/matching/VVarDependencyGraph.hpp"
#include "marco/model/Assigment.hpp"
#include "marco/model/ModEquation.hpp"
#include "marco/model/ModParser.hpp"
#include "marco/model/Model.hpp"
#include "marco/passes/ConstantFold.hpp"
#include "marco/utils/IRange.hpp"

using namespace marco;
using namespace llvm;
using namespace std;
using namespace cl;

OptionCategory mSchedCat("modCollapser options");
opt<string> InputFileName(
		Positional, desc("<input-file>"), init("-"), cat(mSchedCat));

opt<bool> dumpModel(
		"dumpModel",
		desc("dump simulation on stdout while running"),
		init(false),
		cat(mSchedCat));

opt<bool> dumpGraphEarly(
		"dumpGraphEarly", desc("dump graph before running matching"), init(false), cat(mSchedCat));

opt<bool> dumpGraph(
		"dumpGraph", desc("dump graph"), init(false), cat(mSchedCat));

opt<string> outputFile("o", desc("<output-file>"), init("-"), cat(mSchedCat));

opt<size_t> maxIterations(
		"max-iter", desc("max collapsing iterations"), init(100), cat(mSchedCat));

ExitOnError exitOnErr;

int main(int argc, char* argv[])
{
	cl::ParseCommandLineOptions(argc, argv);
	auto errorOrBuffer = MemoryBuffer::getFileOrSTDIN(InputFileName);
	auto buffer = exitOnErr(errorOrToExpected(move(errorOrBuffer)));
	ModParser parser(buffer->getBufferStart());
	Model model = exitOnErr(parser.simulation());

	if (dumpModel)
		model.dump();

	error_code error;
	raw_fd_ostream OS(outputFile, error, sys::fs::F_None);
	if (error)
	{
		errs() << error.message();
		return -1;
	}

	if (dumpGraphEarly)
	{
		VVarDependencyGraph graph(model);
		graph.dump(OS);
		return 0;
	}

	auto outM = exitOnErr(solveScc(move(model), maxIterations));

	if (dumpGraph)
	{
		VVarDependencyGraph graph(outM);
		graph.dump(OS);
		return 0;
	}

	outM.dump(OS);

	return 0;
}
