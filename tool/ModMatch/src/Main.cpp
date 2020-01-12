#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "modelica/lowerer/Lowerer.hpp"
#include "modelica/matching/Matching.hpp"
#include "modelica/model/Assigment.hpp"
#include "modelica/model/EntryModel.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModParser.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;
using namespace cl;

OptionCategory simCCategory("ModMatch options");
opt<string> InputFileName(
		cl::Positional,
		cl::desc("<input-file>"),
		cl::init("-"),
		cl::cat(simCCategory));

opt<bool> dumpModel(
		"dumpModel",
		cl::desc("dump simulation on stdout while running"),
		cl::init(false),
		cl::cat(simCCategory));

opt<string> dumpMatchingGraph(
		"dumpGraph",
		cl::desc("dump the starting matching graph exit"),
		cl::init("-"),
		cl::cat(simCCategory));

opt<int> maxMatchingIterations(
		"iterations",
		cl::desc("maximum number of iterations of the matching algorithm"),
		init(1000),
		cat(simCCategory));

ExitOnError exitOnErr;

int main(int argc, char* argv[])
{
	cl::ParseCommandLineOptions(argc, argv);
	auto errorOrBuffer = MemoryBuffer::getFileOrSTDIN(InputFileName);
	auto buffer = exitOnErr(errorOrToExpected(move(errorOrBuffer)));
	ModParser parser(buffer->getBufferStart());
	auto [init, update] = exitOnErr(parser.simulation());
	auto model = EntryModel(move(update), move(init));

	if (dumpModel)
		model.dump();

	MatchingGraph graph(model);
	graph.match(maxMatchingIterations);

	if (dumpMatchingGraph != "-")
	{
		error_code error;
		raw_fd_ostream OS(dumpMatchingGraph, error, sys::fs::F_None);
		if (error)
		{
			errs() << error.message();
			return -1;
		}

		graph.dumpGraph(OS);
	}

	return 0;
}
