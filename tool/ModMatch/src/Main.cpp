#include <limits>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "modelica/lowerer/Lowerer.hpp"
#include "modelica/matching/Flow.hpp"
#include "modelica/matching/Matching.hpp"
#include "modelica/model/Assigment.hpp"
#include "modelica/model/EntryModel.hpp"
#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModParser.hpp"
#include "modelica/passes/ConstantFold.hpp"
#include "modelica/utils/IRange.hpp"

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

opt<bool> showEmptyEdges(
		"showEmptyEdges",
		cl::desc("show empty edges in dumped graph"),
		cl::init(false),
		cl::cat(simCCategory));

opt<bool> showMapping(
		"showMappings",
		cl::desc("show mappings in dumped graph"),
		cl::init(false),
		cl::cat(simCCategory));

opt<bool> showMatchedCount(
		"showMatchedCount",
		cl::desc("show mappings in dumped graph"),
		cl::init(false),
		cl::cat(simCCategory));

opt<bool> showAlternatives(
		"showAugmentingPathAlternatives",
		cl::desc("show the possible alternatives in the augmenting path"),
		cl::init(false),
		cl::cat(simCCategory));

opt<string> dumpAugmentingPathOnFailure(
		"dumpGraphFailure",
		cl::desc("dump the augumenting graph if a path cannot be found"),
		cl::init("-"),
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

opt<size_t> maxSearchDepth(
		"maxDepth",
		cl::desc("maximun depth of search when looking for an augmenting path"),
		init(numeric_limits<size_t>::max()),
		cat(simCCategory));

opt<size_t> expectedMatches(
		"expectedMatches",
		cl::desc("programs exits with -1 if the matched variables are different "
						 "than this argument"),
		init(0),
		cat(simCCategory));

ExitOnError exitOnErr;

static int dumpGraph(const MatchingGraph& graph)
{
	error_code error;
	raw_fd_ostream OS(dumpMatchingGraph, error, sys::fs::F_None);
	if (error)
	{
		errs() << error.message();
		return -1;
	}

	graph.dumpGraph(OS, showEmptyEdges, showMapping, showMatchedCount);
	return 0;
}

static int dumpAugmentingPath(const AugmentingPath& path)
{
	error_code error;
	raw_fd_ostream OS(dumpMatchingGraph, error, sys::fs::F_None);
	if (error)
	{
		errs() << error.message();
		return -1;
	}

	path.dumpGraph(
			OS, showEmptyEdges, showMapping, showMatchedCount, showAlternatives);
	return 0;
}

int main(int argc, char* argv[])
{
	cl::ParseCommandLineOptions(argc, argv);
	auto errorOrBuffer = MemoryBuffer::getFileOrSTDIN(InputFileName);
	auto buffer = exitOnErr(errorOrToExpected(move(errorOrBuffer)));
	ModParser parser(buffer->getBufferStart());
	auto model = exitOnErr(parser.simulation());
	auto constantFoldedModel = exitOnErr(constantFold(move(model)));

	if (dumpModel)
		constantFoldedModel.dump();

	MatchingGraph graph(constantFoldedModel);

	for (auto i : irange<int>(maxMatchingIterations.getValue()))
	{
		AugmentingPath augmentingPath(graph, maxSearchDepth);
		if (!augmentingPath.valid())
		{
			if (dumpAugmentingPathOnFailure != "-" &&
					augmentingPath.size() == maxSearchDepth)
				return dumpAugmentingPath(augmentingPath);
			break;
		}

		augmentingPath.apply();
	}

	if (dumpMatchingGraph != "-")
		if (auto error = dumpGraph(graph); error != 0)
			return error;

	if (expectedMatches != 0 && expectedMatches != graph.matchedCount())
		return -1;

	return 0;
}
