#include <limits>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "marco/matching/Flow.hpp"
#include "marco/matching/SVarDependencyGraph.hpp"
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

OptionCategory mSchedCat("ModMatch options");
opt<string> InputFileName(
		Positional, desc("<input-file>"), init("-"), cat(mSchedCat));

opt<bool> dumpModel(
		"dumpModel",
		desc("dump simulation on stdout while running"),
		init(false),
		cat(mSchedCat));

opt<bool> dumpGraph(
		"dumpGraph", desc("dump dependency graph"), init(false), cat(mSchedCat));

opt<bool> dumpScc(
		"dumpScc", desc("dump each scc graph"), init(false), cat(mSchedCat));

opt<string> outputFile("o", desc("<output-file>"), init("-"), cat(mSchedCat));

ExitOnError exitOnErr;

int main(int argc, char* argv[])
{
	cl::ParseCommandLineOptions(argc, argv);
	auto errorOrBuffer = MemoryBuffer::getFileOrSTDIN(InputFileName);
	auto buffer = exitOnErr(errorOrToExpected(move(errorOrBuffer)));
	ModParser parser(buffer->getBufferStart());
	Model model = exitOnErr(parser.simulation());
	auto constantFoldedModel = exitOnErr(constantFold(move(model)));

	if (dumpModel)
		constantFoldedModel.dump();

	error_code error;
	raw_fd_ostream OS(outputFile, error, sys::fs::OF_None);
	if (error)
	{
		errs() << error.message();
		return -1;
	}

	if (dumpModel)
	{
		auto assigment = schedule(constantFoldedModel);
		assigment.dump(OS);
		return 0;
	}

	VVarDependencyGraph graph(constantFoldedModel);

	if (dumpGraph)
		graph.dump(OS);

	SccLookup sccs(graph);

	if (dumpScc)
	{
		size_t i = 0;
		for (const auto& scc : sccs)
		{
			string fileName = to_string(i) + "_" + outputFile;
			SVarDependencyGraph scalarGraph(graph, scc);
			raw_fd_ostream OS(outputFile, error, sys::fs::OF_None);
			if (error)
			{
				errs() << error.message();
				return -1;
			}
			scalarGraph.dumpGraph(OS);
		}
	}

	return 0;
}
