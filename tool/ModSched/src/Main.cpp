#include <limits>
#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "modelica/matching/Flow.hpp"
#include "modelica/matching/SVarDependencyGraph.hpp"
#include "modelica/matching/Schedule.hpp"
#include "modelica/matching/VVarDependencyGraph.hpp"
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
	auto [init, update] = exitOnErr(parser.simulation());
	auto model = EntryModel(move(update), move(init));
	auto constantFoldedModel = exitOnErr(constantFold(move(model)));

	if (dumpModel)
		constantFoldedModel.dump();

	error_code error;
	raw_fd_ostream OS(outputFile, error, sys::fs::F_None);
	if (error)
	{
		errs() << error.message();
		return -1;
	}

	if (!dumpModel && !dumpScc)
	{
		auto assigment = schedule(constantFoldedModel);
		assigment.dump(OS);
		return 0;
	}

	VVarDependencyGraph graph(constantFoldedModel);

	if (dumpGraph)
		graph.dump(OS);

	auto sccs = graph.getSCC();

	if (dumpScc)
	{
		size_t i = 0;
		for (const auto& scc : sccs)
		{
			string fileName = to_string(i) + "_" + outputFile;
			SVarDepencyGraph scalarGraph(graph, scc);
			raw_fd_ostream OS(outputFile, error, sys::fs::F_None);
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
