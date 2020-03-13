#include <limits>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "modelica/matching/Flow.hpp"
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

opt<bool> dumpGraph(
		"dumpGraph",
		cl::desc("dump dependency graph"),
		cl::init(false),
		cl::cat(simCCategory));

opt<string> outputFile(
		"o", cl::desc("<output-file>"), cl::init("-"), cl::cat(simCCategory));

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

	VVarDependencyGraph graph(constantFoldedModel);

	if (dumpGraph)
		graph.dump(OS);
	return 0;
}
