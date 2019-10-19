#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "modelica/Parser.hpp"
#include "modelica/omcToModel/OmcToModelPass.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;

cl::OptionCategory astDumperCategory("ASTDumper options");
cl::opt<string> InputFileName(
		cl::Positional,
		cl::desc("<input-file>"),
		cl::init("-"),
		cl::cat(astDumperCategory));

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

	return 0;
}
