#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "modelica/frontend/Parser.hpp"

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
	frontend::Parser parser(buffer->getBufferStart());
	auto ast = exitOnErr(parser.classDefinition());
	ast.dump();

	return 0;
}
