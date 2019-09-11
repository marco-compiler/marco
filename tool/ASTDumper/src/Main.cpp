#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "modelica/Dumper/Dumper.hpp"
#include "modelica/Parser.hpp"

using namespace modelica;
using namespace llvm;
using namespace std;

cl::opt<string> InputFileName(
		cl::Positional, cl::desc("<input-file>"), cl::init("-"));

ExitOnError exitOnErr;
int main(int argc, char* argv[])
{
	cl::ParseCommandLineOptions(argc, argv);
	auto errorOrBuffer = MemoryBuffer::getFileOrSTDIN(InputFileName);
	auto buffer = exitOnErr(errorOrToExpected(move(errorOrBuffer)));
	Parser parser(buffer->getBufferStart());
	UniqueDecl ast = exitOnErr(parser.classDefinition());
	ast = dump(move(ast), outs());

	return 0;
}
