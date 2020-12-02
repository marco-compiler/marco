#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/MemoryBuffer.h>
#include <modelica/frontend/ConstantFolder.hpp>
#include <modelica/frontend/Parser.hpp>
#include <modelica/frontend/SymbolTable.hpp>
#include <modelica/frontend/TypeChecker.hpp>

using namespace modelica;
using namespace llvm;
using namespace std;
using namespace cl;

cl::OptionCategory omcCCat("FunctionLowerer options");
cl::opt<string> InputFileName(
		cl::Positional, cl::desc("<input-file>"), cl::init("-"), cl::cat(omcCCat));

ExitOnError exitOnErr;
int main(int argc, char* argv[])
{
	cl::ParseCommandLineOptions(argc, argv);
	auto errorOrBuffer = MemoryBuffer::getFileOrSTDIN(InputFileName);
	error_code error;

	if (error)
	{
		errs() << error.message();
		return -1;
	}

	auto buffer = exitOnErr(errorOrToExpected(move(errorOrBuffer)));
	Parser parser(buffer->getBufferStart());
	auto ast = exitOnErr(parser.classDefinition());

	TypeChecker checker;
	exitOnErr(checker.checkType(ast, SymbolTable()));

	modelica::ConstantFolder folder;
	exitOnErr(folder.fold(ast, SymbolTable()));
/*
	Model model;
	OmcToModelPass pass(model);
	exitOnErr(pass.lower(ast, SymbolTable()));
	if (dumpModel)
	{
		model.dump(OS);
		return 0;
	}

	auto foldedModel = exitOnErr(constantFold(move(model)));
	exitOnErr(solveDer(foldedModel));
	if (dumpSolvedDerModel)
	{
		foldedModel.dump(OS);
		return 0;
	}

	auto matchedModel = exitOnErr(match(move(foldedModel), 1000));
	if (dumpMatched)
	{
		matchedModel.dump(OS);
		return 0;
	}

	auto collapsed = exitOnErr(solveScc(move(matchedModel), 1000));
	if (dumpCollapsed)
	{
		collapsed.dump(OS);
		return 0;
	}

	auto scheduled = schedule(move(collapsed));
	if (dumpScheduled)
	{
		scheduled.dump(OS);
		return 0;
	}

	auto assModel = exitOnErr(addAproximation(scheduled, timeStep));
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
			entryPointName,
			simulationTime,
			true);

	if (externalLinkage)
		sim.setVarsLinkage(GlobalValue::LinkageTypes::ExternalLinkage);

	if (dumpLowered)
	{
		sim.dump(OS);
		return 0;
	}
	exitOnErr(sim.lower());

	sim.verify();
	sim.dumpBC(OS);
*/
	return 0;
}
