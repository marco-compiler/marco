#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Error.h>
#include <llvm/Transforms/Utils.h>
#include <mlir/Transforms/Passes.h>
#include <modelica/frontend/Parser.h>
#include <modelica/frontend/Passes.h>
#include <modelica/mlirlowerer/CodeGen.h>
#include <modelica/mlirlowerer/passes/ida/IdaSolver.h>

using namespace std;
using namespace llvm;
using namespace modelica;
using namespace codegen;

static cl::OptionCategory modelSolvingOptions("Model solving options");

static cl::opt<int> matchingMaxIterations(
		"matching-max-iterations",
		cl::desc(
				"Maximum number of iterations for the matching phase (default: 1000)"),
		cl::init(1000),
		cl::cat(modelSolvingOptions));
static cl::opt<int> sccMaxIterations(
		"scc-max-iterations",
		cl::desc("Maximum number of iterations for the SCC resolution phase "
						 "(default: 1000)"),
		cl::init(1000),
		cl::cat(modelSolvingOptions));
static cl::opt<codegen::Solver> solver(
		cl::desc("Solvers:"),
		cl::values(clEnumValN(codegen::CleverDAE, "clever-dae", "Clever DAE")),
		cl::init(codegen::CleverDAE),
		cl::cat(modelSolvingOptions));

static cl::OptionCategory codeGenOptions("Code generation options");

static cl::list<string> inputFiles(
		cl::Positional,
		cl::desc("<input files>"),
		cl::OneOrMore,
		cl::cat(codeGenOptions));
static cl::opt<string> outputFile(
		"o", cl::desc("<output-file>"), cl::init("-"), cl::cat(codeGenOptions));
static cl::opt<bool> printModule(
		"print-module",
		cl::desc("Print the ModuleOp right before being solved"),
		cl::init(false),
		cl::cat(codeGenOptions));

static cl::OptionCategory simulationOptions("Simulation options");

static cl::opt<double> startTime(
		"start-time",
		cl::desc("Start time (in seconds) (default: 0)"),
		cl::init(0),
		cl::cat(simulationOptions));
static cl::opt<double> endTime(
		"end-time",
		cl::desc("End time (in seconds) (default: 10)"),
		cl::init(10),
		cl::cat(simulationOptions));
static cl::opt<double> timeStep(
		"time-step",
		cl::desc("Time step (in seconds) (default: 0.1)"),
		cl::init(0.1),
		cl::cat(simulationOptions));
static cl::opt<double> relativeTolerance(
		"rel-tol",
		cl::desc("Relative tolerance (default: 1e-6)"),
		cl::init(1e-6),
		cl::cat(simulationOptions));
static cl::opt<double> absoluteTolerance(
		"abs-tol",
		cl::desc("Absolute tolerance (default: 1e-6)"),
		cl::init(1e-6),
		cl::cat(simulationOptions));

static ExitOnError exitOnErr;

int main(int argc, char *argv[])
{
	SmallVector<const cl::OptionCategory *> categories;
	categories.push_back(&modelSolvingOptions);
	categories.push_back(&codeGenOptions);
	categories.push_back(&simulationOptions);
	HideUnrelatedOptions(categories);

	cl::ParseCommandLineOptions(argc, argv);

	error_code error;
	raw_fd_ostream OS(outputFile, error, sys::fs::F_None);

	if (error)
	{
		errs() << error.message();
		return -1;
	}

	SmallVector<unique_ptr<frontend::Class>, 3> classes;

	if (inputFiles.empty())
		inputFiles.addValue("-");

	for (const auto &inputFile : inputFiles)
	{
		auto errorOrBuffer = MemoryBuffer::getFileOrSTDIN(inputFile);
		auto buffer = exitOnErr(errorOrToExpected(move(errorOrBuffer)));
		frontend::Parser parser(inputFile, buffer->getBufferStart());
		auto ast = exitOnErr(parser.classDefinition());
		classes.push_back(move(ast));
	}

	// Run frontend passes
	frontend::PassManager frontendPassManager;
	frontendPassManager.addPass(frontend::createTypeCheckingPass());
	frontendPassManager.addPass(frontend::createConstantFolderPass());
	exitOnErr(frontendPassManager.run(classes));

	// Create the MLIR module
	mlir::MLIRContext context;

	codegen::ModelicaOptions modelicaOptions;
	modelicaOptions.startTime = startTime;
	modelicaOptions.endTime = endTime;
	modelicaOptions.timeStep = timeStep;
	modelicaOptions.relativeTolerance = relativeTolerance;
	modelicaOptions.absoluteTolerance = absoluteTolerance;

	codegen::MLIRLowerer lowerer(context, modelicaOptions);
	auto module = lowerer.run(classes);

	if (!module)
	{
		errs() << "Failed to emit Modelica IR\n";
		return -1;
	}

	codegen::SolveModelOptions solveModelOptions;
	solveModelOptions.matchingMaxIterations = matchingMaxIterations;
	solveModelOptions.sccMaxIterations = sccMaxIterations;
	solveModelOptions.solver = solver;

	auto model = codegen::getSolvedModel(*module, solveModelOptions);

	if (!model)
	{
		errs() << "Failed to solve Model\n";
		return -1;
	}

	if (printModule)
	{
		model->getOp()->getParentOfType<mlir::ModuleOp>()->dump();
	}

	ida::IdaSolver idaSolver(
			*model, startTime, endTime, relativeTolerance, absoluteTolerance);

	if (failed(idaSolver.init()))
	{
		errs() << "Failed to initialize the IDA solver\n";
		return -1;
	}

	if (failed(idaSolver.run(OS)))
	{
		errs() << "Failed to run the IDA solver\n";
		return -1;
	}

	if (failed(idaSolver.free()))
	{
		errs() << "Failed to correctly free the IDA solver\n";
		return -1;
	}

	return 0;
}
