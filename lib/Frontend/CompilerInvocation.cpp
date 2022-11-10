#include "marco/Diagnostic/Printer.h"
#include "marco/Frontend/CompilerInvocation.h"
#include "marco/Frontend/Options.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/Process.h"
#include <memory>

using namespace ::marco;
using namespace ::marco::diagnostic;
using namespace ::marco::frontend;

//===----------------------------------------------------------------------===//
// Messages
//===----------------------------------------------------------------------===//

namespace
{
  class UnknownOptimizationOptionMessage : public Message
  {
    public:
      UnknownOptimizationOptionMessage(llvm::StringRef option)
        : option(option.str())
      {
      }

      void print(PrinterInstance* printer) const override
      {
        auto& os = printer->getOutputStream();
        os << "Unknown optimization option '" << option << "'\n";
      }

    private:
      std::string option;
  };

  class MissingArgumentValueMessage : public Message
  {
    public:
      MissingArgumentValueMessage(llvm::StringRef argName)
        : argName(argName.str())
      {
      }

      void print(PrinterInstance* printer) const override
      {
        auto& os = printer->getOutputStream();
        os << "Missing value for argument '" << argName << "'\n";
      }

    private:
      std::string argName;
  };

  class UnknownSolverMessage : public Message
  {
    public:
    UnknownSolverMessage(llvm::StringRef solver)
        : solver(solver.str())
    {
    }

    void print(PrinterInstance* printer) const override
    {
      auto& os = printer->getOutputStream();
      os << "Unknown solver '" << solver << "'\n";
    }

    private:
    std::string solver;
  };

  class UnknownArgumentMessage : public Message
  {
    public:
      UnknownArgumentMessage(llvm::StringRef argument)
          : argument(argument.str())
      {
      }

      void print(PrinterInstance* printer) const override
      {
        auto& os = printer->getOutputStream();
        os << "Unknown argument '" << argument << "'\n";
      }

    private:
      std::string argument;
  };
}

//===----------------------------------------------------------------------===//
// CompilerInvocation
//===----------------------------------------------------------------------===//

/// Tweak the frontend configuration based on the frontend action
static void setUpFrontendBasedOnAction(FrontendOptions& options)
{
  assert(options.programAction != InvalidAction && "Frontend action not set!");
}

static void parseFrontendArgs(
    FrontendOptions& options, llvm::opt::ArgList& args, DiagnosticEngine& diagnostics)
{
  // Default action
  options.programAction = EmitObject;

  if (const llvm::opt::Arg* a = args.getLastArg(options::OPT_Action_Group)) {
    switch (a->getOption().getID()) {
      case options::OPT_init_only:
        options.programAction = InitOnly;
        break;

      case options::OPT_emit_flattened:
        options.programAction = EmitFlattened;
        break;

      case options::OPT_emit_ast:
        options.programAction = EmitAST;
        break;

      case options::OPT_emit_final_ast:
        options.programAction = EmitFinalAST;
        break;

      case options::OPT_emit_modelica_dialect:
        options.programAction = EmitModelicaDialect;
        break;

      case options::OPT_emit_llvm_dialect:
        options.programAction = EmitLLVMDialect;
        break;

      case options::OPT_emit_llvm_ir:
        options.programAction = EmitLLVMIR;
        break;

      case options::OPT_compile_only:
        options.programAction = EmitAssembly;
        break;

      case options::OPT_compile_and_assemble_only:
        options.programAction = EmitObject;
        break;

      default: {
        llvm_unreachable("Invalid option in group!");
      }
    }
  }

  options.outputFile = args.getLastArgValue(options::OPT_o);
  options.showHelp = args.hasArg(options::OPT_help);
  options.showVersion = args.hasArg(options::OPT_version);
  options.printStatistics = args.hasArg(options::OPT_print_statistics);

  // Collect the input files and save them in our instance of FrontendOptions
  std::vector<std::string> inputs = args.getAllArgValues(options::OPT_INPUT);
  options.inputs.clear();

  if (inputs.empty()) {
    // '-' is the default input if none is given
    inputs.push_back("-");
  }

  for (size_t i = 0, e = inputs.size(); i != e; ++i) {
    // TODO: expand to handle multiple input types
    options.inputs.emplace_back(std::move(inputs[i]), InputKind(Language::Modelica));
  }

  setUpFrontendBasedOnAction(options);

  if (const llvm::opt::Arg* arg = args.getLastArg(options::OPT_omc_path)) {
    llvm::StringRef value = arg->getValue();
    options.omcPath = value.str();
  }

  for (const auto& omcArg : args.getAllArgValues(options::OPT_omc_arg)) {
    options.omcCustomArgs.push_back(omcArg);
  }

  options.omcBypass = args.hasArg(options::OPT_omc_bypass);

  if (const llvm::opt::Arg* arg = args.getLastArg(options::OPT_filter)) {
    options.variablesFilter = arg->getValue();
  }
}

static void parseCodegenArgs(
    CodegenOptions& options, llvm::opt::ArgList& args, DiagnosticEngine& diagnostics)
{
  // Determine the optimization level
  for (const auto& arg : args.getAllArgValues(options::OPT_opt)) {
    if (arg == "0") {
      options.optLevel.time = 0;
    } else if (arg == "1") {
      options.optLevel.time = 1;
    } else if (arg == "2") {
      options.optLevel.time = 2;
    } else if (arg == "3") {
      options.optLevel.time = 3;
    } else if (arg == "fast") {
      options.optLevel.time = 3;
    } else if (arg == "s") {
      options.optLevel.size = 1;
    } else if (arg == "z") {
      options.optLevel.size = 2;
    } else {
      // "Unknown optimization option: %s"
      diagnostics.emitWarning<UnknownOptimizationOptionMessage>(arg);
    }
  }

  // Set the default options based on the optimization level
  if (options.optLevel.time > 0) {
    options.debug = false;
    options.inlining = true;
    options.outputArraysPromotion = true;
    options.cse = true;
  }

  if (options.optLevel.time > 1) {
    options.assertions = false;
  }

  if (options.optLevel.size > 0) {
    options.debug = false;
    options.cse = true;
  }

  if (options.optLevel.size > 1) {
    options.assertions = false;
  }

  // Continue in processing the user-provided options, which may override the default
  // options given by the optimization level.

  if (!options.debug) {
    options.debug = args.hasArg(marco::frontend::options::OPT_debug);
  }

  options.assertions = args.hasFlag(
      marco::frontend::options::OPT_assertions,
      options::OPT_no_assertions,
      options.assertions);

  options.inlining = args.hasFlag(
      marco::frontend::options::OPT_function_inlining,
      options::OPT_no_function_inlining,
      options.inlining);

  options.outputArraysPromotion = args.hasFlag(
      options::OPT_output_arrays_promotion,
      options::OPT_no_output_arrays_promotion,
      options.outputArraysPromotion);

  options.cse = args.hasFlag(
      marco::frontend::options::OPT_cse,
      options::OPT_no_cse,
      options.cse);

  options.omp = args.hasFlag(
      marco::frontend::options::OPT_omp,
      options::OPT_no_omp,
      options.omp);

  if (const llvm::opt::Arg* arg = args.getLastArg(options::OPT_bit_width)) {
    llvm::StringRef value = arg->getValue();
    llvm::APSInt numericValue(value);
    options.bitWidth = numericValue.getSExtValue();
  }

  // Cross-compilation options

  options.generateMain = args.hasFlag(
      marco::frontend::options::OPT_generate_main,
      options::OPT_no_generate_main,
      options.generateMain);

  if (const llvm::opt::Arg* arg = args.getLastArg(options::OPT_target)) {
    llvm::StringRef value = arg->getValue();
    options.target = value.str();
  } else {
    // Default: native compilation
    options.target = llvm::sys::getDefaultTargetTriple();
  }

  if (const llvm::opt::Arg* arg = args.getLastArg(options::OPT_cpu)) {
    llvm::StringRef value = arg->getValue();
    options.cpu = value.str();

    if (options.cpu == "native") {
      // Get the host CPU name
      options.cpu = llvm::sys::getHostCPUName().str();
    }
  } else {
    // Default: native compilation
    options.cpu = llvm::sys::getHostCPUName().str();
  }
}

static void parseSimulationArgs(
    SimulationOptions& options, llvm::opt::ArgList& args, DiagnosticEngine& diagnostics)
{
  if (const llvm::opt::Arg* arg = args.getLastArg(options::OPT_model)) {
    llvm::StringRef value = arg->getValue();
    options.modelName = value.str();
  }

  // Determine the solver to be used
  for (const auto& arg : args.getAllArgValues(options::OPT_solver)) {
    if (arg == "forward-euler") {
      options.solver = codegen::Solver::forwardEuler();
    } else if (arg == "ida") {
      options.solver = codegen::Solver::ida();
    } else {
      diagnostics.emitError<UnknownSolverMessage>(arg);
    }
  }
}

namespace marco::frontend
{
  bool CompilerInvocation::createFromArgs(
      CompilerInvocation& res,
      llvm::ArrayRef<const char*> commandLineArgs,
      diagnostic::DiagnosticEngine& diagnostics)
  {
    auto numOfErrors = diagnostics.numOfErrors();

    // Parse the arguments
    const llvm::opt::OptTable& opts = marco::frontend::getDriverOptTable();

    const unsigned includedFlagsBitmask = marco::frontend::options::MC1Option;
    unsigned missingArgIndex, missingArgCount;

    llvm::opt::InputArgList args = opts.ParseArgs(
        commandLineArgs, missingArgIndex, missingArgCount, includedFlagsBitmask);

    // Check for missing argument error
    if (missingArgCount != 0) {
      diagnostics.emitError<MissingArgumentValueMessage>(args.getArgString(missingArgIndex));
    }

    // Issue errors on unknown arguments
    for (const auto* a: args.filtered(options::OPT_UNKNOWN)) {
      auto argString = a->getAsString(args);
      diagnostics.emitWarning<UnknownArgumentMessage>(argString);
    }

    parseFrontendArgs(res.frontendOptions(), args, diagnostics);
    parseCodegenArgs(res.codegenOptions(), args, diagnostics);
    parseSimulationArgs(res.simulationOptions(), args, diagnostics);

    return numOfErrors == diagnostics.numOfErrors();
  }

  FrontendOptions& CompilerInvocation::frontendOptions()
  {
    return frontendOptions_;
  }

  const FrontendOptions& CompilerInvocation::frontendOptions() const
  {
    return frontendOptions_;
  }

  CodegenOptions& CompilerInvocation::codegenOptions()
  {
    return codegenOptions_;
  }

  const CodegenOptions& CompilerInvocation::codegenOptions() const
  {
    return codegenOptions_;
  }

  SimulationOptions& CompilerInvocation::simulationOptions()
  {
    return simulationOptions_;
  }

  const SimulationOptions& CompilerInvocation::simulationOptions() const
  {
    return simulationOptions_;
  }
}
