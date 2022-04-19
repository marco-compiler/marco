#include "clang/Basic/AllDiagnostics.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Basic/DiagnosticOptions.h"
#include "clang/Driver/DriverDiagnostic.h"
#include "clang/Driver/Options.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/raw_ostream.h"
#include "marco/Frontend/CompilerInvocation.h"
#include "marco/Frontend/Options.h"
#include <memory>

namespace marco::frontend
{
  CompilerInvocationBase::CompilerInvocationBase()
      : diagnosticOpts_(new clang::DiagnosticOptions())
  {
  }

  CompilerInvocationBase::CompilerInvocationBase(const CompilerInvocationBase& x)
      : diagnosticOpts_(new clang::DiagnosticOptions(x.GetDiagnosticOpts()))
  {
  }

  CompilerInvocationBase::~CompilerInvocationBase() = default;

  // Tweak the frontend configuration based on the frontend action
  static void setUpFrontendBasedOnAction(FrontendOptions& options)
  {
    assert(options.programAction != InvalidAction && "Frontend action not set!");
  }

  static bool parseFrontendArgs(
      FrontendOptions& options, llvm::opt::ArgList& args, clang::DiagnosticsEngine& diags)
  {
    unsigned numErrorsBefore = diags.getNumErrors();

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
      llvm::StringRef value = arg->getValue();
      auto variableFilter = VariableFilter::fromString(value);

      if (!variableFilter) {
        unsigned int diagID = diags.getCustomDiagID(
            clang::DiagnosticsEngine::Warning,
            "Invalid variable filter string. No filtering will take place");

        diags.Report(diagID);

      } else {
        options.variableFilter = *variableFilter;
      }
    }

    return diags.getNumErrors() == numErrorsBefore;
  }

  static bool parseCodegenArgs(
      CodegenOptions& options, llvm::opt::ArgList& args, clang::DiagnosticsEngine& diags)
  {
    unsigned numErrorsBefore = diags.getNumErrors();

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
        unsigned int diagID = diags.getCustomDiagID(
            clang::DiagnosticsEngine::Warning,
            "Unknown optimization option: %s");

        diags.Report(diagID) << arg;
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

    options.generateMain = args.hasFlag(
        marco::frontend::options::OPT_generate_main,
        options::OPT_no_generate_main,
        options.generateMain);

    return diags.getNumErrors() == numErrorsBefore;
  }

  static bool parseSimulationArgs(
      SimulationOptions& options, llvm::opt::ArgList& args, clang::DiagnosticsEngine& diags)
  {
    unsigned numErrorsBefore = diags.getNumErrors();

    if (const llvm::opt::Arg* arg = args.getLastArg(options::OPT_model)) {
      llvm::StringRef value = arg->getValue();
      options.modelName = value.str();
    }

    if (const llvm::opt::Arg* arg = args.getLastArg(options::OPT_start_time)) {
      llvm::StringRef value = arg->getValue();
      llvm::APFloat numericValue(llvm::APFloatBase::IEEEdouble(), value);
      options.startTime = numericValue.convertToDouble();
    }

    if (const llvm::opt::Arg* arg = args.getLastArg(options::OPT_start_time)) {
      llvm::StringRef value = arg->getValue();
      llvm::APFloat numericValue(llvm::APFloatBase::IEEEdouble(), value);
      options.startTime = numericValue.convertToDouble();
    }

    if (const llvm::opt::Arg* arg = args.getLastArg(options::OPT_end_time)) {
      llvm::StringRef value = arg->getValue();
      llvm::APFloat numericValue(llvm::APFloatBase::IEEEdouble(), value);
      options.endTime = numericValue.convertToDouble();
    }

    if (const llvm::opt::Arg* arg = args.getLastArg(options::OPT_time_step)) {
      llvm::StringRef value = arg->getValue();
      llvm::APFloat numericValue(llvm::APFloatBase::IEEEdouble(), value);
      options.timeStep = numericValue.convertToDouble();
    }

    // Determine the solver to be used
    for (const auto& arg : args.getAllArgValues(options::OPT_solver)) {
      if (arg == "forward-euler") {
        options.solver = marco::codegen::Solver::forwardEuler;
      } else if (arg == "ida") {
        options.solver = marco::codegen::Solver::ida;
      } else {
        unsigned int diagID = diags.getCustomDiagID(
            clang::DiagnosticsEngine::Warning,
            "Unknown solver option: %s");

        diags.Report(diagID) << arg;
      }
    }

    // IDA options

    if (const llvm::opt::Arg* arg = args.getLastArg(options::OPT_ida_relative_tolerance)) {
      llvm::StringRef value = arg->getValue();
      llvm::APFloat numericValue(llvm::APFloatBase::IEEEdouble(), value);
      options.ida.relativeTolerance = numericValue.convertToDouble();
    }

    if (const llvm::opt::Arg* arg = args.getLastArg(options::OPT_ida_absolute_tolerance)) {
      llvm::StringRef value = arg->getValue();
      llvm::APFloat numericValue(llvm::APFloatBase::IEEEdouble(), value);
      options.ida.absoluteTolerance = numericValue.convertToDouble();
    }

    options.ida.equidistantTimeGrid = args.hasArg(options::OPT_ida_equidistant_time_grid);

    return diags.getNumErrors() == numErrorsBefore;
  }

  bool CompilerInvocation::createFromArgs(
      CompilerInvocation& res,
      llvm::ArrayRef<const char*> commandLineArgs,
      clang::DiagnosticsEngine& diagnosticEngine)
  {
    bool success = true;

    // Parse the arguments
    const llvm::opt::OptTable& opts = marco::frontend::getDriverOptTable();

    const unsigned includedFlagsBitmask = marco::frontend::options::MC1Option;
    unsigned missingArgIndex, missingArgCount;

    llvm::opt::InputArgList args = opts.ParseArgs(
        commandLineArgs, missingArgIndex, missingArgCount, includedFlagsBitmask);

    // Check for missing argument error
    if (missingArgCount) {
      diagnosticEngine.Report(clang::diag::err_drv_missing_argument)
          << args.getArgString(missingArgIndex)
          << missingArgCount;

      success = false;
    }

    // Issue errors on unknown arguments
    for (const auto* a: args.filtered(options::OPT_UNKNOWN)) {
      auto argString = a->getAsString(args);
      diagnosticEngine.Report(clang::diag::err_drv_unknown_argument) << argString;
      success = false;
    }

    success &= parseFrontendArgs(res.frontendOptions(), args, diagnosticEngine);
    success &= parseCodegenArgs(res.codegenOptions(), args, diagnosticEngine);
    success &= parseSimulationArgs(res.simulationOptions(), args, diagnosticEngine);

    return success;
  }
}