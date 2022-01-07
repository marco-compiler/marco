#include <marco/frontend/CompilerInvocation.h>
#include <clang/Basic/AllDiagnostics.h>
#include <clang/Basic/DiagnosticDriver.h>
#include <clang/Basic/DiagnosticOptions.h>
#include <clang/Driver/DriverDiagnostic.h>
#include <clang/Driver/Options.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Option/Arg.h>
#include <llvm/Option/ArgList.h>
#include <llvm/Option/OptTable.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/FileUtilities.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/raw_ostream.h>
#include <marco/frontend/Options.h>
#include <memory>

namespace marco::frontend
{
  //===----------------------------------------------------------------------===//
  // Initialization.
  //===----------------------------------------------------------------------===//
  CompilerInvocationBase::CompilerInvocationBase()
      : diagnosticOpts_(new clang::DiagnosticOptions()) {}

  CompilerInvocationBase::CompilerInvocationBase(const CompilerInvocationBase& x)
      : diagnosticOpts_(new clang::DiagnosticOptions(x.GetDiagnosticOpts())) {}

  CompilerInvocationBase::~CompilerInvocationBase() = default;

  //===----------------------------------------------------------------------===//
  // Deserialization (from args)
  //===----------------------------------------------------------------------===//
  /*
  static bool parseShowColorsArgs(
      const llvm::opt::ArgList& args, bool defaultColor)
  {
    // Color diagnostics default to auto ("on" if terminal supports) in the driver
    // but default to off in cc1, needing an explicit OPT_fdiagnostics_color.
    // Support both clang's -f[no-]color-diagnostics and gcc's
    // -f[no-]diagnostics-colors[=never|always|auto].
    enum
    {
      Colors_On,
      Colors_Off,
      Colors_Auto
    } ShowColors = defaultColor ? Colors_Auto : Colors_Off;

    for (auto* a: args) {
      const llvm::opt::Option& O = a->getOption();
      if (O.matches(clang::driver::options::OPT_fcolor_diagnostics) ||
          O.matches(clang::driver::options::OPT_fdiagnostics_color)) {
        ShowColors = Colors_On;
      } else if (O.matches(clang::driver::options::OPT_fno_color_diagnostics) ||
          O.matches(clang::driver::options::OPT_fno_diagnostics_color)) {
        ShowColors = Colors_Off;
      } else if (O.matches(clang::driver::options::OPT_fdiagnostics_color_EQ)) {
        llvm::StringRef value(a->getValue());
        if (value == "always") {
          ShowColors = Colors_On;
        } else if (value == "never") {
          ShowColors = Colors_Off;
        } else if (value == "auto") {
          ShowColors = Colors_Auto;
        }
      }
    }

    return ShowColors == Colors_On ||
        (ShowColors == Colors_Auto && llvm::sys::Process::StandardErrHasColors());
  }

  bool ParseDiagnosticArgs(
      clang::DiagnosticOptions& opts,
      llvm::opt::ArgList& args, bool defaultDiagColor)
  {
    opts.ShowColors = parseShowColorsArgs(args, defaultDiagColor);

    return true;
  }
  */

  // Tweak the frontend configuration based on the frontend action
  static void setUpFrontendBasedOnAction(FrontendOptions& opts)
  {
    assert(opts.programAction != InvalidAction && "Frontend action not set!");
  }

  static bool ParseFrontendArgs(
      FrontendOptions& opts, llvm::opt::ArgList& args,
      clang::DiagnosticsEngine& diags)
  {
    unsigned numErrorsBefore = diags.getNumErrors();

    // Default action
    opts.programAction = EmitBitcode;

    if (const llvm::opt::Arg* a = args.getLastArg(marco::frontend::options::OPT_Action_Group)) {
      switch (a->getOption().getID()) {
        default: {
          llvm_unreachable("Invalid option in group!");
        }

        case marco::frontend::options::OPT_init_only:
          opts.programAction = InitOnly;
          break;

        case marco::frontend::options::OPT_emit_ast:
          opts.programAction = EmitAST;
          break;

        case marco::frontend::options::OPT_emit_modelica_dialect:
          opts.programAction = EmitModelicaDialect;
          break;

        case marco::frontend::options::OPT_emit_llvm_dialect:
          opts.programAction = EmitLLVMDialect;
          break;

        case marco::frontend::options::OPT_emit_llvm_ir:
          opts.programAction = EmitLLVMIR;
          break;

        case marco::frontend::options::OPT_emit_bitcode:
          opts.programAction = EmitBitcode;
          break;
      }
    }

    opts.outputFile = args.getLastArgValue(marco::frontend::options::OPT_o);
    opts.showHelp = args.hasArg(marco::frontend::options::OPT_help);
    opts.showVersion = args.hasArg(marco::frontend::options::OPT_version);

    // Get the input kind (from the value passed via `-x`)
    InputKind dashX(Language::Unknown);

    if (const llvm::opt::Arg* a = args.getLastArg(marco::frontend::options::OPT_x)) {
      llvm::StringRef XValue = a->getValue();

      // Principal languages.
      dashX = llvm::StringSwitch<InputKind>(XValue)
          .Case("mo", Language::Modelica)
          .Case("mlir", Language::MLIR)
          .Default(Language::Unknown);

      if (dashX.IsUnknown()) {
        diags.Report(clang::diag::err_drv_invalid_value) << a->getAsString(args) << a->getValue();
      }
    }

    // Collect the input files and save them in our instance of FrontendOptions
    std::vector<std::string> inputs = args.getAllArgValues(marco::frontend::options::OPT_INPUT);
    opts.inputs.clear();

    if (inputs.empty()) {
      // '-' is the default input if none is given
      inputs.push_back("-");
    }

    for (unsigned i = 0, e = inputs.size(); i != e; ++i) {
      InputKind ik = dashX;

      if (ik.IsUnknown()) {
        ik = FrontendOptions::GetInputKindForExtension(llvm::StringRef(inputs[i]).rsplit('.').second);

        if (ik.IsUnknown()) {
          ik = Language::Unknown;
        }
        if (i == 0) {
          dashX = ik;
        }
      }

      opts.inputs.emplace_back(std::move(inputs[i]), ik);
    }

    /*
    if (const llvm::opt::Arg* arg =
        args.getLastArg(clang::driver::options::OPT_finput_charset_EQ)) {
      llvm::StringRef argValue = arg->getValue();
      if (argValue == "utf-8") {
        opts.encoding = Fortran::parser::Encoding::UTF_8;
      } else if (argValue == "latin-1") {
        opts.encoding = Fortran::parser::Encoding::LATIN_1;
      } else {
        diags.Report(clang::diag::err_drv_invalid_value)
            << arg->getAsString(args) << argValue;
      }
    }
     */

    setUpFrontendBasedOnAction(opts);
    //opts.dashX = dashX;

    return diags.getNumErrors() == numErrorsBefore;
  }

  static bool ParseDialectArgs(
      DialectOptions& opts, llvm::opt::ArgList& args,
      clang::DiagnosticsEngine& diags)
  {
    unsigned numErrorsBefore = diags.getNumErrors();

    // TODO

    return diags.getNumErrors() == numErrorsBefore;
  }

  static bool ParseCodegenArgs(
      CodegenOptions& opts, llvm::opt::ArgList& args,
      clang::DiagnosticsEngine& diags)
  {
    unsigned numErrorsBefore = diags.getNumErrors();

    // TODO

    return diags.getNumErrors() == numErrorsBefore;
  }

  /// Parses all diagnostics related arguments and populates the variables
  /// options accordingly. Returns false if new errors are generated.
  static bool parseDiagArgs(
      CompilerInvocation& res, llvm::opt::ArgList& args,
      clang::DiagnosticsEngine& diags)
  {
    unsigned numErrorsBefore = diags.getNumErrors();

    return diags.getNumErrors() == numErrorsBefore;
  }

  bool CompilerInvocation::CreateFromArgs(
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

    // Check for missing argument error.
    if (missingArgCount) {
      diagnosticEngine.Report(clang::diag::err_drv_missing_argument) << args.getArgString(missingArgIndex) << missingArgCount;
      success = false;
    }

    // Issue errors on unknown arguments
    for (const auto *a : args.filtered(clang::driver::options::OPT_UNKNOWN)) {
      auto argString = a->getAsString(args);
      diagnosticEngine.Report(clang::diag::err_drv_unknown_argument) << argString;
      success = false;
    }

    success &= ParseFrontendArgs(res.frontendOptions(), args, diagnosticEngine);
    success &= ParseDialectArgs(res.dialectOptions(), args, diagnosticEngine);
    success &= ParseCodegenArgs(res.codegenOptions(), args, diagnosticEngine);

    return success;
  }
}