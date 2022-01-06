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

  // Tweak the frontend configuration based on the frontend action
  static void setUpFrontendBasedOnAction(FrontendOptions& opts)
  {
    //assert(opts.programAction != InvalidAction && "Fortran frontend action not set!");

    /*
    if (opts.programAction == DebugDumpParsingLog)
      opts.instrumentedParse = true;

    if (opts.programAction == DebugDumpProvenance ||
        opts.programAction == Fortran::frontend::GetDefinition)
      opts.needProvenanceRangeToCharBlockMappings = true;
      */
  }

  static bool ParseFrontendArgs(
      FrontendOptions& opts, llvm::opt::ArgList& args,
      clang::DiagnosticsEngine& diags)
  {
    unsigned numErrorsBefore = diags.getNumErrors();

    // Default action
    opts.programAction = EmitBitcode;

    llvm::errs() << "ARGS\n";

    for (const auto& arg : args) {
      arg->dump();
    }

    // Treat multiple action options as an invocation error. Note that `clang
    // -cc1` does accept multiple action options, but will only consider the
    // rightmost one.
    /*
    if (args.hasMultipleArgs(clang::driver::options::OPT_Action_Group)) {
      const unsigned diagID = diags.getCustomDiagID(
          clang::DiagnosticsEngine::Error, "Only one action option is allowed");
      diags.Report(diagID);
      return false;
    }
     */

    /*
    // Identify the action (i.e. opts.ProgramAction)
    if (const llvm::opt::Arg* a =
        args.getLastArg(clang::driver::options::OPT_Action_Group)) {
      switch (a->getOption().getID()) {
        default: {
          llvm_unreachable("Invalid option in group!");
        }
        case clang::driver::options::OPT_test_io:
          opts.programAction = InputOutputTest;
          break;
        case clang::driver::options::OPT_E:
          opts.programAction = PrintPreprocessedInput;
          break;
        case clang::driver::options::OPT_fsyntax_only:
          opts.programAction = ParseSyntaxOnly;
          break;
        case clang::driver::options::OPT_emit_obj:
          opts.programAction = EmitObj;
          break;
        case clang::driver::options::OPT_fdebug_unparse:
          opts.programAction = DebugUnparse;
          break;
        case clang::driver::options::OPT_fdebug_unparse_no_sema:
          opts.programAction = DebugUnparseNoSema;
          break;
        case clang::driver::options::OPT_fdebug_unparse_with_symbols:
          opts.programAction = DebugUnparseWithSymbols;
          break;
        case clang::driver::options::OPT_fdebug_dump_symbols:
          opts.programAction = DebugDumpSymbols;
          break;
        case clang::driver::options::OPT_fdebug_dump_parse_tree:
          opts.programAction = DebugDumpParseTree;
          break;
        case clang::driver::options::OPT_fdebug_dump_all:
          opts.programAction = DebugDumpAll;
          break;
        case clang::driver::options::OPT_fdebug_dump_parse_tree_no_sema:
          opts.programAction = DebugDumpParseTreeNoSema;
          break;
        case clang::driver::options::OPT_fdebug_dump_provenance:
          opts.programAction = DebugDumpProvenance;
          break;
        case clang::driver::options::OPT_fdebug_dump_parsing_log:
          opts.programAction = DebugDumpParsingLog;
          break;
        case clang::driver::options::OPT_fdebug_measure_parse_tree:
          opts.programAction = DebugMeasureParseTree;
          break;
        case clang::driver::options::OPT_fdebug_pre_fir_tree:
          opts.programAction = DebugPreFIRTree;
          break;
        case clang::driver::options::OPT_fget_symbols_sources:
          opts.programAction = GetSymbolsSources;
          break;
        case clang::driver::options::OPT_fget_definition:
          opts.programAction = GetDefinition;
          break;
        case clang::driver::options::OPT_init_only:
          opts.programAction = InitOnly;
          break;

          // TODO:
          // case clang::driver::options::OPT_emit_llvm:
          // case clang::driver::options::OPT_emit_llvm_only:
          // case clang::driver::options::OPT_emit_codegen_only:
          // case clang::driver::options::OPT_emit_module:
          // (...)
      }

      // Parse the values provided with `-fget-definition` (there should be 3
      // integers)
      if (llvm::opt::OptSpecifier(a->getOption().getID()) ==
          clang::driver::options::OPT_fget_definition) {
        unsigned optVals[3] = {0, 0, 0};

        for (unsigned i = 0; i < 3; i++) {
          llvm::StringRef val = a->getValue(i);

          if (val.getAsInteger(10, optVals[i])) {
            // A non-integer was encountered - that's an error.
            diags.Report(clang::diag::err_drv_invalid_value)
                << a->getOption().getName() << val;
            break;
          }
        }
        opts.getDefVals.line = optVals[0];
        opts.getDefVals.startColumn = optVals[1];
        opts.getDefVals.endColumn = optVals[2];
      }
    }
  */

    /*
    opts.outputFile = args.getLastArgValue(clang::driver::options::OPT_o);
    opts.showHelp = args.hasArg(clang::driver::options::OPT_help);
    opts.showVersion = args.hasArg(clang::driver::options::OPT_version);
     */

    // Get the input kind (from the value passed via `-x`)
    InputKind dashX(Language::Unknown);

    /*
    if (const llvm::opt::Arg* a =
        args.getLastArg(clang::driver::options::OPT_x)) {
      llvm::StringRef XValue = a->getValue();
      // Principal languages.
      dashX = llvm::StringSwitch<InputKind>(XValue)
          .Case("f90", Language::Fortran)
          .Default(Language::Unknown);

      // Some special cases cannot be combined with suffixes.
      if (dashX.IsUnknown()) {
        dashX = llvm::StringSwitch<InputKind>(XValue)
            .Case("ir", Language::LLVM_IR)
            .Default(Language::Unknown);
      }

      if (dashX.IsUnknown()) {
        diags.Report(clang::diag::err_drv_invalid_value)
            << a->getAsString(args) << a->getValue();
      }
    }
     */

    // Collect the input files and save them in our instance of FrontendOptions.
    std::vector<std::string> inputs =
        args.getAllArgValues(clang::driver::options::OPT_INPUT);
    opts.inputs.clear();
    if (inputs.empty()) {
      // '-' is the default input if none is given.
      inputs.push_back("-");
    }
    for (unsigned i = 0, e = inputs.size(); i != e; ++i) {
      InputKind ik = dashX;
      if (ik.IsUnknown()) {
        ik = FrontendOptions::GetInputKindForExtension(
            llvm::StringRef(inputs[i]).rsplit('.').second);
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

  // Generate the path to look for intrinsic modules
  static std::string getIntrinsicDir()
  {
    // TODO: Find a system independent API
    llvm::SmallString<128> driverPath;
    driverPath.assign(llvm::sys::fs::getMainExecutable(nullptr, nullptr));
    llvm::sys::path::remove_filename(driverPath);
    driverPath.append("/../include/flang/");
    return std::string(driverPath);
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

  /// Parses all Dialect related arguments and populates the variables
  /// options accordingly. Returns false if new errors are generated.
  static bool parseDialectArgs(
      CompilerInvocation& res, llvm::opt::ArgList& args,
      clang::DiagnosticsEngine& diags)
  {
    unsigned numErrorsBefore = diags.getNumErrors();

    /*
    // -fdefault* family
    if (args.hasArg(clang::driver::options::OPT_fdefault_real_8)) {
      res.defaultKinds().set_defaultRealKind(8);
      res.defaultKinds().set_doublePrecisionKind(16);
    }
    if (args.hasArg(clang::driver::options::OPT_fdefault_integer_8)) {
      res.defaultKinds().set_defaultIntegerKind(8);
      res.defaultKinds().set_subscriptIntegerKind(8);
      res.defaultKinds().set_sizeIntegerKind(8);
    }
    if (args.hasArg(clang::driver::options::OPT_fdefault_double_8)) {
      if (!args.hasArg(clang::driver::options::OPT_fdefault_real_8)) {
        // -fdefault-double-8 has to be used with -fdefault-real-8
        // to be compatible with gfortran
        const unsigned diagID =
            diags.getCustomDiagID(clang::DiagnosticsEngine::Error,
                "Use of `-fdefault-double-8` requires `-fdefault-real-8`");
        diags.Report(diagID);
      }
      // https://gcc.gnu.org/onlinedocs/gfortran/Fortran-Dialect-Options.html
      res.defaultKinds().set_doublePrecisionKind(8);
    }
    if (args.hasArg(clang::driver::options::OPT_flarge_sizes))
      res.defaultKinds().set_sizeIntegerKind(8);

    // -fopenmp and -fopenacc
    if (args.hasArg(clang::driver::options::OPT_fopenacc)) {
      res.frontendOpts().features.Enable(
          Fortran::common::LanguageFeature::OpenACC);
    }
    if (args.hasArg(clang::driver::options::OPT_fopenmp)) {
      res.frontendOpts().features.Enable(
          Fortran::common::LanguageFeature::OpenMP);
    }

    // -pedantic
    if (args.hasArg(clang::driver::options::OPT_pedantic)) {
      res.set_EnableConformanceChecks();
    }
    // -std=f2018 (currently this implies -pedantic)
    // TODO: Set proper options when more fortran standards
    // are supported.
    if (args.hasArg(clang::driver::options::OPT_std_EQ)) {
      auto standard = args.getLastArgValue(clang::driver::options::OPT_std_EQ);
      // We only allow f2018 as the given standard
      if (standard.equals("f2018")) {
        res.set_EnableConformanceChecks();
      } else {
        const unsigned diagID =
            diags.getCustomDiagID(clang::DiagnosticsEngine::Error,
                "Only -std=f2018 is allowed currently.");
        diags.Report(diagID);
      }
    }
     */

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

    unsigned missingArgIndex, missingArgCount;
    llvm::opt::InputArgList args = opts.ParseArgs(commandLineArgs, missingArgIndex, missingArgCount);

    /*
    // Check for missing argument error.
    if (missingArgCount) {
      diagnosticEngine.Report(clang::diag::err_drv_missing_argument)
          << args.getArgString(missingArgIndex) << missingArgCount;
      success = false;
    }
     */

    success &= ParseFrontendArgs(res.frontendOpts(), args, diagnosticEngine);

    return success;
  }

  void CompilerInvocation::SetDefaultFortranOpts()
  {
    /*
    auto& fortranOptions = fortranOpts();

    std::vector<std::string> searchDirectories{"."s};
    fortranOptions.searchDirectories = searchDirectories;
    fortranOptions.isFixedForm = false;
     */
  }

  /*
  // TODO: When expanding this method, consider creating a dedicated API for
  // this. Also at some point we will need to differentiate between different
  // targets and add dedicated predefines for each.
  void CompilerInvocation::setDefaultPredefinitions()
  {
    auto& fortranOptions = fortranOpts();
    const auto& frontendOptions = frontendOpts();

    // Populate the macro list with version numbers and other predefinitions.
    fortranOptions.predefinitions.emplace_back("__flang__", "1");
    fortranOptions.predefinitions.emplace_back(
        "__flang_major__", FLANG_VERSION_MAJOR_STRING);
    fortranOptions.predefinitions.emplace_back(
        "__flang_minor__", FLANG_VERSION_MINOR_STRING);
    fortranOptions.predefinitions.emplace_back(
        "__flang_patchlevel__", FLANG_VERSION_PATCHLEVEL_STRING);

    // Add predefinitions based on extensions enabled
    if (frontendOptions.features.IsEnabled(
        Fortran::common::LanguageFeature::OpenACC)) {
      fortranOptions.predefinitions.emplace_back("_OPENACC", "202011");
    }
    if (frontendOptions.features.IsEnabled(
        Fortran::common::LanguageFeature::OpenMP)) {
      fortranOptions.predefinitions.emplace_back("_OPENMP", "201511");
    }
  }
   */

  void CompilerInvocation::setFortranOpts()
  {
    /*
    auto& fortranOptions = fortranOpts();
    const auto& frontendOptions = frontendOpts();
    const auto& preprocessorOptions = preprocessorOpts();
    auto& moduleDirJ = moduleDir();

    if (frontendOptions.fortranForm != FortranForm::Unknown) {
      fortranOptions.isFixedForm =
          frontendOptions.fortranForm == FortranForm::FixedForm;
    }
    fortranOptions.fixedFormColumns = frontendOptions.fixedFormColumns;

    fortranOptions.features = frontendOptions.features;
    fortranOptions.encoding = frontendOptions.encoding;

    // Adding search directories specified by -I
    fortranOptions.searchDirectories.insert(
        fortranOptions.searchDirectories.end(),
        preprocessorOptions.searchDirectoriesFromDashI.begin(),
        preprocessorOptions.searchDirectoriesFromDashI.end());

    // Add the ordered list of -intrinsic-modules-path
    fortranOptions.searchDirectories.insert(
        fortranOptions.searchDirectories.end(),
        preprocessorOptions.searchDirectoriesFromIntrModPath.begin(),
        preprocessorOptions.searchDirectoriesFromIntrModPath.end());

    //  Add the default intrinsic module directory at the end
    fortranOptions.searchDirectories.emplace_back(getIntrinsicDir());

    // Add the directory supplied through -J/-module-dir to the list of search
    // directories
    if (moduleDirJ.compare(".") != 0) {
      fortranOptions.searchDirectories.emplace_back(moduleDirJ);
    }

    if (frontendOptions.instrumentedParse) {
      fortranOptions.instrumentedParse = true;
    }

    if (frontendOptions.needProvenanceRangeToCharBlockMappings) {
      fortranOptions.needProvenanceRangeToCharBlockMappings = true;
    }

    if (enableConformanceChecks()) {
      fortranOptions.features.WarnOnAllNonstandard();
    }
     */
  }
}