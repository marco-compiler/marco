#include "marco/Driver/Driver.h"
#include "marco/IO/Command.h"
#include "marco/IO/InputFile.h"
#include "marco/Options/Options.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Program.h"

#define VALUE(string) #string
#define TO_LITERAL(string) VALUE(string)

#define LINKER_STR TO_LITERAL(LINKER)
#define LIBS_PATH_STR TO_LITERAL(LIBS_PATH)
#define DEPENDENCIES_LIBS_PATH_STR TO_LITERAL(DEPENDENCIES_LIBS_PATH)
#define LINK_EXTRA_FLAGS_STR TO_LITERAL(LINK_EXTRA_FLAGS)

using namespace ::marco::driver;
using namespace ::marco::io;

namespace
{
  enum class Solver
  {
    EulerForward,
    IDA,
    Unknown
  };
}

namespace marco::driver
{
  Driver::Driver(llvm::StringRef executablePath)
      : executablePath(executablePath.str())
  {
  }

  int Driver::run(llvm::ArrayRef<const char*> argv) const
  {
    // Parse the arguments
    const llvm::opt::OptTable& opts = options::getDriverOptTable();
    unsigned missingArgIndex, missingArgCount;

    llvm::opt::InputArgList args = opts.ParseArgs(
        argv, missingArgIndex, missingArgCount);

    // Process the 'help' command.
    if (args.hasArg(options::OPT_help)) {
      opts.printHelp(
          llvm::outs(),
          "marco [options] input-files", "MLIR Modelica compiler",
          options::DriverOption,
          llvm::opt::DriverFlag::HelpHidden,
          false);

      return EXIT_SUCCESS;
    }

    // Process the 'version' command.
    if (args.hasArg(options::OPT_version)) {
      llvm::outs() << "MARCO - Modelica Advanced Research COmpiler\n";
      llvm::outs() << "Website: https://github.com/modelica-polimi/marco\n";
      return EXIT_SUCCESS;
    }

    // Get the input files.
    std::vector<std::string> inputFiles =
        args.getAllArgValues(options::OPT_INPUT);

    // Get the output file name and remove it from the arguments, so that it
    // does not leak into other tools.
    auto outputFileName = args.getLastArgValue(options::OPT_o, "-");

    // Get the files to be passed to the frontend and forward the others.
    llvm::SmallVector<InputFile> frontendInputFiles;
    llvm::SmallVector<InputFile> frontendOutputFiles;

    for (size_t i = 0, e = inputFiles.size(); i != e; ++i) {
      InputKind kind = InputKind::getFromFullFileName(inputFiles[i]);
      Language language = kind.getLanguage();

      if (language == Language::Modelica ||
          language == Language::MLIR ||
          language == Language::LLVM_IR) {
        frontendInputFiles.emplace_back(inputFiles[i], kind);
      } else {
        frontendOutputFiles.emplace_back(inputFiles[i], kind);
      }
    }

    // Determine if the link action has to be performed.
    bool shouldLink = !args.hasArg(
        options::OPT_compile_only,
        options::OPT_compile_and_assemble_only,
        options::OPT_init_only,
        options::OPT_emit_flattened,
        options::OPT_emit_ast,
        options::OPT_emit_mlir,
        options::OPT_emit_llvm_ir,
        options::OPT_emit_llvm_bitcode);

    // Compute the name of the file produced by the frontend.
    std::string frontendOutputFileName;

    if (outputFileName == "-") {
      if (shouldLink) {
        // No name specified for the simulation executable.
        llvm::errs() << "No output file name specified\n";
        return EXIT_FAILURE;
      } else {
        frontendOutputFileName = "-";
      }
    } else {
      if (shouldLink) {
        frontendOutputFileName = outputFileName.str() + ".o";
      } else {
        frontendOutputFileName = outputFileName.str();
      }
    }

    llvm::FileRemover frontendOutputFileRemover(
        frontendOutputFileName, shouldLink);

    // Run the frontend.
    if (!frontendInputFiles.empty()) {
      frontendOutputFiles.emplace_back(
          frontendOutputFileName,
          InputKind(Language::Unknown, Format::Object));

      if (int resultCode = executeMC1Tool(
              args, frontendInputFiles, frontendOutputFileName);
          resultCode != EXIT_SUCCESS) {
        return resultCode;
      }
    }

    if (shouldLink) {
      // Determine the solver, which influences the linked libraries.
      Solver solver = Solver::Unknown;

      auto solverString =
          args.getLastArgValue(options::OPT_solver, "euler-forward");

      if (solverString == "euler-forward") {
        solver = Solver::EulerForward;
      } else if (solverString == "ida") {
        solver = Solver::IDA;
      }

      // Create the command to run the linker.
      auto linkerPath = llvm::sys::findProgramByName(LINKER_STR);

      if (!linkerPath) {
        return EXIT_FAILURE;
      }

      Command linkCommand(*linkerPath);

      // Add the input files.
      for (const InputFile& file : frontendOutputFiles) {
        linkCommand.appendArg(file.getFile());
      }

      // Set the output file.
      linkCommand
          .appendArg("-o")
          .appendArg(outputFileName);

      // Collect the paths of the libraries to be linked.
      llvm::SmallVector<std::string> allLibsPaths;
      collectLibraryPaths(LIBS_PATH_STR, allLibsPaths);
      collectLibraryPaths(DEPENDENCIES_LIBS_PATH_STR, allLibsPaths);

      // Add the r-path for the dynamic libraries.
      if (!allLibsPaths.empty()) {
        llvm::SmallVector<std::string> wrappedPaths;

        for (const auto& path : allLibsPaths) {
          wrappedPaths.push_back("\"" + path + "\"");
        }

        std::string rpath = llvm::join(allLibsPaths, ":");
        linkCommand.appendArg("-Wl,-rpath," + rpath);
      }

      // Add the paths of the libraries.
      for (const auto& path : allLibsPaths) {
        linkCommand.appendArg("-L").appendArg(path);
      }

      // Add the main function to the simulation, if not explicitly discarded.
      if (!args.hasArg(options::OPT_no_generate_main)) {
        linkCommand.appendArg("-lMARCORuntimeStarter");
      }

      // Add the main simulation driver.
      linkCommand.appendArg("-lMARCORuntimeSimulation");

      // Add the libraries of the solver.
      if (solver == Solver::EulerForward) {
        linkCommand
            .appendArg("-lMARCORuntimeDriverEulerForward")
            .appendArg("-lMARCORuntimeSolverEulerForward");
      } else if (solver == Solver::IDA) {
        linkCommand
            .appendArg("-lMARCORuntimeDriverIDA")
            .appendArg("-lMARCORuntimeSolverIDA");
      }

      // Add the remaining runtime libraries.
      linkCommand
          .appendArg("-lMARCORuntimeSolverKINSOL")
          .appendArg("-lMARCORuntimePrinterCSV")
          .appendArg("-lMARCORuntimeSupport")
          .appendArg("-lMARCORuntimeCLI")
          .appendArg("-lMARCORuntimeModeling")
          .appendArg("-lMARCORuntimeMultithreading")
          .appendArg("-lMARCORuntimeProfiling")
          .appendArg(LINK_EXTRA_FLAGS_STR);

      if (solver == Solver::IDA) {
        linkCommand
            .appendArg("-lsundials_ida")
            .appendArg("-lsundials_kinsol")
            .appendArg("-lsundials_nvecserial")
            .appendArg("-lsundials_sunlinsolklu")
            .appendArg("-lklu");
      }

      for (const auto& linkerArg :
           args.getAllArgValues(options::OPT_linker_arg)) {
        linkCommand.appendArg(linkerArg);
      }

      // Run the linker.
      if (int resultCode = linkCommand.exec(); resultCode != EXIT_SUCCESS) {
        frontendOutputFileRemover.releaseFile();
        return resultCode;
      }
    } else {
      // No link requested. Keep the output of the frontend.
      frontendOutputFileRemover.releaseFile();
    }

    return EXIT_SUCCESS;
  }

  int Driver::executeMC1Tool(
      const llvm::opt::ArgList& args,
      llvm::ArrayRef<InputFile> inputFiles,
      llvm::StringRef outputFileName) const
  {
    llvm::opt::ArgStringList mc1Args;

    for (const llvm::opt::Arg* arg : args.filtered(
             options::OPT_debug,
             options::OPT_omc_path,
             options::OPT_omc_arg,
             options::OPT_omc_bypass,
             options::OPT_assertions,
             options::OPT_no_assertions,
             options::OPT_bit_width,
             options::OPT_function_inlining,
             options::OPT_no_function_inlining,
             options::OPT_output_arrays_promotion,
             options::OPT_no_output_arrays_promotion,
             options::OPT_read_only_variables_propagation,
             options::OPT_no_read_only_variables_propagation,
             options::OPT_variables_to_parameters_promotion,
             options::OPT_no_variables_to_parameters_promotion,
             options::OPT_cse,
             options::OPT_no_cse,
             options::OPT_omp,
             options::OPT_no_omp,
             options::OPT_opt,
             options::OPT_target,
             options::OPT_cpu,
             options::OPT_model,
             options::OPT_filter,
             options::OPT_solver,
             options::OPT_ida_reduced_system,
             options::OPT_no_ida_reduced_system,
             options::OPT_ida_reduced_derivatives,
             options::OPT_no_ida_reduced_derivatives,
             options::OPT_ida_jacobian_one_sweep,
             options::OPT_no_ida_jacobian_one_sweep,
             options::OPT_init_only,
             options::OPT_emit_flattened,
             options::OPT_emit_ast,
             options::OPT_emit_final_ast,
             options::OPT_emit_mlir,
             options::OPT_emit_llvm_ir,
             options::OPT_emit_llvm_bitcode,
             options::OPT_compile_only,
             options::OPT_compile_and_assemble_only)) {
      arg->renderAsInput(args, mc1Args);
    }

    // Build the command.
    Command command(executablePath);
    command.appendArg("-mc1");
    command.appendArgs(mc1Args);

    // Append the output file.
    command.appendArg("-o").appendArg(outputFileName);

    // Append the input files.
    for (const InputFile& inputFile : inputFiles) {
      command.appendArg(inputFile.getFile());
    }

    // Run the frontend.
    return command.exec();
  }

  void Driver::collectLibraryPaths(
      llvm::StringRef pathsStr,
      llvm::SmallVectorImpl<std::string>& paths) const
  {
    llvm::SmallVector<llvm::StringRef> refs;
    pathsStr.split(refs, ';', -1, false);

    for (llvm::StringRef ref : refs) {
      paths.push_back(ref.str());
    }
  }
}
