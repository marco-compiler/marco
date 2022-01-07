#ifndef MARCO_FRONTEND_FRONTENDACTION_H
#define MARCO_FRONTEND_FRONTENDACTION_H

#include <llvm/Support/Error.h>

#include "FrontendOptions.h"

namespace marco::frontend
{
  class CompilerInstance;

  /**
   * Abstract base class for the actions which can be performed by the frontend.
   */
  class FrontendAction
  {
    public:
      FrontendAction() : instance_(nullptr)
      {
      }

      virtual ~FrontendAction() = default;

      void setCompilerInstance(CompilerInstance& ci)
      {
        instance_ = &ci;
      }

      virtual void execute() = 0;

    protected:
      CompilerInstance& instance()
      {
        assert(instance_ != nullptr && "Compiler instance not registered");
        return *instance_;
      }

      bool runParse();
      bool runFrontendPasses();
      bool runASTConversion();
      bool runDialectConversion();
      bool runLLVMIRGeneration();

    private:
      CompilerInstance* instance_;
  };


  /**
   * Abstract base class for the actions which can be performed by the frontend.
   */
   /*
  class FrontendActionOld
  {
      FrontendInputFile currentInput_;
      CompilerInstance* instance_;

    protected:
      /// Callback to run the program action, using the initialized
      /// compiler instance.
      virtual void ExecuteAction() = 0;

      /// Callback at the end of processing a single input, to determine
      /// if the output files should be erased or not.
      ///
      /// By default it returns true if a compiler error occurred.
      virtual bool ShouldEraseOutputFiles();

      /// Callback at the start of processing a single input.
      ///
      /// \return True on success; on failure ExecutionAction() and
      /// EndSourceFileAction() will not be called.
      virtual bool BeginSourceFileAction() { return true; }

    public:
      FrontendAction() : instance_(nullptr) {}

      virtual ~FrontendAction() = default;

      /// @name Compiler Instance Access
      /// @{

      CompilerInstance& instance() const
      {
        assert(instance_ && "Compiler instance not registered!");
        return *instance_;
      }

      void set_instance(CompilerInstance* value) { instance_ = value; }

      /// @}
      /// @name Current File Information
      /// @{

      const FrontendInputFile& currentInput() const { return currentInput_; }

      llvm::StringRef GetCurrentFile() const
      {
        assert(!currentInput_.IsEmpty() && "No current file!");
        return currentInput_.file();
      }

      llvm::StringRef GetCurrentFileOrBufferName() const
      {
        assert(!currentInput_.IsEmpty() && "No current file!");
        return currentInput_.IsFile()
               ? currentInput_.file()
               : currentInput_.buffer()->getBufferIdentifier();
      }

      void set_currentInput(const FrontendInputFile& currentInput);

      /// Prepare the action for processing the input file \p input.
      ///
      /// This is run after the options and frontend have been initialized,
      /// but prior to executing any per-file processing.
      /// \param ci - The compiler instance this action is being run from. The
      /// action may store and use this object.
      /// \param input - The input filename and kind.
      /// \return True on success; on failure the compilation of this file should
      bool BeginSourceFile(CompilerInstance& ci, const FrontendInputFile& input);

      llvm::Error Execute();

      /// Perform any per-file post processing, deallocate per-file
      /// objects, and run statistics and output file cleanup code.
      void EndSourceFile();

    protected:
      bool runParse();

      // Report fatal semantic errors. Return True if present, false otherwise.
      //bool reportFatalSemanticErrors();

      // Report fatal scanning errors. Return True if present, false otherwise.
      //inline bool reportFatalScanningErrors() {
      //  return reportFatalErrors("Could not scan %0");
      // }

      // Report fatal parsing errors. Return True if present, false otherwise
      inline bool reportFatalParsingErrors()
      {
        return reportFatalErrors("Could not parse %0");
      }

    private:
      template<unsigned N>
      bool reportFatalErrors(const char (& message)[N]);
  };

*/
}

#endif // MARCO_FRONTEND_FRONTENDACTION_H
