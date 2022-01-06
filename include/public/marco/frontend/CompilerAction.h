#ifndef MARCO_FRONTEND_COMPILERACTION_H
#define MARCO_FRONTEND_COMPILERACTION_H

#include "FrontendOptions.h"

namespace marco::frontend
{
  class CompilerInstance;

  class CompilerAction
  {
    protected:
      /// Callback to run the program action, using the initialized
      /// compiler instance.
      virtual void executeAction() = 0;

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
      CompilerAction() : instance_(nullptr) {}

      virtual ~CompilerAction() = default;

      CompilerInstance& instance() const
      {
        assert(instance_ && "Compiler instance not registered!");
        return *instance_;
      }

      void setInstance(CompilerInstance* value)
      {
        instance_ = value;
      }

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

      /**
       * Run the action.
       *
       * @return whether the action has been executed successfully
       */
      virtual llvm::Error execute() = 0;

      /// Perform any per-file post processing, deallocate per-file
      /// objects, and run statistics and output file cleanup code.
      void EndSourceFile();

    private:
      template<unsigned N>
      bool reportFatalErrors(const char (& message)[N]);

      FrontendInputFile currentInput_;
      CompilerInstance* instance_;
  };

  class GroupAction : public CompilerAction
  {
    public:
      llvm::Error execute() override;

    private:

  };

  class IndividualAction : public CompilerAction
  {
    public:
      llvm::Error execute() override;

    private:

  };
}

#endif // MARCO_FRONTEND_COMPILERACTION_H
