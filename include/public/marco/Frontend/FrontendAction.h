#ifndef MARCO_FRONTEND_FRONTENDACTION_H
#define MARCO_FRONTEND_FRONTENDACTION_H

#include "marco/Frontend/FrontendOptions.h"
#include "mlir/Pass/Pass.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/Support/Error.h"
#include <memory>

namespace marco::frontend
{
  class CompilerInstance;

  /// Abstract base class for the actions which can be performed by the
  /// frontend.
  class FrontendAction
  {
    public:
      FrontendAction();

      virtual ~FrontendAction();

      /// @name Compiler instance access
      /// @{

      CompilerInstance& getInstance();

      const CompilerInstance& getInstance() const;

      void setInstance(CompilerInstance* value);

      /// }
      /// @name Current file information
      /// {

      const io::InputFile& getCurrentInput() const;

      llvm::StringRef getCurrentFile() const;

      llvm::StringRef getCurrentFileOrBufferName() const;

      void setCurrentInput(const io::InputFile& currentIntput);

      /// }
      /// @name Public action interface
      /// {

      /// Prepare the action to execute on the given compiler instance.
      bool prepareToExecute(CompilerInstance& ci);

      /// Prepare the action for processing the input file.
      ///
      /// This is run after the options and frontend have been initialized,
      /// but prior to executing any per-file processing.
      bool beginSourceFile(CompilerInstance &ci, const io::InputFile &input);

      /// Run the action.
      llvm::Error execute();

      /// Perform any per-file post processing, deallocate per-file objects,
      /// and run statistics and output file cleanup code.
      void endSourceFile();

      /// }

    protected:
      /// @name Implementation action interface
      /// {

      virtual bool prepareToExecuteAction(CompilerInstance& ci);

      /// Function called before starting to process a single input.
      /// It gives the opportunity to modify the CompilerInvocation or do some
      /// other action before beginSourceFileAction is called.
      /// @return true on success; on failure beginSourceFileAction,
      /// executeAction and endSourceFileAction will not be called.
      virtual bool beginInvocation();

      /// Callback at the start of processing a single input.
      ///
      /// @return true on success; on failure executionAction() and
      /// endSourceFileAction() will not be called.
      virtual bool beginSourceFileAction();

      /// Callback to run the program action, using the initialized compiler
      /// instance.
      virtual void executeAction() = 0;

      /// Function called at the end of processing a single input.
      /// This is guaranteed to only be called following a successful call to
      /// beginSourceFileAction (and beginSourceFile).
      virtual void endSourceFileAction();

      /// Callback at the end of processing a single input, to determine
      /// if the output files should be erased or not.
      ///
      /// By default it returns true if a compiler error occurred.
      virtual bool shouldEraseOutputFiles() const;

      /// }

    private:
      CompilerInstance* instance;
      io::InputFile currentInput;
  };
}

#endif // MARCO_FRONTEND_FRONTENDACTION_H
