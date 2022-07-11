#ifndef MARCO_DIAGNOSTIC_DIAGNOSTIC_H
#define MARCO_DIAGNOSTIC_DIAGNOSTIC_H

#include "marco/Diagnostic/LogMessage.h"
#include "llvm/ADT/Twine.h"
#include <memory>

namespace llvm
{
  class raw_ostream;
}

namespace marco::diagnostic
{
  // The printer header file includes the LLVM raw_ostream header.
  // In order to avoid this, we forward declare the class and include the header file
  // only in the implementation sources.
  class Printer;

  struct DiagnosticOptions
  {
    bool showColors = true;
    bool warningAsError = false;

    static DiagnosticOptions getDefaultOptions();
  };

  class DiagnosticEngine
  {
    public:
      DiagnosticEngine(
        std::unique_ptr<Printer> printer,
        DiagnosticOptions options = DiagnosticOptions::getDefaultOptions());

      DiagnosticOptions& getOptions();

      const DiagnosticOptions& getOptions() const;

      template<typename Message, typename... Args>
      void emitFatalError(Args&&... args)
      {
        ++numOfErrors_;
        emit(Level::FATAL_ERROR, Message(std::forward<Args>(args)...));
      }

      template<typename Message, typename... Args>
      void emitError(Args&&... args)
      {
        ++numOfErrors_;
        emit(Level::ERROR, Message(std::forward<Args>(args)...));
      }

      template<typename Message, typename... Args>
      void emitWarning(Args&&... args)
      {
        Level level = options_.warningAsError ? Level::ERROR : Level::WARNING;

        if (level == Level::ERROR) {
          ++numOfErrors_;
        }

        emit(level, Message(std::forward<Args>(args)...));
      }

      template<typename Message, typename... Args>
      void emitNote(Args&&... args)
      {
        emit(Level::NOTE, Message(std::forward<Args>(args)...));
      }

      size_t numOfErrors() const;

      bool hasErrors() const;

    private:
      void emit(Level level, const Message& message);

    private:
      size_t numOfErrors_;
      std::unique_ptr<Printer> printer_;
      DiagnosticOptions options_;
  };
}

#endif // MARCO_DIAGNOSTIC_DIAGNOSTIC_H
