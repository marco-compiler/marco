#ifndef MARCO_DIAGNOSTIC_DIAGNOSTIC_H
#define MARCO_DIAGNOSTIC_DIAGNOSTIC_H

#include <llvm/ADT/Twine.h>
#include <llvm/Support/raw_ostream.h>
#include <memory>

#include "LogMessage.h"

namespace marco::diagnostic
{
  enum class Level
  {
      ERROR,
      WARNING,
      NOTE
  };

  class LogStream : public llvm::raw_ostream
  {

  };

  template<typename T>
  LogStream& operator<<(LogStream& stream, const T& obj)
  {

  }

  class PrintInstance
  {
    public:
      PrintInstance(llvm::raw_ostream::Colors color, bool showColors);

      void setColor(llvm::raw_ostream& os) const;

      void resetColor(llvm::raw_ostream& os) const;

      void setBold(llvm::raw_ostream& os) const;

      void unsetBold(llvm::raw_ostream& os) const;

    private:
      llvm::raw_ostream::Colors color_;
      bool showColors_;
  };

  class Printer
  {
    public:
      virtual void print(llvm::raw_ostream& os, Level level, const Message& message, bool showColors) const;

      virtual void printDiagnosticLevel(llvm::raw_ostream& os, Level level) const;
  };

  struct DiagnosticOptions
  {
      bool showColors = true;
      bool warningAsError = false;
  };

  class DiagnosticEngine
  {
    public:
      DiagnosticEngine(DiagnosticOptions options, std::unique_ptr<Printer> printer = std::make_unique<Printer>());

      template<typename Message, typename... Args>
      void emitError(Args&&... args) const
      {
        emit(Level::ERROR, Message(std::forward<Args>(args)...));
      }

      template<typename Message, typename... Args>
      void emitWarning(Args&&... args) const
      {
        Level level = options.warningAsError ? Level::ERROR : Level::WARNING;
        emit(level, Message(std::forward<Args>(args)...));
      }

      template<typename Message, typename... Args>
      void emitNote(Args&&... args) const
      {
        emit(Level::NOTE, Message(std::forward<Args>(args)...));
      }

    private:
      void emit(Level level, const Message& message) const;

      DiagnosticOptions options;
      std::unique_ptr<Printer> printer;
  };
}

#endif // MARCO_DIAGNOSTIC_DIAGNOSTIC_H
