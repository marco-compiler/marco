#ifndef MARCO_DIAGNOSTIC_LOGMESSAGE_H
#define MARCO_DIAGNOSTIC_LOGMESSAGE_H

#include "marco/Diagnostic/Level.h"
#include "marco/Diagnostic/Location.h"
#include "llvm/ADT/StringRef.h"

namespace llvm
{
  class raw_ostream;
}

namespace marco::diagnostic
{
  class PrinterInstance;

  class Message
  {
    public:
      Message();

      Message(const Message& other);
      Message(Message&& other);

      virtual ~Message();

      Message& operator=(const Message& other);
      Message& operator=(Message&& other);

      virtual void print(PrinterInstance* printer) const = 0;

    protected:
      void printDiagnosticLevel(llvm::raw_ostream& os, Level level) const;
  };

	class SourceMessage : public Message
	{
		public:
      SourceMessage(SourceRange location);

      SourceMessage(const SourceMessage& other);
      SourceMessage(SourceMessage&& other);

      virtual ~SourceMessage();

      SourceMessage& operator=(const SourceMessage& other);
      SourceMessage& operator=(SourceMessage&& other);

    protected:
      void printFileNameAndPosition(llvm::raw_ostream& os) const;

      void printLines(llvm::raw_ostream& os, std::function<void(llvm::raw_ostream&)> highlightSourceFn) const;

    private:
      void printLine(
        llvm::raw_ostream& os,
        std::function<void(llvm::raw_ostream&)> highlightSourceFn,
        llvm::StringRef line,
        size_t lineMaxDigits,
        int64_t lineNumber,
        int64_t highlightBegin,
        int64_t highlightEnd) const;

      void printLineNumber(
          llvm::raw_ostream& os,
          int64_t lineNumber,
          size_t lineMaxDigits,
          bool shouldPrintLineNumber) const;

    private:
      SourceRange location;
	};
}

#endif // MARCO_DIAGNOSTIC_LOGMESSAGE_H
