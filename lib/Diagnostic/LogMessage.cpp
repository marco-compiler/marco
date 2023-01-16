#include "marco/Diagnostic/Diagnostic.h"
#include "marco/Diagnostic/LogMessage.h"
#include "marco/Diagnostic/Printer.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco;
using namespace ::marco::diagnostic;

namespace
{
  template<typename T>
  size_t numDigits(T value)
  {
    size_t digits = 0;

    while (value) {
      value /= 10;
      ++digits;
    }

    return digits;
  }
}

namespace marco::diagnostic
{
  Message::Message() = default;

  Message::Message(const Message& other) = default;

  Message::Message(Message&& other) = default;

  Message::~Message() = default;

  Message& Message::operator=(const Message& other) = default;

  Message& Message::operator=(Message&& other) = default;

  void Message::printDiagnosticLevel(llvm::raw_ostream& os, Level level) const
  {
    switch (level) {
      case Level::FATAL_ERROR:
        os << "fatal error";
        break;

      case Level::ERROR:
        os << "error";
        break;

      case Level::WARNING:
        os << "warning";
        break;

      case Level::REMARK:
        os << "remark";
        break;

      case Level::NOTE:
        os << "note";
        break;
    }
  }

  GenericStringMessage::GenericStringMessage(llvm::StringRef message)
    : message(message.str())
  {
  }

  void GenericStringMessage::print(PrinterInstance* printer) const
  {
    auto& os = printer->getOutputStream();
    os << message << "\n";
  }

  SourceMessage::SourceMessage(SourceRange location)
    : location(std::move(location))
  {
  }

  SourceMessage::SourceMessage(const SourceMessage& other) = default;

  SourceMessage::SourceMessage(SourceMessage&& other) = default;

  SourceMessage::~SourceMessage() = default;

  SourceMessage& SourceMessage::operator=(const SourceMessage& other) = default;

  SourceMessage& SourceMessage::operator=(SourceMessage&& other) = default;

  void SourceMessage::printFileNameAndPosition(llvm::raw_ostream& os) const
  {
    assert(*location.begin.file == *location.end.file);
    const auto& file = *location.begin.file;

    if (file.filePath() != "-") {
      os << file.filePath() << ":" << std::to_string(location.begin.line);
    }
  }

  void SourceMessage::printLines(llvm::raw_ostream& os, std::function<void(llvm::raw_ostream&)> highlightSourceFn) const
  {
    assert(*location.begin.file == *location.end.file);
    assert(location.begin.line < location.end.line || (location.begin.line == location.end.line && location.begin.column <= location.end.column));

    auto buffer = llvm::MemoryBuffer::getMemBuffer(
        location.begin.file->source(),
        location.begin.file->filePath());

    if (buffer == nullptr) {
      return;
    }

    llvm::line_iterator lineIterator(*buffer, false);

    while (!lineIterator.is_at_end() && lineIterator.line_number() < location.begin.line) {
      ++lineIterator;
    }

    size_t lineMaxDigits = std::max(numDigits(location.begin.line), numDigits(location.end.line));

    while (!lineIterator.is_at_end() && lineIterator.line_number() <= location.end.line) {
      auto line = *lineIterator;
      auto currentLineNumber = lineIterator.line_number();

      if (currentLineNumber == location.begin.line && currentLineNumber == location.end.line) {
        printLine(os, highlightSourceFn, line, lineMaxDigits, currentLineNumber, location.begin.column, location.end.column);

      } else if (currentLineNumber == location.begin.line) {
        printLine(os, highlightSourceFn, line, lineMaxDigits, currentLineNumber, location.begin.column, line.size());

      } else if (currentLineNumber == location.end.line) {
        printLine(os, highlightSourceFn, line, lineMaxDigits, currentLineNumber, 1, location.end.column);

      } else {
        printLine(os, highlightSourceFn, line, lineMaxDigits, currentLineNumber, 1, line.size());
      }

      ++lineIterator;
    }
  }

  void SourceMessage::printLine(
      llvm::raw_ostream& os,
      std::function<void(llvm::raw_ostream&)> highlightSourceFn,
      llvm::StringRef line,
      size_t lineMaxDigits,
      int64_t lineNumber,
      int64_t highlightBegin,
      int64_t highlightEnd) const
  {
    os.resetColor();
    printLineNumber(os, lineNumber, lineMaxDigits, true);

    for (int64_t i = 1; i < highlightBegin; ++i) {
      os << line[i - 1];
    }

    highlightSourceFn(os);

    for (int64_t i = highlightBegin; i <= highlightEnd; ++i) {
      os << line[i - 1];
    }

    os.resetColor();

    for (int64_t i = highlightEnd + 1, e = line.size();
         i < e && line[i - 1] != '\n' && line[i - 1] != '\0'; ++i) {
      os << line[i - 1];
    }

    os << "\n";
    printLineNumber(os, lineNumber, lineMaxDigits, false);

    for (int64_t i = 1; i < highlightBegin; ++i) {
      if (std::isspace(line[i - 1])) {
        os << line[i - 1];
      } else {
        os << " ";
      }
    }

    highlightSourceFn(os);
    os << "^";

    for (int64_t i = highlightBegin + 1; i <= highlightEnd; ++i) {
      os << "~";
    }

    os << "\n\n";
    os.resetColor();
  }

  void SourceMessage::printLineNumber(llvm::raw_ostream& os, int64_t lineNumber, size_t lineMaxDigits, bool shouldPrintLineNumber) const
  {
    constexpr size_t minWidth = 4;

    auto currentLineDigits = numDigits(lineNumber);
    assert(currentLineDigits <= lineMaxDigits);

    size_t width = std::max(minWidth, lineMaxDigits);

    os << " ";

    if (shouldPrintLineNumber) {
      for (size_t i = 0; i < width - currentLineDigits; ++i) {
        os << " ";
      }

      os << lineNumber;
    } else {
      for (size_t i = 0; i < width; ++i) {
        os << " ";
      }
    }

    os << " | ";
  }
}
