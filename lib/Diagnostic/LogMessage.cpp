#include "marco/Diagnostic/Diagnostic.h"
#include "marco/Diagnostic/LogMessage.h"

/*
namespace marco::diagnostic
{
  Message::Message() = default;

  Message::Message(const Message& other) = default;

  Message::Message(Message&& other) = default;

  Message::~Message() = default;

  Message& Message::operator=(const Message& other) = default;

  Message& Message::operator=(Message&& other) = default;

  void Message::beforeMessage(const PrintInstance& print, llvm::raw_ostream& os) const
  {
  }

  void Message::afterMessage(const PrintInstance& print, llvm::raw_ostream& os) const
  {

  }

  void Message::before(llvm::raw_ostream& os) const
  {
    printBeforeMessage(os);
  }

  void Message::after(llvm::raw_ostream& os) const
  {
    printAfterMessage(os);
  }

  SourceMessage::SourceMessage(SourceRange location) : location(std::move(location))
  {
  }

  SourceMessage::SourceMessage(const SourceMessage& other) = default;

  SourceMessage::SourceMessage(SourceMessage&& other) = default;

  SourceMessage::~SourceMessage() = default;

  SourceMessage& SourceMessage::operator=(const SourceMessage& other) = default;

  SourceMessage& SourceMessage::operator=(SourceMessage&& other) = default;

  void SourceMessage::before(Printer* printer, llvm::raw_ostream& os) const
  {
    Message::before(os);
    os.changeColor(llvm::raw_ostream::SAVEDCOLOR, true);
    os << *location.fileName << ":" << location.startLine << ":" << location.startColumn << ": ";
  }

  void SourceMessage::after(Printer* printer, llvm::raw_ostream& os) const
  {


    Message::after(os);
    location.printLines(os, getFormatter());
    // TODO printLines();
  }
}
*/
