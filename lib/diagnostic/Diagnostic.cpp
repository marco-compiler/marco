#include "marco/diagnostic/Diagnostic.h"

/*
static const enum llvm::raw_ostream::Colors noteColor = llvm::raw_ostream::BLACK;
static const enum llvm::raw_ostream::Colors remarkColor = llvm::raw_ostream::BLUE;
static const enum llvm::raw_ostream::Colors warningColor = llvm::raw_ostream::YELLOW;
static const enum llvm::raw_ostream::Colors errorColor = llvm::raw_ostream::RED;
static const enum llvm::raw_ostream::Colors fatalColor = llvm::raw_ostream::RED;

static const enum llvm::raw_ostream::Colors savedColor = llvm::raw_ostream::SAVEDCOLOR;

namespace marco::diagnostic
{
  PrintInstance::PrintInstance(llvm::raw_ostream::Colors color, bool showColors)
    : color_(std::move(color)), showColors_(showColors)
  {
  }

  llvm::raw_ostream::Colors PrintInstance::color() const
  {
    return color_;
  }

  bool PrintInstance::showColors()
  {
    return showColors_;
  }

  void Printer::print(llvm::raw_ostream& os, Level level, const Message& message, bool showColors) const
  {
    auto printInstance =
    message.print()

    message.printBeforeMessage(os);
    printDiagnosticLevel(os, level);

    // Print primary diagnostic messages in bold
    os.changeColor(savedColor, true);
    message.printMessage(os);
    os.resetColor();
    os << "\n";

    message.printAfterMessage(os);
  }

  void Printer::printDiagnosticLevel(llvm::raw_ostream& os, Level level) const
  {
    // Print diagnostic category in bold and color
    switch (level) {
      case Level::NOTE:
        os.changeColor(noteColor, true);
        break;

      case Level::WARNING:
        os.changeColor(warningColor, true);
        break;

      case Level::ERROR:
        os.changeColor(errorColor, true);
        break;
    }

    switch (level) {
      case Level::NOTE:
        os << "note";
        break;

      case Level::WARNING:
        os << "warning";
        break;

      case Level::ERROR:
        os << "error";
        break;
    }

    os << ": ";
    os.resetColor();
  }

  DiagnosticEngine::DiagnosticEngine(DiagnosticOptions options, std::unique_ptr<Printer> printer)
    : options(std::move(options)), printer(std::move(printer))
  {
    if (this->printer == nullptr) {
      this->printer = std::make_unique<Printer>();
    }
  }

  void DiagnosticEngine::emit(Level level, const Message& message) const
  {
    printer->print(llvm::errs(), level, message, options.showColors());
  }
}
*/
