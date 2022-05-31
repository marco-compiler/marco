#include "marco/Diagnostic/Printer.h"

using namespace ::marco;
using namespace ::marco::diagnostic;

// Diagnostic colors
static const enum llvm::raw_ostream::Colors noteColor = llvm::raw_ostream::BLACK;
static const enum llvm::raw_ostream::Colors remarkColor = llvm::raw_ostream::BLUE;
static const enum llvm::raw_ostream::Colors warningColor = llvm::raw_ostream::YELLOW;
static const enum llvm::raw_ostream::Colors errorColor = llvm::raw_ostream::RED;
static const enum llvm::raw_ostream::Colors fatalColor = llvm::raw_ostream::RED;

static const enum llvm::raw_ostream::Colors savedColor = llvm::raw_ostream::SAVEDCOLOR;

namespace marco::diagnostic
{
  //===----------------------------------------------------------------------===//
  // Printer
  //===----------------------------------------------------------------------===//

  llvm::raw_ostream::Colors Printer::diagnosticLevelToColor(marco::diagnostic::Level level) const
  {
    switch (level) {
      case Level::ERROR:
        return errorColor;

      case Level::WARNING:
        return warningColor;

      case Level::NOTE:
        return noteColor;
    }

    llvm_unreachable("Unknown diagnostic level");
    return fatalColor;
  }

  llvm::raw_ostream& Printer::getOutputStream()
  {
    return llvm::errs();
  }

  //===----------------------------------------------------------------------===//
  // PrinterInstance
  //===----------------------------------------------------------------------===//

  PrinterInstance::PrinterInstance(Printer* printer, Level level, bool showColors)
      : printer_(printer),
        color_(printer->diagnosticLevelToColor(level)),
        bold_(false)
  {
    printer->getOutputStream().enable_colors(showColors);
  }

  llvm::raw_ostream& PrinterInstance::getOutputStream()
  {
    return printer_->getOutputStream();
  }

  Level PrinterInstance::diagnosticLevel() const
  {
    return level_;
  }

  llvm::raw_ostream::Colors PrinterInstance::diagnosticLevelColor() const
  {
    return color_;
  }

  void PrinterInstance::setColor(llvm::raw_ostream& os, llvm::raw_ostream::Colors color)
  {
    os.changeColor(color, bold_);
  }

  void PrinterInstance::resetColor(llvm::raw_ostream& os)
  {
    os.resetColor();
  }

  void PrinterInstance::setBold(llvm::raw_ostream& os)
  {
    bold_ = true;
    os.changeColor(savedColor, bold_);
  }

  void PrinterInstance::unsetBold(llvm::raw_ostream& os)
  {
    bold_ = false;
    os.changeColor(savedColor, bold_);
  }
}
