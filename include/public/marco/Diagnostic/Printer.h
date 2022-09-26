#ifndef MARCO_DIAGNOSTIC_PRINTER_H
#define MARCO_DIAGNOSTIC_PRINTER_H

#include "marco/Diagnostic/Diagnostic.h"
#include "llvm/Support/raw_ostream.h"

namespace marco::diagnostic
{
  class Printer
  {
    public:
      virtual llvm::raw_ostream::Colors diagnosticLevelToColor(marco::diagnostic::Level level) const;

      virtual llvm::raw_ostream& getOutputStream();
  };

  class PrinterInstance
  {
    public:
      PrinterInstance(Printer* printer, Level level, bool showColors);

      llvm::raw_ostream& getOutputStream();

      Level diagnosticLevel() const;

      llvm::raw_ostream::Colors diagnosticLevelColor() const;

      void setColor(llvm::raw_ostream& os, llvm::raw_ostream::Colors color);

      void resetColor(llvm::raw_ostream& os);

      void setBold(llvm::raw_ostream& os);

      void unsetBold(llvm::raw_ostream& os);

    private:
      Printer* printer_;
      Level level_;
      llvm::raw_ostream::Colors color_;
      bool bold_;
  };
}

#endif // MARCO_DIAGNOSTIC_PRINTER_H
