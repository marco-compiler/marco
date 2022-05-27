#include "marco/Diagnostic/Diagnostic.h"
#include "marco/Diagnostic/Printer.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco;
using namespace ::marco::diagnostic;

namespace marco::diagnostic
{
  //===----------------------------------------------------------------------===//
  // DiagnosticOptions
  //===----------------------------------------------------------------------===//

  DiagnosticOptions DiagnosticOptions::getDefaultOptions()
  {
    DiagnosticOptions options;
    return options;
  }

  //===----------------------------------------------------------------------===//
  // DiagnosticEngine
  //===----------------------------------------------------------------------===//

  DiagnosticEngine::DiagnosticEngine(std::unique_ptr<Printer> printer, DiagnosticOptions options)
    : printer_(std::move(printer)), options_(std::move(options))
  {
    if (this->printer_ == nullptr) {
      this->printer_ = std::make_unique<Printer>();
    }
  }

  size_t DiagnosticEngine::numOfErrors() const
  {
    return numOfErrors_;
  }

  bool DiagnosticEngine::hasErrors() const
  {
    return numOfErrors_ != 0;
  }

  void DiagnosticEngine::emit(Level level, const Message& message)
  {
    auto printerInstance = std::make_unique<PrinterInstance>(printer_.get(), level, options_.showColors);
    message.print(printerInstance.get());
  }
}
