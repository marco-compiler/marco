#include <clang/Basic/DiagnosticOptions.h>
#include <llvm/Support/raw_ostream.h>
#include <marco/frontend/TextDiagnostic.h>

static const enum llvm::raw_ostream::Colors noteColor = llvm::raw_ostream::BLACK;
static const enum llvm::raw_ostream::Colors remarkColor = llvm::raw_ostream::BLUE;
static const enum llvm::raw_ostream::Colors warningColor = llvm::raw_ostream::MAGENTA;
static const enum llvm::raw_ostream::Colors errorColor = llvm::raw_ostream::RED;
static const enum llvm::raw_ostream::Colors fatalColor = llvm::raw_ostream::RED;

static const enum llvm::raw_ostream::Colors savedColor = llvm::raw_ostream::SAVEDCOLOR;

namespace marco::frontend
{
  TextDiagnostic::TextDiagnostic() = default;

  TextDiagnostic::~TextDiagnostic() = default;

  void TextDiagnostic::PrintDiagnosticLevel(
      llvm::raw_ostream& os, clang::DiagnosticsEngine::Level level, bool showColors)
  {
    if (showColors) {
      // Print diagnostic category in bold and color
      switch (level) {
        case clang::DiagnosticsEngine::Ignored:
          llvm_unreachable("Invalid diagnostic type");

        case clang::DiagnosticsEngine::Note:
          os.changeColor(noteColor, true);
          break;

        case clang::DiagnosticsEngine::Remark:
          os.changeColor(remarkColor, true);
          break;

        case clang::DiagnosticsEngine::Warning:
          os.changeColor(warningColor, true);
          break;

        case clang::DiagnosticsEngine::Error:
          os.changeColor(errorColor, true);
          break;

        case clang::DiagnosticsEngine::Fatal:
          os.changeColor(fatalColor, true);
          break;
      }
    }

    switch (level) {
      case clang::DiagnosticsEngine::Ignored:
        llvm_unreachable("Invalid diagnostic type");

      case clang::DiagnosticsEngine::Note:
        os << "note";
        break;

      case clang::DiagnosticsEngine::Remark:
        os << "remark";
        break;

      case clang::DiagnosticsEngine::Warning:
        os << "warning";
        break;

      case clang::DiagnosticsEngine::Error:
        os << "error";
        break;

      case clang::DiagnosticsEngine::Fatal:
        os << "fatal error";
        break;
    }

    os << ": ";

    if (showColors) {
      os.resetColor();
    }
  }

  void TextDiagnostic::PrintDiagnosticMessage(
      llvm::raw_ostream& os, bool isSupplemental, llvm::StringRef message, bool showColors)
  {
    if (showColors && !isSupplemental) {
      // Print primary diagnostic messages in bold and without color.
      os.changeColor(savedColor, true);
    }

    os << message;

    if (showColors) {
      os.resetColor();
    }

    os << '\n';
  }
}
