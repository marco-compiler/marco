#ifndef MARCO_FRONTEND_TEXTDIAGNOSTIC_H
#define MARCO_FRONTEND_TEXTDIAGNOSTIC_H

#include "clang/Basic/Diagnostic.h"
#include "llvm/ADT/IntrusiveRefCntPtr.h"

namespace marco::frontend
{
  /// Class to encapsulate the logic for formatting and printing a textual
  /// diagnostic message.
  ///
  /// The purpose of this class is to isolate the implementation of printing
  /// beautiful text diagnostics from any particular interfaces. Currently only
  /// simple diagnostics that lack source location information are supported (e.g.
  /// Flang driver errors).
  ///
  /// In the future we can extend this class (akin to Clang) to support more
  /// complex diagnostics that would include macro backtraces, caret diagnostics,
  /// FixIt Hints and code snippets.
  ///
  class TextDiagnostic
  {
    public:
      TextDiagnostic();

      ~TextDiagnostic();

      /// Print the diagnostic level to a llvm::raw_ostream.
      ///
      /// This is a static helper that handles colorizing the level and formatting
      /// it into an arbitrary output stream.
      ///
      /// \param os Where the message is printed
      /// \param level The diagnostic level (e.g. error or warning)
      /// \param showColors Enable colorizing of the message.
      static void PrintDiagnosticLevel(llvm::raw_ostream& os, clang::DiagnosticsEngine::Level level, bool showColors);

      /// Pretty-print a diagnostic message to a llvm::raw_ostream.
      ///
      /// This is a static helper to handle the colorizing and rendering diagnostic
      /// message to a particular ostream. In the future we can
      /// extend it to support e.g. line wrapping. It is
      /// publicly visible as at this stage we don't require any state data to
      /// print a diagnostic.
      ///
      /// \param os Where the message is printed
      /// \param isSupplemental true if this is a continuation note diagnostic
      /// \param message The text actually printed
      /// \param showColors Enable colorizing of the message.
      static void PrintDiagnosticMessage(
          llvm::raw_ostream& os, bool isSupplemental, llvm::StringRef message, bool showColors);
  };
}

#endif // MARCO_FRONTEND_TEXTDIAGNOSTIC_H
