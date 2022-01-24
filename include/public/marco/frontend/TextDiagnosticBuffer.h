#ifndef MARCO_FRONTEND_TEXTDIAGNOSTICBUFFER_H
#define MARCO_FRONTEND_TEXTDIAGNOSTICBUFFER_H

#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceLocation.h"
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace marco::frontend
{
  class TextDiagnosticBuffer : public clang::DiagnosticConsumer
  {
    public:
      using DiagList = std::vector<std::pair<clang::SourceLocation, std::string>>;
      using DiagnosticsLevelAndIndexPairs = std::vector<std::pair<clang::DiagnosticsEngine::Level, size_t>>;

      void HandleDiagnostic(
          clang::DiagnosticsEngine::Level level,
          const clang::Diagnostic& info) override;

      /**
       * Flush the buffered diagnostics to a given diagnostic engine.
       *
       * @param engine  diagnostic engine
       */
      void FlushDiagnostics(clang::DiagnosticsEngine& engine) const;

    private:
      DiagList errors, warnings, remarks, notes;

      // All diagnostics in the order in which they were generated. That order
      // likely doesn't correspond to user input order, but at least it keeps
      // notes in the right places. Each pair is a diagnostic level and an index
      // into the corresponding DiagList above.
      DiagnosticsLevelAndIndexPairs all;
  };
}

#endif // MARCO_FRONTEND_TEXTDIAGNOSTICBUFFER_H
