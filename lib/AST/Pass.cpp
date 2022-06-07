#include "marco/AST/Pass.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  Pass::Pass(diagnostic::DiagnosticEngine& diagnostics)
    : diagnostics_(&diagnostics)
  {
  }

  Pass::Pass(const Pass& other) = default;

  Pass::Pass(Pass&& other) = default;

  Pass& Pass::operator=(Pass&& other) = default;

  Pass::~Pass() = default;

  Pass& Pass::operator=(const Pass& other) = default;

  diagnostic::DiagnosticEngine* Pass::diagnostics()
  {
    return diagnostics_;
  }
}
