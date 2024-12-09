#include "marco/Frontend/FrontendOptions.h"

using namespace ::marco::frontend;

namespace marco::frontend {
bool FrontendOptions::shouldPrintIR() const {
  return !printIRBeforePass.empty() || !printIRAfterPass.empty();
}
} // namespace marco::frontend
