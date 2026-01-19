#include "marco/Frontend/Passes/EquationTargets/EquationTarget.h"

using namespace ::marco::frontend;

namespace marco::frontend {
EquationTarget::EquationTarget(llvm::StringRef name) : name(name.str()) {}

EquationTarget::~EquationTarget() = default;

llvm::StringRef EquationTarget::getName() const { return name; }
} // namespace marco::frontend
