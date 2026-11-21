#include "marco/AST/Modelica/TypePrefix.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void TypePrefix::addToJSON(llvm::json::Object &obj) const {
  if (connectorKind != ConnectorKind::None) {
    obj["connector_kind"] = toString(connectorKind);
  }

  if (variability != Variability::None) {
    obj["variability"] = toString(variability);
  }

  if (causality != Causality::None) {
    obj["causality"] = toString(causality);
  }
}

llvm::StringRef toString(ConnectorKind kind) {
  switch (kind) {
  case ConnectorKind::None:
    return "";

  case ConnectorKind::Flow:
    return "flow";

  case ConnectorKind::Stream:
    return "stream";
  }

  llvm_unreachable("Unknown connector kind");
  return "";
}

llvm::StringRef toString(Variability variability) {
  switch (variability) {
  case Variability::None:
    return "";
  case Variability::Discrete:
    return "discrete";
  case Variability::Parameter:
    return "parameter";
  case Variability::Constant:
    return "constant";
  }

  llvm_unreachable("Unknown variability");
  return "";
}

llvm::StringRef toString(Causality causality) {
  switch (causality) {
  case Causality::None:
    return "";
  case Causality::Input:
    return "input";
  case Causality::Output:
    return "output";
  }

  llvm_unreachable("Unknown causality");
  return "";
}
} // namespace marco::ast::modelica
