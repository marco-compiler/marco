#ifndef MARCO_AST_MODELICA_TYPEPREFIX_H
#define MARCO_AST_MODELICA_TYPEPREFIX_H

#include "marco/AST/Modelica/Node.h"

namespace marco::ast::modelica {
enum class ConnectorKind { None, Flow, Stream };

enum class Variability { None, Discrete, Parameter, Constant };

enum class Causality { None, Input, Output };

/// Type prefix of a component clause.
struct TypePrefix {
  ConnectorKind connectorKind = ConnectorKind::None;
  Variability variability = Variability::None;
  Causality causality = Causality::None;

  void addToJSON(llvm::json::Object &obj) const;
};

llvm::StringRef toString(ConnectorKind kind);

llvm::StringRef toString(Variability variability);

llvm::StringRef toString(Causality causality);
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_TYPEPREFIX_H
