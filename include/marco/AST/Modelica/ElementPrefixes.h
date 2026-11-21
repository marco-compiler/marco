#ifndef MARCO_AST_MODELICA_ELEMENTPREFIXES_H
#define MARCO_AST_MODELICA_ELEMENTPREFIXES_H

#include "marco/AST/Modelica/Node.h"

namespace marco::ast::modelica {
/// Prefixes of an element declaration.
struct ElementPrefixes {
  bool isProtected = false;
  bool isRedeclare = false;
  bool isFinal = false;
  bool isInner = false;
  bool isOuter = false;
  bool isReplaceable = false;

  void addToJSON(llvm::json::Object &obj) const;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_ELEMENTPREFIXES_H
