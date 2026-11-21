#ifndef MARCO_AST_MODELICA_DESCRIPTION_H
#define MARCO_AST_MODELICA_DESCRIPTION_H

#include "marco/AST/Modelica/ClassModification.h"
#include "marco/AST/Modelica/Node.h"

namespace marco::ast::modelica {
/// Description of a declaration, equation or statement: a description string
/// and an optional annotation.
struct Description {
  std::string text;
  Child<ClassModification> annotation;

  void walkChildren(llvm::function_ref<void(Node *)> callback);

  void addToJSON(llvm::json::Object &obj) const;
};
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_DESCRIPTION_H
