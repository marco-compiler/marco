#ifndef MARCO_AST_MODELICA_NAME_H
#define MARCO_AST_MODELICA_NAME_H

#include "marco/Lexer/Location.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"
#include <string>

namespace marco::ast::modelica {
/// Dotted name, as in type specifiers and modifications
/// (e.g., `.Modelica.Units.SI.Voltage`).
struct Name {
  /// Whether the name starts with a dot, meaning a global name.
  bool global = false;

  llvm::SmallVector<std::string, 3> parts;

  SourceRange location = SourceRange::unknown();

  std::string toString() const;
};

/// Get the JSON representation of a name.
llvm::json::Value toJSON(const Name &name);
} // namespace marco::ast::modelica

#endif // MARCO_AST_MODELICA_NAME_H
