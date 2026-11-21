#include "marco/AST/Modelica/ElementPrefixes.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void ElementPrefixes::addToJSON(llvm::json::Object &obj) const {
  auto addFlag = [&](llvm::StringRef name, bool value) {
    if (value) {
      obj[name] = true;
    }
  };

  addFlag("protected", isProtected);
  addFlag("redeclare", isRedeclare);
  addFlag("final", isFinal);
  addFlag("inner", isInner);
  addFlag("outer", isOuter);
  addFlag("replaceable", isReplaceable);
}
} // namespace marco::ast::modelica
