#include "marco/AST/Modelica/ArrayConcatenation.h"

using namespace ::marco::ast::modelica;

namespace marco::ast::modelica {
void ArrayConcatenation::walkChildren(
    llvm::function_ref<void(Node *)> callback) {
  walk(rows, callback);
}

void ArrayConcatenation::addJSONProperties(llvm::json::Object &obj) const {
  obj["rows"] = modelica::toJSON(rows);
}

size_t ArrayConcatenation::getNumRows() const { return rows.size(); }

ChildrenRef<Expression> ArrayConcatenation::getRow(size_t index) const {
  return rows[index];
}

void ArrayConcatenation::addRow(Children<Expression> row) {
  rows.push_back(std::move(row));
}
} // namespace marco::ast::modelica
