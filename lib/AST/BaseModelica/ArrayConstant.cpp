#include "marco/AST/BaseModelica/ArrayConstant.h"

using namespace ::marco;
using namespace ::marco::ast::bmodelica;

namespace marco::ast::bmodelica {
ArrayConstant::ArrayConstant(SourceRange location)
    : ArrayGenerator(ASTNodeKind::Expression_ArrayGenerator_ArrayConstant,
                     std::move(location)) {}

ArrayConstant::ArrayConstant(const ArrayConstant &other)
    : ArrayGenerator(other) {
  setValues(other.values);
}

std::unique_ptr<ASTNode> ArrayConstant::clone() const {
  return std::make_unique<ArrayConstant>(*this);
}

llvm::json::Value ArrayConstant::toJSON() const {
  llvm::json::Object result;

  llvm::SmallVector<llvm::json::Value> valuesJson;

  for (const auto &value : values) {
    valuesJson.push_back(value->toJSON());
  }

  result["values"] = llvm::json::Array(valuesJson);

  addNodeKindToJSON(*this, result);
  return result;
}

size_t ArrayConstant::size() const { return values.size(); }

Expression *ArrayConstant::operator[](size_t index) {
  assert(index < values.size());
  return values[index]->cast<Expression>();
}

const Expression *ArrayConstant::operator[](size_t index) const {
  assert(index < values.size());
  return values[index]->cast<Expression>();
}

void ArrayConstant::setValues(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes) {
  values.clear();

  for (const auto &node : nodes) {
    assert(node->isa<Expression>());
    auto &clone = values.emplace_back(node->clone());
    clone->setParent(this);
  }
}
} // namespace marco::ast::bmodelica
