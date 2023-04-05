#include "marco/AST/Node/Array.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  Array::Array(SourceRange location)
      : Expression(ASTNode::Kind::Expression_Array, std::move(location))
  {
  }

  Array::Array(const Array& other)
      : Expression(other)
  {
    setValues(other.values);
  }

  std::unique_ptr<ASTNode> Array::clone() const
  {
    return std::make_unique<Array>(*this);
  }

  llvm::json::Value Array::toJSON() const
  {
    llvm::json::Object result;

    llvm::SmallVector<llvm::json::Value> valuesJson;

    for (const auto& value : values) {
      valuesJson.push_back(value->toJSON());
    }

    result["values"] = llvm::json::Array(valuesJson);

    addJSONProperties(result);
    return result;
  }

  bool Array::isLValue() const
  {
    return false;
  }

  size_t Array::size() const
  {
    return values.size();
  }

  Expression* Array::operator[](size_t index)
  {
    assert(index < values.size());
    return values[index]->cast<Expression>();
  }

  const Expression* Array::operator[](size_t index) const
  {
    assert(index < values.size());
    return values[index]->cast<Expression>();
  }

  void Array::setValues(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes)
  {
    values.clear();

    for (const auto& node : nodes) {
      assert(node->isa<Expression>());
      auto& clone = values.emplace_back(node->clone());
      clone->setParent(this);
    }
  }
}
