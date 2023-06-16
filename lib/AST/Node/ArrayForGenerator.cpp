#include "marco/AST/Node/ArrayForGenerator.h"
#include "marco/AST/Node/Induction.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  ArrayForGenerator::ArrayForGenerator(SourceRange location)
      : ArrayGenerator(ASTNode::Kind::Expression_ArrayGenerator_ArrayForGenerator, std::move(location))
  {
  }

  ArrayForGenerator::ArrayForGenerator(const ArrayForGenerator& other)
      : ArrayGenerator(other)
  {
    setValue(other.value->clone());
    setIndices(other.indices);
  }

  std::unique_ptr<ASTNode> ArrayForGenerator::clone() const
  {
    return std::make_unique<ArrayForGenerator>(*this);
  }

  llvm::json::Value ArrayForGenerator::toJSON() const
  {
    llvm::json::Object result;

    result["values"] = value->toJSON();

    llvm::SmallVector<llvm::json::Value> indicesJson;
    for (const auto& index : indices) {
      indicesJson.push_back(index->toJSON());
    }
    result["indices"] = llvm::json::Array(indicesJson);

    addJSONProperties(result);
    return result;
  }

  Expression* ArrayForGenerator::getValue()
  {
    return value->cast<Expression>();
  }

  const Expression* ArrayForGenerator::getValue() const
  {
    return value->cast<Expression>();
  }

  void ArrayForGenerator::setValue(std::unique_ptr<ASTNode> node)
  {
    assert(node->isa<Expression>());
    value = std::move(node);
    value->setParent(this);
  }

  unsigned ArrayForGenerator::getNumIndices() const
  {
    return indices.size();
  }

  Induction* ArrayForGenerator::getIndex(unsigned index)
  {
    assert(index < indices.size());
    return indices[index]->cast<Induction>();
  }

  const Induction* ArrayForGenerator::getIndex(unsigned index) const
  {
    assert(index < indices.size());
    return indices[index]->cast<Induction>();
  }

  void ArrayForGenerator::setIndices(llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes)
  {
    indices.clear();
    for (const auto& node : nodes) {
      assert(node->isa<Induction>());
      auto& clone = indices.emplace_back(node->clone());
      clone->setParent(this);
    }
  }
}
