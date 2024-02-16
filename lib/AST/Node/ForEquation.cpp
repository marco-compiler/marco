#include "marco/AST/Node/ForEquation.h"
#include "marco/AST/Node/Equation.h"
#include "marco/AST/Node/ForIndex.h"
#include <memory>

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  ForEquation::ForEquation(SourceRange location)
      : Equation(ASTNode::Kind::Equation_For, std::move(location))
  {
  }

  ForEquation::ForEquation(const ForEquation& other)
      : Equation(other)
  {
    setForIndices(other.forIndices);
    setEquations(other.equations);
  }

  ForEquation::~ForEquation() = default;

  std::unique_ptr<ASTNode> ForEquation::clone() const
  {
    return std::make_unique<ForEquation>(*this);
  }

  llvm::json::Value ForEquation::toJSON() const
  {
    llvm::json::Object result;
    llvm::SmallVector<llvm::json::Value> forIndicesJson;

    for (const auto& forIndex : forIndices) {
      forIndicesJson.push_back(forIndex->toJSON());
    }

    llvm::SmallVector<llvm::json::Value> equationsJson;

    for (const auto& equation : equations) {
      equationsJson.push_back(equation->toJSON());
    }

    result["for_indices"] = llvm::json::Array(forIndicesJson);
    result["equations"] = llvm::json::Array(equationsJson);

    addJSONProperties(result);
    return result;
  }

  size_t ForEquation::getNumOfForIndices() const
  {
    return forIndices.size();
  }

  ForIndex* ForEquation::getForIndex(size_t index)
  {
    assert(index < forIndices.size());
    return forIndices[index]->cast<ForIndex>();
  }

  const ForIndex* ForEquation::getForIndex(size_t index) const
  {
    assert(index < forIndices.size());
    return forIndices[index]->cast<ForIndex>();
  }

  void ForEquation::setForIndices(
      llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes)
  {
    forIndices.clear();

    for (const auto& node : nodes) {
      assert(node->isa<ForIndex>());
      auto& clone = forIndices.emplace_back(node->clone());
      clone->setParent(this);
    }
  }

  size_t ForEquation::getNumOfEquations() const
  {
    return equations.size();
  }

  Equation* ForEquation::getEquation(size_t index)
  {
    assert(index < equations.size());
    return equations[index]->cast<Equation>();
  }

  const Equation* ForEquation::getEquation(size_t index) const
  {
    assert(index < equations.size());
    return equations[index]->cast<Equation>();
  }

  void ForEquation::setEquations(
      llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes)
  {
    equations.clear();

    for (const auto& node : nodes) {
      assert(node->isa<Equation>());
      auto& clone = equations.emplace_back(node->clone());
      clone->setParent(this);
    }
  }
}
