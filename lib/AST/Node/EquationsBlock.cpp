#include "marco/AST/Node/EquationsBlock.h"
#include "marco/AST/Node/Equation.h"
#include "marco/AST/Node/ForEquation.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  EquationsBlock::EquationsBlock(SourceRange location)
      : ASTNode(ASTNode::Kind::EquationsBlock, std::move(location))
  {
  }

  EquationsBlock::EquationsBlock(const EquationsBlock& other)
    : ASTNode(other)
  {
    setEquations(other.equations);
    setForEquations(other.forEquations);
  }

  EquationsBlock::~EquationsBlock() = default;

  std::unique_ptr<ASTNode> EquationsBlock::clone() const
  {
    return std::make_unique<EquationsBlock>(*this);
  }

  llvm::json::Value EquationsBlock::toJSON() const
  {
    llvm::json::Object result;

    llvm::SmallVector<llvm::json::Value> equationsJson;

    for (const auto& equation : equations) {
      equationsJson.push_back(equation->toJSON());
    }

    result["equations"] = llvm::json::Array(equationsJson);

    llvm::SmallVector<llvm::json::Value> forEquationsJson;

    for (const auto& forEquation : forEquations) {
      forEquationsJson.push_back(forEquation->toJSON());
    }

    result["for_equations"] = llvm::json::Array(forEquationsJson);

    addJSONProperties(result);
    return result;
  }

  size_t EquationsBlock::getNumOfEquations() const
  {
    return equations.size();
  }

  Equation* EquationsBlock::getEquation(size_t index)
  {
    assert(index < equations.size());
    return equations[index]->cast<Equation>();
  }

  const Equation* EquationsBlock::getEquation(size_t index) const
  {
    assert(index < equations.size());
    return equations[index]->cast<Equation>();
  }

  void EquationsBlock::setEquations(
      llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes)
  {
    equations.clear();

    for (const auto& node : nodes) {
      addEquation(node->clone());
    }
  }

  void EquationsBlock::addEquation(std::unique_ptr<ASTNode> node)
  {
    assert(node->isa<Equation>());
    auto& clone = equations.emplace_back(node->clone());
    clone->setParent(this);
  }

  size_t EquationsBlock::getNumOfForEquations() const
  {
    return forEquations.size();
  }

  ForEquation* EquationsBlock::getForEquation(size_t index)
  {
    assert(index < forEquations.size());
    return forEquations[index]->cast<ForEquation>();
  }

  const ForEquation* EquationsBlock::getForEquation(size_t index) const
  {
    assert(index < forEquations.size());
    return forEquations[index]->cast<ForEquation>();
  }

  void EquationsBlock::setForEquations(
      llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes)
  {
    forEquations.clear();

    for (const auto& node : nodes) {
      addForEquation(node->clone());
    }
  }

  void EquationsBlock::addForEquation(std::unique_ptr<ASTNode> node)
  {
    assert(node->isa<ForEquation>());
    auto& clone = forEquations.emplace_back(node->clone());
    clone->setParent(this);
  }
}
