#include "marco/AST/Node/ForEquation.h"
#include "marco/AST/Node/Equation.h"
#include "marco/AST/Node/Induction.h"
#include <memory>

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  ForEquation::ForEquation(SourceRange location)
      : ASTNode(ASTNode::Kind::ForEquation, std::move(location))
  {
  }

  ForEquation::ForEquation(const ForEquation& other)
      : ASTNode(other)
  {
    setInductions(other.inductions);
    setEquation(other.equation->clone());
  }

  ForEquation::~ForEquation() = default;

  std::unique_ptr<ASTNode> ForEquation::clone() const
  {
    return std::make_unique<ForEquation>(*this);
  }

  llvm::json::Value ForEquation::toJSON() const
  {
    llvm::json::Object result;

    llvm::SmallVector<llvm::json::Value> inductionsJson;

    for (const auto& induction : inductions) {
      inductionsJson.push_back(induction->toJSON());
    }

    result["inductions"] = llvm::json::Array(inductionsJson);
    result["equation"] = getEquation()->toJSON();

    addJSONProperties(result);
    return result;
  }

  size_t ForEquation::getNumOfInductions() const
  {
    return inductions.size();
  }

  Induction* ForEquation::getInduction(size_t index)
  {
    assert(index < inductions.size());
    return inductions[index]->cast<Induction>();
  }

  const Induction* ForEquation::getInduction(size_t index) const
  {
    assert(index < inductions.size());
    return inductions[index]->cast<Induction>();
  }

  void ForEquation::setInductions(
      llvm::ArrayRef<std::unique_ptr<ASTNode>> nodes)
  {
    inductions.clear();

    for (const auto& node : nodes) {
      assert(node->isa<Induction>());
      auto& clone = inductions.emplace_back(node->clone());
      clone->setParent(this);
    }
  }

  void ForEquation::addOuterInduction(std::unique_ptr<ASTNode> induction)
  {
    assert(induction->isa<Induction>());
    induction->setParent(this);
    inductions.insert(inductions.begin(), std::move(induction));
  }

  Equation* ForEquation::getEquation() const
  {
    assert(equation != nullptr && "Equation not set");
    return equation->cast<Equation>();
  }

  void ForEquation::setEquation(std::unique_ptr<ASTNode> node)
  {
    assert(node->isa<Equation>());
    equation = std::move(node);
    equation->setParent(this);
  }
}
