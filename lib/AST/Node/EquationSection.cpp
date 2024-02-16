#include "marco/AST/Node/EquationSection.h"
#include "marco/AST/Node/Equation.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  EquationSection::EquationSection(SourceRange location)
      : ASTNode(ASTNode::Kind::EquationSection, std::move(location))
  {
  }

  EquationSection::EquationSection(const EquationSection& other)
      : ASTNode(other)
  {
    setEquations(other.equations);
  }

  EquationSection::~EquationSection() = default;

  std::unique_ptr<ASTNode> EquationSection::clone() const
  {
    return std::make_unique<EquationSection>(*this);
  }

  llvm::json::Value EquationSection::toJSON() const
  {
    llvm::json::Object result;
    result["initial"] = isInitial();

    llvm::SmallVector<llvm::json::Value> equationsJson;

    for (const auto& equation : equations) {
      equationsJson.push_back(equation->toJSON());
    }

    result["equations"] = llvm::json::Array(equationsJson);

    addJSONProperties(result);
    return result;
  }

  bool EquationSection::isInitial() const
  {
    return initial;
  }

  void EquationSection::setInitial(bool value)
  {
    initial = value;
  }

  size_t EquationSection::getNumOfEquations() const
  {
    return equations.size();
  }

  bool EquationSection::empty() const
  {
    return equations.empty();
  }

  Equation* EquationSection::getEquation(size_t index)
  {
    assert(index < equations.size());
    return equations[index]->cast<Equation>();
  }

  const Equation* EquationSection::getEquation(size_t index) const
  {
    assert(index < equations.size());
    return equations[index]->cast<Equation>();
  }

  llvm::ArrayRef<std::unique_ptr<ASTNode>> EquationSection::getEquations()
  {
    return equations;
  }

  void EquationSection::setEquations(
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
