#include "marco/AST/Node/Induction.h"
#include "marco/AST/Node/Expression.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  Induction::Induction(SourceRange location)
      : ASTNode(ASTNode::Kind::Induction, std::move(location))
  {
  }

  Induction::Induction(const Induction& other)
      : ASTNode(other),
        inductionVariable(other.inductionVariable),
        inductionIndex(other.inductionIndex)
  {
    setBegin(other.begin->clone());
    setEnd(other.end->clone());
    setStep(other.step->clone());
  }

  Induction::~Induction() = default;

  std::unique_ptr<ASTNode> Induction::clone() const
  {
    return std::make_unique<Induction>(*this);
  }

  llvm::json::Value Induction::toJSON() const
  {
    llvm::json::Object result;
    result["induction_variable"] = getName();
    result["induction_index"] = static_cast<int64_t>(inductionIndex);
    result["begin"] = getBegin()->toJSON();
    result["end"] = getEnd()->toJSON();
    result["step"] = getStep()->toJSON();

    addJSONProperties(result);
    return result;
  }

  llvm::StringRef Induction::getName() const
  {
    return inductionVariable;
  }

  void Induction::setName(llvm::StringRef newName)
  {
    inductionVariable = newName.str();
  }

  Expression* Induction::getBegin()
  {
    assert(begin != nullptr && "Begin expression not set");
    return begin->cast<Expression>();
  }

  const Expression* Induction::getBegin() const
  {
    assert(begin != nullptr && "Begin expression not set");
    return begin->cast<Expression>();
  }

  void Induction::setBegin(std::unique_ptr<ASTNode> node)
  {
    assert(node->isa<Expression>());
    begin = std::move(node);
    begin->setParent(this);
  }

  Expression* Induction::getEnd()
  {
    assert(begin != nullptr && "End expression not set");
    return end->cast<Expression>();
  }

  const Expression* Induction::getEnd() const
  {
    assert(end != nullptr && "End expression not set");
    return end->cast<Expression>();
  }

  void Induction::setEnd(std::unique_ptr<ASTNode> node)
  {
    assert(node->isa<Expression>());
    end = std::move(node);
    end->setParent(this);
  }

  Expression* Induction::getStep()
  {
    assert(begin != nullptr && "Step expression not set");
    return step->cast<Expression>();
  }

  const Expression* Induction::getStep() const
  {
    assert(step != nullptr && "Step expression not set");
    return step->cast<Expression>();
  }

  void Induction::setStep(std::unique_ptr<ASTNode> node)
  {
    assert(node->isa<Expression>());
    step = std::move(node);
    step->setParent(this);
  }

  size_t Induction::getInductionIndex() const
  {
    return inductionIndex;
  }

  void Induction::setInductionIndex(size_t index)
  {
    this->inductionIndex = index;
  }
}
