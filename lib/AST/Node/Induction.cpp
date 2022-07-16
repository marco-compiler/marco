#include "marco/AST/Node/Induction.h"
#include "marco/AST/Node/Expression.h"

using namespace ::marco;
using namespace ::marco::ast;

namespace marco::ast
{
  Induction::Induction(SourceRange location,
                       llvm::StringRef inductionVariable,
                       std::unique_ptr<Expression> begin,
                       std::unique_ptr<Expression> end,
                       std::unique_ptr<Expression> step)
      : ASTNode(std::move(location)),
        inductionVariable(inductionVariable.str()),
        begin(std::move(begin)),
        end(std::move(end)),
        step(std::move(step)),
        inductionIndex(0)
  {
  }

  Induction::Induction(const Induction& other)
      : ASTNode(other),
        inductionVariable(other.inductionVariable),
        begin(other.begin->clone()),
        end(other.end->clone()),
        step(other.step->clone()),
        inductionIndex(other.inductionIndex)
  {
  }

  Induction::Induction(Induction&& other) = default;

  Induction::~Induction() = default;

  Induction& Induction::operator=(const Induction& other)
  {
    Induction result(other);
    swap(*this, result);
    return *this;
  }

  Induction& Induction::operator=(Induction&& other) = default;

  void swap(Induction& first, Induction& second)
  {
    swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

    using std::swap;
    swap(first.inductionVariable, second.inductionVariable);
    swap(first.begin, second.begin);
    swap(first.end, second.end);
    swap(first.step, second.step);
    swap(first.inductionIndex, second.inductionIndex);
  }

  void Induction::print(llvm::raw_ostream& os, size_t indents) const
  {
    os.indent(indents);
    os << "induction var " << getName() << "\n";

    os.indent(indents);
    os << "from ";
    begin->print(os, indents + 1);
    os << "\n";
    os.indent(indents);
    os << "to";
    end->print(os, indents + 1);
    os << "\n";
    os.indent(indents);
    os << "step";
    step->print(os, indents + 1);
  }

  llvm::StringRef Induction::getName() const
  {
    return inductionVariable;
  }

  Expression* Induction::getBegin()
  {
    return begin.get();
  }

  const Expression* Induction::getBegin() const
  {
    return begin.get();
  }

  Expression* Induction::getEnd()
  {
    return end.get();
  }

  const Expression* Induction::getEnd() const
  {
    return end.get();
  }

  Expression* Induction::getStep()
  {
    return step.get();
  }

  const Expression* Induction::getStep() const
  {
    return step.get();
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
