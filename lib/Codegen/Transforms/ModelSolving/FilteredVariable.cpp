#include "marco/Codegen/Transforms/ModelSolving/FilteredVariable.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

namespace marco::codegen
{
  FilteredVariable::FilteredVariable(std::unique_ptr<Variable> variable, IndexSet indices)
    : variable(std::move(variable)),
      indices(std::move(indices))
  {
  }

  FilteredVariable::FilteredVariable(const FilteredVariable& other)
      : variable(other.variable->clone()),
        indices(other.indices)
  {
  }

  FilteredVariable::~FilteredVariable() = default;

  FilteredVariable& FilteredVariable::operator=(const FilteredVariable& other)
  {
    FilteredVariable result(other);
    swap(*this, result);
    return *this;
  }

  FilteredVariable& FilteredVariable::operator=(FilteredVariable&& other) = default;

  void swap(FilteredVariable& first, FilteredVariable& second)
  {
    using std::swap;
    swap(first.variable, second.variable);
    swap(first.indices, second.indices);
  }

  std::unique_ptr<Variable> FilteredVariable::clone() const
  {
    return std::make_unique<FilteredVariable>(*this);
  }

  FilteredVariable::Id FilteredVariable::getId() const
  {
    return variable->getId();
  }

  size_t FilteredVariable::getRank() const
  {
    return variable->getRank();
  }

  long FilteredVariable::getDimensionSize(size_t index) const
  {
    return variable->getDimensionSize(index);
  }

  mlir::Value FilteredVariable::getValue() const
  {
    return variable->getValue();
  }

  MemberCreateOp FilteredVariable::getDefiningOp() const
  {
    return variable->getDefiningOp();
  }

  bool FilteredVariable::isParameter() const
  {
    return variable->isParameter();
  }

  IndexSet FilteredVariable::getIndices() const
  {
    return indices;
  }
}
