#include "marco/codegen/passes/model/Scheduling.h"

namespace marco::codegen
{
  ScheduledEquation::ScheduledEquation(
      std::unique_ptr<MatchedEquation> equation,
      ::marco::modeling::scheduling::Direction schedulingDirection)
    : equation(std::move(equation)),
      schedulingDirection(std::move(schedulingDirection))
  {
  }

  ScheduledEquation::ScheduledEquation(const ScheduledEquation& other)
    : equation(std::make_unique<MatchedEquation>(*other.equation)),
      schedulingDirection(other.schedulingDirection)
  {
  }

  ScheduledEquation::~ScheduledEquation() = default;

  ScheduledEquation& ScheduledEquation::operator=(const ScheduledEquation& other)
  {
    ScheduledEquation result(other);
    swap(*this, result);
    return *this;
  }

  ScheduledEquation& ScheduledEquation::operator=(ScheduledEquation&& other) = default;

  void swap(ScheduledEquation& first, ScheduledEquation& second)
  {
    using std::swap;
    swap(first.equation, second.equation);
    swap(first.schedulingDirection, second.schedulingDirection);
  }

  std::unique_ptr<Equation> ScheduledEquation::clone() const
  {
    return std::make_unique<ScheduledEquation>(*this);
  }

  modelica::EquationOp ScheduledEquation::getOperation() const
  {
    return equation->getOperation();
  }

  const Variables& ScheduledEquation::getVariables() const
  {
    return equation->getVariables();
  }

  void ScheduledEquation::setVariables(Variables variables)
  {
    equation->setVariables(std::move(variables));
  }

  std::vector<Access> ScheduledEquation::getAccesses() const
  {
    return equation->getAccesses();
  }

  ::marco::modeling::DimensionAccess ScheduledEquation::resolveDimensionAccess(
      std::pair<mlir::Value, long> access) const
  {
    return equation->resolveDimensionAccess(std::move(access));
  }

  std::vector<Access> ScheduledEquation::getReads() const
  {
    return equation->getReads();
  }

  Access ScheduledEquation::getWrite() const
  {
    return equation->getWrite();
  }

  size_t ScheduledEquation::getNumOfIterationVars() const
  {
    return scheduledIndexes.size();
  }

  long ScheduledEquation::getRangeBegin(size_t inductionVarIndex) const
  {
    return scheduledIndexes[inductionVarIndex].first;
  }

  long ScheduledEquation::getRangeEnd(size_t inductionVarIndex) const
  {
    return scheduledIndexes[inductionVarIndex].second;
  }

  ::marco::modeling::scheduling::Direction ScheduledEquation::getSchedulingDirection() const
  {
    return schedulingDirection;
  }

  void ScheduledEquation::setScheduledIndexes(size_t inductionVarIndex, long begin, long end)
  {
    scheduledIndexes.resize(inductionVarIndex + 1);
    scheduledIndexes[inductionVarIndex] = std::make_pair(begin, end);
  }
}
