#include "marco/Codegen/Transforms/Model/Scheduling.h"
#include "marco/Codegen/Transforms/Model/EquationImpl.h"

using namespace ::marco::codegen::modelica;
using namespace ::marco::modeling;

namespace marco::codegen
{
  ScheduledEquation::ScheduledEquation(
      std::unique_ptr<MatchedEquation> equation,
      MultidimensionalRange scheduledIndexes,
      scheduling::Direction schedulingDirection)
    : equation(std::move(equation)),
      scheduledIndexes(std::move(scheduledIndexes)),
      schedulingDirection(std::move(schedulingDirection))
  {
    assert(this->equation->getIterationRanges().contains(this->scheduledIndexes));
  }

  ScheduledEquation::ScheduledEquation(const ScheduledEquation& other)
    : equation(std::make_unique<MatchedEquation>(*other.equation)),
      scheduledIndexes(other.scheduledIndexes),
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

  EquationOp ScheduledEquation::cloneIR() const
  {
    return equation->cloneIR();
  }

  void ScheduledEquation::eraseIR()
  {
    equation->eraseIR();
  }

  void ScheduledEquation::dumpIR(llvm::raw_ostream& os) const
  {
    equation->dumpIR(os);
  }

  EquationOp ScheduledEquation::getOperation() const
  {
    return equation->getOperation();
  }

  Variables ScheduledEquation::getVariables() const
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

  mlir::Value ScheduledEquation::getValueAtPath(const EquationPath& path) const
  {
    return equation->getValueAtPath(path);
  }

  std::vector<Access> ScheduledEquation::getReads() const
  {
    return equation->getReads();
  }

  Access ScheduledEquation::getWrite() const
  {
    return equation->getWrite();
  }

  mlir::LogicalResult ScheduledEquation::explicitate(
      mlir::OpBuilder& builder, const EquationPath& path)
  {
    return equation->explicitate(builder, path);
  }

  std::unique_ptr<Equation> ScheduledEquation::cloneAndExplicitate(
      mlir::OpBuilder& builder, const EquationPath& path) const
  {
    return equation->cloneAndExplicitate(builder, path);
  }

  std::unique_ptr<Equation> ScheduledEquation::cloneAndExplicitate(mlir::OpBuilder& builder) const
  {
    return equation->cloneAndExplicitate(builder);
  }

  std::vector<mlir::Value> ScheduledEquation::getInductionVariables() const
  {
    return equation->getInductionVariables();
  }

  mlir::LogicalResult ScheduledEquation::replaceInto(
      mlir::OpBuilder& builder,
      Equation& destination,
      const ::marco::modeling::AccessFunction& destinationAccessFunction,
      const EquationPath& destinationPath) const
  {
    return equation->replaceInto(builder, destination, destinationAccessFunction, destinationPath);
  }

  mlir::FuncOp ScheduledEquation::createTemplateFunction(
      mlir::OpBuilder& builder,
      llvm::StringRef functionName,
      mlir::ValueRange vars,
      ::marco::modeling::scheduling::Direction iterationDirection) const
  {
    return equation->createTemplateFunction(builder, functionName, vars, iterationDirection);
  }

  size_t ScheduledEquation::getNumOfIterationVars() const
  {
    return scheduledIndexes.rank();
  }

  modeling::MultidimensionalRange ScheduledEquation::getIterationRanges() const
  {
    return scheduledIndexes;
  }

  ::marco::modeling::scheduling::Direction ScheduledEquation::getSchedulingDirection() const
  {
    return schedulingDirection;
  }


}
