#include "marco/Codegen/Transforms/ModelSolving/Scheduling.h"
#include "marco/Codegen/Transforms/ModelSolving/EquationImpl.h"

using namespace ::marco::modeling;
using namespace ::mlir::modelica;

namespace marco::codegen
{
  ScheduledEquation::ScheduledEquation(
      std::unique_ptr<MatchedEquation> equation,
      IndexSet scheduledIndexes,
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

  EquationInterface ScheduledEquation::cloneIR() const
  {
    return equation->cloneIR();
  }

  void ScheduledEquation::eraseIR()
  {
    equation->eraseIR();
  }

  void ScheduledEquation::dumpIR() const
  {
    equation->dumpIR();
  }

  void ScheduledEquation::dumpIR(llvm::raw_ostream& os) const
  {
    equation->dumpIR(os);
  }

  EquationInterface ScheduledEquation::getOperation() const
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

  Access ScheduledEquation::getAccessAtPath(const EquationPath& path) const
  {
    return equation->getAccessAtPath(path);
  }

  void ScheduledEquation::traversePath(
      const EquationPath& path,
      std::function<bool(mlir::Value)> traverseFn) const
  {
    equation->traversePath(path, std::move(traverseFn));
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
      mlir::OpBuilder& builder,
      const ::marco::modeling::IndexSet& equationIndices,
      const EquationPath& path)
  {
    return equation->explicitate(builder, equationIndices, path);
  }

  std::unique_ptr<Equation> ScheduledEquation::cloneIRAndExplicitate(
      mlir::OpBuilder& builder,
      const ::marco::modeling::IndexSet& equationIndices,
      const EquationPath& path) const
  {
    return equation->cloneIRAndExplicitate(builder, equationIndices, path);
  }

  std::vector<mlir::Value> ScheduledEquation::getInductionVariables() const
  {
    return equation->getInductionVariables();
  }

  mlir::LogicalResult ScheduledEquation::replaceInto(
      mlir::OpBuilder& builder,
      const IndexSet& equationIndices,
      Equation& destination,
      const ::marco::modeling::AccessFunction& destinationAccessFunction,
      const EquationPath& destinationPath) const
  {
    return equation->replaceInto(builder, equationIndices, destination, destinationAccessFunction, destinationPath);
  }

  mlir::func::FuncOp ScheduledEquation::createTemplateFunction(
      llvm::ThreadPool& threadPool,
      mlir::OpBuilder& builder,
      llvm::StringRef functionName,
      ::marco::modeling::scheduling::Direction iterationDirection,
      std::vector<unsigned int>& usedVariables) const
  {
    return equation->createTemplateFunction(threadPool, builder, functionName, iterationDirection, usedVariables);
  }

  size_t ScheduledEquation::getNumOfIterationVars() const
  {
    return scheduledIndexes.rank();
  }

  modeling::IndexSet ScheduledEquation::getIterationRanges() const
  {
    return scheduledIndexes;
  }

  ::marco::modeling::scheduling::Direction ScheduledEquation::getSchedulingDirection() const
  {
    return schedulingDirection;
  }

  std::unique_ptr<Equation> ScheduledEquation::cloneIRAndExplicitate(mlir::OpBuilder& builder) const
  {
    return equation->cloneIRAndExplicitate(builder, getIterationRanges());
  }

  mlir::LogicalResult ScheduledEquation::getCoefficients(
      mlir::OpBuilder& builder,
      std::vector<mlir::Value>& vector,
      mlir::Value& constantTerm,
      ::marco::modeling::Point equationIndex) const
  {
    return equation->getCoefficients(builder, vector, constantTerm, equationIndex);
  }

  mlir::LogicalResult ScheduledEquation::getSideCoefficients(
      mlir::OpBuilder& builder,
      std::vector<mlir::Value>& coefficients,
      mlir::Value& constantTerm,
      std::vector<mlir::Value> values,
      EquationPath::EquationSide side,
      ::marco::modeling::Point equationIndex) const
  {
    return equation->getSideCoefficients(builder, coefficients, constantTerm, values, side, equationIndex);
  }

  mlir::LogicalResult ScheduledEquation::convertAndCollectSide(
      mlir::OpBuilder& builder,
      std::vector<mlir::Value>& output,
      EquationPath::EquationSide side) const
  {
    return equation->convertAndCollectSide(builder, output, side);
  }

  void ScheduledEquation::replaceSides(
      mlir::OpBuilder& builder,
      mlir::Value lhs,
      mlir::Value rhs) const
  {
    return equation->replaceSides(builder, lhs, rhs);
  }

  size_t ScheduledEquation::getFlatAccessIndex(
      const Access& access,
      const ::marco::modeling::Point equationIndex) const
  {
    return equation->getFlatAccessIndex(access, equationIndex);
  }

  mlir::LogicalResult schedule(
      Model<ScheduledEquationsBlock>& result, const Model<MatchedEquation>& model)
  {
    result.setVariables(model.getVariables());
    std::vector<MatchedEquation*> equations;

    for (const auto& equation : model.getEquations()) {
      equations.push_back(equation.get());
    }

    Scheduler<Variable*, MatchedEquation*> scheduler;
    ScheduledEquationsBlocks scheduledBlocks;

    for (const auto& scc : scheduler.schedule(equations)) {
      Equations<ScheduledEquation> scheduledEquations;

      for (const auto& scheduledEquationInfo : scc) {
        auto clone = std::make_unique<MatchedEquation>(*scheduledEquationInfo.getEquation());

        auto scheduledEquation = std::make_unique<ScheduledEquation>(
            std::move(clone), scheduledEquationInfo.getIndexes(), scheduledEquationInfo.getIterationDirection());

        scheduledEquations.push_back(std::move(scheduledEquation));
      }

      auto scheduledEquationsBlock = std::make_unique<ScheduledEquationsBlock>(scheduledEquations, scc.hasCycle());
      scheduledBlocks.append(std::move(scheduledEquationsBlock));
    }

    result.setScheduledBlocks(std::move(scheduledBlocks));
    return mlir::success();
  }
}
