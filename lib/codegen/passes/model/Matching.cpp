#include "marco/codegen/dialects/modelica/ModelicaDialect.h"
#include "marco/codegen/passes/model/Matching.h"

using namespace ::marco::codegen;
using namespace ::marco::codegen::modelica;

namespace marco::codegen
{
  MatchedEquation::MatchedEquation(std::unique_ptr<Equation> equation, EquationPath matchedPath)
    : equation(std::move(equation)),
      matchedPath(std::move(matchedPath))
  {
  }

  MatchedEquation::MatchedEquation(const MatchedEquation& other)
    : equation(other.equation->clone()),
      matchedIndexes(other.matchedIndexes.begin(), other.matchedIndexes.end()),
      matchedPath(other.matchedPath)
  {
  }

  MatchedEquation::~MatchedEquation() = default;

  MatchedEquation& MatchedEquation::operator=(const MatchedEquation& other)
  {
    MatchedEquation result(other);
    swap(*this, result);
    return *this;
  }

  MatchedEquation& MatchedEquation::operator=(MatchedEquation&& other) = default;

  void swap(MatchedEquation& first, MatchedEquation& second)
  {
    using std::swap;
    swap(first.equation, second.equation);
    swap(first.matchedIndexes, second.matchedIndexes);
    swap(first.matchedPath, second.matchedPath);
  }

  std::unique_ptr<Equation> MatchedEquation::clone() const
  {
    return std::make_unique<MatchedEquation>(*this);
  }

  EquationOp MatchedEquation::cloneIR() const
  {
    return equation->cloneIR();
  }

  void MatchedEquation::eraseIR()
  {
    equation->eraseIR();
  }

  void MatchedEquation::dumpIR() const
  {
    equation->dumpIR();
  }

  EquationOp MatchedEquation::getOperation() const
  {
    return equation->getOperation();
  }

  const Variables& MatchedEquation::getVariables() const
  {
    return equation->getVariables();
  }

  void MatchedEquation::setVariables(Variables variables)
  {
    equation->setVariables(std::move(variables));
  }

  std::vector<Access> MatchedEquation::getAccesses() const
  {
    return equation->getAccesses();
  }

  ::marco::modeling::DimensionAccess MatchedEquation::resolveDimensionAccess(
      std::pair<mlir::Value, long> access) const
  {
    return equation->resolveDimensionAccess(std::move(access));
  }

  mlir::FuncOp MatchedEquation::createTemplateFunction(
      mlir::OpBuilder& builder,
      llvm::StringRef functionName,
      mlir::ValueRange vars,
      ::marco::modeling::scheduling::Direction iterationDirection) const
  {
    return equation->createTemplateFunction(builder, functionName, vars, iterationDirection);
  }

  mlir::Value MatchedEquation::getValueAtPath(const EquationPath& path) const
  {
    return equation->getValueAtPath(path);
  }

  mlir::LogicalResult MatchedEquation::explicitate(
      mlir::OpBuilder& builder, const EquationPath& path)
  {
    return equation->explicitate(builder, path);
  }

  std::unique_ptr<Equation> MatchedEquation::cloneAndExplicitate(
      mlir::OpBuilder& builder, const EquationPath& path) const
  {
    return equation->cloneAndExplicitate(builder, path);
  }

  std::unique_ptr<Equation> MatchedEquation::cloneAndExplicitate(mlir::OpBuilder& builder) const
  {
    return cloneAndExplicitate(builder, getWrite().getPath());
  }

  std::vector<mlir::Value> MatchedEquation::getInductionVariables() const
  {
    return equation->getInductionVariables();
  }

  mlir::LogicalResult MatchedEquation::replaceInto(
      mlir::OpBuilder& builder,
      Equation& destination,
      const ::marco::modeling::AccessFunction& destinationAccessFunction,
      const EquationPath& destinationPath,
      const Access& sourceAccess) const
  {
    return equation->replaceInto(builder, destination, destinationAccessFunction, destinationPath);
  }

  size_t MatchedEquation::getNumOfIterationVars() const
  {
    return matchedIndexes.size();
  }

  long MatchedEquation::getRangeBegin(size_t inductionVarIndex) const
  {
    return matchedIndexes[inductionVarIndex].first;
  }

  long MatchedEquation::getRangeEnd(size_t inductionVarIndex) const
  {
    return matchedIndexes[inductionVarIndex].second;
  }

  std::vector<Access> MatchedEquation::getReads() const
  {
    std::vector<Access> result;

    for (const auto& access : getAccesses()) {
      if (access.getPath() != matchedPath) {
        result.push_back(access);
      }
    }

    return result;
  }

  Access MatchedEquation::getWrite() const
  {
    return getAccessFromPath(matchedPath);
  }

  void MatchedEquation::setMatchedIndexes(size_t inductionVarIndex, long begin, long end)
  {
    matchedIndexes.resize(inductionVarIndex + 1);
    matchedIndexes[inductionVarIndex] = std::make_pair(begin, end);
  }
}
