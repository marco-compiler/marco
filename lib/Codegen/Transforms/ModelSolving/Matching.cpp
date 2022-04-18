#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"

using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

namespace marco::codegen
{
  MatchedEquation::MatchedEquation(
      std::unique_ptr<Equation> equation,
      modeling::MultidimensionalRange matchedIndexes,
      EquationPath matchedPath)
    : equation(std::move(equation)),
      matchedIndexes(std::move(matchedIndexes)),
      matchedPath(std::move(matchedPath))
  {
    assert(this->equation->getIterationRanges().contains(this->matchedIndexes));
  }

  MatchedEquation::MatchedEquation(const MatchedEquation& other)
    : equation(other.equation->clone()),
      matchedIndexes(other.matchedIndexes),
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

  void MatchedEquation::dumpIR(llvm::raw_ostream& os) const
  {
    equation->dumpIR(os);
  }

  EquationOp MatchedEquation::getOperation() const
  {
    return equation->getOperation();
  }

  Variables MatchedEquation::getVariables() const
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

  Access MatchedEquation::getAccessAtPath(const EquationPath& path) const
  {
    return equation->getAccessAtPath(path);
  }

  mlir::LogicalResult MatchedEquation::explicitate(
      mlir::OpBuilder& builder, const EquationPath& path)
  {
    return equation->explicitate(builder, path);
  }

  std::unique_ptr<Equation> MatchedEquation::cloneIRAndExplicitate(
      mlir::OpBuilder& builder, const EquationPath& path) const
  {
    return equation->cloneIRAndExplicitate(builder, path);
  }

  std::unique_ptr<Equation> MatchedEquation::cloneIRAndExplicitate(mlir::OpBuilder& builder) const
  {
    return cloneIRAndExplicitate(builder, getWrite().getPath());
  }

  std::vector<mlir::Value> MatchedEquation::getInductionVariables() const
  {
    return equation->getInductionVariables();
  }

  mlir::LogicalResult MatchedEquation::replaceInto(
      mlir::OpBuilder& builder,
      Equation& destination,
      const ::marco::modeling::AccessFunction& destinationAccessFunction,
      const EquationPath& destinationPath) const
  {
    return equation->replaceInto(builder, destination, destinationAccessFunction, destinationPath);
  }

  size_t MatchedEquation::getNumOfIterationVars() const
  {
    return matchedIndexes.rank();
  }

  modeling::MultidimensionalRange MatchedEquation::getIterationRanges() const
  {
    return matchedIndexes;
  }

  std::vector<Access> MatchedEquation::getReads() const
  {
    std::vector<Access> result;
    auto writeAccess = getWrite();

    for (const auto& access : getAccesses()) {
      if (access.getVariable() != writeAccess.getVariable() ||
          access.getAccessFunction() != writeAccess.getAccessFunction()) {
        result.push_back(access);
      }
    }

    return result;
  }

  Access MatchedEquation::getWrite() const
  {
    return getAccessAtPath(matchedPath);
  }

  mlir::LogicalResult match(
      Model<MatchedEquation>& result,
      Model<Equation>& model,
      const mlir::BlockAndValueMapping& derivatives)
  {
    Variables allVariables = model.getVariables();

    // Filter the variables. State and constant ones must not in fact
    // take part into the matching process as their values are already
    // determined (state variables depend on their derivatives, while
    // constants have a fixed value).

    Variables variables;

    for (const auto& variable : allVariables) {
      mlir::Value var = variable->getValue();

      if (!derivatives.contains(var)) {
        auto nonStateVariable = std::make_unique<Variable>(var);

        if (!nonStateVariable->isConstant()) {
          variables.add(std::move(nonStateVariable));
        }
      }
    }

    model.setVariables(variables);
    result.setVariables(variables);

    model.getEquations().setVariables(model.getVariables());

    MatchingGraph<Variable*, Equation*> matchingGraph;

    for (const auto& variable : model.getVariables()) {
      matchingGraph.addVariable(variable.get());
    }

    for (const auto& equation : model.getEquations()) {
      matchingGraph.addEquation(equation.get());
    }

    // Apply the simplification algorithm to solve the obliged matches
    if (!matchingGraph.simplify()) {
      model.getOperation().emitError("Inconsistency found during the matching simplification process");
      return mlir::failure();
    }

    // Apply the full matching algorithm for the equations and variables that are still unmatched
    if (!matchingGraph.match()) {
      model.getOperation().emitError("Matching failed");
      return mlir::failure();
    }

    Equations<MatchedEquation> matchedEquations;

    for (auto& solution : matchingGraph.getMatch()) {
      auto clone = solution.getEquation()->clone();

      matchedEquations.add(std::make_unique<MatchedEquation>(
          std::move(clone), solution.getIndexes(), solution.getAccess()));
    }

    result.setEquations(matchedEquations);
    return mlir::success();
  }
}
