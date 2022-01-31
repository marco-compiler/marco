#include "marco/codegen/dialects/modelica/ModelicaDialect.h"
#include "marco/codegen/passes/model/Matching.h"

using namespace ::marco::codegen;
using namespace ::marco::codegen::modelica;

static mlir::LogicalResult explicitate(
    mlir::OpBuilder& builder,
    modelica::EquationOp equation,
    size_t argumentIndex,
    EquationPath::EquationSide side)
{
  auto terminator = mlir::cast<EquationSidesOp>(equation.body()->getTerminator());
  assert(terminator.lhs().size() == 1);
  assert(terminator.rhs().size() == 1);

  mlir::Value toExplicitate = side == EquationPath::LEFT ? terminator.lhs()[0] : terminator.rhs()[0];
  mlir::Value otherExp = side == EquationPath::RIGHT ? terminator.lhs()[0] : terminator.rhs()[0];

  mlir::Operation* op = toExplicitate.getDefiningOp();

  if (!op->hasTrait<InvertibleOpInterface::Trait>()) {
    return op->emitError("Operation is not invertible");
  }

  return mlir::cast<InvertibleOpInterface>(op).invert(builder, argumentIndex, otherExp);
}

static mlir::LogicalResult explicitate(
    mlir::OpBuilder& builder, modelica::EquationOp equation, const EquationPath& path)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  auto terminator = mlir::cast<EquationSidesOp>(equation.body()->getTerminator());
  builder.setInsertionPoint(terminator);

  for (auto index : path) {
    if (auto status = explicitate(builder, equation, index, path.getEquationSide()); mlir::failed(status)) {
      return status;
    }
  }

  if (path.getEquationSide() == EquationPath::RIGHT) {
    builder.setInsertionPointAfter(terminator);
    builder.create<EquationSidesOp>(terminator->getLoc(), terminator.rhs(), terminator.lhs());
    terminator->erase();
  }

  return mlir::success();
}

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
      mlir::ValueRange vars) const
  {
    return equation->createTemplateFunction(builder, functionName, vars);
  }

  std::unique_ptr<Equation> MatchedEquation::explicitate(mlir::OpBuilder& builder)
  {
    EquationOp clonedOp = cloneIR();

    if (auto status = ::explicitate(builder, clonedOp, getWrite().getPath()); mlir::failed(status)) {
      return nullptr;
    }

    return Equation::build(clonedOp, getVariables());
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
