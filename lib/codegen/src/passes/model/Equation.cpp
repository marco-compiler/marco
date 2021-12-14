#include <marco/codegen/passes/model/Equation.h>

using namespace marco;
using namespace marco::codegen;
using namespace marco::codegen::modelica;

static long getIntFromAttribute(mlir::Attribute attribute)
{
  if (auto indexAttr = attribute.dyn_cast<mlir::IntegerAttr>())
    return indexAttr.getInt();

  if (auto booleanAttr = attribute.dyn_cast<BooleanAttribute>())
    return booleanAttr.getValue() ? 1 : 0;

  if (auto integerAttr = attribute.dyn_cast<IntegerAttribute>())
    return integerAttr.getValue();

  if (auto realAttr = attribute.dyn_cast<RealAttribute>())
    return realAttr.getValue();

  assert(false && "Unknown attribute type");
  return 0;
}

namespace marco::codegen
{
  class Equation::Impl
  {
    public:
    using Access = Equation::Access;

    Impl(EquationOp equation, llvm::ArrayRef<Variable> variables);

      virtual std::unique_ptr<Equation::Impl> clone() const = 0;

    virtual std::unique_ptr<Impl> cloneIR() const = 0;
    virtual void eraseIR() = 0;

    mlir::Operation* getId() const;

    virtual size_t getNumOfIterationVars() const = 0;
    virtual long getRangeStart(size_t inductionVarIndex) const = 0;
    virtual long getRangeEnd(size_t inductionVarIndex) const = 0;
    virtual void getVariableAccesses(llvm::SmallVectorImpl<Access>& accesses) const = 0;

    EquationOp getOperation() const;

    llvm::Optional<Variable> findVariable(mlir::Value value) const;

    bool isVariable(mlir::Value value) const;

    bool isReferenceAccess(mlir::Value value) const;

    void searchAccesses(
            llvm::SmallVectorImpl<Access>& accesses,
            mlir::Value value,
            EquationPath path) const;

    void searchAccesses(
            llvm::SmallVectorImpl<Access>& accesses,
            mlir::Value value,
            llvm::SmallVectorImpl<matching::DimensionAccess>& dimensionAccesses,
            EquationPath path) const;

    void searchAccesses(
            llvm::SmallVectorImpl<Access>& accesses,
            mlir::Operation* op,
            llvm::SmallVectorImpl<matching::DimensionAccess>& dimensionAccesses,
            EquationPath path) const;

    void resolveAccess(
            llvm::SmallVectorImpl<Access>& accesses,
            mlir::Value value,
            llvm::SmallVectorImpl<matching::DimensionAccess>& dimensionsAccesses,
            EquationPath path) const;

    std::pair<mlir::Value, long> evaluateDimensionAccess(mlir::Value value) const;

    virtual matching::DimensionAccess resolveDimensionAccess(std::pair<mlir::Value, long> access) const = 0;

    mlir::LogicalResult explicitate(const EquationPath& path);

    protected:
    llvm::ArrayRef<Variable> getVariables() const;

    private:
    mlir::LogicalResult explicitate(mlir::OpBuilder& builder, size_t argumentIndex, EquationPath::EquationSide side);

    mlir::Operation* equationOp;
    llvm::ArrayRef<Variable> variables;
  };
}

Equation::Impl::Impl(EquationOp equation, llvm::ArrayRef<Variable> variables)
        : equationOp(equation.getOperation()), variables(variables)
{
  auto terminator = mlir::cast<EquationSidesOp>(equation.body()->getTerminator());
  assert(terminator.lhs().size() == 1);
  assert(terminator.rhs().size() == 1);
}

mlir::Operation* Equation::Impl::getId() const
{
  return equationOp;
}

EquationOp Equation::Impl::getOperation() const
{
  return mlir::cast<EquationOp>(equationOp);
}

llvm::Optional<Variable> Equation::Impl::findVariable(mlir::Value value) const
{
  assert(value.isa<mlir::BlockArgument>());

  auto it = llvm::find_if(variables, [&](const Variable& variable) {
      return value == variable.getValue();
  });

  if (it == variables.end())
    return llvm::None;

  return *it;
}

bool Equation::Impl::isVariable(mlir::Value value) const
{
  if (value.isa<mlir::BlockArgument>())
    return mlir::isa<ModelOp>(value.getParentRegion()->getParentOp());

  return false;
}

bool Equation::Impl::isReferenceAccess(mlir::Value value) const
{
  if (isVariable(value))
    return true;

  mlir::Operation* definingOp = value.getDefiningOp();

  if (auto loadOp = mlir::dyn_cast<LoadOp>(definingOp))
    return isReferenceAccess(loadOp.memory());

  if (auto viewOp = mlir::dyn_cast<SubscriptionOp>(definingOp))
    return isReferenceAccess(viewOp.source());

  return false;
}

void Equation::Impl::searchAccesses(
        llvm::SmallVectorImpl<Access>& accesses,
        mlir::Value value,
        EquationPath path) const
{
  llvm::SmallVector<matching::DimensionAccess> dimensionAccesses;
  searchAccesses(accesses, value, dimensionAccesses, std::move(path));
}

void Equation::Impl::searchAccesses(
        llvm::SmallVectorImpl<Access>& accesses,
        mlir::Value value,
        llvm::SmallVectorImpl<matching::DimensionAccess>& dimensionAccesses,
        EquationPath path) const
{
  if (isVariable(value))
    resolveAccess(accesses, value, dimensionAccesses, std::move(path));
  else if (mlir::Operation* definingOp = value.getDefiningOp(); definingOp != nullptr)
    searchAccesses(accesses, definingOp, dimensionAccesses, std::move(path));
}

void Equation::Impl::searchAccesses(
        llvm::SmallVectorImpl<Access>& accesses,
        mlir::Operation* op,
        llvm::SmallVectorImpl<matching::DimensionAccess>& dimensionAccesses,
        EquationPath path) const
{
  auto processIndexesFn = [&](mlir::ValueRange indexes) {
      for (size_t i = 0, e = indexes.size(); i < e; ++i)
      {
        mlir::Value index = indexes[e - 1 - i];
        auto evaluatedAccess = evaluateDimensionAccess(index);
        dimensionAccesses.push_back(resolveDimensionAccess(evaluatedAccess));
      }
  };

  if (auto loadOp = mlir::dyn_cast<LoadOp>(op))
  {
    processIndexesFn(loadOp.indexes());
    searchAccesses(accesses, loadOp.memory(), dimensionAccesses, std::move(path));
  }
  else if (auto subscriptionOp = mlir::dyn_cast<SubscriptionOp>(op))
  {
    processIndexesFn(subscriptionOp.indexes());
    searchAccesses(accesses, subscriptionOp.source(), dimensionAccesses, std::move(path));
  }
  else
  {
    for (size_t i = 0, e = op->getNumOperands(); i < e; ++i)
    {
      EquationPath::Guard guard(path);
      path.append(i);
      searchAccesses(accesses, op->getOperand(i), path);
    }
  }
}

void Equation::Impl::resolveAccess(
        llvm::SmallVectorImpl<Access>& accesses,
        mlir::Value value,
        llvm::SmallVectorImpl<matching::DimensionAccess>& dimensionsAccesses,
        EquationPath path) const
{
  auto variable = findVariable(value);

  if (variable.hasValue())
  {
    llvm::SmallVector<matching::DimensionAccess, 3> reverted(dimensionsAccesses.rbegin(), dimensionsAccesses.rend());
    mlir::Type type = value.getType();

    auto arrayType = type.cast<ArrayType>();

    if (arrayType.getRank() == 0)
    {
      assert(dimensionsAccesses.empty());
      reverted.push_back(matching::DimensionAccess::constant(0));
      accesses.emplace_back(*variable, matching::AccessFunction(reverted), std::move(path));
    }
    else
    {
      if (arrayType.getShape().size() == dimensionsAccesses.size())
        accesses.emplace_back(*variable, matching::AccessFunction(reverted), std::move(path));
    }
  }
}

std::pair<mlir::Value, long> Equation::Impl::evaluateDimensionAccess(mlir::Value value) const
{
  if (value.isa<mlir::BlockArgument>())
    return std::make_pair(value, 0);

  mlir::Operation* op = value.getDefiningOp();
  assert((mlir::isa<ConstantOp>(op) || mlir::isa<AddOp>(op) || mlir::isa<SubOp>(op)) && "Invalid access pattern");

  if (auto constantOp = mlir::dyn_cast<ConstantOp>(op))
    return std::make_pair(nullptr, getIntFromAttribute(constantOp.value()));

  if (auto addOp = mlir::dyn_cast<AddOp>(op))
  {
    auto first = evaluateDimensionAccess(addOp.lhs());
    auto second = evaluateDimensionAccess(addOp.rhs());

    assert(first.first == nullptr || second.first == nullptr);
    mlir::Value induction = first.first != nullptr ? first.first : second.first;
    return std::make_pair(induction, first.second + second.second);
  }

  auto subOp = mlir::dyn_cast<SubOp>(op);

  auto first = evaluateDimensionAccess(subOp.lhs());
  auto second = evaluateDimensionAccess(subOp.rhs());

  assert(first.first == nullptr || second.first == nullptr);
  mlir::Value induction = first.first != nullptr ? first.first : second.first;
  return std::make_pair(induction, first.second - second.second);
}

llvm::ArrayRef<Variable> Equation::Impl::getVariables() const
{
  return variables;
}

mlir::LogicalResult Equation::Impl::explicitate(const EquationPath& path)
{
  auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());
  mlir::OpBuilder builder(terminator);

  for (auto index : path)
  {
    if (auto status = explicitate(builder, index, path.getEquationSide()); mlir::failed(status))
      return status;
  }

  if (path.getEquationSide() == EquationPath::RIGHT)
  {
    builder.setInsertionPointAfter(terminator);
    builder.create<EquationSidesOp>(terminator->getLoc(), terminator.rhs(), terminator.lhs());
    terminator->erase();
  }

  return mlir::success();
}

mlir::LogicalResult Equation::Impl::explicitate(
        mlir::OpBuilder& builder,
        size_t argumentIndex,
        EquationPath::EquationSide side)
{
  auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());
  assert(terminator.lhs().size() == 1);
  assert(terminator.rhs().size() == 1);

  mlir::Value toExplicitate = side == EquationPath::LEFT ? terminator.lhs()[0] : terminator.rhs()[0];
  mlir::Value otherExp = side == EquationPath::RIGHT ? terminator.lhs()[0] : terminator.rhs()[0];

  mlir::Operation* op = toExplicitate.getDefiningOp();

  if (!op->hasTrait<InvertibleOpInterface::Trait>())
    return op->emitError("Operation is not invertible");

  return mlir::cast<InvertibleOpInterface>(op).invert(builder, argumentIndex, otherExp);
}

/**
 * Scalar Equation with Scalar Assignments.
 *
 * An equation that does not present induction variables, neither
 * explicit or implicit.
 */
class ScalarEquation : public Equation::Impl
{
  public:
  using Access = Equation::Impl::Access;

  ScalarEquation(EquationOp equation, llvm::ArrayRef<Variable> variables);

  std::unique_ptr<Equation::Impl> clone() const override;

  std::unique_ptr<Impl> cloneIR() const override;
  void eraseIR() override;

  size_t getNumOfIterationVars() const override;

  long getRangeStart(size_t inductionVarIndex) const override;
  long getRangeEnd(size_t inductionVarIndex) const override;

  void getVariableAccesses(llvm::SmallVectorImpl<Access>& accesses) const override;

  matching::DimensionAccess resolveDimensionAccess(std::pair<mlir::Value, long> access) const override;
};

ScalarEquation::ScalarEquation(EquationOp equation, llvm::ArrayRef<Variable> variables)
        : Impl(equation, variables)
{
  // Check that the equation is not enclosed in a loop
  assert(equation->getParentOfType<ForEquationOp>() == nullptr);

  // Check that all the values are scalars
  auto terminator = mlir::cast<EquationSidesOp>(equation.body()->getTerminator());

  auto isScalarFn = [](mlir::Value value) {
    auto type = value.getType();
    return type.isa<BooleanType>() || type.isa<IntegerType>() || type.isa<RealType>();
  };

  assert(llvm::all_of(terminator.lhs(), isScalarFn));
  assert(llvm::all_of(terminator.rhs(), isScalarFn));
}

std::unique_ptr<Equation::Impl> ScalarEquation::clone() const
{
  return std::make_unique<ScalarEquation>(*this);
}

std::unique_ptr<Equation::Impl> ScalarEquation::cloneIR() const
{
  EquationOp equationOp = getOperation();
  mlir::OpBuilder builder(equationOp);
  auto clone = mlir::cast<EquationOp>(builder.clone(*equationOp.getOperation()));
  return std::make_unique<ScalarEquation>(clone, getVariables());
}

void ScalarEquation::eraseIR()
{
  getOperation().erase();
}

size_t ScalarEquation::getNumOfIterationVars() const
{
  return 1;
}

long ScalarEquation::getRangeStart(size_t inductionVarIndex) const
{
  return 0;
}

long ScalarEquation::getRangeEnd(size_t inductionVarIndex) const
{
  return 1;
}

void ScalarEquation::getVariableAccesses(llvm::SmallVectorImpl<Access>& accesses) const
{
  auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());

  auto processFn = [&](mlir::Value value, EquationPath path) {
    searchAccesses(accesses, value, std::move(path));
  };

  processFn(terminator.lhs()[0], EquationPath(EquationPath::LEFT));
  processFn(terminator.rhs()[0], EquationPath(EquationPath::RIGHT));
}

matching::DimensionAccess ScalarEquation::resolveDimensionAccess(std::pair<mlir::Value, long> access) const
{
  assert(access.first == nullptr);
  return matching::DimensionAccess::constant(access.second);
}

/**
 * Loop Equation.
 *
 * An equation that present explicit or implicit induction variables.
 */
class LoopEquation : public Equation::Impl
{
  public:
  using Access = Equation::Impl::Access;

  LoopEquation(EquationOp equation, llvm::ArrayRef<Variable> variables);

  std::unique_ptr<Equation::Impl> clone() const override;

  std::unique_ptr<Impl> cloneIR() const override;
  void eraseIR() override;

  size_t getNumOfIterationVars() const override;

  long getRangeStart(size_t inductionVarIndex) const override;
  long getRangeEnd(size_t inductionVarIndex) const override;

  void getVariableAccesses(llvm::SmallVectorImpl<Access>& accesses) const override;

  matching::DimensionAccess resolveDimensionAccess(std::pair<mlir::Value, long> access) const override;

  private:
  size_t getNumberOfExplicitLoops() const;

  ForEquationOp getExplicitLoop(size_t index) const;

  size_t getNumberOfImplicitLoops() const;

  long getImplicitLoopStart(size_t index) const;
  long getImplicitLoopEnd(size_t index) const;
};

/**
 * Check if an equation has explicit or implicit induction variables.
 *
 * @param equation  equation
 * @return true if the equation is surrounded by explicit loops or defines implicit ones
 */
static bool hasInductionVariables(EquationOp equation)
{
  auto hasExplicitLoops = [&]() -> bool {
      return equation->getParentOfType<ForEquationOp>() != nullptr;
  };

  auto hasImplicitLoops = [&]() -> bool {
      auto terminator = mlir::cast<EquationSidesOp>(equation.body()->getTerminator());

      return llvm::any_of(terminator.lhs(), [](mlir::Value value) {
          return value.getType().isa<ArrayType>();
      });
  };

  return hasExplicitLoops() || hasImplicitLoops();
}

LoopEquation::LoopEquation(EquationOp equation, llvm::ArrayRef<Variable> variables)
        : Impl(equation, variables)
{
  // Check that there exists at least one explicit or implicit loop
  assert(hasInductionVariables(equation) && "No explicit or implicit loop found");
}

std::unique_ptr<Equation::Impl> LoopEquation::clone() const
{
  return std::make_unique<LoopEquation>(*this);
}

std::unique_ptr<Equation::Impl> LoopEquation::cloneIR() const
{
  EquationOp equationOp = getOperation();
  mlir::OpBuilder builder(equationOp);

  ForEquationOp parent = equationOp->getParentOfType<ForEquationOp>();
  llvm::SmallVector<ForEquationOp, 3> explicitLoops;

  while (parent != nullptr)
  {
    explicitLoops.push_back(parent);
    parent = parent->getParentOfType<ForEquationOp>();
  }

  mlir::BlockAndValueMapping mapping;
  builder.setInsertionPoint(explicitLoops.back());

  for (auto it = explicitLoops.rbegin(); it != explicitLoops.rend(); ++it)
  {
    auto loop = builder.create<ForEquationOp>(it->getLoc(), it->start(), it->end());
    builder.setInsertionPointToStart(loop.body());
    mapping.map(it->induction(), loop.induction());
  }

  auto clone = mlir::cast<EquationOp>(builder.clone(*equationOp.getOperation(), mapping));
  return std::make_unique<LoopEquation>(clone, getVariables());
}

void LoopEquation::eraseIR()
{
  EquationOp equationOp = getOperation();
  ForEquationOp parent = equationOp->getParentOfType<ForEquationOp>();
  equationOp.erase();

  while (parent != nullptr)
  {
    ForEquationOp newParent = parent->getParentOfType<ForEquationOp>();
    parent->erase();
    parent = newParent;
  }
}

size_t LoopEquation::getNumOfIterationVars() const
{
  return getNumberOfExplicitLoops() + getNumberOfImplicitLoops();
}

long LoopEquation::getRangeStart(size_t inductionVarIndex) const
{
  size_t explicitLoops = getNumberOfExplicitLoops();

  if (inductionVarIndex < explicitLoops)
    return getExplicitLoop(inductionVarIndex).start();

  return getImplicitLoopStart(inductionVarIndex - explicitLoops);
}

long LoopEquation::getRangeEnd(size_t inductionVarIndex) const
{
  size_t explicitLoops = getNumberOfExplicitLoops();

  if (inductionVarIndex < explicitLoops)
    return getExplicitLoop(inductionVarIndex).end() + 1;

  return getImplicitLoopEnd(inductionVarIndex - explicitLoops);
}

void LoopEquation::getVariableAccesses(llvm::SmallVectorImpl<Access>& accesses) const
{
  auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());
  size_t explicitInductions = getNumberOfExplicitLoops();

  auto processFn = [&](mlir::Value value, EquationPath path) {
    llvm::SmallVector<matching::DimensionAccess> implicitDimensionAccesses;
    size_t implicitInductionVar = 0;

    if (auto arrayType = value.getType().dyn_cast<ArrayType>())
    {
      for (size_t i = 0, e = arrayType.getRank(); i < e; ++i)
      {
        auto dimensionAccess = matching::DimensionAccess::relative(explicitInductions + implicitInductionVar, 0);
        implicitDimensionAccesses.push_back(dimensionAccess);
        ++implicitInductionVar;
      }
    }

    searchAccesses(accesses, value, implicitDimensionAccesses, std::move(path));
  };

  processFn(terminator.lhs()[0], EquationPath(EquationPath::LEFT));
  processFn(terminator.rhs()[0], EquationPath(EquationPath::RIGHT));
}

matching::DimensionAccess LoopEquation::resolveDimensionAccess(std::pair<mlir::Value, long> access) const
{
  if (access.first == nullptr)
    return matching::DimensionAccess::constant(access.second);

  llvm::SmallVector<ForEquationOp, 3> loops;
  ForEquationOp parent = getOperation()->getParentOfType<ForEquationOp>();

  while (parent != nullptr)
  {
    loops.push_back(parent);
    parent = parent->getParentOfType<ForEquationOp>();
  }

  auto loopIt = llvm::find_if(loops, [&](ForEquationOp loop) {
    return loop.induction() == access.first;
  });

  size_t inductionVarIndex = loops.end() - loopIt - 1;
  return matching::DimensionAccess::relative(inductionVarIndex, access.second);
}

size_t LoopEquation::getNumberOfExplicitLoops() const
{
  size_t result = 0;
  ForEquationOp parent = getOperation()->getParentOfType<ForEquationOp>();

  while (parent != nullptr)
  {
    ++result;
    parent = parent->getParentOfType<ForEquationOp>();
  }

  return result;
}

ForEquationOp LoopEquation::getExplicitLoop(size_t index) const
{
  llvm::SmallVector<ForEquationOp, 3> loops;
  ForEquationOp parent = getOperation()->getParentOfType<ForEquationOp>();

  while (parent != nullptr)
  {
    loops.push_back(parent);
    parent = parent->getParentOfType<ForEquationOp>();
  }

  assert(index < loops.size());
  return loops[loops.size() - 1 - index];
}

size_t LoopEquation::getNumberOfImplicitLoops() const
{
  size_t result = 0;
  auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());

  if (auto arrayType = terminator.lhs()[0].getType().dyn_cast<ArrayType>())
    result += arrayType.getRank();

  return result;
}

long LoopEquation::getImplicitLoopStart(size_t index) const
{
  assert(index < getNumOfIterationVars() - getNumberOfExplicitLoops());
  return 0;
}

long LoopEquation::getImplicitLoopEnd(size_t index) const
{
  assert(index < getNumOfIterationVars() - getNumberOfExplicitLoops());

  size_t counter = 0;
  auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());

  if (auto arrayType = terminator.lhs()[0].getType().dyn_cast<ArrayType>())
    for (size_t i = 0; i < arrayType.getRank(); ++i, ++counter)
      if (counter == index)
        return arrayType.getShape()[i];

  assert(false && "Implicit loop not found");
  return 0;
}

Equation::Equation(EquationOp equation, llvm::ArrayRef<Variable> variables)
{
  if (hasInductionVariables(equation))
    impl = std::make_unique<LoopEquation>(equation, variables);
  else
    impl = std::make_unique<ScalarEquation>(equation, variables);
}

Equation::Equation(std::unique_ptr<Equation::Impl> impl) : impl(std::move(impl))
{
}

Equation::Equation(const Equation& other)
        : impl(other.impl->clone())
{
}

Equation::~Equation() = default;

Equation& Equation::operator=(const Equation& other)
{
  Equation result(other);
  swap(*this, result);
  return *this;
}

Equation& Equation::operator=(Equation&& other) = default;

namespace marco::codegen
{
  void swap(Equation& first, Equation& second)
  {
    using std::swap;
    swap(first.impl, second.impl);
  }
}

Equation Equation::cloneIR() const
{
  return Equation(impl->cloneIR());
}

void Equation::eraseIR()
{
  impl->eraseIR();
}

Equation::Id Equation::getId() const
{
  return impl->getId();
}

size_t Equation::getNumOfIterationVars() const
{
  return impl->getNumOfIterationVars();
}

long Equation::getRangeStart(size_t inductionVarIndex) const
{
  return impl->getRangeStart(inductionVarIndex);
}

long Equation::getRangeEnd(size_t inductionVarIndex) const
{
  return impl->getRangeEnd(inductionVarIndex);
}

void Equation::getVariableAccesses(llvm::SmallVectorImpl<Equation::Access>& accesses) const
{
  impl->getVariableAccesses(accesses);
}

mlir::LogicalResult Equation::explicitate(const EquationPath& path)
{
  return impl->explicitate(path);
}
