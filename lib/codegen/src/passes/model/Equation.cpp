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

    Impl(EquationOp equation, llvm::ArrayRef<Variable> variables)
            : equationOp(equation.getOperation()), variables(variables)
    {
      auto terminator = mlir::cast<EquationSidesOp>(equation.body()->getTerminator());
      assert(terminator.lhs().size() == 1);
      assert(terminator.rhs().size() == 1);
    }

    virtual std::unique_ptr<Equation::Impl> clone() const = 0;

    mlir::Operation* getId() const
    {
      return equationOp;
    }

    virtual size_t getNumOfIterationVars() const = 0;
    virtual long getRangeStart(size_t inductionVarIndex) const = 0;
    virtual long getRangeEnd(size_t inductionVarIndex) const = 0;
    virtual void getVariableAccesses(llvm::SmallVectorImpl<Access>& accesses) const = 0;

    EquationOp getOperation() const
    {
      return mlir::cast<EquationOp>(equationOp);
    }

    llvm::Optional<Variable> findVariable(mlir::Value value) const
    {
      assert(value.isa<mlir::BlockArgument>());

      auto it = llvm::find_if(variables, [&](const Variable& variable) {
        return value == variable.getValue();
      });

      if (it == variables.end())
        return llvm::None;

      return *it;
    }

    bool isVariable(mlir::Value value) const
    {
      if (value.isa<mlir::BlockArgument>())
        return mlir::isa<ModelOp>(value.getParentRegion()->getParentOp());

      return false;
    }

    bool isReferenceAccess(mlir::Value value) const
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

    void searchAccesses(
            llvm::SmallVectorImpl<Access>& accesses,
            mlir::Value value) const
    {
      llvm::SmallVector<matching::DimensionAccess> dimensionAccesses;
      searchAccesses(accesses, value, dimensionAccesses);
    }

    void searchAccesses(
            llvm::SmallVectorImpl<Access>& accesses,
            mlir::Value value,
            llvm::SmallVectorImpl<matching::DimensionAccess>& dimensionAccesses) const
    {
      if (isVariable(value))
        resolveAccess(accesses, value, dimensionAccesses);
      else if (mlir::Operation* definingOp = value.getDefiningOp(); definingOp != nullptr)
        searchAccesses(accesses, definingOp, dimensionAccesses);
    }

    void searchAccesses(
            llvm::SmallVectorImpl<Access>& accesses,
            mlir::Operation* op,
            llvm::SmallVectorImpl<matching::DimensionAccess>& dimensionAccesses) const
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
        searchAccesses(accesses, loadOp.memory(), dimensionAccesses);
      }
      else if (auto subscriptionOp = mlir::dyn_cast<SubscriptionOp>(op))
      {
        processIndexesFn(subscriptionOp.indexes());
        searchAccesses(accesses, subscriptionOp.source(), dimensionAccesses);
      }
      else
      {
        for (mlir::Value operand : op->getOperands())
          searchAccesses(accesses, operand);
      }
    }

    void resolveAccess(
            llvm::SmallVectorImpl<Access>& accesses,
            mlir::Value value,
            llvm::SmallVectorImpl<matching::DimensionAccess>& dimensionsAccesses) const
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
          accesses.emplace_back(*variable, reverted);
        }
        else
        {
          if (arrayType.getShape().size() == dimensionsAccesses.size())
            accesses.emplace_back(*variable, reverted);
        }
      }
    }

    std::pair<mlir::Value, long> evaluateDimensionAccess(mlir::Value value) const
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

    virtual matching::DimensionAccess resolveDimensionAccess(std::pair<mlir::Value, long> access) const = 0;

    private:
    mlir::Operation* equationOp;
    llvm::ArrayRef<Variable> variables;
  };
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

  ScalarEquation(EquationOp equation, llvm::ArrayRef<Variable> variables)
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

  std::unique_ptr<Equation::Impl> clone() const override
  {
    return std::make_unique<ScalarEquation>(*this);
  }

  size_t getNumOfIterationVars() const override
  {
    return 1;
  }

  long getRangeStart(size_t inductionVarIndex) const override
  {
    return 0;
  }

  long getRangeEnd(size_t inductionVarIndex) const override
  {
    return 1;
  }

  void getVariableAccesses(llvm::SmallVectorImpl<Access>& accesses) const override
  {
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());

    auto processFn = [&](mlir::Value value) {
        searchAccesses(accesses, value);
    };

    processFn(terminator.lhs()[0]);
    processFn(terminator.rhs()[0]);
  }

  matching::DimensionAccess resolveDimensionAccess(std::pair<mlir::Value, long> access) const override
  {
    assert(access.first == nullptr);
    return matching::DimensionAccess::constant(access.second);
  }
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

/**
 * Loop Equation.
 *
 * An equation that present explicit or implicit induction variables.
 */
class LoopEquation : public Equation::Impl
{
  public:
  using Access = Equation::Impl::Access;

  LoopEquation(EquationOp equation, llvm::ArrayRef<Variable> variables)
          : Impl(equation, variables)
  {
    // Check that there exists at least one explicit or implicit loop
    assert(hasInductionVariables(equation) && "No explicit or implicit loop found");
  }

  std::unique_ptr<Equation::Impl> clone() const override
  {
    return std::make_unique<LoopEquation>(*this);
  }

  size_t getNumOfIterationVars() const override
  {
    return getNumberOfExplicitLoops() + getNumberOfImplicitLoops();
  }

  long getRangeStart(size_t inductionVarIndex) const override
  {
    size_t explicitLoops = getNumberOfExplicitLoops();

    if (inductionVarIndex < explicitLoops)
      return getExplicitLoop(inductionVarIndex).start();

    return getImplicitLoopStart(inductionVarIndex - explicitLoops);
  }

  long getRangeEnd(size_t inductionVarIndex) const override
  {
    size_t explicitLoops = getNumberOfExplicitLoops();

    if (inductionVarIndex < explicitLoops)
      return getExplicitLoop(inductionVarIndex).end() + 1;

    return getImplicitLoopEnd(inductionVarIndex - explicitLoops);
  }

  void getVariableAccesses(llvm::SmallVectorImpl<Access>& accesses) const override
  {
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());
    size_t explicitInductions = getNumberOfExplicitLoops();

    auto processFn = [&](mlir::Value value) {
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

        searchAccesses(accesses, value, implicitDimensionAccesses);
    };

    processFn(terminator.lhs()[0]);
    processFn(terminator.rhs()[0]);
  }

  virtual matching::DimensionAccess resolveDimensionAccess(std::pair<mlir::Value, long> access) const override
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

  private:
  size_t getNumberOfExplicitLoops() const
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

  ForEquationOp getExplicitLoop(size_t index) const
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

  size_t getNumberOfImplicitLoops() const
  {
    size_t result = 0;
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());

    if (auto arrayType = terminator.lhs()[0].getType().dyn_cast<ArrayType>())
      result += arrayType.getRank();

    return result;
  }

  long getImplicitLoopStart(size_t index) const
  {
    assert(index < getNumOfIterationVars() - getNumberOfExplicitLoops());
    return 0;
  }

  long getImplicitLoopEnd(size_t index) const
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
};

Equation::Equation(EquationOp equation, llvm::ArrayRef<Variable> variables)
{
  if (hasInductionVariables(equation))
    impl = std::make_unique<LoopEquation>(equation, variables);
  else
    impl = std::make_unique<ScalarEquation>(equation, variables);
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

void Equation::getVariableAccesses(llvm::SmallVectorImpl<matching::Access<Variable>>& accesses) const
{
  impl->getVariableAccesses(accesses);
}
