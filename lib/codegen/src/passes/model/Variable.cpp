#include <marco/codegen/passes/model/Variable.h>

using namespace marco::codegen;
using namespace marco::codegen::modelica;

namespace marco::codegen
{
  class Variable::Impl
  {
    public:
    Impl(mlir::Value value) : value(value)
    {
      assert(value.isa<mlir::BlockArgument>());
      size_t index = value.cast<mlir::BlockArgument>().getArgNumber();
      auto model = value.getParentRegion()->getParentOfType<ModelOp>();
      auto terminator = mlir::cast<YieldOp>(model.init().back().getTerminator());
      assert(index < terminator.values().size());
      definingOp = terminator.values()[index].getDefiningOp();
      assert(mlir::isa<MemberCreateOp>(definingOp));
    }

    virtual std::unique_ptr<Variable::Impl> clone() const = 0;

    mlir::Operation* getId() const
    {
      return definingOp;
    }

    virtual size_t getRank() const = 0;

    virtual long getDimensionSize(size_t index) const = 0;

    mlir::Value getValue() const
    {
      return value;
    }

    MemberCreateOp getDefiningOp() const
    {
      return mlir::cast<MemberCreateOp>(definingOp);
    }

    private:
    mlir::Value value;
    mlir::Operation* definingOp;
  };
}

/**
 * Variable implementation for scalar values.
 * The arrays declaration are kept untouched within the IR, but they
 * are masked by this class as arrays with just one element.
 */
class ScalarVariable : public Variable::Impl
{
public:
    ScalarVariable(mlir::Value value) : Impl(value)
    {
      assert(value.getType().isa<ArrayType>());
      assert(value.getType().cast<ArrayType>().getRank() == 0);
    }

    std::unique_ptr<Variable::Impl> clone() const override
    {
      return std::make_unique<ScalarVariable>(*this);
    }

    size_t getRank() const override
    {
      return 1;
    }

    long getDimensionSize(size_t index) const override
    {
      return 1;
    }
};

/**
 * Variable implementation for array values.
 * The class just acts as a forwarder.
 */
class ArrayVariable : public Variable::Impl
{
public:
    ArrayVariable(mlir::Value value) : Impl(value)
    {
      assert(value.getType().isa<ArrayType>());
      assert(value.getType().cast<ArrayType>().getRank() != 0);
    }

    std::unique_ptr<Variable::Impl> clone() const override
    {
      return std::make_unique<ArrayVariable>(*this);
    }

    size_t getRank() const override
    {
      return getValue().getType().cast<ArrayType>().getRank();
    }

    long getDimensionSize(size_t index) const override
    {
      return getValue().getType().cast<ArrayType>().getShape()[index];
    }
};

Variable::Variable(mlir::Value value)
{
  if (auto arrayType = value.getType().dyn_cast<ArrayType>(); arrayType.getRank() != 0)
    impl = std::make_unique<ArrayVariable>(value);
  else
    impl = std::make_unique<ScalarVariable>(value);
}

Variable::Variable(const Variable& other)
        : impl(other.impl->clone())
{
}

Variable::~Variable() = default;

Variable& Variable::operator=(const Variable& other)
{
  Variable result(other);
  swap(*this, result);
  return *this;
}

Variable& Variable::operator=(Variable&& other) = default;

namespace marco::codegen
{
  void swap(Variable& first, Variable& second)
  {
    using std::swap;
    swap(first.impl, second.impl);
  }
}

Variable::Id Variable::getId() const
{
  return impl->getId();
}

size_t Variable::getRank() const
{
  return impl->getRank();
}

long Variable::getDimensionSize(size_t index) const
{
  return impl->getDimensionSize(index);
}

mlir::Value Variable::getValue() const
{
  return impl->getValue();
}

MemberCreateOp Variable::getDefiningOp() const
{
  return impl->getDefiningOp();
}

bool Variable::isConstant() const
{
  return getDefiningOp().isConstant();
}