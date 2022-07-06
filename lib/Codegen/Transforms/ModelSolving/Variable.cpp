#include "marco/Codegen/Transforms/ModelSolving/Variable.h"
#include "marco/Codegen/Transforms/ModelSolving/VariableImpl.h"
#include "marco/Codegen/Transforms/ModelSolving/ScalarVariable.h"
#include "marco/Codegen/Transforms/ModelSolving/ArrayVariable.h"

using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen
{
  std::unique_ptr<Variable> Variable::build(mlir::Value value)
  {
    if (auto arrayType = value.getType().dyn_cast<ArrayType>(); !arrayType.isScalar()) {
      return std::make_unique<ArrayVariable>(value);
    }

    return std::make_unique<ScalarVariable>(value);
  }

  Variable::~Variable() = default;

  BaseVariable::BaseVariable(mlir::Value value)
    : value(value)
  {
    assert(value.isa<mlir::BlockArgument>());
    size_t index = value.cast<mlir::BlockArgument>().getArgNumber();
    auto model = value.getParentRegion()->getParentOfType<ModelOp>();
    auto terminator = mlir::cast<YieldOp>(model.getInitRegion().back().getTerminator());
    assert(index < terminator.getValues().size());
    definingOp = terminator.getValues()[index].getDefiningOp();
    assert(mlir::isa<MemberCreateOp>(definingOp));
  }

  bool Variable::operator==(mlir::Value value) const
  {
    assert(value.isa<mlir::BlockArgument>());
    return getValue() == value;
  }

  BaseVariable::Id BaseVariable::getId() const
  {
    return definingOp;
  }

  mlir::Value BaseVariable::getValue() const
  {
    return value;
  }

  mlir::modelica::MemberCreateOp BaseVariable::getDefiningOp() const
  {
    return mlir::cast<MemberCreateOp>(definingOp);
  }

  bool BaseVariable::isConstant() const
  {
    return getDefiningOp().isConstant();
  }

  //===----------------------------------------------------------------------===//
  // Variables container
  //===----------------------------------------------------------------------===//

  class Variables::Impl
  {
    public:
      void add(std::unique_ptr<Variable> variable)
      {
        values.push_back(variable->getValue().cast<mlir::BlockArgument>());

        llvm::sort(values, [](const auto& x, const auto& y) {
          auto xArgNumber = x.template cast<mlir::BlockArgument>().getArgNumber();
          auto yArgNumber = y.template cast<mlir::BlockArgument>().getArgNumber();
          return xArgNumber < yArgNumber;
        });

        types.clear();

        for (const auto& value : values) {
          types.push_back(value.getType());
        }

        variables.push_back(std::move(variable));
      }

      size_t size() const
      {
        return variables.size();
      }

      std::unique_ptr<Variable>& operator[](size_t index)
      {
        assert(index < size());
        auto& result = variables[index];
        assert(result->getValue().cast<mlir::BlockArgument>().getArgNumber() == index);
        return result;
      }

      const std::unique_ptr<Variable>& operator[](size_t index) const
      {
        assert(index < size());
        const auto& result = variables[index];
        assert(result->getValue().cast<mlir::BlockArgument>().getArgNumber() == index);
        return result;
      }

      Variables::iterator begin()
      {
        return variables.begin();
      }

      Variables::const_iterator begin() const
      {
        return variables.begin();
      }

      Variables::iterator end()
      {
        return variables.end();
      }

      Variables::const_iterator end() const
      {
        return variables.end();
      }

      mlir::ValueRange getValues() const
      {
        return values;
      }

      mlir::TypeRange getTypes() const
      {
        return types;
      }

    private:
      Variables::Container variables;
      std::vector<mlir::Value> values;
      std::vector<mlir::Type> types;
  };

  Variables::Variables() : impl(std::make_shared<Impl>())
  {
  }

  void Variables::add(std::unique_ptr<Variable> variable)
  {
    impl->add(std::move(variable));
  }

  size_t Variables::size() const
  {
    return impl->size();
  }

  std::unique_ptr<Variable>& Variables::operator[](size_t index)
  {
    return (*impl)[index];
  }

  const std::unique_ptr<Variable>& Variables::operator[](size_t index) const
  {
    return (*impl)[index];
  }

  Variables::iterator Variables::begin()
  {
    return impl->begin();
  }

  Variables::const_iterator Variables::begin() const
  {
    return impl->begin();
  }

  Variables::iterator Variables::end()
  {
    return impl->end();
  }

  Variables::const_iterator Variables::end() const
  {
    return impl->end();
  }

  mlir::ValueRange Variables::getValues() const
  {
    return impl->getValues();
  }

  mlir::TypeRange Variables::getTypes() const
  {
    return impl->getTypes();
  }

  bool Variables::isVariable(mlir::Value value) const
  {
    if (value.isa<mlir::BlockArgument>()) {
      return mlir::isa<ModelOp>(value.getParentRegion()->getParentOp());
    }

    return false;
  }

  bool Variables::isReferenceAccess(mlir::Value value) const
  {
    if (isVariable(value)) {
      return true;
    }

    mlir::Operation* definingOp = value.getDefiningOp();

    if (auto loadOp = mlir::dyn_cast<LoadOp>(definingOp)) {
      return isReferenceAccess(loadOp.getArray());
    }

    if (auto viewOp = mlir::dyn_cast<SubscriptionOp>(definingOp)) {
      return isReferenceAccess(viewOp.getSource());
    }

    return false;
  }
}
