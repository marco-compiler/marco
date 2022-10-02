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
    auto terminator = mlir::cast<YieldOp>(model.getVarsRegion().back().getTerminator());
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
      Impl(llvm::ArrayRef<std::unique_ptr<Variable>> variables)
      {
        for (const auto& variable : variables) {
          this->variables.push_back(variable->clone());
        }

        llvm::sort(this->variables, [](const auto& x, const auto& y) {
          auto xArgNumber = x->getValue().template cast<mlir::BlockArgument>().getArgNumber();
          auto yArgNumber = y->getValue().template cast<mlir::BlockArgument>().getArgNumber();
          return xArgNumber < yArgNumber;
        });
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

      void getValues(llvm::SmallVectorImpl<mlir::Value>& result) const
      {
        for (const auto& variable : variables) {
          result.push_back(variable->getValue());
        }
      }

      void getTypes(llvm::SmallVectorImpl<mlir::Type>& result) const
      {
        for (const auto& variable : variables) {
          result.push_back(variable->getValue().getType());
        }
      }

    private:
      Variables::Container variables;
  };

  Variables::Variables(llvm::ArrayRef<std::unique_ptr<Variable>> variables)
      : impl(std::make_shared<Impl>(variables))
  {
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

  void Variables::getValues(llvm::SmallVectorImpl<mlir::Value>& result) const
  {
    impl->getValues(result);
  }

  void Variables::getTypes(llvm::SmallVectorImpl<mlir::Type>& result) const
  {
    impl->getTypes(result);
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
