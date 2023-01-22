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

  bool BaseVariable::isParameter() const
  {
    return getDefiningOp().isParameter();
  }

  //===----------------------------------------------------------------------===//
  // Variables container
  //===----------------------------------------------------------------------===//

  class Variables::Impl
  {
    public:
      size_t size() const
      {
        return variables.size();
      }

      const std::unique_ptr<Variable>& operator[](size_t index) const
      {
        assert(index < size());
        const auto& result = variables[index];
        return result;
      }

      void add(std::unique_ptr<Variable> variable)
      {
        mlir::Value value = variable->getValue();
        unsigned int argNumber = value.cast<mlir::BlockArgument>().getArgNumber();
        assert(positionsMap.find(argNumber) == positionsMap.end() && "Variable already added");

        variables.push_back(std::move(variable));
        positionsMap[argNumber] = variables.size() - 1;
      }

      llvm::Optional<Variable*> findVariable(mlir::Value value) const
      {
        unsigned int argNumber = value.cast<mlir::BlockArgument>().getArgNumber();

        if (auto it = positionsMap.find(argNumber); it != positionsMap.end()) {
          return variables[it->second].get();
        }

        return llvm::None;
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

    private:
      Variables::Container variables;
      std::map<unsigned int, size_t> positionsMap;
  };

  Variables::Variables()
      : impl(std::make_shared<Impl>())
  {
  }

  size_t Variables::size() const
  {
    return impl->size();
  }

  const std::unique_ptr<Variable>& Variables::operator[](size_t index) const
  {
    return (*impl)[index];
  }

  void Variables::add(std::unique_ptr<Variable> variable)
  {
    impl->add(std::move(variable));
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

  llvm::Optional<Variable*> Variables::findVariable(mlir::Value value) const
  {
    return impl->findVariable(value);
  }
}
