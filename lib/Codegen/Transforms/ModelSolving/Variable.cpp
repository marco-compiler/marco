#include "marco/Codegen/Transforms/ModelSolving/Variable.h"
#include "marco/Codegen/Transforms/ModelSolving/VariableImpl.h"
#include "marco/Codegen/Transforms/ModelSolving/ScalarVariable.h"
#include "marco/Codegen/Transforms/ModelSolving/ArrayVariable.h"

using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen
{
  std::unique_ptr<Variable> Variable::build(VariableOp variableOp)
  {
    if (!variableOp.getMemberType().isScalar()) {
      return std::make_unique<ArrayVariable>(variableOp);
    }

    return std::make_unique<ScalarVariable>(variableOp);
  }

  Variable::~Variable() = default;

  BaseVariable::BaseVariable(VariableOp variableOp)
    : definingOp(variableOp.getOperation())
  {
  }

  bool Variable::operator==(VariableOp variableOp) const
  {
    return getDefiningOp() == variableOp;
  }

  BaseVariable::Id BaseVariable::getId() const
  {
    return definingOp;
  }

  mlir::modelica::VariableOp BaseVariable::getDefiningOp() const
  {
    return mlir::cast<VariableOp>(definingOp);
  }

  bool BaseVariable::isReadOnly() const
  {
    return getDefiningOp().isReadOnly();
  }

  bool BaseVariable::isParameter() const
  {
    return getDefiningOp().isParameter();
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
        VariableOp op = variable->getDefiningOp();

        assert(positionsMap.find(op.getSymName()) == positionsMap.end() &&
               "Variable already added");

        variables.push_back(std::move(variable));
        positionsMap[op.getSymName()] = variables.size() - 1;
      }

      llvm::Optional<Variable*> findVariable(llvm::StringRef name) const
      {
        if (auto it = positionsMap.find(name); it != positionsMap.end()) {
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
      llvm::StringMap<size_t> positionsMap;
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
    mlir::Operation* definingOp = value.getDefiningOp();

    if (mlir::isa<VariableGetOp>(definingOp)) {
      return true;
    }

    if (auto loadOp = mlir::dyn_cast<LoadOp>(definingOp)) {
      return isReferenceAccess(loadOp.getArray());
    }

    if (auto viewOp = mlir::dyn_cast<SubscriptionOp>(definingOp)) {
      return isReferenceAccess(viewOp.getSource());
    }

    return false;
  }

  llvm::Optional<Variable*> Variables::findVariable(llvm::StringRef name) const
  {
    return impl->findVariable(name);
  }
}
