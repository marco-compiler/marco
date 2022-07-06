#include "marco/Codegen/Transforms/ModelSolving/Algorithm.h"

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen
{
  Algorithm::Algorithm(AlgorithmOp op, Variables variables)
    : operation(op.getOperation()),
      variables(variables)
  {
  }

  Algorithm::Algorithm(const Algorithm& other)
    : operation(other.operation),
      variables(other.variables)
  {
  }

  Algorithm::~Algorithm() = default;

  Algorithm& Algorithm::operator=(const Algorithm& other)
  {
    Algorithm result(other);
    swap(*this, result);
    return *this;
  }

  Algorithm& Algorithm::operator=(Algorithm&& other) = default;

  void swap(Algorithm& first, Algorithm& second)
  {
    using std::swap;
    swap(first.operation, second.operation);
    swap(first.variables, second.variables);
  }

  std::unique_ptr<Algorithm> Algorithm::clone() const
  {
    return std::make_unique<Algorithm>(*this);
  }

  void Algorithm::dumpIR() const
  {
    dumpIR(llvm::dbgs());
  }

  void Algorithm::dumpIR(llvm::raw_ostream& os) const
  {
    getOperation()->print(os);
  }

  mlir::modelica::AlgorithmOp Algorithm::getOperation() const
  {
    return mlir::cast<AlgorithmOp>(operation);
  }

  Variables Algorithm::getVariables() const
  {
    return variables;
  }

  void Algorithm::setVariables(Variables newVariables)
  {
    variables = std::move(newVariables);
  }

  std::vector<Access> Algorithm::getAccesses() const
  {
    // TODO
  }

  std::vector<Access> Algorithm::getWrites() const
  {
    std::vector<Access> result;

    for (const auto& op : getOperation().bodyBlock()->getOperations()) {
      if (auto memoryEffect = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
        if (memoryEffect.hasEffect<mlir::MemoryEffects::Write>()) {
          llvm::SmallVector<mlir::MemoryEffects::EffectInstance, 2> effects;
          memoryEffect.getEffects(effects);

          for (const auto& effect : effects) {
            if (mlir::isa<mlir::MemoryEffects::Write>(effect.getEffect())) {
              llvm::SmallVector<Access, 1> accesses;


            }
          }
        }
      }
    }

    return result;
  }

  std::vector<Access> Algorithm::getReads() const
  {
    // TODO
  }

  bool Algorithm::isVariable(mlir::Value value) const
  {
    if (value.isa<mlir::BlockArgument>()) {
      return mlir::isa<ModelOp>(value.getParentRegion()->getParentOp());
    }

    return false;
  }

  bool Algorithm::isReferenceAccess(mlir::Value value) const
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
