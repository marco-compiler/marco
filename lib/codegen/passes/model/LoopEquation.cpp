#include "marco/codegen/passes/model/LoopEquation.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include <memory>

using namespace ::marco::codegen::modelica;
using namespace ::marco::modeling;

namespace marco::codegen
{
  LoopEquation::LoopEquation(EquationOp equation, Variables variables)
      : BaseEquation(equation, variables)
  {
  }

  std::unique_ptr<Equation> LoopEquation::clone() const
  {
    return std::make_unique<LoopEquation>(*this);
  }

  EquationOp LoopEquation::cloneIR() const
  {
    EquationOp equationOp = getOperation();
    mlir::OpBuilder builder(equationOp);
    ForEquationOp parent = equationOp->getParentOfType<ForEquationOp>();
    llvm::SmallVector<ForEquationOp, 3> explicitLoops;

    while (parent != nullptr) {
      explicitLoops.push_back(parent);
      parent = parent->getParentOfType<ForEquationOp>();
    }

    mlir::BlockAndValueMapping mapping;
    builder.setInsertionPoint(explicitLoops.back());

    for (auto it = explicitLoops.rbegin(); it != explicitLoops.rend(); ++it) {
      auto loop = builder.create<ForEquationOp>(it->getLoc(), it->start(), it->end());
      builder.setInsertionPointToStart(loop.body());
      mapping.map(it->induction(), loop.induction());
    }

    return mlir::cast<EquationOp>(builder.clone(*equationOp.getOperation(), mapping));
  }

  size_t LoopEquation::getNumOfIterationVars() const
  {
    return getNumberOfExplicitLoops() + getNumberOfImplicitLoops();
  }

  long LoopEquation::getRangeBegin(size_t inductionVarIndex) const
  {
    size_t explicitLoops = getNumberOfExplicitLoops();

    if (inductionVarIndex < explicitLoops) {
      return getExplicitLoop(inductionVarIndex).start();
    }

    return getImplicitLoopStart(inductionVarIndex - explicitLoops);
  }

  long LoopEquation::getRangeEnd(size_t inductionVarIndex) const
  {
    size_t explicitLoops = getNumberOfExplicitLoops();

    if (inductionVarIndex < explicitLoops) {
      return getExplicitLoop(inductionVarIndex).end() + 1;
    }

    return getImplicitLoopEnd(inductionVarIndex - explicitLoops);
  }

  std::vector<Access> LoopEquation::getAccesses() const
  {
    std::vector<Access> accesses;
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());
    size_t explicitInductions = getNumberOfExplicitLoops();

    auto processFn = [&](mlir::Value value, EquationPath path) {
      std::vector<DimensionAccess> implicitDimensionAccesses;
      size_t implicitInductionVar = 0;

      if (auto arrayType = value.getType().dyn_cast<ArrayType>())
      {
        for (size_t i = 0, e = arrayType.getRank(); i < e; ++i)
        {
          auto dimensionAccess = DimensionAccess::relative(explicitInductions + implicitInductionVar, 0);
          implicitDimensionAccesses.push_back(dimensionAccess);
          ++implicitInductionVar;
        }
      }

      searchAccesses(accesses, value, implicitDimensionAccesses, std::move(path));
    };

    processFn(terminator.lhs()[0], EquationPath(EquationPath::LEFT));
    processFn(terminator.rhs()[0], EquationPath(EquationPath::RIGHT));

    return accesses;
  }

  DimensionAccess LoopEquation::resolveDimensionAccess(std::pair<mlir::Value, long> access) const
  {
    if (access.first == nullptr)
      return DimensionAccess::constant(access.second);

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
    return DimensionAccess::relative(inductionVarIndex, access.second);
  }

  mlir::FuncOp LoopEquation::createTemplateFunction(
      mlir::OpBuilder& builder,
      llvm::StringRef functionName,
      mlir::ValueRange vars) const
  {
    return nullptr;
  }

  std::vector<mlir::Value> LoopEquation::createTemplateFunctionLoops(
      mlir::OpBuilder& builder,
      mlir::ValueRange lowerBounds,
      mlir::ValueRange upperBounds,
      mlir::ValueRange steps) const
  {
    // TODO create loops
    return {};
  }

  void LoopEquation::mapIterationVars(mlir::BlockAndValueMapping& mapping, mlir::ValueRange iterationVars) const
  {
    std::vector<ForEquationOp> loops;
    ForEquationOp parent = getOperation()->getParentOfType<ForEquationOp>();

    while (parent != nullptr) {
      loops.push_back(parent);
      parent = parent->getParentOfType<ForEquationOp>();
    }

    assert(loops.size() <= iterationVars.size());

    for (size_t i = 0, e = loops.size(); i < e; ++i) {
      mapping.map(loops[e - i - 1].induction(), iterationVars[i]);
    }
  }

  size_t LoopEquation::getNumberOfExplicitLoops() const
  {
    size_t result = 0;
    ForEquationOp parent = getOperation()->getParentOfType<ForEquationOp>();

    while (parent != nullptr) {
      ++result;
      parent = parent->getParentOfType<ForEquationOp>();
    }

    return result;
  }

  ForEquationOp LoopEquation::getExplicitLoop(size_t index) const
  {
    llvm::SmallVector<ForEquationOp, 3> loops;
    ForEquationOp parent = getOperation()->getParentOfType<ForEquationOp>();

    while (parent != nullptr) {
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

    if (auto arrayType = terminator.lhs()[0].getType().dyn_cast<ArrayType>()) {
      result += arrayType.getRank();
    }

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

    if (auto arrayType = terminator.lhs()[0].getType().dyn_cast<ArrayType>()) {
      for (size_t i = 0; i < arrayType.getRank(); ++i, ++counter) {
        if (counter == index) {
          return arrayType.getShape()[i];
        }
      }
    }

    assert(false && "Implicit loop not found");
    return 0;
  }
}
