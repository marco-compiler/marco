#include "marco/Codegen/Transforms/ModelSolving/LoopEquation.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace ::marco::modeling;
using namespace ::mlir::modelica;

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

    if (explicitLoops.empty()) {
      builder.setInsertionPointAfter(getOperation());
    } else {
      builder.setInsertionPointAfter(explicitLoops.back());
    }

    for (auto it = explicitLoops.rbegin(); it != explicitLoops.rend(); ++it) {
      long from = it->from().getSExtValue();
      long to = it->to().getSExtValue();
      auto loop = builder.create<ForEquationOp>(it->getLoc(), from, to);
      builder.setInsertionPointToStart(loop.bodyBlock());
      mapping.map(it->induction(), loop.induction());
    }

    return mlir::cast<EquationOp>(builder.clone(*equationOp.getOperation(), mapping));
  }

  void LoopEquation::eraseIR()
  {
    EquationOp equationOp = getOperation();
    ForEquationOp parent = equationOp->getParentOfType<ForEquationOp>();
    equationOp.erase();

    while (parent != nullptr) {
      ForEquationOp newParent = parent->getParentOfType<ForEquationOp>();
      parent->erase();
      parent = newParent;
    }
  }

  void LoopEquation::dumpIR(llvm::raw_ostream& os) const
  {
    EquationOp equationOp = getOperation();
    mlir::Operation* op = equationOp.getOperation();

    while (auto parent = op->getParentOfType<ForEquationOp>()) {
      op = parent.getOperation();
    }

    op->print(os);
  }

  size_t LoopEquation::getNumOfIterationVars() const
  {
    return getNumberOfExplicitLoops() + getNumberOfImplicitLoops();
  }

  MultidimensionalRange LoopEquation::getIterationRanges() const
  {
    std::vector<Range> ranges;

    for (auto& explicitLoop : getExplicitLoops()) {
      ranges.emplace_back(explicitLoop.from().getSExtValue(), explicitLoop.to().getSExtValue() + 1);
    }

    auto implicitLoops = getImplicitLoops();
    ranges.insert(ranges.end(), implicitLoops.begin(), implicitLoops.end());

    return MultidimensionalRange(std::move(ranges));
  }

  std::vector<Access> LoopEquation::getAccesses() const
  {
    std::vector<Access> accesses;
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().bodyBlock()->getTerminator());
    size_t explicitInductions = getNumberOfExplicitLoops();

    auto processFn = [&](mlir::Value value, EquationPath path) {
      std::vector<DimensionAccess> implicitDimensionAccesses;

      if (auto arrayType = value.getType().dyn_cast<ArrayType>()) {
        size_t implicitInductionVar = 0;

        for (size_t i = 0, e = arrayType.getRank(); i < e; ++i) {
          auto dimensionAccess = DimensionAccess::relative(explicitInductions + implicitInductionVar, 0);
          implicitDimensionAccesses.push_back(dimensionAccess);
          ++implicitInductionVar;
        }
      }

      std::reverse(implicitDimensionAccesses.begin(), implicitDimensionAccesses.end());
      searchAccesses(accesses, value, implicitDimensionAccesses, std::move(path));
    };

    processFn(terminator.lhsValues()[0], EquationPath(EquationPath::LEFT));
    processFn(terminator.rhsValues()[0], EquationPath(EquationPath::RIGHT));

    return accesses;
  }

  DimensionAccess LoopEquation::resolveDimensionAccess(std::pair<mlir::Value, long> access) const
  {
    if (access.first == nullptr) {
      return DimensionAccess::constant(access.second);
    }

    llvm::SmallVector<ForEquationOp, 3> loops;
    ForEquationOp parent = getOperation()->getParentOfType<ForEquationOp>();

    while (parent != nullptr) {
      loops.push_back(parent);
      parent = parent->getParentOfType<ForEquationOp>();
    }

    auto loopIt = llvm::find_if(loops, [&](ForEquationOp loop) {
      return loop.induction() == access.first;
    });

    size_t inductionVarIndex = loops.end() - loopIt - 1;
    return DimensionAccess::relative(inductionVarIndex, access.second);
  }

  Access LoopEquation::getAccessAtPath(const EquationPath& path) const
  {
    mlir::Value access = getValueAtPath(path);
    std::vector<Access> accesses;

    size_t explicitInductions = getNumberOfExplicitLoops();
    std::vector<DimensionAccess> implicitDimensionAccesses;

    if (auto arrayType = access.getType().dyn_cast<ArrayType>()) {
      size_t implicitInductionVar = 0;

      for (size_t i = 0, e = arrayType.getRank(); i < e; ++i) {
        auto dimensionAccess = DimensionAccess::relative(explicitInductions + implicitInductionVar, 0);
        implicitDimensionAccesses.push_back(dimensionAccess);
        ++implicitInductionVar;
      }
    }

    std::reverse(implicitDimensionAccesses.begin(), implicitDimensionAccesses.end());
    searchAccesses(accesses, access, implicitDimensionAccesses, std::move(path));

    assert(accesses.size() == 1);
    return accesses[0];
  }

  std::vector<mlir::Value> LoopEquation::getInductionVariables() const
  {
    std::vector<mlir::Value> explicitInductionVariables;

    for (auto explicitLoop : getExplicitLoops()) {
      explicitInductionVariables.push_back(explicitLoop.induction());
    }

    return explicitInductionVariables;
  }

  mlir::LogicalResult LoopEquation::mapInductionVariables(
      mlir::OpBuilder& builder,
      mlir::BlockAndValueMapping& mapping,
      Equation& destination,
      const ::marco::modeling::AccessFunction& transformation) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto destinationInductionVariables = destination.getInductionVariables();

    auto explicitLoops = getExplicitLoops();

    if (explicitLoops.size() > transformation.size()) {
      // Can't map all the induction variables. An IR substitution is not possible.
      return mlir::failure();
    }

    for (size_t i = 0, e = explicitLoops.size(); i < e; ++i) {
      auto dimensionAccess = DimensionAccess::relative(i, 0);
      auto combinedDimensionAccess = transformation.combine(dimensionAccess);

      if (combinedDimensionAccess.isConstantAccess()) {
        builder.setInsertionPointToStart(destination.getOperation().bodyBlock());

        mlir::Value constantAccess = builder.create<ConstantOp>(
            explicitLoops[i].getLoc(), IntegerAttr::get(builder.getContext(), combinedDimensionAccess.getPosition()));

        mapping.map(explicitLoops[i].induction(), constantAccess);
      } else {
        mlir::Value mapped = destinationInductionVariables[combinedDimensionAccess.getInductionVariableIndex()];

        if (combinedDimensionAccess.getOffset() != 0) {
          builder.setInsertionPointToStart(destination.getOperation().bodyBlock());

          mlir::Value offset = builder.create<ConstantOp>(
              explicitLoops[i].getLoc(), IntegerAttr::get(builder.getContext(), combinedDimensionAccess.getOffset()));

          mapped = builder.create<AddOp>(offset.getLoc(), offset.getType(), mapped, offset);
        }

        mapping.map(explicitLoops[i].induction(), mapped);
      }
    }

    return mlir::success();
  }

  mlir::LogicalResult LoopEquation::createTemplateFunctionBody(
      mlir::OpBuilder& builder,
      mlir::BlockAndValueMapping& mapping,
      mlir::ValueRange beginIndexes,
      mlir::ValueRange endIndexes,
      mlir::ValueRange steps,
      ::marco::modeling::scheduling::Direction iterationDirection) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto equation = getOperation();
    auto loc = equation.getLoc();

    auto numberOfExplicitLoops = getNumberOfExplicitLoops();

    assert(beginIndexes.size() == numberOfExplicitLoops + getNumberOfImplicitLoops());
    assert(endIndexes.size() == numberOfExplicitLoops + getNumberOfImplicitLoops());
    assert(steps.size() == numberOfExplicitLoops + getNumberOfImplicitLoops());

    std::vector<mlir::Value> explicitBeginIndices;
    std::vector<mlir::Value> explicitEndIndices;
    std::vector<mlir::Value> explicitSteps;

    std::vector<mlir::Value> implicitBeginIndices;
    std::vector<mlir::Value> implicitEndIndices;
    std::vector<mlir::Value> implicitSteps;

    for (size_t i = 0; i < steps.size(); ++i) {
      if (i < numberOfExplicitLoops) {
        explicitBeginIndices.push_back(beginIndexes[i]);
        explicitEndIndices.push_back(endIndexes[i]);
        explicitSteps.push_back(steps[i]);
      } else {
        implicitBeginIndices.push_back(beginIndexes[i + numberOfExplicitLoops]);
        implicitEndIndices.push_back(endIndexes[i + numberOfExplicitLoops]);
        implicitSteps.push_back(steps[i + numberOfExplicitLoops]);
      }
    }

    createIterationLoops(
        builder, loc, explicitBeginIndices, explicitEndIndices, explicitSteps, iterationDirection,
        [&](mlir::OpBuilder& nestedExplicitBuilder, mlir::ValueRange explicitIndices) {
          auto explicitLoops = getExplicitLoops();
          assert(explicitLoops.size() == explicitIndices.size());

          for (size_t i = 0; i < explicitLoops.size(); ++i) {
            mapping.map(explicitLoops[i].induction(), explicitIndices[i]);
          }

          // Clone the equation body
          for (auto& op : equation.bodyBlock()->getOperations()) {
            if (auto terminator = mlir::dyn_cast<EquationSidesOp>(op)) {
              // Convert the equality into an assignment
              for (auto [lhs, rhs] : llvm::zip(terminator.lhsValues(), terminator.rhsValues())) {
                mlir::Value mappedLhs = mapping.lookup(lhs);
                mlir::Value mappedRhs = mapping.lookup(rhs);

                if (auto mappedLhsArrayType = mappedLhs.getType().dyn_cast<ArrayType>()) {
                  assert(mappedLhsArrayType.getRank() != 0);

                  createIterationLoops(
                      nestedExplicitBuilder, loc, beginIndexes, endIndexes, steps, iterationDirection,
                      [&](mlir::OpBuilder& nestedImplicitBuilder, mlir::ValueRange implicitIndices) {
                        assert(mappedLhs.getType().cast<ArrayType>().getRank() == implicitIndices.size());
                        mlir::Value rhsValue = nestedImplicitBuilder.create<LoadOp>(loc, mappedRhs, implicitIndices);
                        rhsValue = nestedImplicitBuilder.create<CastOp>(loc, mappedLhsArrayType.getElementType(), rhsValue);
                        nestedImplicitBuilder.create<StoreOp>(loc, rhsValue, mappedLhs, implicitIndices);
                      });
                } else {
                  auto loadOp = mlir::cast<LoadOp>(mappedLhs.getDefiningOp());
                  mappedRhs = builder.create<CastOp>(loc, mappedLhs.getType(), mappedRhs);
                  builder.create<StoreOp>(loc, mappedRhs, loadOp.array(), loadOp.indexes());
                }
              }
            } else if (mlir::isa<EquationSideOp>(op)) {
              // Ignore equation sides
              continue;
            } else {
              // Clone all the other operations
              nestedExplicitBuilder.clone(op, mapping);
            }
          }
        });

    return mlir::success();
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

  std::vector<ForEquationOp> LoopEquation::getExplicitLoops() const
  {
    std::vector<ForEquationOp> loops;
    ForEquationOp parent = getOperation()->getParentOfType<ForEquationOp>();

    while (parent != nullptr) {
      loops.push_back(parent);
      parent = parent->getParentOfType<ForEquationOp>();
    }

    std::reverse(loops.begin(), loops.end());
    return loops;
  }

  ForEquationOp LoopEquation::getExplicitLoop(size_t index) const
  {
    auto loops = getExplicitLoops();
    assert(index < loops.size());
    return loops[index];
  }

  size_t LoopEquation::getNumberOfImplicitLoops() const
  {
    size_t result = 0;
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().bodyBlock()->getTerminator());

    if (auto arrayType = terminator.lhsValues()[0].getType().dyn_cast<ArrayType>()) {
      result += arrayType.getRank();
    }

    return result;
  }

  std::vector<Range> LoopEquation::getImplicitLoops() const
  {
    std::vector<Range> result;

    size_t counter = 0;
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().bodyBlock()->getTerminator());

    if (auto arrayType = terminator.lhsValues()[0].getType().dyn_cast<ArrayType>()) {
      for (size_t i = 0; i < arrayType.getRank(); ++i, ++counter) {
        result.emplace_back(0, arrayType.getShape()[i]);
      }
    }

    return result;
  }
}
