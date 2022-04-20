#include "marco/codegen/passes/model/LoopEquation.h"
#include "marco/modeling/MultidimensionalRangeRagged.h"
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

  void LoopEquation::dumpIR() const
  {
    EquationOp equationOp = getOperation();
    mlir::Operation* op = equationOp.getOperation();

    while (auto parent = op->getParentOfType<ForEquationOp>()) {
      op = parent.getOperation();
    }

    op->dump();
  }

  size_t LoopEquation::getNumOfIterationVars() const
  {
    return getNumberOfExplicitLoops() + getNumberOfImplicitLoops();
  }

  RangeRagged getRangeFromDimensionSize(const Shape::DimensionSize &dimension)
  {
    if(dimension.isRagged())
    {
      llvm::SmallVector<RangeRagged,3> arr;

      for(const auto r : dimension.asRagged()){
        arr.push_back(getRangeFromDimensionSize(r));
      }
      return RangeRagged(arr);
    }
    
    return RangeRagged(0,dimension.getNumericValue());
  }

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

    llvm_unreachable("Unknown attribute type");
    return 0;
  }

  RaggedValue getValue(mlir::Value value)
  {
    if(value.isa<mlir::BlockArgument>())
    {
      auto parent = value.getParentBlock()->getParentOp();
      if (auto forBlock = mlir::dyn_cast<ForEquationOp>(parent)) {
        auto start = getValue(forBlock.start());
        auto end = getValue(forBlock.end()) + 1;

        llvm::SmallVector<RaggedValue,3> values;
        for(auto it = start.asValue(); it<end.asValue(); ++it)
          values.push_back(it);

        return RaggedValue(values);
      }
    }
    
    mlir::Operation* op = value.getDefiningOp();

    if (auto constantOp = mlir::dyn_cast<ConstantOp>(op)) {
      return getIntFromAttribute(constantOp.value());
    }
    if (auto addOp = mlir::dyn_cast<AddOp>(op))
    {
      return getValue(addOp.lhs()) + getValue(addOp.rhs());  
    }
    if (auto subOp = mlir::dyn_cast<SubOp>(op))
    {
      return getValue(subOp.lhs()) - getValue(subOp.rhs());  
    }
    
    if (auto loadOp = mlir::dyn_cast<LoadOp>(op))
    {
      assert(loadOp.indexes().empty());
      return getValue(loadOp.memory());
    }
    if (auto subscriptionOp = mlir::dyn_cast<SubscriptionOp>(op))
    {      
      auto array = subscriptionOp.source().getDefiningOp();
      auto index = getValue(subscriptionOp.indexes()[0]);;
      if (auto allocaOp = mlir::dyn_cast<AllocaOp>(array))
      {
        if(allocaOp.hasConstantValues())
        {
          auto foo=[](const mlir::ArrayAttr &array, const RaggedValue &index, auto self){
            if(index.isRagged())
            {
              llvm::SmallVector<RaggedValue,3> values;
              for(auto i: index.asRagged())
              {
                if(i.isRagged())
                  assert(false);
                else
                  values.push_back(getIntFromAttribute(array[i.asValue()]));
              }
              return RaggedValue(values);
            }
            else
            {
              return RaggedValue(getIntFromAttribute(array[index.asValue()]));
            }
          };
          auto rag = foo(allocaOp.getConstantValues(),index,foo);
          return rag;
        }
      }
    }

    assert(false && "unreachable");
  }

  IndexSet LoopEquation::getIterationRanges() const
  {
    std::vector<RangeRagged> ranges;

    for (auto& explicitLoop : getExplicitLoops()) {
      ranges.emplace_back(getValue(explicitLoop.start()), getValue(explicitLoop.end()) + 1);
    }

    std::vector<RangeRagged> implicitLoops;
    
    {
      //get implicit loops
      size_t counter = 0;
      auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());

      if (auto arrayType = terminator.lhs()[0].getType().dyn_cast<ArrayType>()) {
        for (size_t i = 0; i < arrayType.getRank(); ++i, ++counter) {
          implicitLoops.push_back(getRangeFromDimensionSize(arrayType.getShape()[i]));
        }
      }
    }

    ranges.insert(ranges.end(), implicitLoops.begin(), implicitLoops.end());  
    
    auto multi = MultidimensionalRangeRagged(std::move(ranges));
    return getIndexSetFromRaggedRange(multi);
  }

  std::vector<Access> LoopEquation::getAccesses() const
  {
    std::vector<Access> accesses;
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());
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

    processFn(terminator.lhs()[0], EquationPath(EquationPath::LEFT));
    processFn(terminator.rhs()[0], EquationPath(EquationPath::RIGHT));

    return accesses;
  }

  DimensionAccess LoopEquation::resolveDimensionAccess(std::pair<mlir::Value, ::marco::modeling::RaggedValue> access) const
  {
    if (access.first == nullptr) {
      return DimensionAccess::constant(access.second.asValue());
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
    assert(loops.end()!=loopIt);
    
    if (access.second.isRagged()) {
      std::vector<long> array;

      for(auto val: access.second.asRagged())
        array.push_back(val.asValue());
      
      return DimensionAccess::relativeToArray(inductionVarIndex, array);
    }
    return DimensionAccess::relative(inductionVarIndex, access.second.asValue());
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
        builder.setInsertionPointToStart(destination.getOperation().body());

        mlir::Value constantAccess = builder.create<ConstantOp>(
            explicitLoops[i].getLoc(), IntegerAttribute::get(builder.getContext(), combinedDimensionAccess.getPosition()));

        mapping.map(explicitLoops[i].induction(), constantAccess);
      } else {
        mlir::Value mapped = destinationInductionVariables[combinedDimensionAccess.getInductionVariableIndex()];

        if (combinedDimensionAccess.getOffset() != 0) {
          builder.setInsertionPointToStart(destination.getOperation().body());

          mlir::Value offset = builder.create<ConstantOp>(
              explicitLoops[i].getLoc(), IntegerAttribute::get(builder.getContext(), combinedDimensionAccess.getOffset()));

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

    auto conditionFn = [&](mlir::Value index, mlir::Value end) -> mlir::Value {
      assert(iterationDirection == modeling::scheduling::Direction::Forward ||
              iterationDirection == modeling::scheduling::Direction::Backward);

      if (iterationDirection == modeling::scheduling::Direction::Backward) {
        return builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::sgt, index, end).getResult();
      }

      return builder.create<mlir::CmpIOp>(loc, mlir::CmpIPredicate::slt, index, end).getResult();
    };

    auto updateFn = [&](mlir::Value index, mlir::Value step) -> mlir::Value {
      assert(iterationDirection == modeling::scheduling::Direction::Forward ||
          iterationDirection == modeling::scheduling::Direction::Backward);

      if (iterationDirection == modeling::scheduling::Direction::Backward) {
        return builder.create<mlir::SubIOp>(loc, index, step).getResult();
      }

      return builder.create<mlir::AddIOp>(loc, index, step).getResult();
    };

    // Create the explicit loops
    auto explicitLoops = getExplicitLoops();

    std::vector<mlir::Value> inductionVars;

    for (size_t i = 0; i < explicitLoops.size(); ++i) {
      auto whileOp = builder.create<mlir::scf::WhileOp>(loc, builder.getIndexType(), beginIndexes[i]);

      // Check the condition.
      // A naive check can consist in the equality comparison. However, in order to be future-proof with
      // respect to steps greater than one, we need to check if the current value is beyond the end boundary.
      // This in turn requires to know the iteration direction.
      mlir::Block* beforeBlock = builder.createBlock(&whileOp.before(), {}, builder.getIndexType());
      builder.setInsertionPointToStart(beforeBlock);
      mlir::Value condition = conditionFn(whileOp.before().getArgument(0), endIndexes[i]);
      builder.create<mlir::scf::ConditionOp>(loc, condition, whileOp.before().getArgument(0));

      // Execute the loop body
      mlir::Block* afterBlock = builder.createBlock(&whileOp.after(), {}, builder.getIndexType());
      mlir::Value inductionVariable = afterBlock->getArgument(0);
      mapping.map(explicitLoops[i].induction(), inductionVariable);
      builder.setInsertionPointToStart(afterBlock);

      // Update the induction variable
      mlir::Value nextValue = updateFn(inductionVariable, steps[i]);
      builder.create<mlir::scf::YieldOp>(loc, nextValue);
      builder.setInsertionPoint(nextValue.getDefiningOp());
    }

    // Clone the equation body
    for (auto& op : equation.body()->getOperations()) {
      if (auto terminator = mlir::dyn_cast<modelica::EquationSidesOp>(op)) {
        // Convert the equality into an assignment
        for (auto [lhs, rhs] : llvm::zip(terminator.lhs(), terminator.rhs())) {
          auto mappedLhs = mapping.lookup(lhs);
          auto mappedRhs = mapping.lookup(rhs);

          if (auto loadOp = mlir::dyn_cast<LoadOp>(mappedLhs.getDefiningOp())) {
            assert(loadOp.indexes().empty());
            builder.create<AssignmentOp>(loc, mappedRhs, loadOp.memory());
          } else {
            builder.create<AssignmentOp>(loc, mappedRhs, mappedLhs);
          }
        }
      } else {
        // Clone all the other operations
        builder.clone(op, mapping);
      }
    }

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
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());

    if (auto arrayType = terminator.lhs()[0].getType().dyn_cast<ArrayType>()) {
      result += arrayType.getRank();
    }

    return result;
  }



  IndexSet LoopEquation::getImplicitLoops() const
  {
    std::vector<RangeRagged> tmp;

    size_t counter = 0;
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().body()->getTerminator());

    if (auto arrayType = terminator.lhs()[0].getType().dyn_cast<ArrayType>()) {
      for (size_t i = 0; i < arrayType.getRank(); ++i, ++counter) {
        tmp.push_back(getRangeFromDimensionSize(arrayType.getShape()[i]));
      }
    }

    MultidimensionalRangeRagged multi_ragged(tmp);
    return getIndexSetFromRaggedRange(multi_ragged);
  }
}
