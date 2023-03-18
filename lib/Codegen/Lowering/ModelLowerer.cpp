#include "marco/Codegen/Lowering/ModelLowerer.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  ModelLowerer::ModelLowerer(LoweringContext* context, BridgeInterface* bridge)
      : Lowerer(context, bridge)
  {
  }

  std::vector<mlir::Operation*> ModelLowerer::lower(const Model& model)
  {
    std::vector<mlir::Operation*> result;

    mlir::OpBuilder::InsertionGuard guard(builder());
    Lowerer::SymbolScope varScope(symbolTable());

    mlir::Location location = loc(model.getLocation());

    llvm::SmallVector<mlir::Type, 3> variableTypes;
    llvm::SmallVector<mlir::Location, 3> variableLocations;

    // Determine the type of each variable.
    llvm::SmallVector<mlir::Attribute, 3> variableNames;

    for (const auto& member : model.getMembers()) {
      mlir::Type type = lower(member->getType());

      if (!type.isa<ArrayType>()) {
        type = ArrayType::get(llvm::None, type);
      }

      variableTypes.push_back(type);

      mlir::StringAttr nameAttribute = builder().getStringAttr(member->getName());
      variableNames.push_back(nameAttribute);

      variableLocations.push_back(loc(member->getLocation()));
    }

    // Create the model operation and its blocks.
    auto modelOp = builder().create<ModelOp>(location, model.getName());

    // Simulation variables.
    builder().setInsertionPointToStart(modelOp.bodyBlock());

    // First, add the variables to the symbol table.
    for (const auto& member : model.getMembers()) {
      mlir::Location variableLoc = loc(member->getLocation());
      mlir::Type type = lower(member->getType());

      symbolTable().insert(
          member->getName(),
          Reference::variable(
              builder(), variableLoc, member->getName(), type));
    }

    // Then, lower them.
    for (const auto& member : model.getMembers()) {
      lower(*member);
    }

    // Equations.
    symbolTable().insert("time", Reference::time(builder(), location));

    // Create the binding equations.
    for (const auto& member : model.getMembers()) {
      if (member->hasModification()) {
        if (const auto* modification = member->getModification();
            modification->hasExpression()) {
          createBindingEquation(*member, *modification->getExpression());
          builder().setInsertionPointToEnd(modelOp.bodyBlock());
        }
      }
    }

    // Create the 'start' values.
    for (const auto& member : model.getMembers()) {
      if (member->hasStartExpression()) {
        lowerStartAttribute(
            *member,
            *member->getStartExpression(),
            member->getFixedProperty(),
            member->getEachProperty());
      }
    }

    // Create the equations.
    for (const auto& block : model.getEquationsBlocks()) {
      for (const auto& equation : block->getEquations()) {
        lower(*equation, false);
      }

      for (const auto& forEquation : block->getForEquations()) {
        lower(*forEquation, false);
      }
    }

    // Create the initial equations.
    for (const auto& block : model.getInitialEquationsBlocks()) {
      for (const auto& equation : block->getEquations()) {
        lower(*equation, true);
      }

      for (const auto& forEquation : block->getForEquations()) {
        lower(*forEquation, true);
      }
    }

    // Create the algorithms.
    for (const auto& algorithm : model.getAlgorithms()) {
      lower(*algorithm);
    }

    // Add the model operation to the list of top-level operations.
    result.push_back(modelOp);

    // Process the inner classes.
    builder().setInsertionPointAfter(modelOp);

    for (const auto& innerClass : model.getInnerClasses()) {
      for (auto& loweredInnerClass : lower(*innerClass)) {
        result.push_back(std::move(loweredInnerClass));
      }
    }

    return result;
  }

  void ModelLowerer::lower(const Member& member)
  {
    // TODO refactor, it is the same for both functions and models.
    mlir::Location location = loc(member.getLocation());
    mlir::Type type = lower(member.getType());

    VariabilityProperty variabilityProperty = VariabilityProperty::none;
    IOProperty ioProperty = IOProperty::none;

    if (member.isDiscrete()) {
      variabilityProperty = VariabilityProperty::discrete;
    } else if (member.isParameter()) {
      variabilityProperty = VariabilityProperty::parameter;
    } else if (member.isConstant()) {
      variabilityProperty = VariabilityProperty::constant;
    }

    if (member.isInput()) {
      ioProperty = IOProperty::input;
    } else if (member.isOutput()) {
      ioProperty = IOProperty::output;
    }

    auto variableType = VariableType::wrap(type, variabilityProperty, ioProperty);

    llvm::SmallVector<llvm::StringRef> dimensionsConstraints;
    bool hasFixedDimensions = false;

    if (auto arrayType = type.dyn_cast<ArrayType>()) {
      for (const auto& dimension : member.getType().getDimensions()) {
        if (dimension.hasExpression()) {
          dimensionsConstraints.push_back(
              VariableOp::kDimensionConstraintFixed);

          hasFixedDimensions = true;
        } else {
          dimensionsConstraints.push_back(
              VariableOp::kDimensionConstraintUnbounded);
        }
      }
    }

    auto var = builder().create<VariableOp>(
        location, member.getName(), variableType,
        builder().getStrArrayAttr(dimensionsConstraints),
        nullptr);

    if (hasFixedDimensions) {
      mlir::OpBuilder::InsertionGuard guard(builder());

      builder().setInsertionPointToStart(
          &var.getConstraintsRegion().emplaceBlock());

      llvm::SmallVector<mlir::Value> fixedDimensions;

      for (const auto& dimension : member.getType().getDimensions()) {
        if (dimension.hasExpression()) {
          mlir::Location dimensionLoc =
              loc(dimension.getExpression()->getLocation());

          mlir::Value size =
              lower(*dimension.getExpression())[0].get(dimensionLoc);

          size = builder().create<CastOp>(
              location, builder().getIndexType(), size);

          fixedDimensions.push_back(size);
        }
      }

      builder().create<YieldOp>(location, fixedDimensions);
    }
  }

  void ModelLowerer::createBindingEquation(
      const ast::Member& member, const ast::Expression& expression)
  {
    mlir::OpBuilder::InsertionGuard guard(builder());
    mlir::Location location = loc(expression.getLocation());

    auto bindingEquationOp = builder().create<BindingEquationOp>(
        location, member.getName());

    assert(bindingEquationOp.getBodyRegion().empty());

    mlir::Block* bodyBlock = builder().createBlock(
        &bindingEquationOp.getBodyRegion());

    builder().setInsertionPointToStart(bodyBlock);

    // Lower the expression and yield its value.
    mlir::Location expressionLoc = loc(expression.getLocation());
    auto expressionValues = lower(expression);
    assert(expressionValues.size() == 1);

    builder().create<YieldOp>(
        location, expressionValues[0].get(expressionLoc));
  }

  void ModelLowerer::lowerStartAttribute(
      const ast::Member& member,
      const ast::Expression& expression,
      bool fixed,
      bool each)
  {
    mlir::Location location = loc(expression.getLocation());

    auto startOp = builder().create<StartOp>(
        location, member.getName(), fixed, each);

    mlir::OpBuilder::InsertionGuard guard(builder());

    assert(startOp.getBodyRegion().empty());
    mlir::Block* bodyBlock = builder().createBlock(&startOp.getBodyRegion());
    builder().setInsertionPointToStart(bodyBlock);

    mlir::Location valueLoc = loc(expression.getLocation());
    auto value = lower(expression);
    assert(value.size() == 1);

    builder().create<YieldOp>(location, value[0].get(valueLoc));
  }
}
