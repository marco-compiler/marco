#include "marco/Codegen/Lowering/ModelLowerer.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

static VariabilityProperty getVariabilityProperty(const Member& member)
{
  if (member.isDiscrete()) {
    return VariabilityProperty::discrete;
  }

  if (member.isParameter()) {
    return VariabilityProperty::parameter;
  }

  if (member.isConstant()) {
    return VariabilityProperty::constant;
  }

  return VariabilityProperty::none;
}

static IOProperty getIOProperty(const Member& member)
{
  if (member.isInput()) {
    return IOProperty::input;
  }

  if (member.isOutput()) {
    return IOProperty::output;
  }

  return IOProperty::none;
}

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

    auto location = loc(model.getLocation());

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

    mlir::Block* varsBlock = builder().createBlock(&modelOp.getVarsRegion());

    mlir::Block* bodyBlock = builder().createBlock(
        &modelOp.getBodyRegion(), {}, variableTypes, variableLocations);

    {
      // Simulation variables.
      builder().setInsertionPointToStart(varsBlock);
      llvm::SmallVector<mlir::Value, 3> vars;

      for (const auto& member : model.getMembers()) {
        lower(*member);
        vars.push_back(symbolTable().lookup(member->getName()).getReference());
      }

      builder().create<YieldOp>(location, vars);
    }

    {
      // Equations.
      builder().setInsertionPointToStart(bodyBlock);
      symbolTable().insert("time", Reference::time(&builder()));

      for (const auto& member : llvm::enumerate(model.getMembers())) {
        symbolTable().insert(
            member.value()->getName(),
            Reference::memory(
                &builder(),
                modelOp.getBodyRegion().getArgument(member.index())));
      }

      // Create the binding equations.
      for (const auto& member : model.getMembers()) {
        if (member->hasModification()) {
          if (const auto* modification = member->getModification();
              modification->hasExpression()) {
            createBindingEquation(*member, *modification->getExpression());
            builder().setInsertionPointToEnd(bodyBlock);
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
    auto location = loc(member.getLocation());

    const auto& frontendType = member.getType();
    mlir::Type type = lower(frontendType);

    auto memberType = MemberType::wrap(
        type, getVariabilityProperty(member), getIOProperty(member));

    mlir::Value memberOp = builder().create<MemberCreateOp>(
        location, member.getName(), memberType, llvm::None);

    symbolTable().insert(
        member.getName(), Reference::member(&builder(), memberOp));
  }

  void ModelLowerer::createBindingEquation(
      const ast::Member& member, const ast::Expression& expression)
  {
    mlir::OpBuilder::InsertionGuard guard(builder());
    auto location = loc(expression.getLocation());

    auto bindingEquationOp = builder().create<BindingEquationOp>(
        location, symbolTable().lookup(member.getName()).getReference());

    assert(bindingEquationOp.getBodyRegion().empty());

    mlir::Block* bodyBlock = builder().createBlock(
        &bindingEquationOp.getBodyRegion());

    builder().setInsertionPointToStart(bodyBlock);

    // Lower the expression and yield its value.
    auto expressionValues = lower(expression);
    assert(expressionValues.size() == 1);
    builder().create<YieldOp>(location, *expressionValues[0]);
  }

  void ModelLowerer::lowerStartAttribute(
      const ast::Member& member,
      const ast::Expression& expression,
      bool fixed,
      bool each)
  {
    auto location = loc(expression.getLocation());

    auto startOp = builder().create<StartOp>(
        location,
        symbolTable().lookup(member.getName()).getReference(),
        builder().getBoolAttr(fixed),
        builder().getBoolAttr(each));

    mlir::OpBuilder::InsertionGuard guard(builder());

    assert(startOp.getBodyRegion().empty());
    mlir::Block* bodyBlock = builder().createBlock(&startOp.getBodyRegion());
    builder().setInsertionPointToStart(bodyBlock);

    auto value = lower(expression);
    assert(value.size() == 1);
    builder().create<YieldOp>(location, *value[0]);
  }
}
