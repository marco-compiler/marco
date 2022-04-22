#include "marco/Codegen/Lowering/ModelLowerer.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

static IOProperty getIOProperty(const Member& member) {
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

    llvm::SmallVector<mlir::Type, 3> args;

    // Variables
    llvm::SmallVector<mlir::Attribute, 3> variableNames;

    for (const auto& member : model.getMembers()) {
      mlir::Type type = lower(member->getType());

      if (!type.isa<ArrayType>()) {
        type = ArrayType::get(builder().getContext(), type, llvm::None);
      }

      args.push_back(type);

      mlir::StringAttr nameAttribute = builder().getStringAttr(member->getName());
      variableNames.push_back(nameAttribute);
    }

    // Create the operation
    auto modelOp = builder().create<ModelOp>(location);

    {
      // Simulation variables
      mlir::Block* initBlock = builder().createBlock(&modelOp.initRegion());
      builder().setInsertionPointToStart(initBlock);
      llvm::SmallVector<mlir::Value, 3> vars;

      for (const auto& member : model.getMembers()) {
        lower(*member);
        vars.push_back(symbolTable().lookup(member->getName()).getReference());
      }

      builder().create<YieldOp>(location, vars);
    }

    {
      // Body
      mlir::Block* bodyBlock = builder().createBlock(&modelOp.bodyRegion(), {}, args);
      builder().setInsertionPointToStart(bodyBlock);

      symbolTable().insert("time", Reference::time(&builder()));

      for (const auto& member : llvm::enumerate(model.getMembers())) {
        symbolTable().insert(member.value()->getName(), Reference::memory(&builder(), modelOp.bodyRegion().getArgument(member.index())));
      }

      // Members with an assigned value are conceptually the same as equations performing that assignment.
      for (const auto& member : model.getMembers()) {
        if (!member->isParameter() && member->hasInitializer()) {
          createMemberTrivialEquation(modelOp, *member, *member->getInitializer());
        }
      }

      // Create the equations
      for (const auto& equation : model.getEquations()) {
        lower(*equation);
      }

      for (const auto& forEquation : model.getForEquations()) {
        lower(*forEquation);
      }
    }

    result.push_back(modelOp);
    return result;
  }

  void ModelLowerer::lower(const Member& member)
  {
    auto location = loc(member.getLocation());

    const auto& frontendType = member.getType();
    mlir::Type type = lower(frontendType);
    auto memberType = MemberType::wrap(type, member.isParameter(), getIOProperty(member));

    mlir::Value memberOp = builder().create<MemberCreateOp>(
        location, member.getName(), memberType, llvm::None);

    symbolTable().insert(member.getName(), Reference::member(&builder(), memberOp));

    Reference reference = symbolTable().lookup(member.getName());

    if (member.hasStartOverload()) {
      auto values = lower(*member.getStartOverload());
      assert(values.size() == 1);

      if (type.isa<ArrayType>()) {
        builder().create<ArrayFillOp>(location, *reference, *values[0]);
      } else {
        reference.set(*values[0]);
      }
    } else {
      if (auto arrayType = type.dyn_cast<ArrayType>()) {
        mlir::Value zero = builder().create<ConstantOp>(location, getZeroAttr(arrayType.getElementType()));
        builder().create<ArrayFillOp>(location, *reference, zero);
      } else {
        mlir::Value zero = builder().create<ConstantOp>(location, getZeroAttr(type));
        reference.set(zero);
      }
    }
  }

  void ModelLowerer::createMemberTrivialEquation(
      ModelOp modelOp, const ast::Member& member, const ast::Expression& expression)
  {
    assert(member.hasInitializer());
    auto location = loc(expression.getLocation());

    mlir::OpBuilder::InsertionGuard guard(builder());
    builder().setInsertionPointToEnd(modelOp.bodyBlock());

    auto equationOp = builder().create<EquationOp>(location);
    assert(equationOp.bodyRegion().empty());
    mlir::Block* equationBodyBlock = builder().createBlock(&equationOp.bodyRegion());
    builder().setInsertionPointToStart(equationBodyBlock);

    // Right-hand side
    auto rhs = lower(expression);
    assert(rhs.size() == 1);

    // Create the assignment
    mlir::Value lhsTuple = builder().create<EquationSideOp>(location, *symbolTable().lookup(member.getName()));
    mlir::Value rhsTuple = builder().create<EquationSideOp>(location, *rhs[0]);
    builder().create<EquationSidesOp>(location, lhsTuple, rhsTuple);
  }
}
