#include "marco/Codegen/Lowering/ModelLowerer.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

static mlir::Attribute getZeroAttr(mlir::OpBuilder& builder, mlir::Type type)
{
  if (type.isa<BooleanType>()) {
    return BooleanAttr::get(builder.getContext(), false);
  }

  if (type.isa<IntegerType>()) {
    return IntegerAttr::get(builder.getContext(), 0);
  }

  if (type.isa<RealType>()) {
    return RealAttr::get(builder.getContext(), 0);
  }

  return builder.getZeroAttr(type);
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

    // Time variable
    auto timeType = ArrayType::get(builder().getContext(), ArrayAllocationScope::unknown, RealType::get(builder().getContext()), llvm::None);
    args.push_back(timeType);

    // Other variables
    llvm::SmallVector<mlir::Attribute, 3> variableNames;

    for (const auto& member : model.getMembers()) {
      mlir::Type type = lower(member->getType(), ArrayAllocationScope::unknown);

      if (auto arrayType = type.dyn_cast<ArrayType>()) {
        type = arrayType.toUnknownAllocationScope();
      } else {
        type = ArrayType::get(builder().getContext(), ArrayAllocationScope::unknown, type, llvm::None);
      }

      args.push_back(type);

      mlir::StringAttr nameAttribute = builder().getStringAttr(member->getName());
      variableNames.push_back(nameAttribute);
    }

    llvm::ArrayRef<mlir::Attribute> attributeArray(variableNames);

    // Create the operation
    auto modelOp = builder().create<ModelOp>(
        location,
        builder().getF64FloatAttr(context()->options.startTime),
        builder().getF64FloatAttr(context()->options.endTime),
        builder().getF64FloatAttr(context()->options.timeStep));

    {
      // Simulation variables
      mlir::Block* initBlock = builder().createBlock(&modelOp.initRegion());
      builder().setInsertionPointToStart(initBlock);
      llvm::SmallVector<mlir::Value, 3> vars;

      auto memberType = MemberType::get(builder().getContext(), MemberAllocationScope::heap, RealType::get(builder().getContext()), llvm::None);
      mlir::Value time = builder().create<MemberCreateOp>(location, "time", memberType, llvm::None, false);
      vars.push_back(time);

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

      mlir::Value time = modelOp.bodyRegion().getArgument(0);
      symbolTable().insert("time", Reference::memory(&builder(), time));

      for (const auto& member : llvm::enumerate(model.getMembers())) {
        symbolTable().insert(member.value()->getName(), Reference::memory(&builder(), modelOp.bodyRegion().getArgument(member.index() + 1)));
      }

      for (const auto& equation : model.getEquations()) {
        lower(*equation);
      }

      for (const auto& forEquation : model.getForEquations()) {
        lower(*forEquation);
      }

      builder().create<YieldOp>(location, llvm::None);
    }

    result.push_back(modelOp);
    return result;
  }

  void ModelLowerer::lower(const Member& member)
  {
    auto location = loc(member.getLocation());

    const auto& frontendType = member.getType();
    mlir::Type type = lower(frontendType, ArrayAllocationScope::heap);
    auto memberType = MemberType::wrap(type, MemberAllocationScope::heap);
    bool isConstant = member.isParameter();
    mlir::Value memberOp = builder().create<MemberCreateOp>(location, member.getName(), memberType, llvm::None, isConstant);
    symbolTable().insert(member.getName(), Reference::member(&builder(), memberOp));

    Reference ref = symbolTable().lookup(member.getName());

    if (member.hasStartOverload()) {
      auto values = lower(*member.getStartOverload());
      assert(values.size() == 1);

      if (type.isa<ArrayType>()) {
        builder().create<ArrayFillOp>(location, *ref, *values[0]);
      } else {
        ref.set(*values[0]);
      }
    } else if (member.hasInitializer()) {
      mlir::Value value = *lower(*member.getInitializer())[0];
      ref.set(value);
    } else {
      if (auto arrayType = type.dyn_cast<ArrayType>()) {
        mlir::Value zero = builder().create<ConstantOp>(location, getZeroAttr(builder(), arrayType.getElementType()));
        builder().create<ArrayFillOp>(location, *ref, zero);
      } else {
        mlir::Value zero = builder().create<ConstantOp>(location, getZeroAttr(builder(), type));
        ref.set(zero);
      }
    }
  }
}
