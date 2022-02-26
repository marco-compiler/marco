#include "Utils.h"

using namespace ::marco::codegen::modelica;

namespace marco::codegen::test
{
  ModelOp createModel(mlir::OpBuilder& builder, mlir::TypeRange varTypes)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // TODO remove names
    llvm::SmallVector<mlir::Attribute, 3> names;

    for (size_t i = 0; i < varTypes.size(); ++i) {
      names.push_back(builder.getStringAttr("var" + std::to_string(i)));
    }

    llvm::SmallVector<mlir::Type, 3> varArrayTypes;

    for (const auto& type : varTypes) {
      if (auto arrayType = type.dyn_cast<ArrayType>()) {
        varArrayTypes.push_back(arrayType.toAllocationScope(BufferAllocationScope::unknown));
      } else {
        varArrayTypes.push_back(ArrayType::get(builder.getContext(), BufferAllocationScope::unknown, type));
      }
    }

    auto model = builder.create<ModelOp>(
        builder.getUnknownLoc(),
        builder.getArrayAttr(names),
        RealAttribute::get(builder.getContext(), 0),
        RealAttribute::get(builder.getContext(), 10),
        RealAttribute::get(builder.getContext(), 0.1),
        varArrayTypes);

    builder.setInsertionPointToStart(&model.init().front());
    llvm::SmallVector<mlir::Value, 3> members;

    for (const auto& [name, type] : llvm::zip(names, varArrayTypes)) {
      auto arrayType = type.cast<ArrayType>().toAllocationScope(BufferAllocationScope::heap);
      auto memberType = MemberType::get(arrayType);
      auto member = builder.create<MemberCreateOp>(builder.getUnknownLoc(), name.cast<mlir::StringAttr>().getValue(), memberType, llvm::None);
      members.push_back(member.getResult());
    }

    builder.create<YieldOp>(builder.getUnknownLoc(), members);
    return model;
  }

  Variables mapVariables(ModelOp model)
  {
    Variables variables;

    for (const auto& variable : model.body().getArguments()) {
      variables.add(Variable::build(variable));
    }

    return variables;
  }
}
