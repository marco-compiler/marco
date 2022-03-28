#include "mlir/IR/Builders.h"

#include "Utils.h"

using namespace ::mlir::modelica;

namespace marco::codegen::test
{
  ModelOp createModel(mlir::OpBuilder& builder, mlir::TypeRange varTypes)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    llvm::SmallVector<mlir::Attribute, 3> names;

    for (size_t i = 0; i < varTypes.size(); ++i) {
      names.push_back(builder.getStringAttr("var" + std::to_string(i)));
    }

    llvm::SmallVector<mlir::Type, 3> varArrayTypes;

    for (const auto& type : varTypes) {
      if (auto arrayType = type.dyn_cast<ArrayType>()) {
        varArrayTypes.push_back(arrayType);
      } else {
        varArrayTypes.push_back(ArrayType::get(builder.getContext(), type, llvm::None));
      }
    }

    auto modelOp = builder.create<ModelOp>(
        builder.getUnknownLoc(),
        builder.getF64FloatAttr(0),
        builder.getF64FloatAttr(10),
        builder.getF64FloatAttr(0.1));

    mlir::Block* initBlock = builder.createBlock(&modelOp.initRegion());
    builder.setInsertionPointToStart(initBlock);

    llvm::SmallVector<mlir::Value, 3> members;

    for (const auto& [name, type] : llvm::zip(names, varArrayTypes)) {
      auto arrayType = type.cast<ArrayType>();
      auto memberType = MemberType::wrap(arrayType);
      auto member = builder.create<MemberCreateOp>(builder.getUnknownLoc(), name.cast<mlir::StringAttr>().getValue(), memberType, llvm::None);
      members.push_back(member.getResult());
    }

    builder.create<YieldOp>(builder.getUnknownLoc(), members);

    builder.createBlock(&modelOp.bodyRegion(), {}, varArrayTypes);
    return modelOp;
  }

  Variables mapVariables(ModelOp model)
  {
    Variables variables;

    for (const auto& variable : model.bodyRegion().getArguments()) {
      variables.add(Variable::build(variable));
    }

    return variables;
  }

  mlir::modelica::EquationOp createEquation(
      mlir::OpBuilder& builder,
      mlir::modelica::ModelOp model,
      llvm::ArrayRef<std::pair<long, long>> iterationRanges,
      std::function<void(mlir::OpBuilder&, mlir::ValueRange)> bodyFn)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&model.bodyRegion().front());
    auto loc = builder.getUnknownLoc();

    // Create the iteration ranges
    std::vector<mlir::Value> inductionVariables;

    for (const auto& [begin, end] : iterationRanges) {
      assert(begin <= end);
      auto forEquationOp = builder.create<ForEquationOp>(loc, begin, end);
      inductionVariables.push_back(forEquationOp.induction());
      builder.setInsertionPointToStart(forEquationOp.bodyBlock());
    }

    // Create the equation body
    auto equationOp = builder.create<EquationOp>(loc);
    builder.setInsertionPointToStart(equationOp.bodyBlock());
    bodyFn(builder, inductionVariables);

    return equationOp;
  }
}
