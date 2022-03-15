#include "marco/Codegen/Lowering/ClassLowerer.h"
#include "marco/Codegen/Lowering/ModelLowerer.h"
#include "marco/Codegen/Lowering/StandardFunctionLowerer.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  ClassLowerer::ClassLowerer(LoweringContext* context, BridgeInterface* bridge)
      : Lowerer(context, bridge),
        modelLowerer(std::make_unique<ModelLowerer>(context, bridge)),
        standardFunctionLowerer(std::make_unique<StandardFunctionLowerer>(context, bridge))
  {
  }

  std::vector<mlir::Operation*> ClassLowerer::operator()(const ast::PartialDerFunction& function)
  {
    std::vector<mlir::Operation*> result;

    mlir::OpBuilder::InsertionGuard guard(builder());
    auto location = loc(function.getLocation());

    llvm::StringRef derivedFunctionName = function.getDerivedFunction()->get<ReferenceAccess>()->getName();
    llvm::SmallVector<mlir::Attribute, 3> independentVariables;

    for (const auto& independentVariable : function.getIndependentVariables()) {
      auto independentVariableName = independentVariable->get<ReferenceAccess>()->getName();
      independentVariables.push_back(builder().getStringAttr(independentVariableName));
    }

    auto derFunctionOp = builder().create<DerFunctionOp>(
        location, function.getName(), derivedFunctionName, builder().getArrayAttr(independentVariables));

    result.push_back(derFunctionOp);
    return result;
  }

  std::vector<mlir::Operation*> ClassLowerer::operator()(const ast::StandardFunction& function)
  {
    return standardFunctionLowerer->lower(function);
  }

  std::vector<mlir::Operation*> ClassLowerer::operator()(const ast::Model& model)
  {
    return modelLowerer->lower(model);
  }

  std::vector<mlir::Operation*> ClassLowerer::operator()(const ast::Package& package)
  {
    std::vector<mlir::Operation*> result;

    for (const auto& cls : package.getInnerClasses()) {
      for (const auto& op : lower(*cls)) {
        result.push_back(op);
      }
    }

    return result;
  }

  std::vector<mlir::Operation*> ClassLowerer::operator()(const ast::Record& record)
  {
    std::vector<mlir::Operation*> result;

    /*
    llvm::ScopedHashTableScope<mlir::StringRef, Reference> varScope(symbolTable);
    auto location = loc(record.getLocation());

    // Whenever a record is defined, a record constructor function with the
    // same name and in the same scope as the record class must be implicitly
    // defined, so that the record can then be instantiated.

    llvm::SmallVector<mlir::Type, 3> argsTypes;
    llvm::SmallVector<mlir::Type, 3> recordTypes;

    for (const auto& member : record)
    {
      argsTypes.push_back(lower(member.getType(), BufferAllocationScope::unknown));
      recordTypes.push_back(lower(member.getType(), BufferAllocationScope::heap));
    }

    RecordType resultType = builder.getRecordType(recordTypes);

    auto functionType = builder.getFunctionType(argsTypes, resultType);
    auto function = mlir::FuncOp::create(location, record.getName(), functionType);

    auto& entryBlock = *function.addEntryBlock();
    builder.setInsertionPointToStart(&entryBlock);

    llvm::SmallVector<mlir::Value, 3> results;

    for (const auto& [arg, type] : llvm::zip(entryBlock.getArguments(), recordTypes))
    {
      if (auto arrayType = type.dyn_cast<ArrayType>())
        results.push_back(builder.create<ArrayCloneOp>(location, arg, arrayType, false));
      else
        results.push_back(arg);
    }

    mlir::Value result = builder.create<RecordOp>(location, resultType, results);
    builder.create<mlir::ReturnOp>(location, result);

    return { function };
    */

    return result;
  }
}
