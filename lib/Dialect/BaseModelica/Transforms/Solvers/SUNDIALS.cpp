#include "marco/Dialect/BaseModelica/Transforms/Solvers/SUNDIALS.h"

using namespace ::mlir::bmodelica;

namespace mlir::bmodelica
{
  mlir::sundials::VariableGetterOp createGetterFunction(
      mlir::OpBuilder& builder,
      mlir::SymbolTableCollection& symbolTableCollection,
      mlir::Location loc,
      mlir::ModuleOp moduleOp,
      VariableOp variable,
      llvm::StringRef functionName)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(moduleOp.getBody());

    auto variableType = variable.getVariableType();

    auto getterOp = builder.create<mlir::sundials::VariableGetterOp>(
        loc, functionName, variableType.getRank());

    symbolTableCollection.getSymbolTable(moduleOp).insert(getterOp);

    mlir::Block* entryBlock = getterOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto receivedIndices =
        getterOp.getVariableIndices().take_front(variableType.getRank());

    mlir::Value result = builder.create<QualifiedVariableGetOp>(loc, variable);

    if (!receivedIndices.empty()) {
      result = builder.create<TensorExtractOp>(loc, result, receivedIndices);
    }

    if (auto requestedResultType = getterOp.getFunctionType().getResult(0);
        result.getType() != requestedResultType) {
      result = builder.create<CastOp>(loc, requestedResultType, result);
    }

    builder.create<mlir::sundials::ReturnOp>(loc, result);
    return getterOp;
  }

  mlir::sundials::VariableGetterOp createGetterFunction(
      mlir::OpBuilder& builder,
      mlir::SymbolTableCollection& symbolTableCollection,
      mlir::Location loc,
      mlir::ModuleOp moduleOp,
      GlobalVariableOp variable,
      llvm::StringRef functionName)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(moduleOp.getBody());

    auto arrayType = variable.getType().cast<ArrayType>();

    auto getterOp = builder.create<mlir::sundials::VariableGetterOp>(
        loc, functionName, arrayType.getRank());

    symbolTableCollection.getSymbolTable(moduleOp).insert(getterOp);

    mlir::Block* entryBlock = getterOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto receivedIndices =
        getterOp.getVariableIndices().take_front(arrayType.getRank());

    mlir::Value result = builder.create<GlobalVariableGetOp>(loc, variable);

    if (!receivedIndices.empty()) {
      result = builder.create<LoadOp>(loc, result, receivedIndices);
    }

    if (auto requestedResultType = getterOp.getFunctionType().getResult(0);
        result.getType() != requestedResultType) {
      result = builder.create<CastOp>(loc, requestedResultType, result);
    }

    builder.create<mlir::sundials::ReturnOp>(loc, result);
    return getterOp;
  }

  mlir::sundials::VariableSetterOp createSetterFunction(
      mlir::OpBuilder& builder,
      mlir::SymbolTableCollection& symbolTableCollection,
      mlir::Location loc,
      mlir::ModuleOp moduleOp,
      VariableOp variable,
      llvm::StringRef functionName)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(moduleOp.getBody());

    auto variableType = variable.getVariableType();

    auto setterOp = builder.create<mlir::sundials::VariableSetterOp>(
        loc, functionName, variableType.getRank());

    symbolTableCollection.getSymbolTable(moduleOp).insert(setterOp);

    mlir::Block* entryBlock = setterOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto receivedIndices =
        setterOp.getVariableIndices().take_front(variableType.getRank());

    mlir::Value value = setterOp.getValue();

    if (auto requestedValueType = variableType.getElementType();
        value.getType() != requestedValueType) {
      value = builder.create<CastOp>(
          loc, requestedValueType, setterOp.getValue());
    }

    builder.create<QualifiedVariableSetOp>(
        loc, variable, receivedIndices, value);

    builder.create<mlir::sundials::ReturnOp>(loc);
    return setterOp;
  }

  mlir::sundials::VariableSetterOp createSetterFunction(
      mlir::OpBuilder& builder,
      mlir::SymbolTableCollection& symbolTableCollection,
      mlir::Location loc,
      mlir::ModuleOp moduleOp,
      GlobalVariableOp variable,
      llvm::StringRef functionName)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToEnd(moduleOp.getBody());

    auto arrayType = variable.getType().cast<ArrayType>();

    auto setterOp = builder.create<mlir::sundials::VariableSetterOp>(
        loc, functionName, arrayType.getRank());

    symbolTableCollection.getSymbolTable(moduleOp).insert(setterOp);

    mlir::Block* entryBlock = setterOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto receivedIndices =
        setterOp.getVariableIndices().take_front(arrayType.getRank());

    mlir::Value value = setterOp.getValue();

    if (auto requestedValueType = arrayType.getElementType();
        value.getType() != requestedValueType) {
      value = builder.create<CastOp>(
          loc, requestedValueType, setterOp.getValue());
    }

    mlir::Value array = builder.create<GlobalVariableGetOp>(loc, variable);
    builder.create<StoreOp>(loc, value, array, receivedIndices);

    builder.create<mlir::sundials::ReturnOp>(loc);
    return setterOp;
  }

  GlobalVariableOp createGlobalADSeed(
      mlir::OpBuilder& builder,
      mlir::ModuleOp moduleOp,
      mlir::Location loc,
      llvm::StringRef name,
      mlir::Type type)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());

    mlir::Attribute initialValue = nullptr;
    auto arrayType = type.cast<ArrayType>();
    mlir::Type elementType = arrayType.getElementType();

    if (elementType.isa<BooleanType>()) {
      llvm::SmallVector<bool> values(arrayType.getNumElements(), false);
      initialValue = DenseBooleanElementsAttr::get(arrayType, values);
    } else if (elementType.isa<IntegerType>()) {
      llvm::SmallVector<int64_t> values(arrayType.getNumElements(), 0);
      initialValue = DenseIntegerElementsAttr::get(arrayType, values);
    } else if (elementType.isa<RealType>()) {
      llvm::SmallVector<double> values(arrayType.getNumElements(), 0);
      initialValue = DenseRealElementsAttr::get(arrayType, values);
    }

    return builder.create<GlobalVariableOp>(loc, name, type, initialValue);
  }

  void setGlobalADSeed(
      mlir::OpBuilder& builder,
      mlir::Location loc,
      GlobalVariableOp seedVariableOp,
      mlir::ValueRange indices,
      mlir::Value value)
  {
    mlir::Value seed = builder.create<GlobalVariableGetOp>(loc, seedVariableOp);
    builder.create<StoreOp>(loc, value, seed, indices);
  }
}
