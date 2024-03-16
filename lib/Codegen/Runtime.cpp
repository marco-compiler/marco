#include "marco/Codegen/Runtime.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"

using namespace ::marco;
using namespace ::marco::codegen;

namespace marco::codegen
{
  std::string RuntimeFunctionsMangling::getMangledFunction(
      llvm::StringRef baseName,
      llvm::StringRef mangledReturnType,
      llvm::ArrayRef<std::string> argsMangledTypes) const
  {
    std::string result = "_M" + baseName.str() + "_" + mangledReturnType.str();

    for (const auto& argType : argsMangledTypes) {
      result += "_" + argType;
    }

    return result;
  }

  std::string RuntimeFunctionsMangling::getVoidType() const
  {
    return "void";
  }

  std::string RuntimeFunctionsMangling::getIntegerType(
      unsigned int bitWidth) const
  {
    return "i" + std::to_string(bitWidth);
  }

  std::string RuntimeFunctionsMangling::getFloatingPointType(
      unsigned int bitWidth) const
  {
    return "f" + std::to_string(bitWidth);
  }

  std::string RuntimeFunctionsMangling::getArrayType(
      llvm::StringRef mangledElementType) const
  {
    return "a" + mangledElementType.str();
  }

  std::string RuntimeFunctionsMangling::getPointerType(
      llvm::StringRef mangledElementType) const
  {
    return "p" + mangledElementType.str();
  }

  std::string RuntimeFunctionsMangling::getVoidPointerType() const
  {
    return getPointerType(getVoidType());
  }

  mlir::LLVM::LLVMFuncOp getOrDeclareHeapAllocFn(
      mlir::OpBuilder& builder,
      mlir::ModuleOp moduleOp,
      mlir::SymbolTableCollection& symbolTableCollection)
  {
    RuntimeFunctionsMangling mangling;

    auto mangledResultType = mangling.getVoidPointerType();

    llvm::SmallVector<std::string, 1> mangledArgsTypes;
    mangledArgsTypes.push_back(mangling.getIntegerType(64));

    auto functionName = mangling.getMangledFunction(
        "heapAlloc", mangledResultType, mangledArgsTypes);

    auto funcOp =
        symbolTableCollection.lookupSymbolIn<mlir::LLVM::LLVMFuncOp>(
            moduleOp, builder.getStringAttr(functionName));

    if (funcOp) {
      return funcOp;
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());

    auto resultType =
        mlir::LLVM::LLVMPointerType::get(builder.getContext());

    llvm::SmallVector<mlir::Type, 1> argTypes;
    argTypes.push_back(builder.getI64Type());

    auto functionType =
        mlir::LLVM::LLVMFunctionType::get(resultType, argTypes, false);

    auto newFuncOp = builder.create<mlir::LLVM::LLVMFuncOp>(
        moduleOp.getLoc(), functionName, functionType);

    symbolTableCollection.getSymbolTable(moduleOp).insert(newFuncOp);
    return newFuncOp;
  }

  mlir::LLVM::LLVMFuncOp getOrDeclareHeapFreeFn(
      mlir::OpBuilder& builder,
      mlir::ModuleOp moduleOp,
      mlir::SymbolTableCollection& symbolTableCollection)
  {
    RuntimeFunctionsMangling mangling;

    auto mangledResultType = mangling.getVoidType();

    llvm::SmallVector<std::string, 1> mangledArgsTypes;
    mangledArgsTypes.push_back(mangling.getVoidPointerType());

    auto functionName = mangling.getMangledFunction(
        "heapFree", mangledResultType, mangledArgsTypes);

    auto funcOp =
        symbolTableCollection.lookupSymbolIn<mlir::LLVM::LLVMFuncOp>(
            moduleOp, builder.getStringAttr(functionName));

    if (funcOp) {
      return funcOp;
    }

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(moduleOp.getBody());

    auto resultType = mlir::LLVM::LLVMVoidType::get(builder.getContext());

    llvm::SmallVector<mlir::Type, 1> argTypes;
    argTypes.push_back(builder.getI64Type());

    auto functionType =
        mlir::LLVM::LLVMFunctionType::get(resultType, argTypes, false);

    auto newFuncOp = builder.create<mlir::LLVM::LLVMFuncOp>(
        moduleOp.getLoc(), functionName, functionType);

    symbolTableCollection.getSymbolTable(moduleOp).insert(newFuncOp);
    return newFuncOp;
  }
}
