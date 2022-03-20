#include "marco/Codegen/Runtime.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"

using namespace ::marco;
using namespace ::marco::codegen;

namespace marco::codegen
{
  std::string RuntimeFunctionsMangling::getMangledFunction(
      llvm::StringRef baseName, llvm::StringRef mangledReturnType, llvm::ArrayRef<std::string> argsMangledTypes) const
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

  std::string RuntimeFunctionsMangling::getIntegerType(unsigned int bitWidth) const
  {
    return "i" + std::to_string(bitWidth);
  }

  std::string RuntimeFunctionsMangling::getFloatingPointType(unsigned int bitWidth) const
  {
    return "f" + std::to_string(bitWidth);
  }

  std::string RuntimeFunctionsMangling::getArrayType(llvm::StringRef mangledElementType) const
  {
    return "a" + mangledElementType.str();
  }

  std::string RuntimeFunctionsMangling::getPointerType(llvm::StringRef mangledElementType) const
  {
    return "p" + mangledElementType.str();
  }

  std::string RuntimeFunctionsMangling::getVoidPointerType() const
  {
    return getPointerType(getVoidType());
  }

  mlir::LLVM::LLVMFuncOp lookupOrCreateHeapAllocFn(mlir::ModuleOp moduleOp, mlir::Type indexType)
  {
    RuntimeFunctionsMangling mangling;

    auto mangledReturnType = mangling.getVoidPointerType();

    llvm::SmallVector<std::string, 1> mangledArgsTypes;
    mangledArgsTypes.push_back(mangling.getIntegerType(indexType.getIntOrFloatBitWidth()));

    auto functionName = mangling.getMangledFunction("heapAlloc", mangledReturnType, mangledArgsTypes);

    return mlir::LLVM::lookupOrCreateFn(
        moduleOp, functionName, indexType,
        mlir::LLVM::LLVMPointerType::get(mlir::IntegerType::get(moduleOp->getContext(), 8)));
  }
}
