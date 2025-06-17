#ifndef MARCO_CODEGEN_RUNTIME_H
#define MARCO_CODEGEN_RUNTIME_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace marco::codegen {
class RuntimeFunctionsMangling {
public:
  std::string
  getMangledFunction(llvm::StringRef baseName,
                     llvm::StringRef mangledReturnType,
                     llvm::ArrayRef<std::string> argsMangledTypes) const;

  std::string getVoidType() const;

  std::string getIntegerType(unsigned int bitWidth) const;

  std::string getFloatingPointType(unsigned int bitWidth) const;

  std::string getArrayType(llvm::StringRef mangledElementType) const;

  std::string getPointerType(llvm::StringRef mangledElementType) const;

  std::string getVoidPointerType() const;
};
} // namespace marco::codegen

#endif // MARCO_CODEGEN_RUNTIME_H
