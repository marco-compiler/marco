#ifndef MARCO_CODEGEN_TRANSFORMS_MODEL_TYPECONVERTER_H
#define MARCO_CODEGEN_TRANSFORMS_MODEL_TYPECONVERTER_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/MLIRContext.h"
#include <memory>
#include <vector>

namespace marco::codegen
{
  class TypeConverter : public mlir::LLVMTypeConverter
  {
    public:
      TypeConverter(mlir::MLIRContext* context, mlir::LowerToLLVMOptions options, unsigned int bitWidth);

    private:
      std::vector<std::unique_ptr<mlir::TypeConverter>> typeConverters;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODEL_TYPECONVERTER_H
