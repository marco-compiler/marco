#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_TYPECONVERTER_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_TYPECONVERTER_H

#include "marco/Codegen/Conversion/ModelicaCommon/LLVMTypeConverter.h"
#include "marco/Codegen/Conversion/IDAToLLVM/LLVMTypeConverter.h"
#include "marco/Codegen/Conversion/KINSOLToLLVM/LLVMTypeConverter.h"
#include "mlir/IR/MLIRContext.h"
#include <memory>
#include <vector>

namespace marco::codegen
{
  class ModelTypeConverter : public mlir::modelica::LLVMTypeConverter
  {
    public:
    ModelTypeConverter(mlir::MLIRContext* context, const mlir::LowerToLLVMOptions& options, unsigned int bitWidth);

    private:
      mlir::ida::LLVMTypeConverter idaTypeConverter;
      mlir::kinsol::LLVMTypeConverter kinsolTypeConverter;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_TYPECONVERTER_H
