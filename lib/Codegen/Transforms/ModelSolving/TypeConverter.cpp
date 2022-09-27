#include "marco/Codegen/Transforms/ModelSolving/TypeConverter.h"
#include "marco/Codegen/Conversion/IDAToLLVM/LLVMTypeConverter.h"
#include "marco/Codegen/Conversion/ModelicaCommon/LLVMTypeConverter.h"

using namespace ::marco;
using namespace ::marco::codegen;

namespace marco::codegen
{
  ModelTypeConverter::ModelTypeConverter(mlir::MLIRContext* context, const mlir::LowerToLLVMOptions& options, unsigned int bitWidth)
      : mlir::modelica::LLVMTypeConverter(context, options, bitWidth),
        idaTypeConverter(context, options)
  {
    addConversion([&](mlir::ida::InstanceType type) {
      return idaTypeConverter.convertType(type);
    });

    addConversion([&](mlir::ida::VariableType type) {
      return idaTypeConverter.convertType(type);
    });

    addConversion([&](mlir::ida::EquationType type) {
      return idaTypeConverter.convertType(type);
    });
  }
}
