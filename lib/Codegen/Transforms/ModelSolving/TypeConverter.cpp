#include "marco/Codegen/Transforms/ModelSolving/TypeConverter.h"
#include "marco/Codegen/Conversion/IDACommon/LLVMTypeConverter.h"
#include "marco/Codegen/Conversion/KINSOLToLLVM/LLVMTypeConverter.h"
#include "marco/Codegen/Conversion/ModelicaCommon/LLVMTypeConverter.h"

using namespace ::marco;
using namespace ::marco::codegen;

namespace marco::codegen
{
  ModelTypeConverter::ModelTypeConverter(mlir::MLIRContext* context, const mlir::LowerToLLVMOptions& options, unsigned int bitWidth)
      : mlir::modelica::LLVMTypeConverter(context, options, bitWidth),
        idaTypeConverter(context, options),
        kinsolTypeConverter(context, options)
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

    addConversion([&](mlir::kinsol::InstanceType type) {
      return kinsolTypeConverter.convertType(type);
    });

    addConversion([&](mlir::kinsol::VariableType type) {
      return kinsolTypeConverter.convertType(type);
    });

    addConversion([&](mlir::kinsol::EquationType type) {
      return kinsolTypeConverter.convertType(type);
    });
  }
}
