#include "marco/Codegen/Transforms/ModelSolving/TypeConverter.h"
#include "marco/Codegen/Conversion/IDAToLLVM/TypeConverter.h"
#include "marco/Codegen/Conversion/ModelicaCommon/LLVMTypeConverter.h"

using namespace ::marco;
using namespace ::marco::codegen;

namespace marco::codegen
{
  TypeConverter::TypeConverter(mlir::MLIRContext* context, mlir::LowerToLLVMOptions options, unsigned int bitWidth)
    : mlir::LLVMTypeConverter(context, options)
  {
    typeConverters.push_back(std::make_unique<mlir::modelica::LLVMTypeConverter>(&getContext(), options, bitWidth));
    typeConverters.push_back(std::make_unique<mlir::ida::TypeConverter>(context, options));

    addConversion([&](mlir::Type type) -> mlir::Type {
      for (const auto& typeConverter : typeConverters) {
        if (auto res = typeConverter->convertType(type); res != nullptr) {
          return res;
        }
      }

      return nullptr;
    });

    addTargetMaterialization([&](mlir::OpBuilder& builder, mlir::Type resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value> {
      for (const auto& typeConverter : typeConverters) {
        if (auto res = typeConverter->materializeTargetConversion(builder, loc, resultType, inputs); res != nullptr) {
          return res;
        }
      }

      return llvm::None;
    });

    addSourceMaterialization([&](mlir::OpBuilder& builder, mlir::Type resultType, mlir::ValueRange inputs, mlir::Location loc) -> llvm::Optional<mlir::Value> {
      for (const auto& typeConverter : typeConverters) {
        if (auto res = typeConverter->materializeSourceConversion(builder, loc, resultType, inputs); res != nullptr) {
          return res;
        }
      }

      return llvm::None;
    });
  }
}
