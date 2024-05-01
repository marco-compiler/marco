#include "gmock/gmock.h"
#include "marco/Codegen/Conversion/ModelicaCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace ::mlir::bmodelica;

TEST(TypeConverterTest, array)
{
  /*
  mlir::MLIRContext context;
  context.loadDialect<ModelicaDialect>();
  context.loadDialect<mlir::memref::MemRefDialect>();
  context.loadDialect<mlir::LLVM::LLVMDialect>();
  mlir::LowerToLLVMOptions options(&context);

  TypeConverter typeConverter(&context, options, 64);
  auto arrayType = ArrayType::get(&context, IntegerType::get(&context), { 3, 5 });
  typeConverter.convertType(arrayType).dump();
   */
}
