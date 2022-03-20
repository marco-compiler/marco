#ifndef MARCO_CODEGEN_ARRAYDESCRIPTOR_H
#define MARCO_CODEGEN_ARRAYDESCRIPTOR_H

#include "mlir/IR/Builders.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace marco::codegen
{
  /// Helper class to produce LLVM dialect operations extracting or inserting
  /// values to a struct representing an array descriptor.
  class ArrayDescriptor
  {
    public:
      ArrayDescriptor(mlir::LLVMTypeConverter* typeConverter, mlir::Value value);

      /// Allocate an empty descriptor.
      static ArrayDescriptor undef(
          mlir::OpBuilder& builder, mlir::LLVMTypeConverter* typeConverter, mlir::Location location, mlir::Type descriptorType);

      mlir::Value operator*();

      /// Build IR to extract the pointer to the memory buffer.
      mlir::Value getPtr(mlir::OpBuilder& builder, mlir::Location location);

      /// Build IR to set the pointer to the memory buffer.
      void setPtr(mlir::OpBuilder& builder, mlir::Location location, mlir::Value ptr);

      /// Build IR to extract the rank.
      mlir::Value getRank(mlir::OpBuilder& builder, mlir::Location location);

      /// Build IR to set the rank.
      void setRank(mlir::OpBuilder& builder, mlir::Location location, mlir::Value rank);

      /// Build IR to extract the size of a dimension.
      mlir::Value getSize(mlir::OpBuilder& builder, mlir::Location location, unsigned int dimension);

      /// Build IR to extract the size of a dimension.
      mlir::Value getSize(mlir::OpBuilder& builder, mlir::Location location, mlir::Value dimension);

      /// Build IR to set the size of a dimension.
      void setSize(mlir::OpBuilder& builder, mlir::Location location, unsigned int dimension, mlir::Value size);

      /// Emit IR computing the memory (in bytes) that is necessary to store the descriptor.
      ///
      /// This assumes the descriptor to be
      ///   { type*, i32, i32[rank] }
      /// and densely packed, so the total size is
      ///   sizeof(pointer) + (1 + rank) * sizeof(i32).
      mlir::Value computeSize(mlir::OpBuilder& builder, mlir::Location loc);

      mlir::Type getRankType() const;

      mlir::Type getSizesContainerType() const;

      mlir::Type getSizeType() const;

    private:
      mlir::LLVMTypeConverter* typeConverter;
      mlir::Value value;
      mlir::Type descriptorType;
  };

  /// Helper class to produce LLVM dialect operations extracting or inserting
  /// values to a struct representing an unsized array descriptor.
  class UnsizedArrayDescriptor
  {
    public:
      explicit UnsizedArrayDescriptor(mlir::Value value);

      /// Allocate an empty descriptor.
      static UnsizedArrayDescriptor undef(mlir::OpBuilder& builder, mlir::Location location, mlir::Type descriptorType);

      mlir::Value operator*();

      /// Build IR to extract the rank.
      mlir::Value getRank(mlir::OpBuilder& builder, mlir::Location location);

      /// Build IR to set the rank.
      void setRank(mlir::OpBuilder& builder, mlir::Location location, mlir::Value rank);

      /// Build IR to extract the pointer to array descriptor.
      mlir::Value getPtr(mlir::OpBuilder& builder, mlir::Location location);

      /// Build IR to set the pointer to the array descriptor.
      void setPtr(mlir::OpBuilder& builder, mlir::Location location, mlir::Value ptr);

    private:
      mlir::Value value;
      mlir::Type descriptorType;
  };
}

#endif // MARCO_CODEGEN_ARRAYDESCRIPTOR_H
