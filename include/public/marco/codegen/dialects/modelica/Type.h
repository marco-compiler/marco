#pragma once

#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/Types.h"

namespace marco::codegen::modelica
{
	class ModelicaDialect;

	enum class MemberAllocationScope
	{
		stack,
		heap
	};

	class MemberTypeStorage : public mlir::TypeStorage
	{
		public:
		using Shape = llvm::SmallVector<long, 3>;
		using KeyTy = std::tuple<MemberAllocationScope, mlir::Type, Shape>;

		MemberTypeStorage() = delete;

		bool operator==(const KeyTy& key) const;
		static unsigned int hashKey(const KeyTy& key);
		static MemberTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy &key);

		[[nodiscard]] MemberAllocationScope getAllocationScope() const;
		[[nodiscard]] Shape getShape() const;
		[[nodiscard]] mlir::Type getElementType() const;

		private:
		MemberTypeStorage(MemberAllocationScope allocationScope, mlir::Type elementType, const Shape& shape);

		MemberAllocationScope allocationScope;
		mlir::Type elementType;
		Shape shape;
	};

	enum class BufferAllocationScope
	{
		unknown,
		stack,
		heap
	};

	class ArrayTypeStorage : public mlir::TypeStorage
	{
		public:
		using Shape = llvm::SmallVector<long, 3>;
		using KeyTy = std::tuple<BufferAllocationScope, mlir::Type, Shape>;

		ArrayTypeStorage() = delete;

		bool operator==(const KeyTy& key) const;
		static unsigned int hashKey(const KeyTy& key);
		static ArrayTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy &key);

		[[nodiscard]] BufferAllocationScope getAllocationScope() const;
		[[nodiscard]] Shape getShape() const;
		[[nodiscard]] mlir::Type getElementType() const;

		private:
		ArrayTypeStorage(BufferAllocationScope allocationScope, mlir::Type elementType, const Shape& shape);

		BufferAllocationScope allocationScope;
		mlir::Type elementType;
		Shape shape;
	};

	class UnsizedArrayTypeStorage : public mlir::TypeStorage
	{
		public:
		using KeyTy = mlir::Type;

		UnsizedArrayTypeStorage() = delete;

		bool operator==(const KeyTy& key) const;
		static unsigned int hashKey(const KeyTy& key);
		static UnsizedArrayTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy &key);

		[[nodiscard]] mlir::Type getElementType() const;
		[[nodiscard]] unsigned int getRank() const;

		private:
		UnsizedArrayTypeStorage(mlir::Type elementType);

		mlir::Type elementType;
	};

	class StructTypeStorage : public mlir::TypeStorage
	{
		public:
		using KeyTy = llvm::ArrayRef<mlir::Type>;

		StructTypeStorage() = delete;
		bool operator==(const KeyTy& key) const;
		static unsigned int hashKey(const KeyTy& key);
		static StructTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy& key);

		[[nodiscard]] llvm::ArrayRef<mlir::Type> getElementTypes() const;

		private:
		StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes);

		llvm::ArrayRef<mlir::Type> elementTypes;
	};

	class BooleanType : public mlir::Type::TypeBase<BooleanType, mlir::Type, mlir::TypeStorage>
	{
		public:
		using Base::Base;

		static BooleanType get(mlir::MLIRContext* context);
	};

	class IntegerType : public mlir::Type::TypeBase<IntegerType, mlir::Type, mlir::TypeStorage>
	{
		public:
		using Base::Base;

		static IntegerType get(mlir::MLIRContext* context);
	};

	class RealType : public mlir::Type::TypeBase<RealType, mlir::Type, mlir::TypeStorage>
	{
		public:
		using Base::Base;

		static RealType get(mlir::MLIRContext* context);
	};

	class ArrayType;

	class MemberType : public mlir::Type::TypeBase<MemberType, mlir::Type, MemberTypeStorage>
	{
		public:
		using Base::Base;
		using Shape = MemberTypeStorage::Shape;

		static MemberType get(mlir::MLIRContext* context, MemberAllocationScope allocationScope, mlir::Type elementType, llvm::ArrayRef<long> shape = {});
		static MemberType get(ArrayType arrayType);

		[[nodiscard]] MemberAllocationScope getAllocationScope() const;
		[[nodiscard]] mlir::Type getElementType() const;
		[[nodiscard]] Shape getShape() const;
		[[nodiscard]] unsigned int getRank() const;

		[[nodiscard]] ArrayType toArrayType() const;
		[[nodiscard]] mlir::Type unwrap() const;
	};

	class UnsizedArrayType;

	class ArrayType : public mlir::Type::TypeBase<ArrayType, mlir::Type, ArrayTypeStorage>
	{
		public:
		using Base::Base;
		using Shape = ArrayTypeStorage::Shape;

		static ArrayType get(mlir::MLIRContext* context, BufferAllocationScope allocationScope, mlir::Type elementType, llvm::ArrayRef<long> shape = {});

		[[nodiscard]] BufferAllocationScope getAllocationScope() const;

		[[nodiscard]] mlir::Type getElementType() const;

		[[nodiscard]] Shape getShape() const;

		[[nodiscard]] unsigned int getRank() const;

		[[nodiscard]] unsigned int getConstantDimensions() const;
		[[nodiscard]] unsigned int getDynamicDimensions() const;

		[[nodiscard]] long rawSize() const;

		[[nodiscard]] bool hasConstantShape() const;

		[[nodiscard]] bool isScalar() const;

		[[nodiscard]] ArrayType slice(unsigned int subscriptsAmount);
		[[nodiscard]] ArrayType toAllocationScope(BufferAllocationScope scope);
		[[nodiscard]] ArrayType toUnknownAllocationScope();
		[[nodiscard]] ArrayType toMinAllowedAllocationScope();
		[[nodiscard]] ArrayType toElementType(mlir::Type elementType);
		[[nodiscard]] UnsizedArrayType toUnsized();

		[[nodiscard]] bool canBeOnStack() const;
	};

	class UnsizedArrayType : public mlir::Type::TypeBase<UnsizedArrayType, mlir::Type, UnsizedArrayTypeStorage>
	{
		public:
		using Base::Base;

		static UnsizedArrayType get(mlir::MLIRContext* context, mlir::Type elementType);

		[[nodiscard]] mlir::Type getElementType() const;
	};

	class StructType : public mlir::Type::TypeBase<StructType, mlir::Type, StructTypeStorage>
	{
		public:
		using Base::Base;

		static StructType get(mlir::MLIRContext* context, llvm::ArrayRef<mlir::Type> elementTypes);

		llvm::ArrayRef<mlir::Type> getElementTypes();
	};

	mlir::Type parseModelicaType(mlir::DialectAsmParser& parser);
	void printModelicaType(mlir::Type type, mlir::DialectAsmPrinter& printer);
}
