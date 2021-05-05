#pragma once

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Types.h>

namespace modelica::codegen
{
	class ModelicaDialect;

	class IntegerTypeStorage : public mlir::TypeStorage
	{
		public:
		using KeyTy = unsigned int;

		IntegerTypeStorage() = delete;
		bool operator==(const KeyTy& key) const;
		static unsigned int hashKey(const KeyTy& key);
		static IntegerTypeStorage* construct(mlir::TypeStorageAllocator&allocator, unsigned int bitWidth);

		[[nodiscard]] unsigned int getBitWidth() const;

		private:
		explicit IntegerTypeStorage(unsigned int bitWidth);

		unsigned int bitWidth;
	};

	class RealTypeStorage : public mlir::TypeStorage
	{
		public:
		using KeyTy = unsigned int;

		RealTypeStorage() = delete;
		bool operator==(const KeyTy& key) const;
		static unsigned int hashKey(const KeyTy& key);
		static RealTypeStorage* construct(mlir::TypeStorageAllocator&allocator, unsigned int bitWidth);

		[[nodiscard]] unsigned int getBitWidth() const;

		private:
		explicit RealTypeStorage(unsigned int bitWidth);

		unsigned int bitWidth;
	};

	enum BufferAllocationScope { unknown, stack, heap };

	class PointerTypeStorage : public mlir::TypeStorage
	{
		public:
		using Shape = llvm::SmallVector<long, 3>;
		using KeyTy = std::tuple<BufferAllocationScope, mlir::Type, Shape>;

		PointerTypeStorage() = delete;

		bool operator==(const KeyTy& key) const;
		static unsigned int hashKey(const KeyTy& key);
		static PointerTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy &key);

		[[nodiscard]] BufferAllocationScope getAllocationScope() const;
		[[nodiscard]] Shape getShape() const;
		[[nodiscard]] mlir::Type getElementType() const;

		private:
		PointerTypeStorage(BufferAllocationScope allocationScope, mlir::Type elementType, const Shape& shape);

		BufferAllocationScope allocationScope;
		mlir::Type elementType;
		Shape shape;
	};

	class UnsizedPointerTypeStorage : public mlir::TypeStorage
	{
		public:
		using KeyTy = std::tuple<mlir::Type, unsigned int>;

		UnsizedPointerTypeStorage() = delete;

		bool operator==(const KeyTy& key) const;
		static unsigned int hashKey(const KeyTy& key);
		static UnsizedPointerTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy &key);

		[[nodiscard]] mlir::Type getElementType() const;
		[[nodiscard]] unsigned int getRank() const;

		private:
		UnsizedPointerTypeStorage(mlir::Type elementType, unsigned int rank);

		mlir::Type elementType;
		unsigned int rank;
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

	class IntegerType : public mlir::Type::TypeBase<IntegerType, mlir::Type, IntegerTypeStorage>
	{
		public:
		using Base::Base;

		static IntegerType get(mlir::MLIRContext* context, unsigned int bitWidth);
		[[nodiscard]] unsigned int getBitWidth() const;
	};

	class RealType : public mlir::Type::TypeBase<RealType, mlir::Type, RealTypeStorage>
	{
		public:
		using Base::Base;

		static RealType get(mlir::MLIRContext* context, unsigned int bitWidth);
		[[nodiscard]] unsigned int getBitWidth() const;
	};

	class UnsizedPointerType;

	class PointerType : public mlir::Type::TypeBase<PointerType, mlir::Type, PointerTypeStorage>
	{
		public:
		using Base::Base;
		using Shape = PointerTypeStorage::Shape;

		static PointerType get(mlir::MLIRContext* context, BufferAllocationScope allocationScope, mlir::Type elementType, llvm::ArrayRef<long> shape = {});

		[[nodiscard]] BufferAllocationScope getAllocationScope() const;

		[[nodiscard]] mlir::Type getElementType() const;

		[[nodiscard]] Shape getShape() const;

		[[nodiscard]] unsigned int getRank() const;

		[[nodiscard]] unsigned int getConstantDimensions() const;
		[[nodiscard]] unsigned int getDynamicDimensions() const;

		[[nodiscard]] long rawSize() const;

		[[nodiscard]] bool hasConstantShape() const;

		[[nodiscard]] PointerType slice(unsigned int subscriptsAmount);
		[[nodiscard]] PointerType toAllocationScope(BufferAllocationScope scope);
		[[nodiscard]] PointerType toUnknownAllocationScope();
		[[nodiscard]] PointerType toMinAllowedAllocationScope();
		[[nodiscard]] UnsizedPointerType toUnsized();

		[[nodiscard]] bool canBeOnStack() const;
	};

	class UnsizedPointerType : public mlir::Type::TypeBase<UnsizedPointerType, mlir::Type, UnsizedPointerTypeStorage>
	{
		public:
		using Base::Base;

		static UnsizedPointerType get(mlir::MLIRContext* context, mlir::Type elementType, unsigned int rank);

		[[nodiscard]] mlir::Type getElementType() const;
		[[nodiscard]] unsigned int getRank() const;
	};

	class OpaquePointerType : public mlir::Type::TypeBase<OpaquePointerType, mlir::Type, mlir::TypeStorage>
	{
		public:
		using Base::Base;
		static OpaquePointerType get(mlir::MLIRContext* context);
	};

	class StructType : public mlir::Type::TypeBase<StructType, mlir::Type, StructTypeStorage>
	{
		public:
		using Base::Base;

		static StructType get(mlir::MLIRContext* context, llvm::ArrayRef<mlir::Type> elementTypes);

		llvm::ArrayRef<mlir::Type> getElementTypes();
	};

	void printModelicaType(mlir::Type type, mlir::DialectAsmPrinter& printer);
}
