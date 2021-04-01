#pragma once

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Types.h>

namespace modelica
{
	class ModelicaDialect;

	class IntegerTypeStorage;
	class RealTypeStorage;
	class PointerTypeStorage;
	//class UnrankedPointerTypeStorage;
	class RecordTypeStorage;

	class BooleanType : public mlir::Type::TypeBase<BooleanType, mlir::Type, mlir::TypeStorage> {
		public:
		using Base::Base;
		static BooleanType get(mlir::MLIRContext* context);
	};

	class IntegerType : public mlir::Type::TypeBase<IntegerType, mlir::Type, IntegerTypeStorage> {
		public:
		using Base::Base;
		static IntegerType get(mlir::MLIRContext* context, unsigned int bitWidth);
		[[nodiscard]] unsigned int getBitWidth() const;
	};

	class RealType : public mlir::Type::TypeBase<RealType, mlir::Type, RealTypeStorage> {
		public:
		using Base::Base;
		static RealType get(mlir::MLIRContext* context, unsigned int bitWidth);
		[[nodiscard]] unsigned int getBitWidth() const;
	};

	enum BufferAllocationScope { unknown, stack, heap };

	class PointerType : public mlir::Type::TypeBase<PointerType, mlir::Type, PointerTypeStorage> {
		public:
		using Base::Base;
		using Shape = llvm::SmallVector<long, 3>;

		static PointerType get(mlir::MLIRContext* context, BufferAllocationScope allocationScope, mlir::Type elementType, llvm::ArrayRef<long> shape = {});

		[[nodiscard]] BufferAllocationScope getAllocationScope() const;

		[[nodiscard]] mlir::Type getElementType() const;

		[[nodiscard]] Shape getShape() const;

		[[nodiscard]] unsigned int getRank() const;

		[[nodiscard]] unsigned int getConstantDimensions() const;
		[[nodiscard]] unsigned int getDynamicDimensions() const;

		[[nodiscard]] bool hasConstantShape() const;

		[[nodiscard]] PointerType slice(unsigned int subscriptsAmount);
		[[nodiscard]] PointerType toAllocationScope(BufferAllocationScope scope);
		[[nodiscard]] PointerType toUnknownAllocationScope();
		[[nodiscard]] PointerType toElementType(mlir::Type type);

		[[nodiscard]] bool canBeOnStack() const;
	};

	/*
	class UnrankedPointerType : public mlir::Type::TypeBase<UnrankedPointerType, mlir::Type, UnrankedPointerTypeStorage> {
		public:
		using Base::Base;

		static UnrankedPointerType get(mlir::MLIRContext* context, mlir::Type elementType, unsigned int rank);

		[[nodiscard]] mlir::Type getElementType() const;
		[[nodiscard]] unsigned int getRank() const;
	};
	 */

	class RecordType : public mlir::Type::TypeBase<RecordType, mlir::Type, RecordTypeStorage>
	    {
		public:
		using Base::Base;

		static RecordType get(mlir::MLIRContext* context, llvm::ArrayRef<mlir::Type> elementTypes);

		llvm::ArrayRef<mlir::Type> getElementTypes();
	};

	void printModelicaType(mlir::Type type, mlir::DialectAsmPrinter& printer);
}
