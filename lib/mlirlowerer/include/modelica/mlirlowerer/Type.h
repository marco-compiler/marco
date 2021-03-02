#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <llvm/ADT/SmallVector.h>

namespace modelica
{
	class ModelicaDialect;

	class IntegerTypeStorage;
	class RealTypeStorage;
	class PointerTypeStorage;

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

	class PointerType : public mlir::Type::TypeBase<PointerType, mlir::Type, PointerTypeStorage> {
		public:
		using Base::Base;
		using Shape = llvm::SmallVector<long, 3>;

		/// Return a sequence type with the specified shape and element type
		static PointerType get(mlir::MLIRContext* context, bool heap, mlir::Type elementType, const Shape& shape = {});

		[[nodiscard]] bool isOnHeap() const;

		/// The element type of this sequence
		[[nodiscard]] mlir::Type getElementType() const;

		/// The shape of the sequence. If the sequence has an unknown shape, the shape
		/// returned will be empty.
		[[nodiscard]] Shape getShape() const;

		[[nodiscard]] unsigned int getRank() const;

		[[nodiscard]] unsigned int getConstantDimensions() const;
		[[nodiscard]] unsigned int getDynamicDimensions() const;

		[[nodiscard]] bool hasConstantShape() const;
	};

	void printModelicaType(ModelicaDialect* dialect, mlir::Type type, mlir::DialectAsmPrinter& printer);
}
