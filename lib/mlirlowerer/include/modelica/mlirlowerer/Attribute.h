#pragma once

#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Attributes.h>

namespace modelica
{
	class ModelicaDialect;

	class BooleanAttributeStorage;
	class IntegerAttributeStorage;
	class RealAttributeStorage;

	class BooleanAttribute : public mlir::Attribute::AttrBase<BooleanAttribute, mlir::Attribute, BooleanAttributeStorage> {
		public:
		using Base::Base;

		static constexpr llvm::StringRef getAttrName();
		static BooleanAttribute get(mlir::Type type, bool value);
		[[nodiscard]] bool getValue() const;
	};

	class IntegerAttribute : public mlir::Attribute::AttrBase<IntegerAttribute, mlir::Attribute, IntegerAttributeStorage> {
		public:
		using Base::Base;

		static constexpr llvm::StringRef getAttrName();
		static IntegerAttribute get(mlir::Type type, long value);
		[[nodiscard]] long getValue() const;
	};

	class RealAttribute : public mlir::Attribute::AttrBase<RealAttribute, mlir::Attribute, RealAttributeStorage> {
		public:
		using Base::Base;

		static constexpr llvm::StringRef getAttrName();
		static RealAttribute get(mlir::Type type, double value);
		[[nodiscard]] double getValue() const;
	};

	void printModelicaAttribute(mlir::Attribute attribute, mlir::DialectAsmPrinter& printer);
}
