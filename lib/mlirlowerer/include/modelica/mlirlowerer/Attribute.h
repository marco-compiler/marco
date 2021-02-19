#pragma once

#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <llvm/ADT/SmallVector.h>

namespace modelica
{
	class ModelicaDialect;

	class BooleanAttributeStorage;
	class IndexAttributeStorage;

	class BooleanAttribute : public mlir::Attribute::AttrBase<BooleanAttribute, mlir::Attribute, BooleanAttributeStorage> {
		public:
		using Base::Base;

		static constexpr llvm::StringRef getAttrName();
		static BooleanAttribute get(mlir::MLIRContext* context, bool value);
		[[nodiscard]] bool getValue() const;
	};

	class IndexAttribute : public mlir::Attribute::AttrBase<IndexAttribute, mlir::Attribute, IndexAttributeStorage> {
		public:
		using Base::Base;

		static constexpr llvm::StringRef getAttrName();
		static IndexAttribute get(mlir::MLIRContext* context, long value);
		[[nodiscard]] long getValue() const;
	};
}

