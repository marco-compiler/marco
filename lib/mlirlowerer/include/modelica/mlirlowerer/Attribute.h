#pragma once

#include <llvm/ADT/SmallVector.h>
#include <map>
#include <mlir/IR/Attributes.h>

namespace modelica::codegen
{
	class ModelicaDialect;

	class BooleanAttributeStorage;
	class BooleanArrayAttributeStorage;
	class IntegerAttributeStorage;
	class IntegerArrayAttributeStorage;
	class RealAttributeStorage;
	class RealArrayAttributeStorage;
	class InverseFunctionAttributeStorage;

	class BooleanAttribute : public mlir::Attribute::AttrBase<BooleanAttribute, mlir::Attribute, BooleanAttributeStorage>
	{
		public:
		using Base::Base;

		static constexpr llvm::StringRef getAttrName();
		static BooleanAttribute get(mlir::Type type, bool value);
		[[nodiscard]] bool getValue() const;
	};

	class BooleanArrayAttribute : public mlir::Attribute::AttrBase<BooleanArrayAttribute, mlir::Attribute, BooleanArrayAttributeStorage>
	{
		public:
		using Base::Base;

		static constexpr llvm::StringRef getAttrName();
		static BooleanArrayAttribute get(mlir::Type type, llvm::ArrayRef<bool> values);
		[[nodiscard]] llvm::ArrayRef<bool> getValue() const;
	};

	class IntegerAttribute : public mlir::Attribute::AttrBase<IntegerAttribute, mlir::Attribute, IntegerAttributeStorage>
	{
		public:
		using Base::Base;

		static constexpr llvm::StringRef getAttrName();
		static IntegerAttribute get(mlir::Type type, long value);
		[[nodiscard]] long getValue() const;
	};

	class IntegerArrayAttribute : public mlir::Attribute::AttrBase<IntegerArrayAttribute, mlir::Attribute, IntegerArrayAttributeStorage>
	{
		public:
		using Base::Base;

		static constexpr llvm::StringRef getAttrName();
		static IntegerArrayAttribute get(mlir::Type type, llvm::ArrayRef<long> values);
		[[nodiscard]] llvm::ArrayRef<llvm::APInt> getValue() const;
	};

	class RealAttribute : public mlir::Attribute::AttrBase<RealAttribute, mlir::Attribute, RealAttributeStorage>
	{
		public:
		using Base::Base;

		static constexpr llvm::StringRef getAttrName();
		static RealAttribute get(mlir::Type type, double value);
		[[nodiscard]] double getValue() const;
	};

	class RealArrayAttribute : public mlir::Attribute::AttrBase<RealArrayAttribute, mlir::Attribute, RealArrayAttributeStorage>
	{
		public:
		using Base::Base;

		static constexpr llvm::StringRef getAttrName();
		static RealArrayAttribute get(mlir::Type type, llvm::ArrayRef<double> values);
		[[nodiscard]] llvm::ArrayRef<llvm::APFloat> getValue() const;
	};

	class InverseFunctionAttribute : public mlir::Attribute::AttrBase<InverseFunctionAttribute, mlir::Attribute, InverseFunctionAttributeStorage>
	{
		public:
		using Base::Base;

		static constexpr llvm::StringRef getAttrName();
		static InverseFunctionAttribute get(mlir::MLIRContext* context, unsigned int invertedArg, llvm::StringRef function, llvm::ArrayRef<unsigned int> args);

		[[nodiscard]] unsigned int getInvertedArgumentIndex() const;
		[[nodiscard]] llvm::StringRef getFunction() const;
		[[nodiscard]] llvm::ArrayRef<unsigned int> getArgumentsIndexes() const;
	};

	void printModelicaAttribute(mlir::Attribute attribute, mlir::DialectAsmPrinter& printer);
}
