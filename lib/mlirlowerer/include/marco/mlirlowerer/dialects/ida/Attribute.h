#pragma once

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <mlir/IR/Attributes.h>

namespace marco::codegen::ida
{
	class IdaDialect;

	class BooleanAttributeStorage : public mlir::AttributeStorage
	{
		public:
		using KeyTy = std::tuple<mlir::Type, bool>;

		BooleanAttributeStorage() = delete;
		bool operator==(const KeyTy& key) const;
		static unsigned int hashKey(const KeyTy& key);
		static KeyTy getKey(mlir::Type type, bool value);
		static BooleanAttributeStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy key);

		[[nodiscard]] bool getValue() const;

		private:
		BooleanAttributeStorage(mlir::Type type, bool value);

		mlir::Type type;
		bool value;
	};

	class IntegerAttributeStorage : public mlir::AttributeStorage
	{
		public:
		using KeyTy = std::tuple<mlir::Type, llvm::APInt>;

		bool operator==(const KeyTy& key) const;
		static unsigned int hashKey(const KeyTy& key);
		static KeyTy getKey(mlir::Type type, int64_t value);
		static IntegerAttributeStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy key);

		[[nodiscard]] int64_t getValue() const;

		private:
		IntegerAttributeStorage(mlir::Type type, llvm::APInt value);

		mlir::Type type;
		llvm::APInt value;
	};

	class RealAttributeStorage : public mlir::AttributeStorage
	{
		public:
		using KeyTy = std::tuple<mlir::Type, llvm::APFloat>;

		bool operator==(const KeyTy& key) const;
		static unsigned int hashKey(const KeyTy& key);
		static KeyTy getKey(mlir::Type type, double value);
		static RealAttributeStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy key);

		[[nodiscard]] double getValue() const;

		private:
		RealAttributeStorage(mlir::Type type, llvm::APFloat value);

		mlir::Type type;
		llvm::APFloat value;
	};

	class BooleanAttribute : public mlir::Attribute::AttrBase<BooleanAttribute, mlir::Attribute, BooleanAttributeStorage>
	{
		public:
		using Base::Base;

		static constexpr llvm::StringRef getAttrName();
		static BooleanAttribute get(mlir::MLIRContext* context, bool value);
		static BooleanAttribute get(mlir::Type type, bool value);
		[[nodiscard]] bool getValue() const;
	};

	class IntegerAttribute : public mlir::Attribute::AttrBase<IntegerAttribute, mlir::Attribute, IntegerAttributeStorage>
	{
		public:
		using Base::Base;

		static constexpr llvm::StringRef getAttrName();
		static IntegerAttribute get(mlir::MLIRContext* context, int64_t value);
		static IntegerAttribute get(mlir::Type type, int64_t value);
		[[nodiscard]] int64_t getValue() const;
	};

	class RealAttribute : public mlir::Attribute::AttrBase<RealAttribute, mlir::Attribute, RealAttributeStorage>
	{
		public:
		using Base::Base;

		static constexpr llvm::StringRef getAttrName();
		static RealAttribute get(mlir::MLIRContext* context, double value);
		static RealAttribute get(mlir::Type type, double value);
		[[nodiscard]] double getValue() const;
	};

	mlir::Attribute parseIdaAttribute(mlir::DialectAsmParser& parser, mlir::Type type);
	void printIdaAttribute(mlir::Attribute attribute, mlir::DialectAsmPrinter& printer);
}
