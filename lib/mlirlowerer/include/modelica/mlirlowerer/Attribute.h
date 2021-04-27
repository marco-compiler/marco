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
	class InverseFunctionsAttributeStorage;

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

	template<typename ValueType, typename BaseIterator>
	class InvertibleArgumentsIterator
	{
		public:
		using iterator_category = std::forward_iterator_tag;
		using value_type = ValueType;
		using difference_type = std::ptrdiff_t;
		using pointer = ValueType*;
		using reference = ValueType&;

		InvertibleArgumentsIterator(BaseIterator iterator) : iterator(iterator)
		{
		}

		operator bool() const { return iterator(); }

		bool operator==(const InvertibleArgumentsIterator& it) const
		{
			return it.iterator == iterator;
		}

		bool operator!=(const InvertibleArgumentsIterator& it) const
		{
			return it.iterator != iterator;
		}

		InvertibleArgumentsIterator& operator++()
		{
			iterator++;
			return *this;
		}

		InvertibleArgumentsIterator operator++(int)
		{
			auto temp = *this;
			iterator++;
			return temp;
		}

		value_type& operator*()
		{
			return iterator->first;
		}

		const value_type& operator*() const
		{
			return iterator->first;
		}

		private:
		BaseIterator iterator;
	};

	class InverseFunctionsAttribute : public mlir::Attribute::AttrBase<InverseFunctionsAttribute, mlir::Attribute, InverseFunctionsAttributeStorage>
	{
		public:
		using Map = std::map<unsigned int, std::pair<llvm::StringRef, llvm::ArrayRef<unsigned int>>>;
		using iterator = InvertibleArgumentsIterator<unsigned int, Map::iterator>;
		using const_iterator = InvertibleArgumentsIterator<const unsigned int, Map::const_iterator>;

		using Base::Base;

		static constexpr llvm::StringRef getAttrName();
		static InverseFunctionsAttribute get(mlir::MLIRContext* context, Map inverseFunctionsList);

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator cbegin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator cend() const;

		[[nodiscard]] bool isInvertible(unsigned int argumentIndex) const;
		[[nodiscard]] llvm::StringRef getFunction(unsigned int argumentIndex) const;
		[[nodiscard]] llvm::ArrayRef<unsigned int> getArgumentsIndexes(unsigned int argumentIndex) const;
	};

	void printModelicaAttribute(mlir::Attribute attribute, mlir::DialectAsmPrinter& printer);
}
