#pragma once

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/SmallVector.h>
#include <map>
#include <mlir/IR/Attributes.h>

namespace marco::codegen::modelica
{
	class ModelicaDialect;

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

	class BooleanArrayAttributeStorage : public mlir::AttributeStorage
	{
		public:
		using KeyTy = std::tuple<mlir::Type, llvm::ArrayRef<bool>>;

		bool operator==(const KeyTy& key) const;
		static unsigned int hashKey(const KeyTy& key);
		static KeyTy getKey(mlir::Type type, llvm::ArrayRef<bool> values);
		static BooleanArrayAttributeStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy key);

		[[nodiscard]] llvm::ArrayRef<bool> getValue() const;

		private:
		BooleanArrayAttributeStorage(mlir::Type type, llvm::ArrayRef<bool> value);

		mlir::Type type;
		llvm::SmallVector<bool, 3> values;
	};

	class IntegerAttributeStorage : public mlir::AttributeStorage
	{
		public:
		using KeyTy = std::tuple<mlir::Type, llvm::APInt>;

		bool operator==(const KeyTy& key) const;
		static unsigned int hashKey(const KeyTy& key);
		static KeyTy getKey(mlir::Type type, long value);
		static IntegerAttributeStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy key);

		[[nodiscard]] long getValue() const;

		private:
		IntegerAttributeStorage(mlir::Type type, llvm::APInt value);

		mlir::Type type;
		llvm::APInt value;
	};

	class IntegerArrayAttributeStorage : public mlir::AttributeStorage
	{
		public:
		using KeyTy = std::tuple<mlir::Type, llvm::ArrayRef<llvm::APInt>>;

		bool operator==(const KeyTy& key) const;
		static unsigned int hashKey(const KeyTy& key);
		static KeyTy getKey(mlir::Type type, llvm::ArrayRef<llvm::APInt> values);
		static IntegerArrayAttributeStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy key);

		[[nodiscard]] llvm::ArrayRef<llvm::APInt> getValue() const;

		private:
		IntegerArrayAttributeStorage(mlir::Type type, llvm::ArrayRef<llvm::APInt> value);

		mlir::Type type;
		llvm::SmallVector<llvm::APInt, 3> values;
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

	class RealArrayAttributeStorage : public mlir::AttributeStorage
	{
		public:
		using KeyTy = std::tuple<mlir::Type, llvm::ArrayRef<llvm::APFloat>>;

		bool operator==(const KeyTy& key) const;
		static unsigned int hashKey(const KeyTy& key);
		static KeyTy getKey(mlir::Type type, llvm::ArrayRef<llvm::APFloat> values);
		static RealArrayAttributeStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy key);

		[[nodiscard]] llvm::ArrayRef<llvm::APFloat> getValue() const;

		private:
		RealArrayAttributeStorage(mlir::Type type, llvm::ArrayRef<llvm::APFloat> value);

		mlir::Type type;
		llvm::SmallVector<llvm::APFloat, 3> values;
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

	class InverseFunctionsAttributeStorage : public mlir::AttributeStorage
	{
		public:
		using Map = std::map<unsigned int, std::pair<llvm::StringRef, llvm::ArrayRef<unsigned int>>>;
		using iterator = InvertibleArgumentsIterator<unsigned int, Map::iterator>;
		using const_iterator = InvertibleArgumentsIterator<const unsigned int, Map::const_iterator>;

		using KeyTy = Map;

		bool operator==(const KeyTy& key) const;
		static unsigned int hashKey(const KeyTy& key);
		static InverseFunctionsAttributeStorage* construct(mlir::AttributeStorageAllocator& allocator, Map map);

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator cbegin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator cend() const;

		[[nodiscard]] bool isInvertible(unsigned int argumentIndex) const;
		[[nodiscard]] llvm::StringRef getFunction(unsigned int argumentIndex) const;
		[[nodiscard]] llvm::ArrayRef<unsigned int> getArgumentsIndexes(unsigned int argumentIndex) const;

		private:
		InverseFunctionsAttributeStorage(Map map);

		Map map;
	};

	class DerivativeAttributeStorage: public mlir::AttributeStorage
	{
		public:
		using KeyTy = std::tuple<llvm::StringRef, unsigned int>;

		bool operator==(const KeyTy& key) const;
		static unsigned int hashKey(const KeyTy& key);
		static DerivativeAttributeStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy key);

		[[nodiscard]] llvm::StringRef getName() const;
		[[nodiscard]] unsigned int getOrder() const;

		private:
		DerivativeAttributeStorage(llvm::StringRef name, unsigned int order);

		llvm::StringRef name;
		unsigned int order;
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
		static IntegerAttribute get(mlir::MLIRContext* context, long value);
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
		static RealAttribute get(mlir::MLIRContext* context, double value);
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

	class InverseFunctionsAttribute : public mlir::Attribute::AttrBase<InverseFunctionsAttribute, mlir::Attribute, InverseFunctionsAttributeStorage>
	{
		public:
		using Map = InverseFunctionsAttributeStorage::Map;
		using iterator = InverseFunctionsAttributeStorage::iterator;
		using const_iterator = InverseFunctionsAttributeStorage::const_iterator;

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

	class DerivativeAttribute : public mlir::Attribute::AttrBase<DerivativeAttribute, mlir::Attribute, DerivativeAttributeStorage>
	{
		public:
		using Base::Base;

		static constexpr llvm::StringRef getAttrName();
		static DerivativeAttribute get(mlir::MLIRContext* context, llvm::StringRef name, unsigned int order);

		[[nodiscard]] llvm::StringRef getName() const;
		[[nodiscard]] unsigned int getOrder() const;
	};

	mlir::Attribute parseModelicaAttribute(mlir::DialectAsmParser& parser, mlir::Type type);
	void printModelicaAttribute(mlir::Attribute attribute, mlir::DialectAsmPrinter& printer);
}
