#include <mlir/IR/DialectImplementation.h>
#include <modelica/mlirlowerer/Attribute.h>
#include <modelica/mlirlowerer/Type.h>

using namespace modelica;

class modelica::BooleanAttributeStorage : public mlir::AttributeStorage
{
	public:
	using KeyTy = std::tuple<mlir::Type, bool>;

	BooleanAttributeStorage() = delete;

	bool operator==(const KeyTy& key) const
	{
		return key == KeyTy(type, value);
	}

	static unsigned int hashKey(const KeyTy& key)
	{
		return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
	}

	static KeyTy getKey(mlir::Type type, bool value) {
		return KeyTy(type, value);
	}

	static BooleanAttributeStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy key)
	{
		return new (allocator.allocate<BooleanAttributeStorage>()) BooleanAttributeStorage(std::get<0>(key), std::get<1>(key));
	}

	[[nodiscard]] bool getValue() const
	{
		return value;
	}

	private:
	BooleanAttributeStorage(mlir::Type type, bool value)
			: AttributeStorage(type), type(type), value(value)
	{
	}

	mlir::Type type;
	bool value;
};

class modelica::BooleanArrayAttributeStorage : public mlir::AttributeStorage
{
	public:
	using KeyTy = std::tuple<mlir::Type, llvm::ArrayRef<bool>>;

	bool operator==(const KeyTy& key) const
	{
		return key == KeyTy(type, values);
	}

	static unsigned int hashKey(const KeyTy& key)
	{
		return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
	}

	static KeyTy getKey(mlir::Type type, llvm::ArrayRef<bool> values)
	{
		assert(type.isa<PointerType>() && type.cast<PointerType>().getElementType().isa<BooleanType>());
		return KeyTy(type, values);
	}

	static BooleanArrayAttributeStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy key)
	{
		return new (allocator.allocate<BooleanArrayAttributeStorage>()) BooleanArrayAttributeStorage(std::get<0>(key), std::get<1>(key));
	}

	[[nodiscard]] llvm::ArrayRef<bool> getValue() const
	{
		return values;
	}

	private:
	BooleanArrayAttributeStorage(mlir::Type type, llvm::ArrayRef<bool> value)
			: AttributeStorage(type), type(type), values(value.begin(), value.end())
	{
	}

	mlir::Type type;
	llvm::SmallVector<bool, 3> values;
};

class modelica::IntegerAttributeStorage : public mlir::AttributeStorage
{
	public:
	using KeyTy = std::tuple<mlir::Type, llvm::APInt>;

	bool operator==(const KeyTy& key) const
	{
		return key == KeyTy(type, value);
	}

	static unsigned int hashKey(const KeyTy& key)
	{
		return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
	}

	static KeyTy getKey(mlir::Type type, long value)
	{
		assert(type.isa<IntegerType>());
		auto integerType = type.cast<IntegerType>();
		return KeyTy(type, llvm::APInt(integerType.getBitWidth(), value, true));
	}

	static IntegerAttributeStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy key)
	{
		return new (allocator.allocate<IntegerAttributeStorage>()) IntegerAttributeStorage(std::get<0>(key), std::get<1>(key));
	}

	[[nodiscard]] long getValue() const
	{
		return value.getSExtValue();
	}

	private:
	IntegerAttributeStorage(mlir::Type type, llvm::APInt value)
			: AttributeStorage(type), type(type), value(std::move(value))
	{
	}

	mlir::Type type;
	llvm::APInt value;
};

class modelica::IntegerArrayAttributeStorage : public mlir::AttributeStorage
{
	public:
	using KeyTy = std::tuple<mlir::Type, llvm::ArrayRef<llvm::APInt>>;

	bool operator==(const KeyTy& key) const
	{
		return key == KeyTy(type, values);
	}

	static unsigned int hashKey(const KeyTy& key)
	{
		return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
	}

	static KeyTy getKey(mlir::Type type, llvm::ArrayRef<llvm::APInt> values)
	{
		assert(type.isa<PointerType>() && type.cast<PointerType>().getElementType().isa<IntegerType>());
		return KeyTy(type, values);
	}

	static IntegerArrayAttributeStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy key)
	{
		return new (allocator.allocate<IntegerArrayAttributeStorage>()) IntegerArrayAttributeStorage(std::get<0>(key), std::get<1>(key));
	}

	[[nodiscard]] llvm::ArrayRef<llvm::APInt> getValue() const
	{
		return values;
	}

	private:
	IntegerArrayAttributeStorage(mlir::Type type, llvm::ArrayRef<llvm::APInt> value)
			: AttributeStorage(type), type(type), values(value.begin(), value.end())
	{
	}

	mlir::Type type;
	llvm::SmallVector<llvm::APInt, 3> values;
};

class modelica::RealAttributeStorage : public mlir::AttributeStorage
{
	public:
	using KeyTy = std::tuple<mlir::Type, llvm::APFloat>;

	bool operator==(const KeyTy& key) const
	{
		return key == KeyTy(type, value);
	}

	static unsigned int hashKey(const KeyTy& key)
	{
		return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
	}

	static KeyTy getKey(mlir::Type type, double value) {
		return KeyTy(type, value);
	}

	static RealAttributeStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy key)
	{
		return new (allocator.allocate<RealAttributeStorage>()) RealAttributeStorage(std::get<0>(key), std::get<1>(key));
	}

	[[nodiscard]] double getValue() const
	{
		return value.convertToDouble();
	}

	private:
	RealAttributeStorage(mlir::Type type, llvm::APFloat value)
			: AttributeStorage(type), type(type), value(std::move(value))
	{
	}

	mlir::Type type;
	llvm::APFloat value;
};

class modelica::RealArrayAttributeStorage : public mlir::AttributeStorage
{
	public:
	using KeyTy = std::tuple<mlir::Type, llvm::ArrayRef<llvm::APFloat>>;

	bool operator==(const KeyTy& key) const
	{
		return key == KeyTy(type, values);
	}

	static unsigned int hashKey(const KeyTy& key)
	{
		return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
	}

	static KeyTy getKey(mlir::Type type, llvm::ArrayRef<llvm::APFloat> values)
	{
		assert(type.isa<PointerType>() && type.cast<PointerType>().getElementType().isa<RealType>());
		return KeyTy(type, values);
	}

	static RealArrayAttributeStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy key)
	{
		return new (allocator.allocate<RealArrayAttributeStorage>()) RealArrayAttributeStorage(std::get<0>(key), std::get<1>(key));
	}

	[[nodiscard]] llvm::ArrayRef<llvm::APFloat> getValue() const
	{
		return values;
	}

	private:
	RealArrayAttributeStorage(mlir::Type type, llvm::ArrayRef<llvm::APFloat> value)
			: AttributeStorage(type), type(type), values(value.begin(), value.end())
	{
	}

	mlir::Type type;
	llvm::SmallVector<llvm::APFloat, 3> values;
};

constexpr llvm::StringRef BooleanAttribute::getAttrName()
{
	return "bool";
}

BooleanAttribute BooleanAttribute::get(mlir::Type type, bool value)
{
	assert(type.isa<BooleanType>());
	return Base::get(type.getContext(), type, value);
}

bool BooleanAttribute::getValue() const
{
	return getImpl()->getValue();
}

constexpr llvm::StringRef BooleanArrayAttribute::getAttrName()
{
	return "bool[]";
}

BooleanArrayAttribute BooleanArrayAttribute::get(mlir::Type type, llvm::ArrayRef<bool> values)
{
	assert(type.isa<PointerType>() && type.cast<PointerType>().getElementType().isa<BooleanType>());
	return Base::get(type.getContext(), type, values);
}

llvm::ArrayRef<bool> BooleanArrayAttribute::getValue() const
{
	return getImpl()->getValue();
}

constexpr llvm::StringRef IntegerAttribute::getAttrName()
{
	return "int";
}

IntegerAttribute IntegerAttribute::get(mlir::Type type, long value)
{
	assert(type.isa<IntegerType>());
	return Base::get(type.getContext(), type, value);
}

long IntegerAttribute::getValue() const
{
	return getImpl()->getValue();
}

constexpr llvm::StringRef IntegerArrayAttribute::getAttrName()
{
	return "int[]";
}

IntegerArrayAttribute IntegerArrayAttribute::get(mlir::Type type, llvm::ArrayRef<long> values)
{
	assert(type.isa<PointerType>() && type.cast<PointerType>().getElementType().isa<IntegerType>());
	auto baseType = type.cast<PointerType>().getElementType().cast<IntegerType>();
	llvm::SmallVector<llvm::APInt, 3> vals;

	for (const auto& value : values)
		vals.emplace_back(baseType.getBitWidth(), value, true);

	return Base::get(type.getContext(), type, vals);
}

llvm::ArrayRef<llvm::APInt> IntegerArrayAttribute::getValue() const
{
	return getImpl()->getValue();
}

constexpr llvm::StringRef RealAttribute::getAttrName()
{
	return "real";
}

RealAttribute RealAttribute::get(mlir::Type type, double value)
{
	assert(type.isa<RealType>());
	return Base::get(type.getContext(), type, value);
}

double RealAttribute::getValue() const
{
	return getImpl()->getValue();
}

constexpr llvm::StringRef RealArrayAttribute::getAttrName()
{
	return "real[]";
}

RealArrayAttribute RealArrayAttribute::get(mlir::Type type, llvm::ArrayRef<double> values)
{
	assert(type.isa<PointerType>() && type.cast<PointerType>().getElementType().isa<RealType>());
	llvm::SmallVector<llvm::APFloat, 3> vals(values.begin(), values.end());
	return Base::get(type.getContext(), type, vals);
}

llvm::ArrayRef<llvm::APFloat> RealArrayAttribute::getValue() const
{
	return getImpl()->getValue();
}

void modelica::printModelicaAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter& printer)
{
	auto& os = printer.getStream();

	if (auto attribute = attr.dyn_cast<BooleanAttribute>()) {
		os << (attribute.getValue() ? "true" : "false");
		return;
	}

	if (auto attribute = attr.dyn_cast<IntegerAttribute>()) {
		os << "int<" << std::to_string(attribute.getValue()) << ">";
		return;
	}

	if (auto attribute = attr.dyn_cast<RealAttribute>()) {
		os << "real<" << std::to_string(attribute.getValue()) << ">";
		return;
	}
}
