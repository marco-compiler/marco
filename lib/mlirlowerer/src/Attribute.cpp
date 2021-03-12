#include <mlir/IR/DialectImplementation.h>
#include <modelica/mlirlowerer/Attribute.h>
#include <modelica/mlirlowerer/Type.h>

using namespace modelica;

class modelica::BooleanAttributeStorage : public mlir::AttributeStorage {
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

	static BooleanAttributeStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy key) {
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

class modelica::IntegerAttributeStorage : public mlir::AttributeStorage {
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

	static KeyTy getKey(mlir::Type type, long value) {
		assert(type.isa<IntegerType>());
		auto integerType = type.cast<IntegerType>();
		return KeyTy(type, llvm::APInt(integerType.getBitWidth(), value, true));
	}

	static IntegerAttributeStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy key) {
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

class modelica::RealAttributeStorage : public mlir::AttributeStorage {
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

	static KeyTy getKey(mlir::Type type, unsigned int bitWidth, double value) {
		return KeyTy(type, value);
	}

	static RealAttributeStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy key) {
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

void modelica::printModelicaAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter& printer) {
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
