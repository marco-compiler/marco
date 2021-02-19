#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h>
#include <modelica/mlirlowerer/Attribute.h>

using namespace modelica;

class modelica::BooleanAttributeStorage : public mlir::AttributeStorage {
	public:
	using KeyTy = bool;

	static unsigned hashKey(const KeyTy& key)
	{
		return llvm::hash_combine(key);
	}

	bool operator==(const KeyTy& key) const
	{
		return key == value;
	}

	static BooleanAttributeStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy key) {
		return new (allocator.allocate<IndexAttributeStorage>()) BooleanAttributeStorage(key);
	}

	[[nodiscard]] bool getValue() const
	{
		return value;
	}

	private:
	BooleanAttributeStorage(long value) : value(value)
	{
	}

	bool value;
};

class modelica::IndexAttributeStorage : public mlir::AttributeStorage {
	public:
	using KeyTy = long;

	static unsigned hashKey(const KeyTy& key)
	{
		return llvm::hash_combine(key);
	}

	bool operator==(const KeyTy& key) const
	{
		return key == value;
	}

	static IndexAttributeStorage* construct(mlir::AttributeStorageAllocator& allocator, KeyTy key) {
		return new (allocator.allocate<IndexAttributeStorage>()) IndexAttributeStorage(key);
	}

	[[nodiscard]] long getValue() const
	{
		return value;
	}

	private:
	IndexAttributeStorage(long value) : value(value)
	{
	}

	long value;
};

constexpr llvm::StringRef IndexAttribute::getAttrName()
{
	return "index";
}

IndexAttribute IndexAttribute::get(mlir::MLIRContext* context, long value)
{
	return Base::get(context, value);
}

long IndexAttribute::getValue() const
{
	return getImpl()->getValue();
}

constexpr llvm::StringRef BooleanAttribute::getAttrName()
{
	return "boolean";
}

BooleanAttribute BooleanAttribute::get(mlir::MLIRContext* context, bool value)
{
	return Base::get(context, value);
}

bool BooleanAttribute::getValue() const
{
	return getImpl()->getValue();
}
