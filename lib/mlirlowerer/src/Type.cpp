#include <mlir/IR/DialectImplementation.h>
#include <modelica/mlirlowerer/Type.h>

using namespace modelica;

class modelica::IntegerTypeStorage : public mlir::TypeStorage {
	public:
	using KeyTy = unsigned int;

	IntegerTypeStorage() = delete;

	bool operator==(const KeyTy& key) const
	{
		return key == getBitWidth();
	}

	static unsigned int hashKey(const KeyTy& key)
	{
		return llvm::hash_combine(key);
	}

	static IntegerTypeStorage *construct(mlir::TypeStorageAllocator&allocator, unsigned int bitWidth) {
		auto* storage = allocator.allocate<PointerTypeStorage>();
		return new (storage) IntegerTypeStorage(bitWidth);
	}

	[[nodiscard]] unsigned int getBitWidth() const
	{
		return bitWidth;
	}

	private:
	explicit IntegerTypeStorage(unsigned int bitWidth) : bitWidth(bitWidth)
	{
	}

	unsigned int bitWidth;
};

class modelica::RealTypeStorage : public mlir::TypeStorage {
	public:
	using KeyTy = unsigned int;

	RealTypeStorage() = delete;

	bool operator==(const KeyTy& key) const
	{
		return key == getBitWidth();
	}

	static unsigned int hashKey(const KeyTy& key)
	{
		return llvm::hash_combine(key);
	}

	static RealTypeStorage *construct(mlir::TypeStorageAllocator&allocator, unsigned int bitWidth) {
		auto* storage = allocator.allocate<PointerTypeStorage>();
		return new (storage) RealTypeStorage(bitWidth);
	}

	[[nodiscard]] unsigned int getBitWidth() const
	{
		return bitWidth;
	}

	private:
	explicit RealTypeStorage(unsigned int bitWidth) : bitWidth(bitWidth)
	{
	}

	unsigned int bitWidth;
};

llvm::hash_code hash_value(const PointerType::Shape& shape) {
	if (shape.size()) {
		return llvm::hash_combine_range(shape.begin(), shape.end());
	}
	return llvm::hash_combine(0);
}

class modelica::PointerTypeStorage : public mlir::TypeStorage {
	public:
	using KeyTy = std::tuple<BufferAllocationScope, mlir::Type, PointerType::Shape>;

	PointerTypeStorage() = delete;

	bool operator==(const KeyTy& key) const {
		return key == KeyTy{getAllocationScope(), getElementType(), getShape()};
	}

	static unsigned int hashKey(const KeyTy& key) {
		auto shapeHash{hash_value(std::get<PointerType::Shape>(key))};
		shapeHash = llvm::hash_combine(shapeHash, std::get<mlir::Type>(key));
		return llvm::hash_combine(std::get<BufferAllocationScope>(key), std::get<mlir::Type>(key), shapeHash);
	}

	static PointerTypeStorage *construct(mlir::TypeStorageAllocator& allocator, const KeyTy &key) {
		auto *storage = allocator.allocate<PointerTypeStorage>();
		return new (storage) PointerTypeStorage{std::get<BufferAllocationScope>(key), std::get<mlir::Type>(key), std::get<PointerType::Shape>(key)};
	}

	[[nodiscard]] BufferAllocationScope getAllocationScope() const
	{
		return allocationScope;
	}

	[[nodiscard]] PointerType::Shape getShape() const
	{
		return shape;
	}

	[[nodiscard]] mlir::Type getElementType() const
	{
		return elementType;
	}

	private:
	PointerTypeStorage(BufferAllocationScope allocationScope, mlir::Type elementType, const PointerType::Shape& shape)
			: allocationScope(allocationScope),
				elementType(elementType),
				shape(std::move(shape))
	{
	}

	BufferAllocationScope allocationScope;
	mlir::Type elementType;
	PointerType::Shape shape;
};

BooleanType BooleanType::get(mlir::MLIRContext* context)
{
	return Base::get(context);
}

IntegerType IntegerType::get(mlir::MLIRContext* context, unsigned int bitWidth)
{
	return Base::get(context, bitWidth);
}

unsigned int IntegerType::getBitWidth() const
{
	return getImpl()->getBitWidth();
}

RealType RealType::get(mlir::MLIRContext* context, unsigned int bitWidth)
{
	return Base::get(context, bitWidth);
}

unsigned int RealType::getBitWidth() const
{
	return getImpl()->getBitWidth();
}

PointerType PointerType::get(mlir::MLIRContext* context, BufferAllocationScope allocationScope, mlir::Type elementType, llvm::ArrayRef<long> shape)
{
	return Base::get(context, allocationScope, elementType, Shape(shape.begin(), shape.end()));
}

BufferAllocationScope PointerType::getAllocationScope() const
{
	return getImpl()->getAllocationScope();
}

mlir::Type PointerType::getElementType() const
{
	return getImpl()->getElementType();
}

PointerType::Shape PointerType::getShape() const
{
	return getImpl()->getShape();
}

unsigned int PointerType::getRank() const
{
	return getShape().size();
}

unsigned int PointerType::getConstantDimensions() const
{
	auto dimensions = getShape();
	unsigned int count = 0;

	for (auto dimension : dimensions) {
		if (dimension > 0)
			count++;
	}

	return count;
}

unsigned int PointerType::getDynamicDimensions() const
{
	auto dimensions = getShape();
	unsigned int count = 0;

	for (auto dimension : dimensions) {
		if (dimension  == -1)
			count++;
	}

	return count;
}

bool PointerType::hasConstantShape() const
{
	auto dimensions = getShape();

	for (auto dimension : dimensions)
		if (dimension == -1)
			return false;

	return true;
}

void modelica::printModelicaType(mlir::Type type, mlir::DialectAsmPrinter& printer) {
	auto& os = printer.getStream();

	if (type.isa<BooleanType>()) {
		os << "bool";
		return;
	}

	if (type.isa<IntegerType>()) {
		os << "int";
		return;
	}

	if (type.dyn_cast<RealType>()) {
		os << "real";
		return;
	}

	if (auto pointerType = type.dyn_cast<PointerType>()) {
		os << "ptr<";

		if (pointerType.getAllocationScope() == stack)
			os << "stack, ";
		else if (pointerType.getAllocationScope() == heap)
			os << "heap, ";

		auto dimensions = pointerType.getShape();

		for (const auto& dimension : dimensions)
			os << (dimension == -1 ? "?" : std::to_string(dimension)) << "x";

		printer.printType(pointerType.getElementType());
		os << ">";
		return;
	}
}
