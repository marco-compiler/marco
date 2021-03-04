#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/DialectImplementation.h>
#include <modelica/mlirlowerer/Type.h>

using namespace modelica;

class modelica::IntegerTypeStorage : public mlir::TypeStorage {
	public:
	using KeyTy = unsigned int;

	IntegerTypeStorage() = delete;

	static unsigned hashKey(const KeyTy& key)
	{
		return llvm::hash_combine(key);
	}

	bool operator==(const KeyTy& key) const
	{
		return key == getBitWidth();
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

	static unsigned hashKey(const KeyTy& key)
	{
		return llvm::hash_combine(key);
	}

	bool operator==(const KeyTy& key) const
	{
		return key == getBitWidth();
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
	using KeyTy = std::tuple<bool, mlir::Type, PointerType::Shape>;

	PointerTypeStorage() = delete;

	static unsigned hashKey(const KeyTy& key) {
		auto shapeHash{hash_value(std::get<PointerType::Shape>(key))};
		shapeHash = llvm::hash_combine(shapeHash, std::get<mlir::Type>(key));
		return llvm::hash_combine(std::get<bool>(key), std::get<mlir::Type>(key), shapeHash);
	}

	bool operator==(const KeyTy& key) const {
		return key == KeyTy{isOnHeap(), getElementType(), getShape()};
	}

	static PointerTypeStorage *construct(mlir::TypeStorageAllocator& allocator, const KeyTy &key) {
		auto *storage = allocator.allocate<PointerTypeStorage>();
		return new (storage) PointerTypeStorage{std::get<bool>(key), std::get<mlir::Type>(key), std::get<PointerType::Shape>(key)};
	}

	[[nodiscard]] PointerType::Shape getShape() const
	{
		return shape;
	}

	[[nodiscard]] mlir::Type getElementType() const
	{
		return elementType;
	}

	[[nodiscard]] bool isOnHeap() const
	{
		return heap;
	}

	private:
	PointerTypeStorage(bool heap, mlir::Type elementType, const PointerType::Shape& shape)
			: heap(heap),
				elementType(elementType),
				shape(std::move(shape))
	{
	}

	bool heap;
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

PointerType PointerType::get(mlir::MLIRContext* context, bool heap, mlir::Type elementType, const Shape shape)
{
	return Base::get(context, heap, elementType, shape);
}

bool PointerType::isOnHeap() const
{
	return getImpl()->isOnHeap();
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
	return getConstantDimensions() == getRank();
}

void modelica::printModelicaType(ModelicaDialect* dialect, mlir::Type ty, mlir::DialectAsmPrinter& printer) {
	auto& os = printer.getStream();

	if (auto type = ty.dyn_cast<BooleanType>()) {
		os << "bool";
		return;
	}

	if (auto type = ty.dyn_cast<IntegerType>()) {
		os << "int";
		return;
	}

	if (auto type = ty.dyn_cast<RealType>()) {
		os << "real";
		return;
	}

	if (auto type = ty.dyn_cast<PointerType>()) {
		os << "ptr<" << (type.isOnHeap() ? "true" : "false") << ", ";
		auto dimensions = type.getShape();

		for (const auto& dimension : dimensions)
			os << (dimension == -1 ? "?" : std::to_string(dimension)) << "x";

		printer.printType(type.getElementType());
		os << ">";
		return;
	}
}
