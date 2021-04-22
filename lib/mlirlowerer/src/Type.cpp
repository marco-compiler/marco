#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/DialectImplementation.h>
#include <modelica/mlirlowerer/Type.h>

using namespace modelica::codegen;

namespace modelica::codegen
{
	class IntegerTypeStorage : public mlir::TypeStorage {
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

		static IntegerTypeStorage* construct(mlir::TypeStorageAllocator&allocator, unsigned int bitWidth) {
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

	class RealTypeStorage : public mlir::TypeStorage {
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

		static RealTypeStorage* construct(mlir::TypeStorageAllocator&allocator, unsigned int bitWidth) {
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

	class PointerTypeStorage : public mlir::TypeStorage {
		public:
		using KeyTy = std::tuple<BufferAllocationScope, mlir::Type, PointerType::Shape>;

		PointerTypeStorage() = delete;

		bool operator==(const KeyTy& key) const {
			return key == KeyTy{getAllocationScope(), getElementType(), getShape()};
		}

		static unsigned int hashKey(const KeyTy& key) {
			auto shapeHash{hash_value(std::get<PointerType::Shape>(key))};
			return llvm::hash_combine(std::get<BufferAllocationScope>(key), std::get<mlir::Type>(key), shapeHash);
		}

		static PointerTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy &key) {
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

/*
class UnrankedPointerTypeStorage : public mlir::TypeStorage {
	public:
	using KeyTy = std::tuple<mlir::Type, unsigned int>;

	UnrankedPointerTypeStorage() = delete;

	bool operator==(const KeyTy& key) const {
		return key == KeyTy{getElementType(), getRank()};
	}

	static unsigned int hashKey(const KeyTy& key) {
		return llvm::hash_combine(std::get<mlir::Type>(key), std::get<unsigned int>(key));
	}

	static UnrankedPointerTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy &key) {
		auto *storage = allocator.allocate<UnrankedPointerTypeStorage>();
		return new (storage) UnrankedPointerTypeStorage{std::get<mlir::Type>(key), std::get<unsigned int>(key)};
	}

	[[nodiscard]] mlir::Type getElementType() const
	{
		return elementType;
	}

	[[nodiscard]] unsigned int getRank() const
	{
		return rank;
	}

	private:
	UnrankedPointerTypeStorage(mlir::Type elementType, unsigned int rank)
			: elementType(elementType),
				rank(rank)
	{
	}

	mlir::Type elementType;
	unsigned int rank;
};
 */

	class StructTypeStorage : public mlir::TypeStorage {
		public:
		using KeyTy = llvm::ArrayRef<mlir::Type>;

		StructTypeStorage() = delete;

		bool operator==(const KeyTy& key) const {
			return key == KeyTy{getElementTypes()};
		}

		static unsigned int hashKey(const KeyTy& key) {
			return llvm::hash_combine(key);
		}

		static StructTypeStorage* construct(mlir::TypeStorageAllocator& allocator, const KeyTy& key) {
			llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);
			return new (allocator.allocate<StructTypeStorage>()) StructTypeStorage(elementTypes);
		}

		[[nodiscard]] llvm::ArrayRef<mlir::Type> getElementTypes() const
		{
			return elementTypes;
		}

		private:
		StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
				: elementTypes(elementTypes)
		{
		}

		llvm::ArrayRef<mlir::Type> elementTypes;
	};
}

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

long PointerType::rawSize() const
{
	long result = 1;

	for (long size : getShape())
	{
		if (size == -1)
			return -1;

		result *= size;
	}

	return result;
}

bool PointerType::hasConstantShape() const
{
	return llvm::all_of(getShape(), [](long size) {
		return size != -1;
	});
}

PointerType PointerType::slice(unsigned int subscriptsAmount)
{
	auto shape = getShape();
	assert(subscriptsAmount <= shape.size() && "Too many subscriptions");
	llvm::SmallVector<long, 3> resultShape;

	for (size_t i = subscriptsAmount, e = shape.size(); i < e; ++i)
		resultShape.push_back(shape[i]);

	return PointerType::get(getContext(), getAllocationScope(), getElementType(), resultShape);
}

PointerType PointerType::toAllocationScope(BufferAllocationScope scope)
{
	return PointerType::get(getContext(), scope, getElementType(), getShape());
}

PointerType PointerType::toUnknownAllocationScope()
{
	return toAllocationScope(unknown);
}

PointerType PointerType::toMinAllowedAllocationScope()
{
	if (getAllocationScope() == heap)
		return *this;

	if (canBeOnStack())
		return toAllocationScope(stack);

	return toAllocationScope(heap);
}

PointerType PointerType::toElementType(mlir::Type type)
{
	return PointerType::get(getContext(), getAllocationScope(), type, getShape());
}

bool PointerType::canBeOnStack() const
{
	return hasConstantShape();
}

/*
UnrankedPointerType UnrankedPointerType::get(mlir::MLIRContext* context, mlir::Type elementType, unsigned int rank)
{
	return Base::get(context, elementType, rank);
}

mlir::Type UnrankedPointerType::getElementType() const
{
	return getImpl()->getElementType();
}

unsigned int UnrankedPointerType::getRank() const
{
	return getImpl()->getRank();
}
*/

StructType StructType::get(mlir::MLIRContext* context, llvm::ArrayRef<mlir::Type> elementTypes)
{
	return Base::get(context, elementTypes);
}

llvm::ArrayRef<mlir::Type> StructType::getElementTypes()
{
	return getImpl()->getElementTypes();
}

void modelica::codegen::printModelicaType(mlir::Type type, mlir::DialectAsmPrinter& printer) {
	auto& os = printer.getStream();

	if (type.isa<BooleanType>())
	{
		os << "bool";
		return;
	}

	if (type.isa<IntegerType>())
	{
		os << "int";
		return;
	}

	if (type.dyn_cast<RealType>())
	{
		os << "real";
		return;
	}

	if (auto pointerType = type.dyn_cast<PointerType>())
	{
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

	/*
	if (auto pointerType = type.dyn_cast<UnrankedPointerType>())
	{
		os << "ptr<*x" << pointerType.getElementType() << ">";
		return;
	}
	*/

	if (auto structType = type.dyn_cast<StructType>())
	{
		os << "struct<";

		for (auto subtype : llvm::enumerate(structType.getElementTypes()))
		{
			if (subtype.index() != 0)
				os << ", ";

			os << subtype.value();
		}

		os << ">";
	}
}
