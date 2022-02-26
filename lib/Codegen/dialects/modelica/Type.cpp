#include "marco/Codegen/dialects/modelica/Type.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace marco::codegen::modelica;

bool MemberTypeStorage::operator==(const KeyTy& key) const
{
	return key == KeyTy{getAllocationScope(), getElementType(), getShape()};
}

unsigned int MemberTypeStorage::hashKey(const KeyTy& key)
{
	auto hashValue = [](const MemberTypeStorage::Shape& s) -> llvm::hash_code
	{
		if (!s.empty()) {
			return llvm::hash_combine_range(s.begin(), s.end());
		}

		return llvm::hash_combine(0);
	};

	auto shapeHash{hashValue(std::get<MemberTypeStorage::Shape>(key))};
	return llvm::hash_combine(std::get<MemberAllocationScope>(key), std::get<mlir::Type>(key), shapeHash);
}

MemberTypeStorage* MemberTypeStorage::construct(mlir::TypeStorageAllocator& allocator, const KeyTy &key)
{
	auto *storage = allocator.allocate<MemberTypeStorage>();
	return new (storage) MemberTypeStorage{std::get<MemberAllocationScope>(key), std::get<mlir::Type>(key), std::get<ArrayType::Shape>(key)};
}

MemberAllocationScope MemberTypeStorage::getAllocationScope() const
{
	return allocationScope;
}

MemberTypeStorage::Shape MemberTypeStorage::getShape() const
{
	return shape;
}

mlir::Type MemberTypeStorage::getElementType() const
{
	return elementType;
}

MemberTypeStorage::MemberTypeStorage(MemberAllocationScope allocationScope, mlir::Type elementType, const Shape& shape)
		: allocationScope(allocationScope),
			elementType(elementType),
			shape(std::move(shape))
{
}

bool ArrayTypeStorage::operator==(const KeyTy& key) const
{
	return key == KeyTy{getAllocationScope(), getElementType(), getShape()};
}

unsigned int ArrayTypeStorage::hashKey(const KeyTy& key)
{
	auto hashValue = [](const ArrayTypeStorage::Shape& s) -> llvm::hash_code
	{
		if (s.size()) {
			return llvm::hash_combine_range(s.begin(), s.end());
		}
		return llvm::hash_combine(0);
	};

	auto shapeHash{hashValue(std::get<ArrayType::Shape>(key))};
	return llvm::hash_combine(std::get<BufferAllocationScope>(key), std::get<mlir::Type>(key), shapeHash);
}

ArrayTypeStorage* ArrayTypeStorage::construct(mlir::TypeStorageAllocator& allocator, const KeyTy &key)
{
	auto *storage = allocator.allocate<ArrayTypeStorage>();
	return new (storage) ArrayTypeStorage{std::get<BufferAllocationScope>(key), std::get<mlir::Type>(key), std::get<ArrayType::Shape>(key)};
}

BufferAllocationScope ArrayTypeStorage::getAllocationScope() const
{
	return allocationScope;
}

ArrayType::Shape ArrayTypeStorage::getShape() const
{
	return shape;
}

mlir::Type ArrayTypeStorage::getElementType() const
{
	return elementType;
}

ArrayTypeStorage::ArrayTypeStorage(BufferAllocationScope allocationScope, mlir::Type elementType, const Shape& shape)
		: allocationScope(allocationScope),
			elementType(elementType),
			shape(std::move(shape))
{
}

bool UnsizedArrayTypeStorage::operator==(const KeyTy& key) const
{
	return key == KeyTy{getElementType()};
}

unsigned int UnsizedArrayTypeStorage::hashKey(const KeyTy& key)
{
	return llvm::hash_combine(key);
}

UnsizedArrayTypeStorage* UnsizedArrayTypeStorage::construct(mlir::TypeStorageAllocator& allocator, const KeyTy &key)
{
	auto *storage = allocator.allocate<UnsizedArrayTypeStorage>();
	return new (storage) UnsizedArrayTypeStorage{key};
}

mlir::Type UnsizedArrayTypeStorage::getElementType() const
{
	return elementType;
}

UnsizedArrayTypeStorage::UnsizedArrayTypeStorage(mlir::Type elementType)
		: elementType(elementType)
{
}

bool StructTypeStorage::operator==(const KeyTy& key) const
{
	return key == KeyTy{getElementTypes()};
}

unsigned int StructTypeStorage::hashKey(const KeyTy& key)
{
	return llvm::hash_combine(key);
}

StructTypeStorage* StructTypeStorage::construct(mlir::TypeStorageAllocator& allocator, const KeyTy& key)
{
	llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);
	return new (allocator.allocate<StructTypeStorage>()) StructTypeStorage(elementTypes);
}

llvm::ArrayRef<mlir::Type> StructTypeStorage::getElementTypes() const
{
	return elementTypes;
}

StructTypeStorage::StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
		: elementTypes(elementTypes)
{
}

BooleanType BooleanType::get(mlir::MLIRContext* context)
{
	return Base::get(context);
}

IntegerType IntegerType::get(mlir::MLIRContext* context)
{
	return Base::get(context);
}

RealType RealType::get(mlir::MLIRContext* context)
{
	return Base::get(context);
}

namespace marco::codegen::modelica
{
	static BufferAllocationScope memberToBufferAllocationScope(MemberAllocationScope scope)
	{
		switch (scope)
		{
			case MemberAllocationScope::stack:
				return BufferAllocationScope::stack;

			case MemberAllocationScope::heap:
				return BufferAllocationScope::heap;
		}

		assert(false && "Unexpected allocation scope");
		return BufferAllocationScope::heap;
	}

	static MemberAllocationScope bufferToMemberAllocationScope(BufferAllocationScope scope)
	{
		switch (scope)
		{
			case BufferAllocationScope::stack:
				return MemberAllocationScope::stack;

			case BufferAllocationScope::heap:
			case BufferAllocationScope::unknown:
				return MemberAllocationScope::heap;
		}

		assert(false && "Unexpected allocation scope");
		return MemberAllocationScope::heap;
	}
}

MemberType MemberType::get(mlir::MLIRContext* context, MemberAllocationScope allocationScope, mlir::Type elementType, llvm::ArrayRef<long> shape)
{
	return Base::get(context, allocationScope, elementType, Shape(shape.begin(), shape.end()));
}

MemberType MemberType::get(ArrayType arrayType)
{
	auto shape = arrayType.getShape();

	return Base::get(
			arrayType.getContext(),
			bufferToMemberAllocationScope(arrayType.getAllocationScope()),
			arrayType.getElementType(),
			Shape(shape.begin(), shape.end()));
}

MemberAllocationScope MemberType::getAllocationScope() const
{
	return getImpl()->getAllocationScope();
}

mlir::Type MemberType::getElementType() const
{
	return getImpl()->getElementType();
}

MemberType::Shape MemberType::getShape() const
{
	return getImpl()->getShape();
}

unsigned int MemberType::getRank() const
{
	return getShape().size();
}

ArrayType MemberType::toArrayType() const
{
	return ArrayType::get(
			getContext(),
			memberToBufferAllocationScope(getAllocationScope()),
			getElementType(),
			getShape());
}

mlir::Type MemberType::unwrap() const
{
	if (getRank() == 0)
		return getElementType();

	return toArrayType();
}

ArrayType ArrayType::get(mlir::MLIRContext* context, BufferAllocationScope allocationScope, mlir::Type elementType, llvm::ArrayRef<long> shape)
{
	return Base::get(context, allocationScope, elementType, Shape(shape.begin(), shape.end()));
}

BufferAllocationScope ArrayType::getAllocationScope() const
{
	return getImpl()->getAllocationScope();
}

mlir::Type ArrayType::getElementType() const
{
	return getImpl()->getElementType();
}

ArrayType::Shape ArrayType::getShape() const
{
	return getImpl()->getShape();
}

unsigned int ArrayType::getRank() const
{
	return getShape().size();
}

unsigned int ArrayType::getConstantDimensions() const
{
	auto dimensions = getShape();
	unsigned int count = 0;

	for (auto dimension : dimensions) {
		if (dimension > 0)
			count++;
	}

	return count;
}

unsigned int ArrayType::getDynamicDimensions() const
{
	auto dimensions = getShape();
	unsigned int count = 0;

	for (auto dimension : dimensions) {
		if (dimension  == -1)
			count++;
	}

	return count;
}

long ArrayType::rawSize() const
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

bool ArrayType::hasConstantShape() const
{
	return llvm::all_of(getShape(), [](long size) {
		return size != -1;
	});
}

bool ArrayType::isScalar() const
{
	return getRank() == 0;
}

ArrayType ArrayType::slice(unsigned int subscriptsAmount)
{
	auto shape = getShape();
	assert(subscriptsAmount <= shape.size() && "Too many subscriptions");
	llvm::SmallVector<long, 3> resultShape;

	for (size_t i = subscriptsAmount, e = shape.size(); i < e; ++i)
		resultShape.push_back(shape[i]);

	return ArrayType::get(getContext(), getAllocationScope(), getElementType(), resultShape);
}

ArrayType ArrayType::toAllocationScope(BufferAllocationScope scope)
{
	return ArrayType::get(getContext(), scope, getElementType(), getShape());
}

ArrayType ArrayType::toUnknownAllocationScope()
{
	return toAllocationScope(BufferAllocationScope::unknown);
}

ArrayType ArrayType::toMinAllowedAllocationScope()
{
	if (getAllocationScope() == BufferAllocationScope::heap)
		return *this;

	if (canBeOnStack())
		return toAllocationScope(BufferAllocationScope::stack);

	return toAllocationScope(BufferAllocationScope::heap);
}

ArrayType ArrayType::toElementType(mlir::Type elementType)
{
	return ArrayType::get(getContext(), getAllocationScope(), elementType, getShape());
}

UnsizedArrayType ArrayType::toUnsized()
{
	return UnsizedArrayType::get(getContext(), getElementType());
}

bool ArrayType::canBeOnStack() const
{
	return hasConstantShape();
}

UnsizedArrayType UnsizedArrayType::get(mlir::MLIRContext* context, mlir::Type elementType)
{
	return Base::get(context, elementType);
}

mlir::Type UnsizedArrayType::getElementType() const
{
	return getImpl()->getElementType();
}

StructType StructType::get(mlir::MLIRContext* context, llvm::ArrayRef<mlir::Type> elementTypes)
{
	return Base::get(context, elementTypes);
}

llvm::ArrayRef<mlir::Type> StructType::getElementTypes()
{
	return getImpl()->getElementTypes();
}

namespace marco::codegen::modelica
{
	mlir::Type parseModelicaType(mlir::DialectAsmParser& parser)
	{
		auto builder = parser.getBuilder();

		if (mlir::succeeded(parser.parseOptionalKeyword("bool")))
			return BooleanType::get(builder.getContext());

		if (mlir::succeeded(parser.parseOptionalKeyword("int")))
			return IntegerType::get(builder.getContext());

		if (mlir::succeeded(parser.parseOptionalKeyword("real")))
			return RealType::get(builder.getContext());

		if (mlir::succeeded(parser.parseOptionalKeyword("member")))
		{
			if (parser.parseLess())
				return mlir::Type();

			MemberAllocationScope scope;

			if (mlir::succeeded(parser.parseOptionalKeyword("stack")))
			{
				scope = MemberAllocationScope::stack;
			}
			else if (mlir::succeeded(parser.parseOptionalKeyword("heap")))
			{
				scope = MemberAllocationScope::heap;
			}
			else
			{
				parser.emitError(parser.getCurrentLocation()) << "unexpected member allocation scope";
				return mlir::Type();
			}

			if (parser.parseComma())
				return mlir::Type();

			llvm::SmallVector<int64_t, 3> dimensions;

			if (parser.parseDimensionList(dimensions))
				return mlir::Type();

			mlir::Type baseType;

			if (parser.parseType(baseType) ||
					parser.parseGreater())
				return mlir::Type();

			llvm::SmallVector<long, 3> castedDims(dimensions.begin(), dimensions.end());
			return MemberType::get(builder.getContext(), scope, baseType, castedDims);
		}

		if (mlir::succeeded(parser.parseOptionalKeyword("array")))
		{
			if (parser.parseLess())
				return mlir::Type();

			if (mlir::succeeded(parser.parseOptionalStar()))
			{
				mlir::Type baseType;

				if (parser.parseType(baseType) ||
						parser.parseGreater())
					return mlir::Type();

				return UnsizedArrayType::get(builder.getContext(), baseType);
			}

			BufferAllocationScope scope = BufferAllocationScope::unknown;

			if (mlir::succeeded(parser.parseOptionalKeyword("stack")))
			{
				scope = BufferAllocationScope::stack;

				if (parser.parseComma())
					return mlir::Type();
			}
			else if (mlir::succeeded(parser.parseOptionalKeyword("heap")))
			{
				scope = BufferAllocationScope::heap;

				if (parser.parseComma())
					return mlir::Type();
			}

			llvm::SmallVector<int64_t, 3> dimensions;

			if (parser.parseDimensionList(dimensions))
				return mlir::Type();

			mlir::Type baseType;

			if (parser.parseType(baseType) ||
					parser.parseGreater())
				return mlir::Type();

			llvm::SmallVector<long, 3> castedDims(dimensions.begin(), dimensions.end());
			return ArrayType::get(builder.getContext(), scope, baseType, castedDims);
		}

		if (mlir::succeeded(parser.parseOptionalKeyword("struct")))
		{
			if (mlir::failed(parser.parseLess()))
				return mlir::Type();

			llvm::SmallVector<mlir::Type, 3> types;

			do {
				mlir::Type type;

				if (parser.parseType(type))
					return mlir::Type();

				types.push_back(type);
			} while (succeeded(parser.parseOptionalComma()));

			if (mlir::failed(parser.parseGreater()))
				return mlir::Type();

			return StructType::get(builder.getContext(), types);
		}

		parser.emitError(parser.getCurrentLocation()) << "unknown type";
		return mlir::Type();
	}

	void printModelicaType(mlir::Type type, mlir::DialectAsmPrinter& printer) {
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

		if (auto memberType = type.dyn_cast<MemberType>())
		{
			os << "member<";

			if (memberType.getAllocationScope() == MemberAllocationScope::stack)
				os << "stack, ";
			else if (memberType.getAllocationScope() == MemberAllocationScope::heap)
				os << "heap, ";

			auto dimensions = memberType.getShape();

			for (const auto& dimension : dimensions)
				os << (dimension == -1 ? "?" : std::to_string(dimension)) << "x";

			printer.printType(memberType.getElementType());
			os << ">";
			return;
		}

		if (auto arrayType = type.dyn_cast<ArrayType>())
		{
			os << "array<";

			if (arrayType.getAllocationScope() == BufferAllocationScope::stack)
				os << "stack, ";
			else if (arrayType.getAllocationScope() == BufferAllocationScope::heap)
				os << "heap, ";

			auto dimensions = arrayType.getShape();

			for (const auto& dimension : dimensions)
				os << (dimension == -1 ? "?" : std::to_string(dimension)) << "x";

			printer.printType(arrayType.getElementType());
			os << ">";
			return;
		}

		if (auto arrayType = type.dyn_cast<UnsizedArrayType>())
		{
			os << "array<*x" << arrayType.getElementType() << ">";
			return;
		}

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
}

