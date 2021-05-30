#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <modelica/mlirlowerer/Type.h>

using namespace modelica::codegen;

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
	return new (storage) MemberTypeStorage{std::get<MemberAllocationScope>(key), std::get<mlir::Type>(key), std::get<PointerType::Shape>(key)};
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

bool PointerTypeStorage::operator==(const KeyTy& key) const
{
	return key == KeyTy{getAllocationScope(), getElementType(), getShape()};
}

unsigned int PointerTypeStorage::hashKey(const KeyTy& key)
{
	auto hashValue = [](const PointerTypeStorage::Shape& s) -> llvm::hash_code
	{
		if (s.size()) {
			return llvm::hash_combine_range(s.begin(), s.end());
		}
		return llvm::hash_combine(0);
	};

	auto shapeHash{hashValue(std::get<PointerType::Shape>(key))};
	return llvm::hash_combine(std::get<BufferAllocationScope>(key), std::get<mlir::Type>(key), shapeHash);
}

PointerTypeStorage* PointerTypeStorage::construct(mlir::TypeStorageAllocator& allocator, const KeyTy &key)
{
	auto *storage = allocator.allocate<PointerTypeStorage>();
	return new (storage) PointerTypeStorage{std::get<BufferAllocationScope>(key), std::get<mlir::Type>(key), std::get<PointerType::Shape>(key)};
}

BufferAllocationScope PointerTypeStorage::getAllocationScope() const
{
	return allocationScope;
}

PointerType::Shape PointerTypeStorage::getShape() const
{
	return shape;
}

mlir::Type PointerTypeStorage::getElementType() const
{
	return elementType;
}

PointerTypeStorage::PointerTypeStorage(BufferAllocationScope allocationScope, mlir::Type elementType, const Shape& shape)
		: allocationScope(allocationScope),
			elementType(elementType),
			shape(std::move(shape))
{
}

bool UnsizedPointerTypeStorage::operator==(const KeyTy& key) const
{
	return key == KeyTy{getElementType()};
}

unsigned int UnsizedPointerTypeStorage::hashKey(const KeyTy& key)
{
	return llvm::hash_combine(key);
}

UnsizedPointerTypeStorage* UnsizedPointerTypeStorage::construct(mlir::TypeStorageAllocator& allocator, const KeyTy &key)
{
	auto *storage = allocator.allocate<UnsizedPointerTypeStorage>();
	return new (storage) UnsizedPointerTypeStorage{key};
}

mlir::Type UnsizedPointerTypeStorage::getElementType() const
{
	return elementType;
}

UnsizedPointerTypeStorage::UnsizedPointerTypeStorage(mlir::Type elementType)
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

namespace modelica::codegen
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
	}

	static MemberAllocationScope bufferToMemberAllocationScope(BufferAllocationScope scope)
	{
		switch (scope)
		{
			case BufferAllocationScope::stack:
				return MemberAllocationScope::stack;

			case BufferAllocationScope::heap:
				return MemberAllocationScope::heap;

			case BufferAllocationScope::unknown:
				assert(false && "Unexpected unknown allocation scope");
				return MemberAllocationScope::heap;
		}
	}
}

MemberType MemberType::get(mlir::MLIRContext* context, MemberAllocationScope allocationScope, mlir::Type elementType, llvm::ArrayRef<long> shape)
{
	return Base::get(context, allocationScope, elementType, Shape(shape.begin(), shape.end()));
}

MemberType MemberType::get(PointerType pointerType)
{
	auto shape = pointerType.getShape();

	return Base::get(
			pointerType.getContext(),
			bufferToMemberAllocationScope(pointerType.getAllocationScope()),
			pointerType.getElementType(),
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

PointerType MemberType::toPointerType() const
{
	return PointerType::get(
			getContext(),
			memberToBufferAllocationScope(getAllocationScope()),
			getElementType(),
			getShape());
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

bool PointerType::isScalar() const
{
	return getRank() == 0;
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
	return toAllocationScope(BufferAllocationScope::unknown);
}

PointerType PointerType::toMinAllowedAllocationScope()
{
	if (getAllocationScope() == BufferAllocationScope::heap)
		return *this;

	if (canBeOnStack())
		return toAllocationScope(BufferAllocationScope::stack);

	return toAllocationScope(BufferAllocationScope::heap);
}

UnsizedPointerType PointerType::toUnsized()
{
	return UnsizedPointerType::get(getContext(), getElementType());
}

bool PointerType::canBeOnStack() const
{
	return hasConstantShape();
}

UnsizedPointerType UnsizedPointerType::get(mlir::MLIRContext* context, mlir::Type elementType)
{
	return Base::get(context, elementType);
}

mlir::Type UnsizedPointerType::getElementType() const
{
	return getImpl()->getElementType();
}

OpaquePointerType OpaquePointerType::get(mlir::MLIRContext* context)
{
	return Base::get(context);
}

StructType StructType::get(mlir::MLIRContext* context, llvm::ArrayRef<mlir::Type> elementTypes)
{
	return Base::get(context, elementTypes);
}

llvm::ArrayRef<mlir::Type> StructType::getElementTypes()
{
	return getImpl()->getElementTypes();
}

namespace modelica::codegen
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

			llvm::SmallVector<long, 3> dimensions;

			if (parser.parseDimensionList(dimensions))
				return mlir::Type();

			mlir::Type baseType;

			if (parser.parseType(baseType) ||
					parser.parseGreater())
				return mlir::Type();

			return MemberType::get(builder.getContext(), scope, baseType, dimensions);
		}

		if (mlir::succeeded(parser.parseOptionalKeyword("ptr")))
		{
			if (parser.parseLess())
				return mlir::Type();

			if (mlir::succeeded(parser.parseOptionalStar()))
			{
				mlir::Type baseType;

				if (parser.parseType(baseType) ||
						parser.parseGreater())
					return mlir::Type();

				return UnsizedPointerType::get(builder.getContext(), baseType);
			}

			BufferAllocationScope scope = BufferAllocationScope::unknown;

			if (mlir::succeeded(parser.parseOptionalKeyword("stack")))
			{
				scope = BufferAllocationScope::stack;
			}
			else if (mlir::succeeded(parser.parseOptionalKeyword("heap")))
			{
				scope = BufferAllocationScope::heap;
			}
			else if (mlir::failed(parser.parseKeyword("unknown")))
			{
				parser.emitError(parser.getCurrentLocation()) << "unexpected buffer allocation scope";
				return mlir::Type();
			}

			if (parser.parseComma())
				return mlir::Type();

			llvm::SmallVector<long, 3> dimensions;

			if (parser.parseDimensionList(dimensions))
				return mlir::Type();

			mlir::Type baseType;

			if (parser.parseType(baseType) ||
					parser.parseGreater())
				return mlir::Type();

			return PointerType::get(builder.getContext(), scope, baseType, dimensions);
		}

		if (mlir::succeeded(parser.parseOptionalKeyword("opaque_ptr")))
			return OpaquePointerType::get(builder.getContext());

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

		if (auto pointerType = type.dyn_cast<PointerType>())
		{
			os << "ptr<";

			if (pointerType.getAllocationScope() == BufferAllocationScope::stack)
				os << "stack, ";
			else if (pointerType.getAllocationScope() == BufferAllocationScope::heap)
				os << "heap, ";

			auto dimensions = pointerType.getShape();

			for (const auto& dimension : dimensions)
				os << (dimension == -1 ? "?" : std::to_string(dimension)) << "x";

			printer.printType(pointerType.getElementType());
			os << ">";
			return;
		}

		if (auto pointerType = type.dyn_cast<UnsizedPointerType>())
		{
			os << "ptr<*x" << pointerType.getElementType() << ">";
			return;
		}

		if (type.isa<OpaquePointerType>())
		{
			os << "opaque_ptr";
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

