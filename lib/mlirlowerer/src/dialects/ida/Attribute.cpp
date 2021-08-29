#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <marco/mlirlowerer/dialects/ida/Attribute.h>
#include <marco/mlirlowerer/dialects/ida/Type.h>
#include <numeric>

using namespace marco::codegen::ida;

bool BooleanAttributeStorage::operator==(const KeyTy& key) const
{
	return key == KeyTy(type, value);
}

unsigned int BooleanAttributeStorage::hashKey(const KeyTy& key)
{
	return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
}

BooleanAttributeStorage::KeyTy BooleanAttributeStorage::getKey(mlir::Type type, bool value)
{
	return KeyTy(type, value);
}

BooleanAttributeStorage* BooleanAttributeStorage::construct(mlir::AttributeStorageAllocator& allocator, KeyTy key)
{
	return new (allocator.allocate<BooleanAttributeStorage>()) BooleanAttributeStorage(std::get<0>(key), std::get<1>(key));
}

bool BooleanAttributeStorage::getValue() const
{
	return value;
}

BooleanAttributeStorage::BooleanAttributeStorage(mlir::Type type, bool value)
		: AttributeStorage(type), type(type), value(value)
{
}

bool IntegerAttributeStorage::operator==(const KeyTy& key) const
{
	return key == KeyTy(type, value);
}

unsigned int IntegerAttributeStorage::hashKey(const KeyTy& key)
{
	return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
}

IntegerAttributeStorage::KeyTy IntegerAttributeStorage::getKey(mlir::Type type, int64_t value)
{
	assert(type.isa<IntegerType>());
	return KeyTy(type, llvm::APInt(sizeof(int64_t) * 8, value, true));
}

IntegerAttributeStorage* IntegerAttributeStorage::construct(mlir::AttributeStorageAllocator& allocator, KeyTy key)
{
	return new (allocator.allocate<IntegerAttributeStorage>()) IntegerAttributeStorage(std::get<0>(key), std::get<1>(key));
}

int64_t IntegerAttributeStorage::getValue() const
{
	return value.getSExtValue();
}

IntegerAttributeStorage::IntegerAttributeStorage(mlir::Type type, llvm::APInt value)
		: AttributeStorage(type), type(type), value(std::move(value))
{
}

bool RealAttributeStorage::operator==(const KeyTy& key) const
{
	return key == KeyTy(type, value);
}

unsigned int RealAttributeStorage::hashKey(const KeyTy& key)
{
	return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
}

RealAttributeStorage::KeyTy RealAttributeStorage::getKey(mlir::Type type, double value) {
	return KeyTy(type, value);
}

RealAttributeStorage* RealAttributeStorage::construct(mlir::AttributeStorageAllocator& allocator, KeyTy key)
{
	return new (allocator.allocate<RealAttributeStorage>()) RealAttributeStorage(std::get<0>(key), std::get<1>(key));
}

double RealAttributeStorage::getValue() const
{
	return value.convertToDouble();
}

RealAttributeStorage::RealAttributeStorage(mlir::Type type, llvm::APFloat value)
		: AttributeStorage(type), type(type), value(std::move(value))
{
}

constexpr llvm::StringRef BooleanAttribute::getAttrName()
{
	return "bool";
}

BooleanAttribute BooleanAttribute::get(mlir::MLIRContext* context, bool value)
{
	return BooleanAttribute::get(BooleanType::get(context), value);
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

IntegerAttribute IntegerAttribute::get(mlir::MLIRContext* context, int64_t value)
{
	return IntegerAttribute::get(IntegerType::get(context), value);
}

IntegerAttribute IntegerAttribute::get(mlir::Type type, int64_t value)
{
	assert(type.isa<IntegerType>());
	return Base::get(type.getContext(), type, value);
}

int64_t IntegerAttribute::getValue() const
{
	return getImpl()->getValue();
}

constexpr llvm::StringRef RealAttribute::getAttrName()
{
	return "real";
}

RealAttribute RealAttribute::get(mlir::MLIRContext* context, double value)
{
	return RealAttribute::get(RealType::get(context), value);
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

namespace marco::codegen::ida
{
	mlir::Attribute parseIdaAttribute(mlir::DialectAsmParser& parser, mlir::Type type)
	{
		if (mlir::succeeded(parser.parseOptionalKeyword("bool")))
		{
			if (parser.parseLess())
				return mlir::Attribute();

			bool value = false;

			if (mlir::succeeded(parser.parseOptionalKeyword("true")))
			{
				value = true;
			}
			else
			{
				if (parser.parseKeyword("false"))
					return mlir::Attribute();

				value = false;
			}

			if (parser.parseGreater())
				return mlir::Attribute();

			return BooleanAttribute::get(
					BooleanType::get(parser.getBuilder().getContext()), value);
		}

		if (mlir::succeeded(parser.parseOptionalKeyword("int")))
		{
			if (parser.parseLess())
				return mlir::Attribute();

			int64_t value = 0;

			if (parser.parseInteger(value))
				return mlir::Attribute();

			if (parser.parseGreater())
				return mlir::Attribute();

			return IntegerAttribute::get(
					IntegerType::get(parser.getBuilder().getContext()), value);
		}

		if (mlir::succeeded(parser.parseOptionalKeyword("real")))
		{
			if (parser.parseLess())
				return mlir::Attribute();

			double value = 0;

			if (parser.parseFloat(value))
				return mlir::Attribute();

			if (parser.parseGreater())
				return mlir::Attribute();

			return RealAttribute::get(
					RealType::get(parser.getBuilder().getContext()), value);
		}

		parser.emitError(parser.getCurrentLocation()) << "unknown attribute";
		return mlir::Attribute();
	}

	void printIdaAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter& printer)
	{
		if (BooleanAttribute attribute = attr.dyn_cast<BooleanAttribute>())
		{
			printer << "bool<" << (attribute.getValue() ? "true" : "false") << ">";
			return;
		}

		if (IntegerAttribute attribute = attr.dyn_cast<IntegerAttribute>())
		{
			printer << "int<" << std::to_string(attribute.getValue()) << ">";
			return;
		}

		if (RealAttribute attribute = attr.dyn_cast<RealAttribute>())
		{
			printer << "real<" << std::to_string(attribute.getValue()) << ">";
			return;
		}
	}
}
