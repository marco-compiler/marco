#include <mlir/IR/Builders.h>
#include <mlir/IR/DialectImplementation.h>
#include <marco/mlirlowerer/dialects/modelica/Attribute.h>
#include <marco/mlirlowerer/dialects/modelica/Type.h>
#include <numeric>

using namespace marco::codegen::modelica;

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

bool BooleanArrayAttributeStorage::operator==(const KeyTy& key) const
{
	return key == KeyTy(type, values);
}

unsigned int BooleanArrayAttributeStorage::hashKey(const KeyTy& key)
{
	return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
}

BooleanArrayAttributeStorage::KeyTy BooleanArrayAttributeStorage::getKey(mlir::Type type, llvm::ArrayRef<bool> values)
{
	assert(type.isa<ArrayType>() && type.cast<ArrayType>().getElementType().isa<BooleanType>());
	return KeyTy(type, values);
}

BooleanArrayAttributeStorage* BooleanArrayAttributeStorage::construct(mlir::AttributeStorageAllocator& allocator, KeyTy key)
{
	return new (allocator.allocate<BooleanArrayAttributeStorage>()) BooleanArrayAttributeStorage(std::get<0>(key), std::get<1>(key));
}

llvm::ArrayRef<bool> BooleanArrayAttributeStorage::getValue() const
{
	return values;
}

BooleanArrayAttributeStorage::BooleanArrayAttributeStorage(mlir::Type type, llvm::ArrayRef<bool> value)
		: AttributeStorage(type), type(type), values(value.begin(), value.end())
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

IntegerAttributeStorage::KeyTy IntegerAttributeStorage::getKey(mlir::Type type, long value)
{
	assert(type.isa<IntegerType>());
	return KeyTy(type, llvm::APInt(sizeof(long) * 8, value, true));
}

IntegerAttributeStorage* IntegerAttributeStorage::construct(mlir::AttributeStorageAllocator& allocator, KeyTy key)
{
	return new (allocator.allocate<IntegerAttributeStorage>()) IntegerAttributeStorage(std::get<0>(key), std::get<1>(key));
}

long IntegerAttributeStorage::getValue() const
{
	return value.getSExtValue();
}

IntegerAttributeStorage::IntegerAttributeStorage(mlir::Type type, llvm::APInt value)
		: AttributeStorage(type), type(type), value(std::move(value))
{
}

bool IntegerArrayAttributeStorage::operator==(const KeyTy& key) const
{
	return key == KeyTy(type, values);
}

unsigned int IntegerArrayAttributeStorage::hashKey(const KeyTy& key)
{
	return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
}

IntegerArrayAttributeStorage::KeyTy IntegerArrayAttributeStorage::getKey(mlir::Type type, llvm::ArrayRef<llvm::APInt> values)
{
	assert(type.isa<ArrayType>() && type.cast<ArrayType>().getElementType().isa<IntegerType>());
	return KeyTy(type, values);
}

IntegerArrayAttributeStorage* IntegerArrayAttributeStorage::construct(mlir::AttributeStorageAllocator& allocator, KeyTy key)
{
	return new (allocator.allocate<IntegerArrayAttributeStorage>()) IntegerArrayAttributeStorage(std::get<0>(key), std::get<1>(key));
}

llvm::ArrayRef<llvm::APInt> IntegerArrayAttributeStorage::getValue() const
{
	return values;
}

IntegerArrayAttributeStorage::IntegerArrayAttributeStorage(mlir::Type type, llvm::ArrayRef<llvm::APInt> value)
		: AttributeStorage(type), type(type), values(value.begin(), value.end())
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

bool RealArrayAttributeStorage::operator==(const KeyTy& key) const
{
	return key == KeyTy(type, values);
}

unsigned int RealArrayAttributeStorage::hashKey(const KeyTy& key)
{
	return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
}

RealArrayAttributeStorage::KeyTy RealArrayAttributeStorage::getKey(mlir::Type type, llvm::ArrayRef<llvm::APFloat> values)
{
	assert(type.isa<ArrayType>() && type.cast<ArrayType>().getElementType().isa<RealType>());
	return KeyTy(type, values);
}

RealArrayAttributeStorage* RealArrayAttributeStorage::construct(mlir::AttributeStorageAllocator& allocator, KeyTy key)
{
	return new (allocator.allocate<RealArrayAttributeStorage>()) RealArrayAttributeStorage(std::get<0>(key), std::get<1>(key));
}

llvm::ArrayRef<llvm::APFloat> RealArrayAttributeStorage::getValue() const
{
	return values;
}

RealArrayAttributeStorage::RealArrayAttributeStorage(mlir::Type type, llvm::ArrayRef<llvm::APFloat> value)
		: AttributeStorage(type), type(type), values(value.begin(), value.end())
{
}

bool InverseFunctionsAttributeStorage::operator==(const InverseFunctionsAttributeStorage::KeyTy& key) const
{
	return key == KeyTy(map);
}

unsigned int InverseFunctionsAttributeStorage::hashKey(const KeyTy& key)
{
	return llvm::hash_combine_range(key.begin(), key.end());
}

InverseFunctionsAttributeStorage* InverseFunctionsAttributeStorage::construct(mlir::AttributeStorageAllocator& allocator, Map map)
{
	std::map<unsigned int, std::pair<llvm::StringRef, llvm::ArrayRef<unsigned int>>> copiedMap;

	for (const auto& entry : map)
	{
		copiedMap[entry.first] = std::make_pair(
				allocator.copyInto(entry.second.first),
				allocator.copyInto(entry.second.second));
	}

	return new (allocator.allocate<InverseFunctionsAttributeStorage>()) InverseFunctionsAttributeStorage(copiedMap);
}

[[nodiscard]] InverseFunctionsAttributeStorage::iterator InverseFunctionsAttributeStorage::begin()
{
	return iterator(map.begin());
}

[[nodiscard]] InverseFunctionsAttributeStorage::const_iterator InverseFunctionsAttributeStorage::cbegin() const
{
	return const_iterator(map.begin());
}

[[nodiscard]] InverseFunctionsAttributeStorage::iterator InverseFunctionsAttributeStorage::end()
{
	return iterator(map.end());
}

InverseFunctionsAttributeStorage::const_iterator InverseFunctionsAttributeStorage::cend() const
{
	return const_iterator(map.end());
}

bool InverseFunctionsAttributeStorage::isInvertible(unsigned int argumentIndex) const
{
	return map.find(argumentIndex) != map.end();
}

llvm::StringRef InverseFunctionsAttributeStorage::getFunction(unsigned int argumentIndex) const
{
	return map.find(argumentIndex)->second.first;
}

llvm::ArrayRef<unsigned int> InverseFunctionsAttributeStorage::getArgumentsIndexes(unsigned int argumentIndex) const
{
	return map.find(argumentIndex)->second.second;
}

InverseFunctionsAttributeStorage::InverseFunctionsAttributeStorage(Map map)
		: map(std::move(map))
{
}

bool DerivativeAttributeStorage::operator==(const KeyTy& key) const
{
	return key == KeyTy(name, order);
}

unsigned int DerivativeAttributeStorage::hashKey(const KeyTy& key)
{
	return llvm::hash_combine(std::get<0>(key), std::get<1>(key));
}

DerivativeAttributeStorage* DerivativeAttributeStorage::construct(mlir::AttributeStorageAllocator& allocator, KeyTy key)
{
	llvm::StringRef name = allocator.copyInto(std::get<0>(key));
	unsigned int order = std::get<1>(key);

	return new (allocator.allocate<DerivativeAttributeStorage>()) DerivativeAttributeStorage(name, order);
}

llvm::StringRef DerivativeAttributeStorage::getName() const
{
	return name;
}

unsigned int DerivativeAttributeStorage::getOrder() const
{
	return order;
}

DerivativeAttributeStorage::DerivativeAttributeStorage(llvm::StringRef name, unsigned int order)
		: name(name), order(order)
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

constexpr llvm::StringRef BooleanArrayAttribute::getAttrName()
{
	return "bool[]";
}

BooleanArrayAttribute BooleanArrayAttribute::get(mlir::Type type, llvm::ArrayRef<bool> values)
{
	assert(type.isa<ArrayType>() && type.cast<ArrayType>().getElementType().isa<BooleanType>());
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

IntegerAttribute IntegerAttribute::get(mlir::MLIRContext* context, long value)
{
	return IntegerAttribute::get(IntegerType::get(context), value);
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
	assert(type.isa<ArrayType>() && type.cast<ArrayType>().getElementType().isa<IntegerType>());
	llvm::SmallVector<llvm::APInt, 3> vals;

	for (const auto& value : values)
		vals.emplace_back(sizeof(value) * 8, value, true);

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

constexpr llvm::StringRef RealArrayAttribute::getAttrName()
{
	return "real[]";
}

RealArrayAttribute RealArrayAttribute::get(mlir::Type type, llvm::ArrayRef<double> values)
{
	assert(type.isa<ArrayType>() && type.cast<ArrayType>().getElementType().isa<RealType>());
	llvm::SmallVector<llvm::APFloat, 3> vals(values.begin(), values.end());
	return Base::get(type.getContext(), type, vals);
}

llvm::ArrayRef<llvm::APFloat> RealArrayAttribute::getValue() const
{
	return getImpl()->getValue();
}

constexpr llvm::StringRef InverseFunctionsAttribute::getAttrName()
{
	return "inverseFunction";
}

InverseFunctionsAttribute InverseFunctionsAttribute::get(
		mlir::MLIRContext* context,
		InverseFunctionsAttribute::Map inverseFunctionsList)
{
	return Base::get(context, inverseFunctionsList);
}

InverseFunctionsAttribute::iterator InverseFunctionsAttribute::begin()
{
	return getImpl()->begin();
}

InverseFunctionsAttribute::const_iterator InverseFunctionsAttribute::cbegin() const
{
	return getImpl()->cbegin();
}

InverseFunctionsAttribute::iterator InverseFunctionsAttribute::end()
{
	return getImpl()->end();
}

InverseFunctionsAttribute::const_iterator InverseFunctionsAttribute::cend() const
{
	return getImpl()->cend();
}

bool InverseFunctionsAttribute::isInvertible(unsigned int argumentIndex) const
{
	return getImpl()->isInvertible(argumentIndex);
}

llvm::StringRef InverseFunctionsAttribute::getFunction(unsigned int argumentIndex) const
{
	return getImpl()->getFunction(argumentIndex);
}

llvm::ArrayRef<unsigned int> InverseFunctionsAttribute::getArgumentsIndexes(unsigned int argumentIndex) const
{
	return getImpl()->getArgumentsIndexes(argumentIndex);
}

constexpr llvm::StringRef DerivativeAttribute::getAttrName()
{
	return "derivative";
}

DerivativeAttribute DerivativeAttribute::get(mlir::MLIRContext* context,
																						 llvm::StringRef name,
																						 unsigned int order)
{
	return Base::get(context, name, order);
}

llvm::StringRef DerivativeAttribute::getName() const
{
	return getImpl()->getName();
}

unsigned int DerivativeAttribute::getOrder() const
{
	return getImpl()->getOrder();
}

namespace marco::codegen::modelica
{
	mlir::Attribute parseModelicaAttribute(mlir::DialectAsmParser& parser, mlir::Type type)
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

			long value = 0;

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

		// TODO: parse inverse

		if (mlir::succeeded(parser.parseOptionalKeyword("derivative")))
		{
			mlir::StringAttr name;
			unsigned int order = 1;

			if (parser.parseLess() ||
					parser.parseAttribute(name) ||
					parser.parseComma() ||
					parser.parseInteger(order) ||
					parser.parseGreater())
				return mlir::Attribute();

			return DerivativeAttribute::get(
					parser.getBuilder().getContext(), name.getValue(), order);
		}

		parser.emitError(parser.getCurrentLocation()) << "unknown attribute";
		return mlir::Attribute();
	}

	void printModelicaAttribute(mlir::Attribute attr, mlir::DialectAsmPrinter& printer)
	{
		if (auto attribute = attr.dyn_cast<BooleanAttribute>())
		{
			printer << "bool<" << (attribute.getValue() ? "true" : "false") << ">";
			return;
		}

		if (auto attribute = attr.dyn_cast<IntegerAttribute>())
		{
			printer << "int<" << std::to_string(attribute.getValue()) << ">";
			return;
		}

		if (auto attribute = attr.dyn_cast<RealAttribute>())
		{
			printer << "real<" << std::to_string(attribute.getValue()) << ">";
			return;
		}

		if (auto attribute = attr.dyn_cast<InverseFunctionsAttribute>())
		{
			printer << "inverse: {";

			printer << std::accumulate(
					attribute.cbegin(), attribute.cend(), std::string(),
					[&](const std::string& result, const unsigned int& invertibleArg) {
						auto args = attribute.getArgumentsIndexes(invertibleArg);

						std::string argsString = std::accumulate(
								args.begin(), args.end(), std::string(),
								[](const std::string& result, const unsigned int& index) {
									std::string str = std::to_string(index);
									return result.empty() ? str : result + ", " + str;
								});

						std::string str = std::to_string(invertibleArg) + ": " +
															attribute.getFunction(invertibleArg).str() + "(" +
															argsString + ")";

						return result.empty() ? str : result + ", " + str;
					});

			printer << "}";
		}

		if (auto attribute = attr.dyn_cast<DerivativeAttribute>())
		{
			printer << "derivative" << "<\"" << attribute.getName() << "\", " << attribute.getOrder() << ">";
		}
	}
}
