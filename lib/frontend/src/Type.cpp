#include <iostream>
#include <modelica/frontend/Type.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

namespace modelica
{
	raw_ostream& operator<<(raw_ostream& stream, const BuiltinType& obj)
	{
		return stream << toString(obj);
	}

	string toString(BuiltinType type)
	{
		switch (type)
		{
			case BuiltinType::None:
				return "none";
			case BuiltinType::Integer:
				return "integer";
			case BuiltinType::Float:
				return "float";
			case BuiltinType::String:
				return "string";
			case BuiltinType::Boolean:
				return "boolean";
			case BuiltinType::Tuple:
				return "tuple";
			case BuiltinType::Unknown:
				return "unknown";
		}

		assert(false && "Unexpected type");
	}
}	 // namespace modelica

Type::Type(BuiltinType type, ArrayRef<size_t> dim)
		: dimensions(iterator_range<ArrayRef<size_t>::iterator>(move(dim))),
			type(type)
{
	assert(!dimensions.empty());
}

bool Type::operator==(const Type& other) const
{
	return type == other.type && dimensions == other.dimensions;
}

bool Type::operator!=(const Type& other) const { return !(*this == other); }

[[nodiscard]] size_t& Type::operator[](int index) { return dimensions[index]; }

[[nodiscard]] size_t Type::operator[](int index) const
{
	return dimensions[index];
}

void Type::dump() const { dump(outs(), 0); }

void Type::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << type;

	if (!isScalar())
		for (size_t dim : dimensions)
		{
			os << dim;
			os << " ";
		}
}

SmallVectorImpl<size_t>& Type::getDimensions() { return dimensions; }

const SmallVectorImpl<size_t>& Type::getDimensions() const
{
	return dimensions;
}

size_t Type::dimensionsCount() const { return dimensions.size(); }

size_t Type::size() const
{
	size_t toReturn = 1;

	for (size_t dim : dimensions)
		toReturn *= dim;

	return toReturn;
}

bool Type::isScalar() const
{
	return dimensions.size() == 1 && dimensions[0] == 1;
}

llvm::SmallVectorImpl<size_t>::iterator Type::begin()
{
	return dimensions.begin();
}

llvm::SmallVectorImpl<size_t>::const_iterator Type::begin() const
{
	return dimensions.begin();
}

llvm::SmallVectorImpl<size_t>::iterator Type::end() { return dimensions.end(); }

llvm::SmallVectorImpl<size_t>::const_iterator Type::end() const
{
	return dimensions.end();
}

BuiltinType Type::getBuiltIn() const { return type; }

Type Type::subscript(size_t times) const
{
	assert(!isScalar());

	if (dimensions.size() == times)
		return Type(type);

	assert(times > dimensions.size());

	return Type(
			type,
			llvm::SmallVector<size_t, 3>(
					dimensions.begin() + times, dimensions.end()));
}

Type Type::Int() { return Type(typeToBuiltin<int>()); }

Type Type::Float() { return Type(typeToBuiltin<float>()); }

Type Type::tuple() { return Type(BuiltinType::Tuple); }

Type Type::unknown() { return Type(BuiltinType::Unknown); }
