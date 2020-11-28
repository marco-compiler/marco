#include <iostream>
#include <modelica/frontend/Type.hpp>
#include <numeric>
#include <stack>

using namespace llvm;
using namespace modelica;
using namespace std;

raw_ostream& modelica::operator<<(raw_ostream& stream, const BuiltInType& obj)
{
	return stream << toString(obj);
}

string modelica::toString(BuiltInType type)
{
	switch (type)
	{
		case BuiltInType::None:
			return "none";
		case BuiltInType::Integer:
			return "integer";
		case BuiltInType::Float:
			return "float";
		case BuiltInType::String:
			return "string";
		case BuiltInType::Boolean:
			return "boolean";
		case BuiltInType::Unknown:
			return "unknown";
	}

	assert(false && "Unexpected type");
}

UserDefinedType::UserDefinedType(ArrayRef<Type> types)
{
	for (const auto& type : types)
		this->types.emplace_back(std::make_unique<Type>(type));
}

UserDefinedType::UserDefinedType(const UserDefinedType& other)
{
	for (const auto& type : other.types)
		types.emplace_back(std::make_unique<Type>(*type));
}

UserDefinedType& UserDefinedType::operator=(const UserDefinedType& other)
{
	if (this == &other)
		return *this;

	types.clear();

	for (const auto& type : other.types)
		types.emplace_back(std::make_unique<Type>(*type));

	return *this;
}

bool UserDefinedType::operator==(const UserDefinedType& other) const
{
	return std::equal(
			begin(),
			end(),
			other.begin(),
			other.end(),
			[](const UniqueType& lhs, const UniqueType& rhs) {
				return *lhs == *rhs;
			});
}

bool UserDefinedType::operator!=(const UserDefinedType& other) const
{
	return !(*this == other);
}

Type& UserDefinedType::operator[](size_t index)
{
	assert(index < types.size());
	return *types[index];
}

Type UserDefinedType::operator[](size_t index) const
{
	assert(index < types.size());
	return *types[index];
}

void UserDefinedType::dump() const { dump(outs(), 0); }

void UserDefinedType::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << toString(*this);
}

size_t UserDefinedType::size() const { return types.size(); }

SmallVectorImpl<UniqueType>::const_iterator UserDefinedType::begin() const
{
	return types.begin();
}

SmallVectorImpl<UniqueType>::const_iterator UserDefinedType::end() const
{
	return types.end();
}

raw_ostream& modelica::operator<<(
		raw_ostream& stream, const UserDefinedType& obj)
{
	return stream << toString(obj);
}

string modelica::toString(UserDefinedType obj)
{
	return "{" +
				 accumulate(
						 obj.begin(),
						 obj.end(),
						 string(),
						 [](const string& a, const auto& b) -> string {
							 return a + (a.length() > 0 ? ", " : "") + toString(*b);
						 }) +
				 "}";
}

Type::Type(BuiltInType type, ArrayRef<size_t> dim)
		: content(move(type)), dimensions(dim.begin(), dim.end())
{
	assert(holds_alternative<BuiltInType>(content));
	assert(!dimensions.empty());
}

Type::Type(UserDefinedType type, ArrayRef<size_t> dim)
		: content(move(type)), dimensions(dim.begin(), dim.end())
{
	assert(holds_alternative<UserDefinedType>(content));
	assert(!dimensions.empty());
}

Type::Type(llvm::ArrayRef<Type> members, ArrayRef<size_t> dim)
		: Type(UserDefinedType(move(members)), move(dim))
{
}

bool Type::operator==(const Type& other) const
{
	return dimensions == other.dimensions && content == other.content;
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
	os << toString(*this);
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

Type Type::subscript(size_t times) const
{
	assert(!isScalar());

	if (dimensions.size() == times)
		return visit([](const auto& t) { return Type(t); });

	assert(times > dimensions.size());

	return visit([&](const auto& t) {
		return Type(
				t,
				SmallVector<size_t, 3>(dimensions.begin() + times, dimensions.end()));
	});
}

raw_ostream& modelica::operator<<(raw_ostream& stream, const Type& obj)
{
	return stream << toString(obj);
}

string modelica::toString(Type obj)
{
	auto visitor = [](const auto& t) { return toString(t); };

	auto dimensionsToStringLambda = [](const string& a, const auto& b) -> string {
		return a + (a.length() > 0 ? "," : "") + to_string(b);
	};

	auto& dimensions = obj.getDimensions();
	string size = obj.isScalar() ? ""
															 : "[" +
																		 accumulate(
																				 dimensions.begin(),
																				 dimensions.end(),
																				 string(),
																				 dimensionsToStringLambda) +
																		 "] ";
	return size + obj.visit(visitor);
}

Type Type::Int() { return Type(BuiltInType::Integer); }

Type Type::Float() { return Type(BuiltInType::Float); }

Type Type::unknown() { return Type(BuiltInType::Unknown); }
