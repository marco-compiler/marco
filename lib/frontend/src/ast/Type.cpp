#include <iostream>
#include <modelica/frontend/AST.h>
#include <numeric>

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
		this->types.emplace_back(std::make_shared<Type>(type));
}

bool UserDefinedType::operator==(const UserDefinedType& other) const
{
	return std::equal(
			begin(),
			end(),
			other.begin(),
			other.end(),
			[](const Type& lhs, const Type& rhs) {
				return lhs == rhs;
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

UserDefinedType::iterator UserDefinedType::begin()
{
    return types.begin();
}

UserDefinedType::const_iterator UserDefinedType::begin() const
{
	return types.begin();
}

UserDefinedType::iterator UserDefinedType::end()
{
    return types.end();
}

UserDefinedType::const_iterator UserDefinedType::end() const
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
							 return a + (a.length() > 0 ? ", " : "") + toString(b);
						 }) +
				 "}";
}

ArrayDimension::ArrayDimension(long size) : size(size)
{
}

ArrayDimension::ArrayDimension(Expression size)
		: size(std::make_shared<Expression>(move(size)))
{
}

bool ArrayDimension::operator==(const ArrayDimension& other) const
{
	return size == other.size;
}

bool ArrayDimension::operator!=(const ArrayDimension& other) const { return !(*this == other); }

bool ArrayDimension::hasExpression() const
{
	return std::holds_alternative<ExpressionPtr>(size);
}

bool ArrayDimension::isDynamic() const
{
	return hasExpression() || getNumericSize() == -1;
}

long ArrayDimension::getNumericSize() const
{
	assert(holds_alternative<long>(size));
	return get<long>(size);
}

Expression ArrayDimension::getExpression()
{
	assert(hasExpression());
	return *get<ArrayDimension::ExpressionPtr>(size);
}

Expression ArrayDimension::getExpression() const
{
	assert(hasExpression());
	return *get<ArrayDimension::ExpressionPtr>(size);
}

raw_ostream& modelica::operator<<(raw_ostream& stream, const ArrayDimension& obj)
{
	return stream << toString(obj);
}

string modelica::toString(const ArrayDimension& obj)
{
	if (obj.hasExpression())
		return toString(obj.getExpression());

	return to_string(obj.getNumericSize());
}

Type::Type(BuiltInType type, ArrayRef<ArrayDimension> dim)
		: content(move(type)), dimensions(dim.begin(), dim.end())
{
	assert(holds_alternative<BuiltInType>(content));
	assert(!dimensions.empty());
}

Type::Type(UserDefinedType type, ArrayRef<ArrayDimension> dim)
		: content(move(type)), dimensions(dim.begin(), dim.end())
{
	assert(holds_alternative<UserDefinedType>(content));
	assert(!dimensions.empty());
}

Type::Type(llvm::ArrayRef<Type> members, ArrayRef<ArrayDimension> dim)
		: Type(UserDefinedType(move(members)), move(dim))
{
}

bool Type::operator==(const Type& other) const
{
	return dimensions == other.dimensions && content == other.content;
}

bool Type::operator!=(const Type& other) const { return !(*this == other); }

ArrayDimension& Type::operator[](int index) { return dimensions[index]; }

ArrayDimension Type::operator[](int index) const
{
	return dimensions[index];
}

void Type::dump() const { dump(outs(), 0); }

void Type::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << toString(*this);
}

SmallVectorImpl<ArrayDimension>& Type::getDimensions() { return dimensions; }

const SmallVectorImpl<ArrayDimension>& Type::getDimensions() const
{
	return dimensions;
}

void Type::setDimensions(llvm::ArrayRef<ArrayDimension> dims)
{
	dimensions.clear();
	dimensions.insert(dimensions.begin(), dims.begin(), dims.end());
}

size_t Type::dimensionsCount() const { return dimensions.size(); }

size_t Type::size() const
{
	long result = 1;

	for (const auto& dimension : dimensions)
	{
		if (dimension.hasExpression())
			return -1;

		result *= dimension.getNumericSize();
	}

	return result;
}

bool Type::isScalar() const
{
	return dimensions.size() == 1 &&
				 !dimensions[0].hasExpression() &&
				 dimensions[0].getNumericSize() == 1;
}

Type::dimensions_iterator Type::begin()
{
	return dimensions.begin();
}

Type::dimensions_const_iterator Type::begin() const
{
	return dimensions.begin();
}

Type::dimensions_iterator Type::end() { return dimensions.end(); }

Type::dimensions_const_iterator Type::end() const
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
				SmallVector<ArrayDimension, 3>(dimensions.begin() + times, dimensions.end()));
	});
}

raw_ostream& modelica::operator<<(raw_ostream& stream, const Type& obj)
{
	return stream << toString(obj);
}

class ArrayDimensionToStringVisitor
{
	public:
	string operator()(const long& value) { return to_string(value); }
	string operator()(const ArrayDimension::ExpressionPtr& expression) { return toString(*expression); };
};

string modelica::toString(Type obj)
{
	auto visitor = [](const auto& t) { return toString(t); };

	auto dimensionsToStringLambda = [](const string& a, ArrayDimension& b) -> string {
		return a + (a.length() > 0 ? "," : "") + b.visit(ArrayDimensionToStringVisitor());
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

Type Type::unknown() { return Type(BuiltInType::Unknown); }
