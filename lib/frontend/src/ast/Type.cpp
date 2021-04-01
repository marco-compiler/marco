#include <iostream>
#include <modelica/frontend/AST.h>
#include <numeric>

using namespace modelica::frontend;

namespace modelica::frontend
{
	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const BuiltInType& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(BuiltInType type)
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
}

PackedType::PackedType(llvm::ArrayRef<Type> types)
{
	for (const auto& type : types)
		this->types.emplace_back(std::make_shared<Type>(type));
}

bool PackedType::operator==(const PackedType& other) const
{
	if (types.size() != other.types.size())
		return false;

	auto pairs = llvm::zip(types, other.types);
	return std::all_of(pairs.begin(), pairs.end(),
										 [](const auto& pair)
										 {
											 const auto& [x, y] = pair;
											 return *x == *y;
										 });
}

bool PackedType::operator!=(const PackedType& other) const
{
	return !(*this == other);
}

Type& PackedType::operator[](size_t index)
{
	assert(index < types.size());
	return *types[index];
}

Type PackedType::operator[](size_t index) const
{
	assert(index < types.size());
	return *types[index];
}

void PackedType::dump() const { dump(llvm::outs(), 0); }

void PackedType::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << toString(*this);
}

bool PackedType::hasConstantShape() const
{
	return std::all_of(types.begin(), types.end(),
										 [](const auto& type) { return type->hasConstantShape(); });
}

size_t PackedType::size() const { return types.size(); }

PackedType::iterator PackedType::begin()
{
    return types.begin();
}

PackedType::const_iterator PackedType::begin() const
{
	return types.begin();
}

PackedType::iterator PackedType::end()
{
    return types.end();
}

PackedType::const_iterator PackedType::end() const
{
	return types.end();
}

namespace modelica::frontend
{
	llvm::raw_ostream& operator<<(
			llvm::raw_ostream& stream, const PackedType& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(PackedType obj)
	{
		return "{" +
					 accumulate(
							 obj.begin(),
							 obj.end(),
							 std::string(),
							 [](const std::string& a, const auto& b) -> std::string {
								 return a + (a.length() > 0 ? ", " : "") + toString(b);
							 }) +
					 "}";
	}
}

UserDefinedType::UserDefinedType(std::string name, llvm::ArrayRef<Type> types)
		: name(std::move(name))
{
	for (const auto& type : types)
		this->types.emplace_back(std::make_shared<Type>(type));
}

bool UserDefinedType::operator==(const UserDefinedType& other) const
{
	if (types.size() != other.types.size())
		return false;

	auto pairs = llvm::zip(types, other.types);
	return std::all_of(pairs.begin(), pairs.end(),
										 [](const auto& pair)
										 {
											 const auto& [x, y] = pair;
											 return *x == *y;
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

void UserDefinedType::dump() const { dump(llvm::outs(), 0); }

void UserDefinedType::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << toString(*this);
}

llvm::StringRef UserDefinedType::getName() const
{
	return name;
}

bool UserDefinedType::hasConstantShape() const
{
	return std::all_of(types.begin(), types.end(),
										 [](const auto& type) { return type->hasConstantShape(); });
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

namespace modelica::frontend
{
	llvm::raw_ostream& operator<<(
			llvm::raw_ostream& stream, const UserDefinedType& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(UserDefinedType obj)
	{
		return "{" +
					 accumulate(
							 obj.begin(),
							 obj.end(),
							 std::string(),
							 [](const std::string& a, const auto& b) -> std::string {
								 return a + (a.length() > 0 ? ", " : "") + toString(b);
							 }) +
					 "}";
	}
}

ArrayDimension::ArrayDimension(long size) : size(size)
{
}

ArrayDimension::ArrayDimension(Expression size)
		: size(std::make_shared<Expression>(std::move(size)))
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
	assert(std::holds_alternative<long>(size));
	return std::get<long>(size);
}

Expression& ArrayDimension::getExpression()
{
	assert(hasExpression());
	return *std::get<ArrayDimension::ExpressionPtr>(size);
}

const Expression& ArrayDimension::getExpression() const
{
	assert(hasExpression());
	return *std::get<ArrayDimension::ExpressionPtr>(size);
}

namespace modelica::frontend
{
	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const ArrayDimension& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(const ArrayDimension& obj)
	{
		if (obj.hasExpression())
			return toString(obj.getExpression());

		return std::to_string(obj.getNumericSize());
	}
}

Type::Type(BuiltInType type, llvm::ArrayRef<ArrayDimension> dim)
		: content(std::move(type)),
			dimensions(dim.begin(), dim.end())
{
	assert(std::holds_alternative<BuiltInType>(content));
	assert(!dimensions.empty());
}

Type::Type(PackedType type, llvm::ArrayRef<ArrayDimension> dim)
		: content(std::move(type)),
			dimensions(dim.begin(), dim.end())
{
	assert(std::holds_alternative<PackedType>(content));
	assert(!dimensions.empty());
}

Type::Type(UserDefinedType type, llvm::ArrayRef<ArrayDimension> dim)
		: content(std::move(type)),
			dimensions(dim.begin(), dim.end())
{
	assert(std::holds_alternative<UserDefinedType>(content));
	assert(!dimensions.empty());
}

bool Type::operator==(const Type& other) const
{
	return dimensions == other.dimensions && content == other.content;
}

bool Type::operator!=(const Type& other) const { return !(*this == other); }

ArrayDimension& Type::operator[](int index) { return dimensions[index]; }

const ArrayDimension& Type::operator[](int index) const
{
	return dimensions[index];
}

void Type::dump() const { dump(llvm::outs(), 0); }

void Type::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << toString(*this);
}

llvm::SmallVectorImpl<ArrayDimension>& Type::getDimensions() { return dimensions; }

const llvm::SmallVectorImpl<ArrayDimension>& Type::getDimensions() const
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

bool Type::hasConstantShape() const
{
	for (const auto& dimension : dimensions)
		if (dimension.isDynamic())
			return false;

	if (isA<PackedType>())
		return get<PackedType>().hasConstantShape();

	return true;
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
		return visit([&](const auto& t) { return Type(t); });

	assert(times > dimensions.size());

	return visit([&](const auto& type) {
		return Type(
				type,
				llvm::SmallVector<ArrayDimension, 3>(dimensions.begin() + times, dimensions.end()));
	});
}

class ArrayDimensionToStringVisitor
{
	public:
	std::string operator()(const long& value) { return std::to_string(value); }
	std::string operator()(const ArrayDimension::ExpressionPtr& expression) { return toString(*expression); };
};

namespace modelica::frontend
{
	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Type& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(Type obj)
	{
		auto visitor = [](const auto& t) { return toString(t); };

		auto dimensionsToStringLambda = [](const std::string& a, ArrayDimension& b) -> std::string {
			return a + (a.length() > 0 ? "," : "") + b.visit(ArrayDimensionToStringVisitor());
		};

		auto& dimensions = obj.getDimensions();
		std::string size = obj.isScalar() ? ""
																			: "[" +
																				accumulate(
																						dimensions.begin(),
																						dimensions.end(),
																						std::string(),
																						dimensionsToStringLambda) +
																				"] ";
		return size + obj.visit(visitor);
	}
}

Type Type::unknown() { return Type(BuiltInType::Unknown); }
