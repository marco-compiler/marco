#include "marco/ast/AST.h"
#include <memory>
#include <numeric>

using namespace marco::ast;

Array::Array(SourceRange location,
						 Type type,
						 llvm::ArrayRef<std::unique_ptr<Expression>> values)
		: ASTNode(std::move(location)),
			type(std::move(type))
{
	for (const auto& value : values)
		this->values.push_back(value->clone());
}

Array::Array(const Array& other)
		: ASTNode(other),
			type(other.type)
{
	for (const auto& value : other.values)
		this->values.push_back(value->clone());
}

Array::Array(Array&& other) = default;

Array::~Array() = default;

Array& Array::operator=(const Array& other)
{
	Array result(other);
	swap(*this, result);
	return *this;
}

Array& Array::operator=(Array&& other) = default;

namespace marco::ast
{
	void swap(Array& first, Array& second)
	{
		swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

		using std::swap;
		swap(first.type, second.type);
		impl::swap(first.values, second.values);
	}
}

void Array::print(llvm::raw_ostream& os, size_t indents) const
{
	for (const auto& value : values)
		value->print(os, indents);
}

bool Array::isLValue() const
{
	return false;
}

bool Array::operator==(const Array& other) const
{
	if (type != other.type)
		return false;

	if (values.size() != other.values.size())
		return false;

	auto pairs = llvm::zip(values, other.values);

	return std::all_of(
			pairs.begin(), pairs.end(),
			[](const auto& pair) {
				const auto& [x, y] = pair;
				return *x == *y;
			});
}

bool Array::operator!=(const Array& other) const
{
	return !(*this == other);
}

Expression* Array::operator[](size_t index)
{
	assert(index < values.size());
	return values[index].get();
}

const Expression* Array::operator[](size_t index) const
{
	assert(index < values.size());
	return values[index].get();
}

Type& Array::getType()
{
	return type;
}

const Type& Array::getType() const
{
	return type;
}

void Array::setType(Type tp)
{
	type = std::move(tp);
}

size_t Array::size() const
{
	return values.size();
}

Array::iterator Array::begin()
{
	return values.begin();
}

Array::const_iterator Array::begin() const
{
	return values.begin();
}

Array::iterator Array::end()
{
	return values.end();
}

Array::const_iterator Array::end() const
{
	return values.end();
}

namespace marco::ast
{
	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Array& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(const Array& obj)
	{
		return "(" +
					 accumulate(
							 obj.begin(), obj.end(), std::string(),
							 [](const std::string& result, const std::unique_ptr<Expression>& element)
											{
												std::string str = toString(*element);
												return result.empty() ? str : result + "," + str;
											}) +
					 ")";
	}
}
