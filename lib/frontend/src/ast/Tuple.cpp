#include <marco/frontend/AST.h>
#include <marco/utils/IRange.hpp>
#include <numeric>

using namespace marco::frontend;

Tuple::Tuple(SourceRange location,
						 Type type,
						 llvm::ArrayRef<std::unique_ptr<Expression>> expressions)
		: ASTNode(std::move(location)),
			type(std::move(type))
{
	for (const auto& expression : expressions)
		this->expressions.push_back(expression->clone());
}

Tuple::Tuple(const Tuple& other)
		: ASTNode(other),
			type(other.type)
{
	for (const auto& expression : other.expressions)
		this->expressions.push_back(expression->clone());
}

Tuple::Tuple(Tuple&& other) = default;

Tuple::~Tuple() = default;

Tuple& Tuple::operator=(const Tuple& other)
{
	Tuple result(other);
	swap(*this, result);
	return *this;
}

Tuple& Tuple::operator=(Tuple&& other) = default;

namespace marco::frontend
{
	void swap(Tuple& first, Tuple& second)
	{
		swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

		using std::swap;
		swap(first.type, second.type);
		impl::swap(first.expressions, second.expressions);
	}
}

void Tuple::print(llvm::raw_ostream& os, size_t indents) const
{
	for (const auto& expression : *this)
		expression->print(os, indents);
}

bool Tuple::isLValue() const
{
	return false;
}

bool Tuple::operator==(const Tuple& other) const
{
	if (expressions.size() != other.expressions.size())
		return false;

	auto pairs = llvm::zip(expressions, other.expressions);

	return std::all_of(
			pairs.begin(), pairs.end(),
			[](const auto& pair) {
				const auto& [x, y] = pair;
				return *x == *y;
			});
}

bool Tuple::operator!=(const Tuple& other) const
{
	return !(*this == other);
}

Expression* Tuple::operator[](size_t index)
{
	return getArg(index);
}

const Expression* Tuple::operator[](size_t index) const
{
	return getArg(index);
}

Type& Tuple::getType()
{
	return type;
}

const Type& Tuple::getType() const
{
	return type;
}

void Tuple::setType(Type tp)
{
	type = std::move(tp);
}

Expression* Tuple::getArg(size_t index)
{
	assert(index < expressions.size());
	return expressions[index].get();
}

const Expression* Tuple::getArg(size_t index) const
{
	assert(index < expressions.size());
	return expressions[index].get();
}

size_t Tuple::size() const
{
	return expressions.size();
}

Tuple::iterator Tuple::begin()
{
	return expressions.begin();
}

Tuple::const_iterator Tuple::begin() const
{
	return expressions.begin();
}

Tuple::iterator Tuple::end()
{
	return expressions.end();
}

Tuple::const_iterator Tuple::end() const
{
	return expressions.end();
}

namespace marco::frontend
{
	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Tuple& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(const Tuple& obj)
	{
		return "(" +
					 accumulate(std::next(obj.begin()), obj.end(), std::string(),
											[](const std::string& result, const auto& element)
											{
												std::string str = toString(*element);
												return result.empty() ? str : result + "," + str;
											}) +
					 ")";
	}
}
