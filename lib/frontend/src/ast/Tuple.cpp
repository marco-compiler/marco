#include <modelica/frontend/AST.h>
#include <modelica/utils/IRange.hpp>
#include <numeric>

using namespace modelica::frontend;

// TODO type
Tuple::Tuple(SourcePosition location,
						 llvm::ArrayRef<std::unique_ptr<Expression>> expressions)
		: ExpressionCRTP<Tuple>(ASTNodeKind::EXPRESSION_TUPLE, std::move(location), Type::unknown())
{
	for (const auto& expression : expressions)
		this->expressions.push_back(expression->cloneExpression());
}

Tuple::Tuple(const Tuple& other)
		: ExpressionCRTP<Tuple>(static_cast<ExpressionCRTP&>(*this))
{
	for (const auto& expression : other.expressions)
		this->expressions.push_back(expression->cloneExpression());
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

namespace modelica::frontend
{
	void swap(Tuple& first, Tuple& second)
	{
		swap(static_cast<impl::ExpressionCRTP<Tuple>&>(first),
				 static_cast<impl::ExpressionCRTP<Tuple>&>(second));

		using std::swap;
		swap(first.expressions, second.expressions);
	}
}

void Tuple::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "type: ";
	getType().dump(os);
	os << "\n";

	for (const auto& expression : *this)
		expression.dump(os, indents);
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
	assert(index < expressions.size());
	return expressions[index].get();
}

const Expression* Tuple::operator[](size_t index) const
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

namespace modelica::frontend
{
	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Tuple& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(const Tuple& obj)
	{
		return "(" +
					 accumulate(++obj.begin(), obj.end(), std::string(),
											[](const std::string& result, const Expression& element)
											{
												std::string str = toString(element);
												return result.empty() ? str : result + "," + str;
											}) +
					 ")";
	}
}
