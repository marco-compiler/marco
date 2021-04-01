#include <modelica/frontend/AST.h>
#include <modelica/utils/IRange.hpp>
#include <numeric>

using namespace modelica;

Tuple::Tuple(SourcePosition location, llvm::ArrayRef<Expression> expressions)
		: location(std::move(location))
{
	for (const auto& exp : expressions)
		this->expressions.push_back(std::make_shared<Expression>(exp));
}

Tuple::Tuple(const Tuple& other)
		: location(other.location)
{
	for (const auto& exp : other.expressions)
		this->expressions.push_back(std::make_shared<Expression>(*exp));
}

Tuple& Tuple::operator=(const Tuple& other)
{
	if (this == &other)
		return *this;

	location = other.location;
	expressions.clear();

	for (const auto& exp : other.expressions)
		expressions.push_back(std::make_shared<Expression>(*exp));

	return *this;
}

bool Tuple::operator==(const Tuple& other) const
{
	if (expressions.size() != other.expressions.size())
		return false;

	auto pairs = llvm::zip(expressions, other.expressions);
	return std::all_of(pairs.begin(), pairs.end(),
										 [](const auto& pair)
										 {
											 const auto& [x, y] = pair;
											 return *x == *y;
										 });
}

bool Tuple::operator!=(const Tuple& other) const { return !(*this == other); }

Expression& Tuple::operator[](size_t index) { return *expressions[index]; }

const Expression& Tuple::operator[](size_t index) const
{
	return *expressions[index];
}

void Tuple::dump() const { dump(llvm::outs(), 0); }

void Tuple::dump(llvm::raw_ostream& os, size_t indents) const
{
	for (const auto& exp : expressions)
		exp->dump(os, indents);
}

SourcePosition Tuple::getLocation() const
{
	return location;
}

size_t Tuple::size() const { return expressions.size(); }

Tuple::iterator Tuple::begin()
{
	return expressions.begin();
}

Tuple::const_iterator Tuple::begin() const
{
	return expressions.begin();
}

Tuple::iterator Tuple::end() { return expressions.end(); }

Tuple::const_iterator Tuple::end() const
{
	return expressions.end();
}

llvm::raw_ostream& modelica::operator<<(llvm::raw_ostream& stream, const Tuple& obj)
{
	return stream << toString(obj);
}

std::string modelica::toString(const Tuple& obj)
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
