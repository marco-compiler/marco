#include <modelica/frontend/Expression.hpp>
#include <modelica/frontend/Tuple.hpp>
#include <modelica/utils/IRange.hpp>
#include <numeric>

using namespace llvm;
using namespace modelica;
using namespace std;

Tuple::Tuple(SourcePosition location, ArrayRef<Expression> expressions)
		: location(move(location))
{
	for (const auto& exp : expressions)
		this->expressions.push_back(std::make_unique<Expression>(exp));
}

Tuple::Tuple(const Tuple& other)
		: location(other.location)
{
	for (const auto& exp : other.expressions)
		this->expressions.push_back(std::make_unique<Expression>(*exp));
}

Tuple& Tuple::operator=(const Tuple& other)
{
	if (this == &other)
		return *this;

	location = other.location;
	expressions.clear();

	for (const auto& exp : other.expressions)
		expressions.push_back(std::make_unique<Expression>(*exp));

	return *this;
}

bool Tuple::operator==(const Tuple& other) const
{
	if (size() != other.size())
		return false;

	for (auto i : irange(size()))
		if ((*this)[i] != other[i])
			return false;

	return true;
}

bool Tuple::operator!=(const Tuple& other) const { return !(*this == other); }

Expression& Tuple::operator[](size_t index) { return *expressions[index]; }

const Expression& Tuple::operator[](size_t index) const
{
	return *expressions[index];
}

void Tuple::dump() const { dump(outs(), 0); }

void Tuple::dump(raw_ostream& os, size_t indents) const
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
				 accumulate(++obj.begin(), obj.end(), string(),
										[](const string& result, const Expression& element)
										{
											string str = toString(element);
											return result.empty() ? str : result + "," + str;
										}) +
				 ")";
}
