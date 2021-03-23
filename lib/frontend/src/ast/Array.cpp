#include <modelica/frontend/AST.h>
#include <numeric>

using namespace modelica;

Array::Array(SourcePosition location, llvm::ArrayRef<Expression> values)
		: location(std::move(location))
{
	for (const auto& value : values)
		this->values.push_back(std::make_shared<Expression>(value));
}

bool Array::operator==(const Array& other) const
{
	if (size() != other.size())
		return false;

	auto pairs = llvm::zip(values, other.values);

	for (auto [ x, y ] : pairs)
		if (x != y)
			return false;

	return true;

	//return std::all_of(pairs.begin(), pairs.end(),
	//									 [](auto& x, auto& y) { return x == y; });
}

bool Array::operator!=(const Array& other) const { return !(*this == other); }

Expression& Array::operator[](size_t index)
{
	assert(index < size());
	return *values[index];
}

const Expression& Array::operator[](size_t index) const
{
	assert(index < size());
	return *values[index];
}

void Array::dump() const { dump(llvm::outs(), 0); }

void Array::dump(llvm::raw_ostream& os, size_t indents) const
{
	for (const auto& value : values)
		value->dump(os, indents);
}

SourcePosition Array::getLocation() const
{
	return location;
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

llvm::raw_ostream& modelica::operator<<(llvm::raw_ostream& stream, const Array& obj)
{
	return stream << toString(obj);
}

std::string modelica::toString(const Array& obj)
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
