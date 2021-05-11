#include <memory>
#include <modelica/frontend/AST.h>
#include <numeric>

using namespace modelica::frontend;

Array::Array(SourcePosition location,
						 llvm::ArrayRef<std::unique_ptr<Expression>> values,
						 Type type)
		: ExpressionCRTP<Array>(
					ASTNodeKind::EXPRESSION_ARRAY, std::move(location), std::move(type))
{
	for (const auto& value : values)
		this->values.push_back(value->cloneExpression());
}

Array::Array(const Array& other)
		: ExpressionCRTP<Array>(static_cast<ExpressionCRTP<Array>&>(*this))
{
	for (const auto& value : other.values)
		this->values.push_back(value->cloneExpression());
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

namespace modelica::frontend
{
	void swap(Array& first, Array& second)
	{
		swap(static_cast<impl::ExpressionCRTP<Array>&>(first),
				 static_cast<impl::ExpressionCRTP<Array>&>(second));

		impl::swap(first.values, second.values);
	}
}

void Array::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "type: ";
	getType().dump(os);
	os << "\n";

	for (const auto& value : values)
		value->dump(os, indents);
}

bool Array::isLValue() const
{
	return false;
}

bool Array::operator==(const Array& other) const
{
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

namespace modelica::frontend
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
