#include <modelica/frontend/AST.h>

using namespace modelica;
using namespace modelica::frontend;

Expression::Expression(Array content)
		: content(std::move(content))
{
}

Expression::Expression(Call content)
		: content(std::move(content))
{
}

Expression::Expression(Constant content)
		: content(std::move(content))
{
}

Expression::Expression(Operation content)
		: content(std::move(content))
{
}

Expression::Expression(ReferenceAccess content)
		: content(std::move(content))
{
}

Expression::Expression(Tuple content)
		: content(std::move(content))
{
}

Expression::Expression(const Expression& other)
		: content(other.content)
{
}

Expression::Expression(Expression&& other) = default;

Expression::~Expression() = default;

Expression& Expression::operator=(const Expression& other)
{
	Expression result(other);
	swap(*this, result);
	return *this;
}

Expression& Expression::operator=(Expression&& other) = default;

namespace modelica::frontend
{
	void swap(Expression& first, Expression& second)
	{
		using std::swap;
		swap(first.content, second.content);
	}
}

void Expression::print(llvm::raw_ostream& os, size_t indents) const
{
	visit([&os, indents](const auto& obj) {
		obj.dump(os, indents);
	});
}

bool Expression::operator==(const Expression& rhs) const
{
	return content == rhs.content;
}

bool Expression::operator!=(const Expression& rhs) const
{
	return !(rhs == *this);
}

SourceRange Expression::getLocation() const
{
	return visit([](const auto& obj) {
		return obj.getLocation();
	});
}

Type& Expression::getType()
{
	return std::visit([](auto& obj) -> Type& {
		return obj.getType();
	}, content);
}

const Type& Expression::getType() const
{
	return std::visit([](const auto& obj) -> const Type& {
		return obj.getType();
	}, content);
}

void Expression::setType(Type tp)
{
	visit([&tp](auto& obj) {
		obj.setType(std::move(tp));
	});
}

bool Expression::isLValue() const
{
	return visit([](const auto& obj) {
		return obj.isLValue();
	});
}

namespace modelica::frontend
{
	llvm::raw_ostream& operator<<(llvm::raw_ostream& stream, const Expression& obj)
	{
		return stream << toString(obj);
	}

	std::string toString(const Expression& obj)
	{
		// TODO
		return "";
		//return obj.visit([](const auto& obj) { return toString(obj); });
	}
}
