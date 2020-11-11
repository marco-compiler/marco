#include <modelica/frontend/Call.hpp>
#include <modelica/frontend/Expression.hpp>
#include <modelica/utils/IRange.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

Call::Call(Expression fun, ArrayRef<Expression> args)
		: function(std::make_unique<Expression>(move(fun)))
{
	for (const auto& arg : args)
		this->args.emplace_back(std::make_unique<Expression>(arg));
}

Call::Call(const Call& other)
		: function(std::make_unique<Expression>(*other.function))
{
	assert(other.function != nullptr);
	assert(find(other.args, nullptr) == other.args.end());

	for (const auto& exp : other.args)
		args.emplace_back(std::make_unique<Expression>(*exp));
}

Call& Call::operator=(const Call& other)
{
	assert(other.function != nullptr);
	assert(find(other.args, nullptr) == other.args.end());

	if (this == &other)
		return *this;

	function = std::make_unique<Expression>(*other.function);
	args.clear();

	for (const auto& exp : other.args)
		args.emplace_back(std::make_unique<Expression>(*exp));

	return *this;
}

bool Call::operator==(const Call& other) const
{
	if (argumentsCount() != other.argumentsCount())
		return false;

	if (*function != *other.function)
		return false;

	for (auto i : irange(argumentsCount()))
		if (*args[i] != other[i])
			return false;

	return true;
}

bool Call::operator!=(const Call& other) const { return !(*this == other); }

Expression& Call::operator[](size_t index)
{
	assert(index <= argumentsCount());
	return *args[index];
}

const Expression& Call::operator[](size_t index) const
{
	assert(index <= argumentsCount());
	return *args[index];
}

void Call::dump() const { dump(outs(), 0); }

void Call::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "call\n";

	function->dump(os, indents + 1);

	for (const auto& exp : args)
		exp->dump(os, indents + 1);
}

Expression& Call::getFunction() { return *function; }

const Expression& Call::getFunction() const { return *function; }

size_t Call::argumentsCount() const { return args.size(); }

SmallVectorImpl<Call::UniqueExpr>::iterator Call::begin()
{
	return args.begin();
}

SmallVectorImpl<Call::UniqueExpr>::const_iterator Call::begin() const
{
	return args.begin();
}

SmallVectorImpl<Call::UniqueExpr>::iterator Call::end() { return args.end(); }

SmallVectorImpl<Call::UniqueExpr>::const_iterator Call::end() const
{
	return args.end();
}
