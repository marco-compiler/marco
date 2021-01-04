#include <modelica/frontend/Call.hpp>
#include <modelica/frontend/Expression.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

Call::Call(SourcePosition location, Expression function, ArrayRef<Expression> args)
		: location(move(location)),
			function(std::make_unique<Expression>(move(function)))
{
	for (const auto& arg : args)
		this->args.emplace_back(std::make_unique<Expression>(arg));
}

Call::Call(const Call& other)
		: location(other.location),
			function(std::make_unique<Expression>(*other.function))
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

	location = other.location;
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

	auto pairs = llvm::zip(args, other.args);

	return std::all_of(
			pairs.begin(), pairs.end(),
			[](const auto& pair) { return *get<0>(pair) == *get<1>(pair); });
}

bool Call::operator!=(const Call& other) const { return !(*this == other); }

Expression& Call::operator[](size_t index)
{
	assert(index < argumentsCount());
	return *args[index];
}

const Expression& Call::operator[](size_t index) const
{
	assert(index < argumentsCount());
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

SourcePosition Call::getLocation() const
{
	return location;
}

Expression& Call::getFunction() { return *function; }

const Expression& Call::getFunction() const { return *function; }

size_t Call::argumentsCount() const { return args.size(); }

Call::args_iterator Call::begin()
{
	return args.begin();
}

Call::args_const_iterator Call::begin() const
{
	return args.begin();
}

Call::args_iterator Call::end() { return args.end(); }

Call::args_const_iterator Call::end() const
{
	return args.end();
}
