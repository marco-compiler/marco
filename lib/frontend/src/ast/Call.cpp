#include <modelica/frontend/AST.h>
#include <numeric>

using namespace modelica;

Call::Call(SourcePosition location,
					 Expression function,
					 llvm::ArrayRef<Expression> args,
					 unsigned int elementWiseRank)
		: location(std::move(location)),
			function(std::make_shared<Expression>(std::move(function))),
			elementWiseRank(elementWiseRank)
{
	for (const auto& arg : args)
		this->args.emplace_back(std::make_shared<Expression>(arg));
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
			[](const auto& pair) { return *std::get<0>(pair) == *std::get<1>(pair); });
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

void Call::dump() const { dump(llvm::outs(), 0); }

void Call::dump(llvm::raw_ostream& os, size_t indents) const
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

bool Call::isElementWise() const
{
	return elementWiseRank != 0;
}

unsigned int Call::getElementWiseRank() const
{
	return elementWiseRank;
}

void Call::setElementWiseRank(unsigned int rank)
{
	elementWiseRank = rank;
}

llvm::raw_ostream& modelica::operator<<(llvm::raw_ostream& stream, const Call& obj)
{
	return stream << toString(obj);
}

std::string modelica::toString(const Call& obj)
{
	return toString(obj.getFunction()) + "(" +
				 accumulate(obj.begin(), obj.end(), std::string(),
										[](const std::string& result, const Expression& argument)
										{
											std::string str = toString(argument);
											return result.empty() ? str : result + "," + str;
										}) +
				 ")";
}
