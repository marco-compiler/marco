#include <modelica/mlirlowerer/passes/model/Expression.h>

using namespace modelica::codegen::model;

Operation::Operation(llvm::ArrayRef<Expression> args)
{
	for (auto& arg : args)
		this->args.push_back(std::make_shared<Expression>(arg));
}

Operation::ExpressionPtr Operation::operator[](size_t index)
{
	assert(index < size());
	return args[index];
}

const Operation::ExpressionPtr Operation::operator[](size_t index) const
{
	assert(index < size());
	return args[index];
}

size_t Operation::size() const
{
	return args.size();
}

Operation::iterator Operation::begin()
{
	return args.begin();
}

Operation::const_iterator Operation::begin() const
{
	return args.begin();
}

Operation::iterator Operation::end()
{
	return args.end();
}

Operation::const_iterator Operation::end() const
{
	return args.end();
}

size_t Operation::childrenCount() const
{
	return size();
}
