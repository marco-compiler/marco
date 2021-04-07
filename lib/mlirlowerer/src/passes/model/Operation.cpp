#include <modelica/mlirlowerer/passes/model/Operation.h>

using namespace modelica::codegen::model;

Operation::Operation(llvm::ArrayRef<std::shared_ptr<Expression>> args)
		: args(args.begin(), args.end())
{
}

std::shared_ptr<Expression> Operation::operator[](size_t index)
{
	assert(index < size());
	return args[index];
}

std::shared_ptr<Expression> Operation::operator[](size_t index) const
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
