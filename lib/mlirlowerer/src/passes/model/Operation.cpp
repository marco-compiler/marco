#include <marco/mlirlowerer/passes/model/Expression.h>
#include <marco/mlirlowerer/passes/model/Operation.h>

namespace marco::codegen::model
{
	Operation::Operation(llvm::ArrayRef<Expression> args)
	{
		for (const auto& arg : args)
			this->args.emplace_back(std::make_shared<Expression>(arg));
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
}

