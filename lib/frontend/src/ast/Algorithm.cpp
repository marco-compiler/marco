#include <modelica/frontend/AST.h>

using namespace modelica::frontend;

Algorithm::Algorithm(SourcePosition location, llvm::ArrayRef<std::unique_ptr<Statement>> statements)
		: ASTNodeCRTP<Algorithm>(ASTNodeKind::ALGORITHM, std::move(location))
{
	for (const auto& statement : statements)
		this->statements.push_back(statement->cloneStatement());
}

Algorithm::Algorithm(const Algorithm& other)
		: ASTNodeCRTP<Algorithm>(static_cast<const ASTNodeCRTP<Algorithm>&>(other))
{
	for (const auto& statement : other.statements)
		this->statements.push_back(statement->cloneStatement());
}

Algorithm::Algorithm(Algorithm&& other) = default;

Algorithm::~Algorithm() = default;

Algorithm& Algorithm::operator=(const Algorithm& other)
{
	Algorithm result(other);
	swap(*this, result);
	return *this;
}

Algorithm& Algorithm::operator=(Algorithm&& other) = default;

namespace modelica::frontend
{
	void swap(Algorithm& first, Algorithm& second)
	{
		swap(static_cast<impl::ASTNodeCRTP<Algorithm>&>(first),
				 static_cast<impl::ASTNodeCRTP<Algorithm>&>(second));

		std::swap(first.returnCheckName, second.returnCheckName);
		impl::swap(first.statements, second.statements);

		/*
		Algorithm::Container<std::unique_ptr<Statement>> statementsTmp;

		for (const auto& statement : first.statements)
			statementsTmp.push_back(statement->cloneStatement());

		first.statements = std::move(second.statements);
		second.statements = std::move(statementsTmp);
		 */
	}
}

void Algorithm::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "algorithm\n";

	for (const auto& statement : *this)
		statement->dump(os, indents + 1);
}

Statement* Algorithm::operator[](size_t index)
{
	assert(index < statements.size());
	return statements[index].get();
}

const Statement* Algorithm::operator[](size_t index) const
{
	assert(index < statements.size());
	return statements[index].get();
}

llvm::StringRef Algorithm::getReturnCheckName() const
{
	return returnCheckName;
}

void Algorithm::setReturnCheckName(llvm::StringRef name)
{
	returnCheckName = name.str();
}

size_t Algorithm::size() const
{
	return statements.size();
}

Algorithm::statements_iterator Algorithm::begin()
{
	return statements.begin();
}

Algorithm::statements_const_iterator Algorithm::begin() const
{
	return statements.begin();
}

Algorithm::statements_iterator Algorithm::end()
{
	return statements.end();
}

Algorithm::statements_const_iterator Algorithm::end() const
{
	return statements.end();
}
