#include <modelica/frontend/AST.h>

using namespace modelica;

Algorithm::Algorithm(SourcePosition location, llvm::ArrayRef<Statement> statements)
		: location(std::move(location))
{
	for (const auto& statement : statements)
		this->statements.emplace_back(std::make_shared<Statement>(statement));
}

Statement& Algorithm::operator[](size_t index)
{
	return *statements[index];
}

const Statement& Algorithm::operator[](size_t index) const
{
	return *statements[index];
}

void Algorithm::dump() const { dump(llvm::outs(), 0); }

void Algorithm::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "algorithm\n";

	for (const auto& statement : statements)
		statement->visit([&](const auto& obj) { obj.dump(os, indents + 1); });
}

SourcePosition Algorithm::getLocation() const
{
	return location;
}

const std::string& Algorithm::getReturnCheckName() const
{
	return returnCheckName;
}

void Algorithm::setReturnCheckName(std::string name)
{
	this->returnCheckName = name;
}

Algorithm::Container<Statement>& Algorithm::getStatements() { return statements; }

const Algorithm::Container<Statement>& Algorithm::getStatements() const
{
	return statements;
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
