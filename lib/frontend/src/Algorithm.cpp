#include <modelica/frontend/Algorithm.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

Algorithm::Algorithm(SourcePosition location, ArrayRef<Statement> statements)
		: location(move(location)),
			statements(statements.begin(), statements.end())
{
}

void Algorithm::dump() const { dump(outs(), 0); }

void Algorithm::dump(llvm::raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "algorithm\n";

	for (const auto& statement : statements)
		statement.visit([&](const auto& obj) { obj.dump(os, indents + 1); });
}

SourcePosition Algorithm::getLocation() const
{
	return location;
}

const string& Algorithm::getReturnCheckName() const
{
	return returnCheckName;
}

void Algorithm::setReturnCheckName(string name)
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
