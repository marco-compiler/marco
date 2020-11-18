#include <modelica/frontend/Statement.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

AssignmentStatement::AssignmentStatement(
		Expression destination, Expression expression)
		: destination(move(destination)), expression(move(expression))
{
}

AssignmentStatement::AssignmentStatement(
		Tuple destinations, Expression expression)
		: destination(move(destinations)), expression(move(expression))
{
}

AssignmentStatement::AssignmentStatement(
		initializer_list<Expression> destinations, Expression expression)
		: destination(Tuple(move(destinations))), expression(move(expression))
{
}

void AssignmentStatement::dump() const { dump(outs(), 0); }

void AssignmentStatement::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "destinations:\n";
	visit([&](const auto& obj) { obj.dump(os, indents + 1); }, destination);

	os.indent(indents);
	os << "assigned expression:\n";
	expression.dump(os, indents + 1);
}

vector<Expression*> AssignmentStatement::getDestinations()
{
	vector<Expression*> destinations;

	if (destinationIsA<Expression>())
		destinations.push_back(&getDestination<Expression>());
	else
	{
		for (auto& exp : getDestination<Tuple>())
			destinations.push_back(&*exp);
	}

	return destinations;
}

void AssignmentStatement::setDestination(Expression dest)
{
	destination = move(dest);
}

void AssignmentStatement::setDestination(Tuple dest)
{
	destination = move(dest);
}

Expression& AssignmentStatement::getExpression() { return expression; }

const Expression& AssignmentStatement::getExpression() const
{
	return expression;
}

void ForStatement::dump() const { dump(outs(), 0); }

void ForStatement::dump(raw_ostream& os, size_t indents) const {}

IfBlock::IfBlock(Expression condition, ArrayRef<Statement> statements)
		: condition(move(condition))
{
	for (const auto& statement : statements)
		this->statements.push_back(std::make_unique<Statement>(statement));
}

IfBlock::IfBlock(llvm::ArrayRef<Statement> statements)
		: IfBlock(Expression::trueExp(), statements)
{
}

IfBlock::IfBlock(const IfBlock& other): condition(other.condition)
{
	statements.clear();

	for (const auto& statement : other.statements)
		statements.push_back(std::make_unique<Statement>(*statement));
}

IfBlock& IfBlock::operator=(const IfBlock& other)
{
	if (this == &other)
		return *this;

	condition = other.condition;
	statements.clear();

	for (const auto& statement : other.statements)
		statements.push_back(std::make_unique<Statement>(*statement));

	return *this;
}

UniqueStatement& IfBlock::operator[](size_t index)
{
	assert(index < statements.size());
	return statements[index];
}

const UniqueStatement& IfBlock::operator[](size_t index) const
{
	assert(index < statements.size());
	return statements[index];
}

void IfBlock::dump() const { dump(outs(), 0); }

void IfBlock::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "condition:\n";
	condition.dump(os, indents + 1);

	os.indent(indents);
	os << "body:\n";

	for (const auto& statement : statements)
		statement->dump(os, indents + 1);
}

Expression& IfBlock::getCondition() { return condition; }

const Expression& IfBlock::getCondition() const { return condition; }

size_t IfBlock::size() const { return statements.size(); }

llvm::SmallVectorImpl<UniqueStatement>::iterator IfBlock::begin()
{
	return statements.begin();
}

llvm::SmallVectorImpl<UniqueStatement>::const_iterator IfBlock::begin() const
{
	return statements.begin();
}

llvm::SmallVectorImpl<UniqueStatement>::iterator IfBlock::end()
{
	return statements.end();
}

llvm::SmallVectorImpl<UniqueStatement>::const_iterator IfBlock::end() const
{
	return statements.end();
}

IfStatement::IfStatement(llvm::ArrayRef<IfBlock> blocks)
		: blocks(blocks.begin(), blocks.end())
{
	assert(this->blocks.size() > 1);
}

IfBlock& IfStatement::operator[](size_t index)
{
	assert(index < blocks.size());
	return blocks[index];
}

const IfBlock& IfStatement::operator[](size_t index) const
{
	assert(index < blocks.size());
	return blocks[index];
}

void IfStatement::dump() const { dump(outs(), 0); }

void IfStatement::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "if statement\n";

	for (const auto& block : blocks)
		block.dump(os, indents + 1);
}

size_t IfStatement::size() const { return blocks.size(); }

llvm::SmallVectorImpl<IfBlock>::iterator IfStatement::begin()
{
	return blocks.begin();
}

llvm::SmallVectorImpl<IfBlock>::const_iterator IfStatement::begin() const
{
	return blocks.begin();
}

llvm::SmallVectorImpl<IfBlock>::iterator IfStatement::end()
{
	return blocks.end();
}

llvm::SmallVectorImpl<IfBlock>::const_iterator IfStatement::end() const
{
	return blocks.end();
}

Statement::Statement(AssignmentStatement statement): content(move(statement)) {}

Statement::Statement(ForStatement statement): content(move(statement)) {}

Statement::Statement(IfStatement statement): content(move(statement)) {}

void Statement::dump() const { dump(outs(), 0); }

void Statement::dump(raw_ostream& os, size_t indents) const
{
	visit([&](const auto& statement) { statement.dump(os, indents); });
}
