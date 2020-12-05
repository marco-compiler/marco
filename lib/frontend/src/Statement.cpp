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

	if (holds_alternative<Expression>(destination))
		destinations.push_back(&get<Expression>(destination));
	else
	{
		for (auto& exp : get<Tuple>(destination))
			destinations.push_back(&exp);
	}

	return destinations;
}

vector<const Expression*> AssignmentStatement::getDestinations() const
{
	vector<const Expression*> destinations;

	if (holds_alternative<Expression>(destination))
		destinations.push_back(&get<Expression>(destination));
	else
	{
		for (auto& exp : get<Tuple>(destination))
			destinations.push_back(&exp);
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

IfStatement::IfStatement(llvm::ArrayRef<Block> blocks)
		: blocks(blocks.begin(), blocks.end())
{
	assert(this->blocks.size() > 1);
}

IfStatement::Block& IfStatement::operator[](size_t index)
{
	assert(index < blocks.size());
	return blocks[index];
}

const IfStatement::Block& IfStatement::operator[](size_t index) const
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

IfStatement::blocks_iterator IfStatement::begin()
{
	return blocks.begin();
}

IfStatement::blocks_const_iterator IfStatement::begin() const
{
	return blocks.begin();
}

IfStatement::blocks_iterator IfStatement::end()
{
	return blocks.end();
}

IfStatement::blocks_const_iterator IfStatement::end() const
{
	return blocks.end();
}

ForStatement::ForStatement(
		Induction induction, llvm::ArrayRef<Statement> statements)
		: induction(move(induction))
{
	for (const auto& statement : statements)
		this->statements.push_back(std::make_unique<Statement>(statement));
}

ForStatement::ForStatement(const ForStatement& other)
		: induction(other.induction)
{
	statements.clear();

	for (const auto& statement : other.statements)
		statements.push_back(std::make_unique<Statement>(*statement));
}

ForStatement& ForStatement::operator=(const ForStatement& other)
{
	if (this == &other)
		return *this;

	induction = other.induction;
	statements.clear();

	for (const auto& statement : other.statements)
		statements.push_back(std::make_unique<Statement>(*statement));

	return *this;
}

Statement& ForStatement::operator[](size_t index)
{
	assert(index < statements.size());
	return *statements[index];
}

const Statement& ForStatement::operator[](size_t index) const
{
	assert(index < statements.size());
	return *statements[index];
}

void ForStatement::dump() const { dump(outs(), 0); }

void ForStatement::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "induction:\n";
	induction.dump(os, indents + 1);

	os.indent(indents);
	os << "body:\n";

	for (const auto& statement : statements)
		statement->dump(os, indents + 1);
}

Induction& ForStatement::getInduction() { return induction; }

const Induction& ForStatement::getInduction() const { return induction; }

size_t ForStatement::size() const { return statements.size(); }

ForStatement::statements_iterator ForStatement::begin()
{
	return statements.begin();
}

ForStatement::statements_const_iterator ForStatement::begin() const
{
	return statements.begin();
}

ForStatement::statements_iterator ForStatement::end()
{
	return statements.end();
}

ForStatement::statements_const_iterator ForStatement::end() const
{
	return statements.end();
}

WhileStatement::WhileStatement(
		Expression condition, llvm::ArrayRef<Statement> body)
		: ConditionalBlock<Statement>(move(condition), move(body))
{
}

WhenStatement::WhenStatement(
		Expression condition, llvm::ArrayRef<Statement> body)
		: ConditionalBlock<Statement>(move(condition), move(body))
{
}

void BreakStatement::dump() const { dump(outs(), 0); }

void BreakStatement::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "break\n";
}

void ReturnStatement::dump() const { dump(outs(), 0); }

void ReturnStatement::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << "return\n";
}

Statement::Statement(AssignmentStatement statement): content(move(statement)) {}

Statement::Statement(IfStatement statement): content(move(statement)) {}

Statement::Statement(ForStatement statement): content(move(statement)) {}

Statement::Statement(BreakStatement statement): content(move(statement)) {}

Statement::Statement(ReturnStatement statement): content(move(statement)) {}

void Statement::dump() const { dump(outs(), 0); }

void Statement::dump(raw_ostream& os, size_t indents) const
{
	visit([&](const auto& statement) { statement.dump(os, indents); });
}

Statement::assignments_iterator Statement::begin() { return assignments_iterator(this, this); }

Statement::assignments_const_iterator Statement::begin() const
{
	return assignments_const_iterator(this, this);
}

Statement::assignments_iterator Statement::end() { return assignments_iterator(this, nullptr); }

Statement::assignments_const_iterator Statement::end() const
{
	return assignments_const_iterator(this, nullptr);
}
