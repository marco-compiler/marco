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

UniqueStatement& ForStatement::operator[](size_t index)
{
	assert(index < statements.size());
	return statements[index];
}

const UniqueStatement& ForStatement::operator[](size_t index) const
{
	assert(index < statements.size());
	return statements[index];
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

llvm::SmallVectorImpl<UniqueStatement>::iterator ForStatement::begin()
{
	return statements.begin();
}

llvm::SmallVectorImpl<UniqueStatement>::const_iterator ForStatement::begin()
		const
{
	return statements.begin();
}

llvm::SmallVectorImpl<UniqueStatement>::iterator ForStatement::end()
{
	return statements.end();
}

llvm::SmallVectorImpl<UniqueStatement>::const_iterator ForStatement::end() const
{
	return statements.end();
}

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

Statement::iterator Statement::begin() { return iterator(this, this); }

Statement::iterator Statement::end() { return iterator(this, nullptr); }

class AssignmentsIteratorVisitor
{
	public:
	AssignmentsIteratorVisitor(stack<Statement*>* stack): statements(stack){};

	AssignmentStatement* operator()(ForStatement& forStatement)
	{
		for (auto i = forStatement.size(); i > 0; i--)
			statements->push(forStatement[i - 1].get());

		return nullptr;
	}

	AssignmentStatement* operator()(IfStatement& ifStatement)
	{
		for (auto i = ifStatement.size(); i > 0; i--)
		{
			auto& block = ifStatement[i - 1];

			for (auto j = block.size(); j > 0; j--)
				statements->push(block[j - 1].get());
		}

		return nullptr;
	}

	AssignmentStatement* operator()(AssignmentStatement& statement)
	{
		return &statement;
	}

	private:
	stack<Statement*>* statements;
};

AssignmentsIterator::AssignmentsIterator()
		: AssignmentsIterator(nullptr, nullptr)
{
}

AssignmentsIterator::AssignmentsIterator(Statement* root, Statement* start)
		: root(root)
{
	if (start != nullptr)
		statements.push(start);

	fetchNext();
}

AssignmentsIterator::operator bool() const { return !statements.empty(); }

bool AssignmentsIterator::operator==(const AssignmentsIterator& it) const
{
	return root == it.root && statements.size() == it.statements.size() &&
				 current == it.current;
}

bool AssignmentsIterator::operator!=(const AssignmentsIterator& it) const
{
	return !(*this == it);
}

AssignmentsIterator& AssignmentsIterator::operator++()
{
	fetchNext();
	return *this;
}

AssignmentsIterator AssignmentsIterator::operator++(int)
{
	auto temp = *this;
	fetchNext();
	return temp;
}

AssignmentsIterator::value_type& AssignmentsIterator::operator*()
{
	assert(current != nullptr);
	return *current;
}

const AssignmentsIterator::value_type& AssignmentsIterator::operator*() const
{
	assert(current != nullptr);
	return *current;
}

AssignmentsIterator::value_type* AssignmentsIterator::operator->()
{
	return current;
}

void AssignmentsIterator::fetchNext()
{
	bool found = false;

	while (!found && !statements.empty())
	{
		auto& statement = statements.top();
		statements.pop();
		auto* assignment =
				statement->visit(AssignmentsIteratorVisitor(&statements));

		if (assignment != nullptr)
		{
			current = assignment;
			found = true;
		}
	}

	if (!found)
		current = nullptr;
}
