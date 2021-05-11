#include <modelica/frontend/ast/Modification.h>

using namespace modelica::frontend;

Modification::Modification(SourcePosition location,
													 std::unique_ptr<ClassModification> classModification)
		: ASTNodeCRTP<Modification>(ASTNodeKind::MODIFICATION, std::move(location)),
			classModification(std::move(classModification)),
			expression(llvm::None)
{
}

Modification::Modification(SourcePosition location,
													 std::unique_ptr<ClassModification> classModification,
													 std::unique_ptr<Expression> expression)
		: ASTNodeCRTP<Modification>(ASTNodeKind::MODIFICATION, std::move(location)),
			classModification(std::move(classModification)),
			expression(std::move(expression))
{
}

Modification::Modification(SourcePosition location,
													 std::unique_ptr<Expression> expression)
		: ASTNodeCRTP<Modification>(ASTNodeKind::MODIFICATION, std::move(location)),
			classModification(llvm::None),
			expression(std::move(expression))
{
}

Modification::Modification(const Modification& other)
		: ASTNodeCRTP<Modification>(static_cast<ASTNodeCRTP<Modification>&>(*this)),
			classModification(other.classModification.hasValue() ? llvm::Optional((*other.classModification)->clone()) : llvm::None),
			expression(other.expression.hasValue() ? llvm::Optional((*other.expression)->cloneExpression()) : llvm::None)
{
}

Modification::Modification(Modification&& other) = default;

Modification::~Modification() = default;

Modification& Modification::operator=(const Modification& other)
{
	Modification result(other);
	swap(*this, result);
	return *this;
}

Modification& Modification::operator=(Modification&& other) = default;

namespace modelica::frontend
{
	void swap(Modification& first, Modification& second)
	{
		swap(static_cast<impl::ASTNodeCRTP<Modification>&>(first),
				 static_cast<impl::ASTNodeCRTP<Modification>&>(second));

		using std::swap;
		swap(first.classModification, second.classModification);
		swap(first.expression, second.expression);
	}
}

void Modification::dump(llvm::raw_ostream& os, size_t indents) const
{
	// TODO
}

bool Modification::hasClassModification() const
{
	return classModification.hasValue();
}

ClassModification* Modification::getClassModification()
{
	assert(hasClassModification());
	return classModification->get();
}

const ClassModification* Modification::getClassModification() const
{
	assert(hasClassModification());
	return classModification->get();
}

bool Modification::hasExpression() const
{
	return expression.hasValue();
}

Expression* Modification::getExpression()
{
	assert(hasExpression());
	return expression->get();
}

const Expression* Modification::getExpression() const
{
	assert(hasExpression());
	return expression->get();
}

ClassModification::ClassModification(SourcePosition location,
																		 llvm::ArrayRef<std::unique_ptr<Argument>> arguments)
		: ASTNodeCRTP<ClassModification>(static_cast<ASTNodeCRTP<ClassModification>&>(*this))
{
	for (const auto& arg : arguments)
		this->arguments.push_back(arg->cloneArgument());
}

ClassModification::ClassModification(const ClassModification& other)
		: ASTNodeCRTP<ClassModification>(static_cast<ASTNodeCRTP<ClassModification>&>(*this))
{
	for (const auto& arg : other.arguments)
		this->arguments.push_back(arg->cloneArgument());
}

ClassModification::ClassModification(ClassModification&& other) = default;

ClassModification::~ClassModification() = default;

ClassModification& ClassModification::operator=(const ClassModification& other)
{
	ClassModification result(other);
	swap(*this, result);
	return *this;
}

ClassModification& ClassModification::operator=(ClassModification&& other) = default;

namespace modelica::frontend
{
	void swap(ClassModification& first, ClassModification& second)
	{
		swap(static_cast<impl::ASTNodeCRTP<ClassModification>&>(first),
				 static_cast<impl::ASTNodeCRTP<ClassModification>&>(second));

		using std::swap;
		impl::swap(first.arguments, second.arguments);
	}
}

void ClassModification::dump(llvm::raw_ostream& os, size_t indents) const
{
	// TODO
}

ClassModification::iterator ClassModification::begin()
{
	return arguments.begin();
}

ClassModification::const_iterator ClassModification::begin() const
{
	return arguments.begin();
}

ClassModification::iterator ClassModification::end()
{
	return arguments.end();
}

ClassModification::const_iterator ClassModification::end() const
{
	return arguments.end();
}

Argument::Argument(ASTNodeKind kind, SourcePosition location)
		: ASTNodeCRTP<Argument>(kind, std::move(location))
{
}

Argument::Argument(const Argument& other)
		: ASTNodeCRTP<Argument>(static_cast<ASTNodeCRTP<Argument>&>(*this))
{
}

Argument::Argument(Argument&& other) = default;

Argument::~Argument() = default;

Argument& Argument::operator=(const Argument& other)
{
	if (this != &other)
	{
		static_cast<ASTNodeCRTP<Argument>&>(*this) =
				static_cast<const ASTNodeCRTP<Argument>&>(other);
	}

	return *this;
}

Argument& Argument::operator=(Argument&& other) = default;

namespace modelica::frontend
{
	void swap(Argument& first, Argument& second)
	{
		swap(static_cast<impl::ASTNodeCRTP<Argument>&>(first),
				 static_cast<impl::ASTNodeCRTP<Argument>&>(second));

		using std::swap;
	}
}

ElementModification::ElementModification(SourcePosition location,
																				 bool each,
																				 bool final,
																				 llvm::StringRef name,
																				 std::unique_ptr<Modification>& modification)
		: ArgumentCRTP<ElementModification>(ASTNodeKind::ARGUMENT_ELEMENT_MODIFICATION, std::move(location)),
			each(each),
			final(final),
			name(std::move(name)),
			modification(modification->clone())
{
}

ElementModification::ElementModification(SourcePosition location,
																				 bool each,
																				 bool final,
																				 llvm::StringRef name)
		: ArgumentCRTP<ElementModification>(ASTNodeKind::ARGUMENT_ELEMENT_MODIFICATION, std::move(location)),
			each(each),
			final(final),
			name(std::move(name))
{
}

ElementModification::ElementModification(const ElementModification& other)
		: ArgumentCRTP<ElementModification>(static_cast<ArgumentCRTP<ElementModification>&>(*this))
{
}

ElementModification::ElementModification(ElementModification&& other) = default;

ElementModification::~ElementModification() = default;

ElementModification& ElementModification::operator=(const ElementModification& other)
{
	ElementModification result(other);
	swap(*this, result);
	return *this;
}

ElementModification& ElementModification::operator=(ElementModification&& other) = default;

namespace modelica::frontend
{
	void swap(ElementModification& first, ElementModification& second)
	{
		swap(static_cast<impl::ArgumentCRTP<ElementModification>&>(first),
				 static_cast<impl::ArgumentCRTP<ElementModification>&>(second));

		using std::swap;
	}
}

void ElementModification::dump(llvm::raw_ostream& os, size_t indents) const
{
	// TODO
}

bool ElementModification::hasEachProperty() const
{
	return each;
}

bool ElementModification::hasFinalProperty() const
{
	return final;
}

llvm::StringRef ElementModification::getName() const
{
	return name;
}

bool ElementModification::hasModification() const
{
	return modification.hasValue();
}

Modification* ElementModification::getModification()
{
	assert(hasModification());
	return modification->get();
}

const Modification* ElementModification::getModification() const
{
	assert(hasModification());
	return modification->get();
}

ElementReplaceable::ElementReplaceable(SourcePosition location)
		: ArgumentCRTP<ElementReplaceable>(ASTNodeKind::ARGUMENT_ELEMENT_REPLACEABLE, std::move(location))
{
}

ElementReplaceable::ElementReplaceable(const ElementReplaceable& other)
		: ArgumentCRTP<ElementReplaceable>(static_cast<ArgumentCRTP<ElementReplaceable>&>(*this))
{
}

ElementReplaceable::ElementReplaceable(ElementReplaceable&& other) = default;

ElementReplaceable::~ElementReplaceable() = default;

ElementReplaceable& ElementReplaceable::operator=(const ElementReplaceable& other)
{
	ElementReplaceable result(other);
	swap(*this, result);
	return *this;
}

ElementReplaceable& ElementReplaceable::operator=(ElementReplaceable&& other) = default;

namespace modelica::frontend
{
	void swap(ElementReplaceable& first, ElementReplaceable& second)
	{
		swap(static_cast<impl::ArgumentCRTP<ElementReplaceable>&>(first),
				 static_cast<impl::ArgumentCRTP<ElementReplaceable>&>(second));

		using std::swap;
	}
}

void ElementReplaceable::dump(llvm::raw_ostream& os, size_t indents) const
{
	// TODO
}

ElementRedeclaration::ElementRedeclaration(SourcePosition location)
		: ArgumentCRTP<ElementRedeclaration>(ASTNodeKind::ARGUMENT_ELEMENT_REDECLARATION, std::move(location))
{
}

ElementRedeclaration::ElementRedeclaration(const ElementRedeclaration& other)
		: ArgumentCRTP<ElementRedeclaration>(static_cast<ArgumentCRTP<ElementRedeclaration>&>(*this))
{
}

ElementRedeclaration::ElementRedeclaration(ElementRedeclaration&& other) = default;

ElementRedeclaration::~ElementRedeclaration() = default;

ElementRedeclaration& ElementRedeclaration::operator=(const ElementRedeclaration& other)
{
	ElementRedeclaration result(other);
	swap(*this, result);
	return *this;
}

ElementRedeclaration& ElementRedeclaration::operator=(ElementRedeclaration&& other) = default;

namespace modelica::frontend
{
	void swap(ElementRedeclaration& first, ElementRedeclaration& second)
	{
		swap(static_cast<impl::ArgumentCRTP<ElementRedeclaration>&>(first),
				 static_cast<impl::ArgumentCRTP<ElementRedeclaration>&>(second));

		using std::swap;
	}
}

void ElementRedeclaration::dump(llvm::raw_ostream& os, size_t indents) const
{
	// TODO
}
