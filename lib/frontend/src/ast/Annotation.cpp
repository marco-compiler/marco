#include <boost/algorithm/string.hpp>
#include <modelica/frontend/AST.h>

using namespace modelica::frontend;

Annotation::Annotation(SourcePosition location)
		: ASTNodeCRTP<Annotation>(ASTNodeKind::ANNOTATION, std::move(location))
{
}

Annotation::Annotation(SourcePosition location,
											 std::unique_ptr<ClassModification> properties)
		: ASTNodeCRTP<Annotation>(ASTNodeKind::ANNOTATION, std::move(location)),
			properties(std::move(properties))
{
}

Annotation::Annotation(const Annotation& other)
		: ASTNodeCRTP<Annotation>(static_cast<ASTNodeCRTP<Annotation>&>(*this)),
			properties(other.properties->clone())
{
}

Annotation::Annotation(Annotation&& other) = default;

Annotation::~Annotation() = default;

Annotation& Annotation::operator=(const Annotation& other)
{
	Annotation result(other);
	swap(*this, result);
	return *this;
}

Annotation& Annotation::operator=(Annotation&& other) = default;

namespace modelica::frontend
{
	void swap(Annotation& first, Annotation& second)
	{
		swap(static_cast<impl::ASTNodeCRTP<Annotation>&>(first),
				 static_cast<impl::ASTNodeCRTP<Annotation>&>(second));

		using std::swap;
		swap(first.properties, second.properties);
	}
}

void Annotation::dump(llvm::raw_ostream& os, size_t indents) const
{
	// TODO
}

/**
 * Inline property AST structure:
 *
 *          class-modification
 *                  |
 *            argument-list
 *             /         \
 *       argument         ...
 *          |
 *  element-modification
 *    /           \
 *  name        modification
 * inline           |
 *              expression
 *                true
 */
bool Annotation::getInlineProperty() const
{
	for (const auto& argument : *properties)
	{
		if (argument->isa<ElementModification>())
		{
			const auto& elementModification = argument->cast<ElementModification>();

			if (boost::iequals(elementModification->getName(), "inline") &&
					elementModification->hasModification())
			{
				const auto& modification = elementModification->getModification();
				return modification->getExpression()->cast<Constant>()->get<BuiltInType::Boolean>();
			}
		}
	}

	return false;
}

InverseFunctionAnnotation Annotation::getInverseFunctionAnnotation() const
{
	InverseFunctionAnnotation result;

	for (const auto& argument : *properties)
	{
		const auto& elementModification = argument->cast<ElementModification>();

		if (boost::iequals(elementModification->getName().str(), "inverse"))
		{
			assert(elementModification->hasModification());
			const auto& modification = elementModification->getModification();
			assert(modification->hasClassModification());

			for (const auto& inverseDeclaration : *modification->getClassModification())
			{
				const auto& inverseDeclarationFullExpression = inverseDeclaration->cast<ElementModification>();
				assert(inverseDeclarationFullExpression->hasModification());
				const auto& callExpression = inverseDeclarationFullExpression->getModification();
				assert(callExpression->hasExpression());
				const auto& call = callExpression->getExpression()->cast<Call>();

				llvm::SmallVector<std::string, 3> args;

				for (const auto& arg : *call)
					args.push_back(arg->cast<ReferenceAccess>()->getName().str());

				result.addInverse(inverseDeclarationFullExpression->getName().str(), call->getFunction()->cast<ReferenceAccess>()->getName(), args);
			}
		}
	}

	return result;
}
