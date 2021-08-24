#include <boost/algorithm/string.hpp>
#include <marco/frontend/AST.h>

using namespace marco::frontend;

Annotation::Annotation(SourceRange location)
		: ASTNode(std::move(location))
{
}

Annotation::Annotation(SourceRange location,
											 std::unique_ptr<ClassModification> properties)
		: ASTNode(std::move(location)),
			properties(std::move(properties))
{
}

Annotation::Annotation(const Annotation& other)
		: ASTNode(other),
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

namespace marco::frontend
{
	void swap(Annotation& first, Annotation& second)
	{
		swap(static_cast<ASTNode&>(first), static_cast<ASTNode&>(second));

		using std::swap;
		swap(first.properties, second.properties);
	}
}

void Annotation::print(llvm::raw_ostream& os, size_t indents) const
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
		if (auto* elementModification = argument->dyn_get<ElementModification>())
		{
			if (boost::iequals(elementModification->getName(), "inline") &&
					elementModification->hasModification())
			{
				const auto& modification = elementModification->getModification();
				return modification->getExpression()->get<Constant>()->get<BuiltInType::Boolean>();
			}
		}
	}

	return false;
}

bool Annotation::hasDerivativeAnnotation() const
{
	for (const auto& argument : *properties)
		if (auto* elementModification = argument->dyn_get<ElementModification>())
			if (elementModification->getName() == "derivative")
				return true;

	return false;
}

/**
 * Derivative property AST structure:
 *
 *          class-modification
 *                  |
 *            argument-list
 *             /         \
 *       argument         ...
 *          |
 *  element-modification
 *    /                \
 *  name            modification
 * derivative       /         \
 *             expression   class-modification
 *               foo                |
 *                           argument-list
 *                             /       \
 *                        argument     ...
 *                           |
 *                   element-modification
 *                    /              \
 *                  name          modification
 *                  order             |
 *                                expression
 *                                  <int>
 */
DerivativeAnnotation Annotation::getDerivativeAnnotation() const
{
	assert(hasDerivativeAnnotation());

	for (const auto& argument : *properties)
	{
		if (auto* elementModification = argument->dyn_get<ElementModification>())
		{
			if (elementModification->getName() != "derivative")
				continue;

			auto* modification = elementModification->getModification();
			auto name = modification->getExpression()->get<ReferenceAccess>()->getName();
			unsigned int order = 1;

			if (modification->hasClassModification())
				for (const auto& derivativeArgument : *modification->getClassModification())
					if (auto* derivativeModification = derivativeArgument->dyn_get<ElementModification>())
						if (derivativeModification->getName() == "order")
							order = derivativeModification->getModification()->getExpression()->get<Constant>()->get<BuiltInType::Integer>();

			return DerivativeAnnotation(name, order);
		}
	}

	// Normally unreachable
	return DerivativeAnnotation("", 1);
}

InverseFunctionAnnotation Annotation::getInverseFunctionAnnotation() const
{
	InverseFunctionAnnotation result;

	for (const auto& argument : *properties)
	{
		if (auto* elementModification = argument->dyn_get<ElementModification>())
		{
			if (boost::iequals(elementModification->getName().str(), "inverse"))
			{
				assert(elementModification->hasModification());
				const auto& modification = elementModification->getModification();
				assert(modification->hasClassModification());

				for (const auto& inverseDeclaration : *modification->getClassModification())
				{
					const auto& inverseDeclarationFullExpression = inverseDeclaration->get<ElementModification>();
					assert(inverseDeclarationFullExpression->hasModification());
					const auto& callExpression = inverseDeclarationFullExpression->getModification();
					assert(callExpression->hasExpression());
					const auto* call = callExpression->getExpression()->get<Call>();

					llvm::SmallVector<std::string, 3> args;

					for (const auto& arg : *call)
						args.push_back(arg->get<ReferenceAccess>()->getName().str());

					result.addInverse(inverseDeclarationFullExpression->getName().str(),
														call->getFunction()->get<ReferenceAccess>()->getName(),
														args);
				}
			}
		}
	}

	return result;
}
