#include <boost/algorithm/string.hpp>
#include <modelica/frontend/ast/Annotation.h>
#include <modelica/frontend/ast/Function.h>
#include <modelica/frontend/ast/Modification.h>

using namespace modelica::frontend;

Annotation::Annotation()
		: properties(std::make_shared<ClassModification>())
{
}

Annotation::Annotation(ClassModification properties)
		: properties(std::make_shared<ClassModification>(properties))
{
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
		if (argument.isA<ElementModification>())
		{
			const auto& elementModification = argument.get<ElementModification>();

			if (boost::iequals(elementModification.getName(), "inline") &&
					elementModification.hasModification())
			{
				const auto& modification = elementModification.getModification();
				return modification.getExpression().get<Constant>().get<BuiltInType::Boolean>();
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
		const auto& elementModification = argument.get<ElementModification>();

		if (boost::iequals(elementModification.getName(), "inverse"))
		{
			assert(elementModification.hasModification());
			const auto& modification = elementModification.getModification();
			assert(modification.hasClassModification());

			for (const auto& inverseDeclaration : modification.getClassModification())
			{
				const auto& inverseDeclarationFullExpression = inverseDeclaration.get<ElementModification>();
				assert(inverseDeclarationFullExpression.hasModification());
				const auto& callExpression = inverseDeclarationFullExpression.getModification();
				assert(callExpression.hasExpression());
				const auto& call = callExpression.getExpression().get<Call>();

				llvm::SmallVector<std::string, 3> args;

				for (const auto& arg : call)
					args.push_back(arg.get<ReferenceAccess>().getName());

				result.addInverse(inverseDeclarationFullExpression.getName(), call.getFunction().get<ReferenceAccess>().getName(), args);
			}
		}
	}

	return result;
}
