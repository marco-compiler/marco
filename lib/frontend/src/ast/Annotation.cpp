#include <boost/algorithm/string.hpp>
#include <modelica/frontend/ast/Annotation.h>
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
 * Inline           |
 *              expression
 *                true
 */
bool Annotation::getInlineProperty()
{
	for (const auto& argument : *properties)
	{
		if (argument.isA<ElementModification>())
		{
			const auto& elementModification = argument.get<ElementModification>();

			if (boost::iequals(elementModification.getName(), "Inline") &&
					elementModification.hasModification())
			{
				const auto& modification = elementModification.getModification();
				return modification.getExpression().get<Constant>().get<BuiltInType::Boolean>();
			}
		}
	}

	return false;
}
