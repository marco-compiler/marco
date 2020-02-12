#include "modelica/omcToModel/OmcToModelPass.hpp"

#include "modelica/Dumper/Dumper.hpp"
#include "modelica/model/Assigment.hpp"
#include "modelica/model/ModExp.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;

ModExp modExpFromBinaryExp(
		const BinaryExpr* exp,
		const EntryModel& model,
		const StringMap<int>& inductionLookUpTable);
ModExp modExpFromBinaryExp(
		const UnaryExpr* exp,
		const EntryModel& model,
		const StringMap<int>& inductionLookUpTable);

ModExp modExpFromASTExp(
		const Expr* ASTexp,
		const EntryModel& model,
		const StringMap<int>& inductionLookUpTable)
{
	if (isa<BoolLiteralExpr>(ASTexp))
	{
		auto boolLit = dyn_cast<BoolLiteralExpr>(ASTexp);
		return ModExp::constExp(boolLit->getValue());
	}

	if (isa<IntLiteralExpr>(ASTexp))
	{
		auto intLit = dyn_cast<IntLiteralExpr>(ASTexp);
		return ModExp::constExp(intLit->getValue());
	}

	if (isa<FloatLiteralExpr>(ASTexp))
	{
		auto intLit = dyn_cast<FloatLiteralExpr>(ASTexp);
		return ModExp::constExp<float>(intLit->getValue());
	}

	if (isa<BinaryExpr>(ASTexp))
	{
		auto binaryExp = dyn_cast<BinaryExpr>(ASTexp);
		return modExpFromBinaryExp(binaryExp, model, inductionLookUpTable);
	}

	if (isa<UnaryExpr>(ASTexp))
	{
		auto unaryExp = dyn_cast<UnaryExpr>(ASTexp);
		return modExpFromBinaryExp(unaryExp, model, inductionLookUpTable);
	}

	if (isa<ComponentReferenceExpr>(ASTexp))
	{
		auto ref = dyn_cast<ComponentReferenceExpr>(ASTexp);
		auto refName = ref->getName();
		if (const auto& i = inductionLookUpTable.find(refName);
				i != inductionLookUpTable.end())
			return ModExp::induction(ModExp(ModConst(i->second)));

		auto type = model.getVar(refName).getInit().getModType();

		return ModExp(refName, type);
	}

	if (isa<ArraySubscriptionExpr>(ASTexp))
	{
		auto subScript = dyn_cast<ArraySubscriptionExpr>(ASTexp);
		auto sourceArr = subScript->getSourceExpr();

		auto exp = modExpFromASTExp(sourceArr, model, inductionLookUpTable);

		for (int a = 0; a < subScript->subscriptedDimensionsCount(); a++)
		{
			auto subScriptExp = modExpFromASTExp(
					subScript->getSubscriptionExpr(a), model, inductionLookUpTable);
			auto oneBasedSubscript = subScriptExp - ModExp::constExp<int>(1);

			exp = ModExp::at(move(exp), move(oneBasedSubscript));
		}
		return exp;
	}

	if (isa<DerFunctionCallExpr>(ASTexp))
	{
		auto call = dyn_cast<DerFunctionCallExpr>(ASTexp);
		auto par = modExpFromASTExp(call->at(0), model, inductionLookUpTable);

		auto type = par.getModType();
		return ModExp(ModCall("der", { move(par) }, move(type)));
	}

	assert(false && "Unandled expression");	 // NOLINT
	return ModExp::constExp(0);
}
ModExp modExpFromBinaryExp(
		const UnaryExpr* exp,
		const EntryModel& model,
		const StringMap<int>& inductionLookUpTable)
{
	auto left = modExpFromASTExp(exp->getOperand(), model, inductionLookUpTable);

	switch (exp->getOpCode())
	{
		case UnaryExprOp::Minus:
			return ModExp::negate(move(left));
		default:
			assert(false && "unandled bin op");	 // NOLINT
			return left;
	}
	assert(false && "unreachable");	 // NOLINT
	return left;
}

ModExp modExpFromBinaryExp(
		const BinaryExpr* exp,
		const EntryModel& model,
		const StringMap<int>& inductionLookUpTable)
{
	auto left = modExpFromASTExp(exp->getLeftHand(), model, inductionLookUpTable);
	auto right =
			modExpFromASTExp(exp->getRightHand(), model, inductionLookUpTable);
	switch (exp->getOpCode())
	{
		case BinaryExprOp::Sum:
			return left + right;
		case BinaryExprOp::Subtraction:
			return left - right;
		case BinaryExprOp::Multiply:
			return left * right;
		case BinaryExprOp::Division:
			return left / right;
		case BinaryExprOp::GreatureEqual:
			return left >= right;
		case BinaryExprOp::Greater:
			return left > right;
		case BinaryExprOp::LessEqual:
			return left <= right;
		case BinaryExprOp::Less:
			return left < right;
		case BinaryExprOp::PowerOf:
			return ModExp::elevate(left, right);

		default:
			assert(false && "unandled bin op");	 // NOLINT
			return left;
	}
	assert(false && "unreachable");	 // NOLINT
	return left;
}

SmallVector<size_t, 3> dimFromArraySubscription(
		const ArraySubscriptionExpr* subscription)
{
	SmallVector<size_t, 3> toReturn;

	for (auto& iter : subscription->children())
	{
		if (iter == nullptr)
			continue;
		assert(isa<IntLiteralExpr>(iter.get()));	// NOLINT
		auto dim = dyn_cast<IntLiteralExpr>(iter.get());
		toReturn.push_back(dim->getValue());
	}

	return toReturn;
}

bool insertArray(
		StringRef varName,
		SmallVector<size_t, 3> dimensions,
		const ElementModification* innerMod,
		EntryModel& model)
{
	for (const auto& child : *innerMod)
	{
		if (!isa<SimpleModification>(child.get()))
			continue;

		auto simpleMod = dyn_cast<SimpleModification>(child.get());
		if (!isa<FloatLiteralExpr>(simpleMod->getExpression()))
			continue;

		auto ref = dyn_cast<FloatLiteralExpr>(simpleMod->getExpression());

		ModExp sourceExp(ModConst(ref->getValue()));

		ModType outType(sourceExp.getModType().getBuiltin(), dimensions);
		return !model.emplaceVar(
				varName.str(), ModExp(ModCall("fill", { move(sourceExp) }, outType)));
	}
	return false;
}

static InductionVar inductionFromEq(const ForEquation* eq)
{
	auto forExp = eq->getForExpression(0);
	assert(isa<RangeExpr>(forExp));	 // NOLINT

	auto rangeExp = dyn_cast<RangeExpr>(forExp);
	assert(isa<IntLiteralExpr>(rangeExp->getStart()));	// NOLINT
	assert(isa<IntLiteralExpr>(rangeExp->getStop()));		// NOLINT
	auto begin = dyn_cast<IntLiteralExpr>(rangeExp->getStart());
	auto end = dyn_cast<IntLiteralExpr>(rangeExp->getStop());

	return InductionVar(begin->getValue(), end->getValue() + 1);
}

static ModEquation handleEq(
		const SimpleEquation* eq,
		const StringMap<int>& inductionLookUpTable,
		const SmallVector<InductionVar, 3>& inds,
		const EntryModel& model)
{
	auto left = modExpFromASTExp(eq->getLeftHand(), model, inductionLookUpTable);
	auto right =
			modExpFromASTExp(eq->getRightHand(), model, inductionLookUpTable);
	return ModEquation(move(left), move(right), inds);
}

unique_ptr<SimpleEquation> OmcToModelPass::visit(
		unique_ptr<SimpleEquation> decl)
{
	if (forEqNestingLevel != 0)
		return decl;
	StringMap<int> inductionLookUpTable;
	SmallVector<InductionVar, 3> inds;
	model.addEquation(handleEq(decl.get(), inductionLookUpTable, inds, model));
	return decl;
}

unique_ptr<ForEquation> OmcToModelPass::visit(unique_ptr<ForEquation> eq)
{
	if (forEqNestingLevel++ != 0)
		return eq;

	SmallVector<InductionVar, 3> inds;
	StringMap<int> inductionLookUpTable;
	const Equation* nav = eq.get();
	const ForEquation* previous = eq.get();
	int indVars = 0;
	while (isa<ForEquation>(nav))
	{
		auto ptr = dyn_cast<ForEquation>(nav);
		inds.push_back(inductionFromEq(ptr));
		inductionLookUpTable[ptr->getNames()[0]] = indVars;
		indVars++;
		previous = ptr;
		nav = ptr->getEquation(0);
	}

	for (size_t a = 0; a < previous->equationsCount(); a++)
	{
		auto eq = previous->getEquation(a);
		if (isa<SimpleEquation>(eq))
			model.addEquation(handleEq(
					dyn_cast<SimpleEquation>(eq), inductionLookUpTable, inds, model));
	}

	return eq;
}

unique_ptr<ComponentClause> OmcToModelPass::visit(
		unique_ptr<ComponentClause> clause)
{
	auto decl = dyn_cast<ComponentDeclaration>(clause->getComponent(0));
	string varName = decl->getIdent();
	auto type = decl->getIdent();
	SmallVector<size_t, 3> dimensions = { 1 };

	for (const auto& child : *clause)
	{
		if (isa<ArraySubscriptionDecl>(child.get()))
		{
			auto dim = cast<ArraySubscriptionDecl>(child.get());
			auto subsc = dim->getArraySubscript();
			if (subsc == nullptr)
				continue;

			dimensions =
					dimFromArraySubscription(dyn_cast<ArraySubscriptionExpr>(subsc));
		}
	}

	for (const auto& child : *decl)
	{
		if (isa<SimpleModification>(child))
		{
			handleSimpleMod(varName, *dyn_cast<SimpleModification>(child.get()));
			break;
		}

		if (isa<OverridingClassModification>(child))
		{
			auto mod = dyn_cast<OverridingClassModification>(child.get());
			auto simpleMod = mod->getSimpleModification();
			handleSimpleMod(varName, *dyn_cast<SimpleModification>(simpleMod));
			break;
		}
		if (isa<ClassModification>(child))
		{
			auto mod = dyn_cast<ClassModification>(child.get());
			for (const auto& elmMod : *mod)
			{
				if (!isa<ElementModification>(elmMod))
					continue;

				auto innerMod = dyn_cast<ElementModification>(elmMod.get());
				if (innerMod != nullptr && !innerMod->getName().empty() &&
						innerMod->getName()[0] == "start")
				{
					insertArray(varName, dimensions, innerMod, model);
					return clause;
				}
			}
			ModExp sourceExp = ModExp::constExp<float>(0);

			ModType outType(BultinModTypes::FLOAT, dimensions);
			auto val = model.emplaceVar(
					varName, ModExp(ModCall("fill", { move(sourceExp) }, outType)));
			return clause;
		}
	}

	return clause;
}

bool OmcToModelPass::handleSimpleMod(
		StringRef name, const SimpleModification& mod)
{
	StringMap<int> inductionLookUpTable;
	return !model.emplaceVar(
			name.str(),
			modExpFromASTExp(mod.getExpression(), model, inductionLookUpTable),
			false);
}
