#include "modelica/omcToModel/OmcToModelPass.hpp"

#include "modelica/Dumper/Dumper.hpp"

using namespace modelica;
using namespace std;
using namespace llvm;

ModExp modExpFromBinaryExp(const BinaryExpr* exp, const EntryModel& model);

ModExp modExpFromASTExp(const Expr* ASTexp, const EntryModel& model)
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
		return modExpFromBinaryExp(binaryExp, model);
	}

	if (isa<ComponentReferenceExpr>(ASTexp))
	{
		auto ref = dyn_cast<ComponentReferenceExpr>(ASTexp);
		auto refName = ref->getName();

		auto type = model.getVar(refName).getInit().getModType();

		return ModExp(refName, type);
	}

	assert(false && "Unandled expression");	 // NOLINT
	return ModExp::constExp(0);
}

ModExp modExpFromBinaryExp(const BinaryExpr* exp, const EntryModel& model)
{
	auto left = modExpFromASTExp(exp->getLeftHand(), model);
	auto right = modExpFromASTExp(exp->getRightHand(), model);
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

		default:
			assert(false && "unandled bin op");	 // NOLINT
			return left;
	}
	assert(false && "unreachable");	 // NOLINT
	return left;
}

vector<size_t> dimFromArraySubscription(
		const ArraySubscriptionExpr* subscription)
{
	vector<size_t> toReturn;

	for (auto& iter : subscription->children())
	{
		if (iter == nullptr)
			continue;
		assert(isa<IntLiteralExpr>(iter.get()));
		auto dim = dyn_cast<IntLiteralExpr>(iter.get());
		toReturn.push_back(dim->getValue());
	}

	return toReturn;
}

bool insertArray(
		StringRef varName,
		vector<size_t> dimensions,
		const ElementModification* innerMod,
		EntryModel& model)
{
	for (const auto& child : *innerMod)
	{
		if (!isa<SimpleModification>(child.get()))
			continue;

		auto simpleMod = dyn_cast<SimpleModification>(child.get());
		if (!isa<ComponentReferenceExpr>(simpleMod->getExpression()))
			continue;

		auto ref = dyn_cast<ComponentReferenceExpr>(simpleMod->getExpression());

		auto intExp = model.getVar(ref->getName()).getInit();

		vector<int> intDim;
		for (auto i : dimensions)
			intDim.push_back(i);
		ModConst<int> constDim(intDim);
		ModExp arg(constDim, ModType(BultinModTypes::INT, intDim.size()));
		ModExp sourceExp(ref->getName(), intExp.getModType());

		ModType outType(intExp.getModType().getBuiltin(), dimensions);
		return !model.emplaceVar(
				varName.str(),
				ModExp(ModCall("fill", { move(sourceExp), move(arg) }, outType)));
	}
	return false;
}

unique_ptr<ComponentClause> OmcToModelPass::visit(
		unique_ptr<ComponentClause> clause)
{
	auto decl = dyn_cast<ComponentDeclaration>(clause->getComponent(0));
	string varName = decl->getIdent();
	auto type = decl->getIdent();
	vector<size_t> dimensions = { 1 };

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
			vector<int> intDim;
			for (auto i : dimensions)
				intDim.push_back(i);
			ModConst<int> constDim(intDim);
			ModExp arg(constDim, ModType(BultinModTypes::INT, intDim.size()));
			ModExp sourceExp = ModExp::constExp<float>(0);

			ModType outType(BultinModTypes::FLOAT, dimensions);
			auto val = model.emplaceVar(
					varName,
					ModExp(ModCall("fill", { move(sourceExp), move(arg) }, outType)));
			return clause;
		}
	}
	outs() << varName << "\n";
	for (auto i = model.varbegin(); i != model.varend(); i++)
	{
		outs() << i->first() << " = ";
		i->second.getInit().dump();
		outs() << "\n";
	}

	return clause;
}

bool OmcToModelPass::handleSimpleMod(
		StringRef name, const SimpleModification& mod)
{
	return !model.emplaceVar(
			name.str(), modExpFromASTExp(mod.getExpression(), model));
}
