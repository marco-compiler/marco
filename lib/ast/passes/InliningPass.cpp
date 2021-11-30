#include <cstdio>
#include <marco/ast/AST.h>
#include <marco/ast/Errors.h>
#include <marco/ast/passes/InliningPass.h>
#include <numeric>
#include <queue>
#include <stack>
#include <sstream>

using namespace marco;
using namespace marco::ast;


InlineExpanser::InlineExpanser()
{
}

template<>
llvm::Error InlineExpanser::run<Class>(Class& cls)
{	
	return cls.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return run<deconst>(cls);
	});
}

llvm::Error InlineExpanser::run(llvm::ArrayRef<std::unique_ptr<Class>> classes)
{
	//InlineExpanser starting point
	for (const auto& cls : classes)
		if (auto error = run<Class>(*cls); error)
			return error;

	return llvm::Error::success();
}

template<>
llvm::Error InlineExpanser::run<PartialDerFunction>(Class& cls)
{
	return llvm::Error::success();
}

template<>
llvm::Error InlineExpanser::run<StandardFunction>(Class& cls)
{
	// SymbolTable::ScopeTy varScope(symbolTable);
	// auto* function = cls.get<StandardFunction>();

	// // Populate the symbol table
	// symbolTable.insert(function->getName(), Symbol(cls));

	// for (const auto& member : function->getMembers())
	// 	symbolTable.insert(member->getName(), Symbol(*member));

	return llvm::Error::success();
}


template<>
llvm::Error InlineExpanser::run<Model>(Class& cls)
{
	auto* model = cls.get<Model>();
	// SymbolTable::ScopeTy varScope(symbolTable);

	// // Populate the symbol table
	// symbolTable.insert(model->getName(), Symbol(cls));

	// for (auto& member : model->getMembers())
	// 	symbolTable.insert(member->getName(), Symbol(*member));


	// handle the members declaration exploding the inlineable record members
	// e.g. Complex x;	->	Real 'x.re';
	//						Real 'x.im';						
	auto& members = model->getMembers_mut();

	llvm::SmallVector<std::unique_ptr<Member>,3> members_to_add;

	for(auto it = members.begin(); it!=members.end();){
		bool to_remove=false;
		auto &member = **it;
		auto type = member.getType();

		if(type.isa<Record*>() ){
			const Record* record = type.get<Record*>();

			if(record->shouldBeInlined()){
				to_remove = true;

				for(const auto &m:(*record)){

					auto new_type = m->getType();
					
					//todo: add member + record's member dimensions (now it's assumed the record doesn't use arrays)
					new_type.setDimensions(type.getDimensions()); 

					auto new_m = Member::build(
						member.getLocation(), 
						(member.getName()+"."+m->getName()).str(),
						new_type,
						member.getTypePrefix());//todo handle also init and public section

					members_to_add.emplace_back(std::move(new_m));
				}
			}
		}

		if(to_remove){
			it = members.erase(it);
		}else{
			it++;
		}
	}

	for(auto &it:members_to_add)
		members.emplace_back(std::move(it));
	
	// and, similarly, handle the equations exploding the inlineable records ones
	llvm::SmallVector<std::unique_ptr<Equation>, 3> new_equations;

	for (auto& eq : model->getEquations()){
		auto error = run(*eq);
		if (error)
			return error;

		auto *lhs = eq->getLhsExpression();
		auto *rhs = eq->getRhsExpression();

		assert(lhs->getType() == rhs->getType());

		if(lhs->isa<RecordInstance>() && rhs->isa<RecordInstance>()){
			auto lt = lhs->get<RecordInstance>();
			auto rt = rhs->get<RecordInstance>();

			const Record* record = lt->getRecordType();

			llvm::SmallVector<std::unique_ptr<Expression>, 3> destinations;
			llvm::SmallVector<std::unique_ptr<Expression>, 3> expressions;


			for (const auto &member : (*record)){
				destinations.push_back(lt->getMemberValue(member->getName()).clone());
				expressions.push_back(rt->getMemberValue(member->getName()).clone());
			}

			eq = Equation::build(eq->getLocation(),std::move(destinations[0]),std::move(expressions[0]));

			for(size_t i=1; i<destinations.size(); ++i){
				auto e = Equation::build(eq->getLocation(),std::move(destinations[i]),std::move(expressions[i]));
				new_equations.emplace_back(std::move(e));
			}
		}
	}

	for(auto &e : new_equations)
		model->addEquation(std::move(e));
		
	//todo : explode inlinable records related ones, just as normal equations
	for (auto& eq : model->getForEquations())
		if (auto error = run(*eq); error)
			return error;

	for (auto& algorithm : model->getAlgorithms())
		if (auto error = run(*algorithm); error)
			return error;


	return llvm::Error::success();
}

template<>
llvm::Error InlineExpanser::run<Package>(Class& cls)
{
	SymbolTable::ScopeTy varScope(symbolTable);
	auto* package = cls.get<Package>();

	// // Populate the symbol table
	symbolTable.insert(package->getName(), Symbol(cls));

	for (auto& innerClass : *package)
		symbolTable.insert(innerClass->getName(), Symbol(*innerClass));

	for (auto& innerClass : *package)
		if (auto error = run<Class>(*innerClass); error)
			return error;

	return llvm::Error::success();
}

template<>
llvm::Error InlineExpanser::run<Record>(Class& cls)
{
	SymbolTable::ScopeTy varScope(symbolTable);
	auto* record = cls.get<Record>();

	// Populate the symbol table
	symbolTable.insert(record->getName(), Symbol(cls));

	for (auto& member : *record)
		symbolTable.insert(member->getName(), Symbol(*member));

	// for (auto& member : *record)
	// 	if (auto error = run(*member); error)
	// 		return error;

	return llvm::Error::success();
}

llvm::Error InlineExpanser::run(Equation& equation)
{
	if (auto error = run<Expression>(*equation.getLhsExpression()); error)
		return error;

	if (auto error = run<Expression>(*equation.getRhsExpression()); error)
		return error;

	return llvm::Error::success();
}

llvm::Error InlineExpanser::run(ForEquation& forEquation)
{
	//todo : handle as the normal equations (e.g. decomposing records related ones, thus adding new equations. see: run<Model>(...) function)
	SymbolTable::ScopeTy varScope(symbolTable);

	for (auto& ind : forEquation.getInductions())
	{
		symbolTable.insert(ind->getName(), Symbol(*ind));

		if (auto error = run<Expression>(*ind->getBegin()); error)
			return error;

		if (auto error = run<Expression>(*ind->getEnd()); error)
			return error;
	}

	if (auto error = run(*forEquation.getEquation()); error)
		return error;

	return llvm::Error::success();
}

template<>
llvm::Error InlineExpanser::run<Expression>(Expression& expression)
{
	return expression.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return run<deconst>(expression);
	});
}

template<>
llvm::Error InlineExpanser::run<Array>(Expression& expression)
{
	return llvm::Error::success();
}


template<>
llvm::Error InlineExpanser::run<Call>(Expression& expression)
{
	assert(expression.isa<Call>());
	auto *call = expression.get<Call>();

	auto actual_args = call->getArgs();
	for (auto& exp : actual_args)
	{
		if (auto error = run<Expression>(*exp); error)
			return error;
	}

	llvm::StringRef functionName = call->getFunction()->get<ReferenceAccess>()->getName();


	if (!symbolTable.count(functionName))
		return llvm::Error::success();
	
	auto* cls = symbolTable.lookup(functionName).dyn_get<Class>();

	if(!cls)
		return llvm::Error::success();
	
	
	if ( auto* standardFunction = cls->dyn_get<StandardFunction>(); standardFunction && standardFunction->shouldBeInlined())
	{
		auto formal_args = standardFunction->getArgs();

		assert(formal_args.size() == actual_args.size()); //todo check default params

		if(standardFunction->isCustomRecordConstructor())
		{
			// todo handle default args, now it only works if the custom constructor has the same signature as the default one
			auto recordType = standardFunction->getType().getResults()[0];

			expression = *Expression::recordInstance(expression.getLocation(),recordType,actual_args);
			return llvm::Error::success();
		}

		auto algs = standardFunction->getAlgorithms();
		assert(algs.size()==1 && "only functions with one algorithm (and with one statement) are inline-able.");
		auto stms = algs[0]->getBody();
		assert(stms.size()==1 && "only functions with one statement are inline-able.");

		if(auto assignment = stms[0]->dyn_get<AssignmentStatement>())
		{
			//inlineable function -> procede to substitute the call with the body of the function
			
			TranslationTable::ScopeTy varScope(translationTable);

			// populate the translation table (the formal args need to be replaced with the actual args)
			for(size_t i=0; i<formal_args.size();++i){
				translationTable.insert(formal_args[i]->getName(),actual_args[i].get());
			}

			// we need to visit the new expression to handle nested calls
			Expression new_expression(*(assignment->getExpression()));
			if(auto error=run<Expression>(new_expression);error)
				return error;

			expression = new_expression;

			return llvm::Error::success();
		}

		assert(false && "only functions containing one assignment statement are inlinable");
	}
	else if(auto *record = cls->dyn_get<Record>();record && record->shouldBeInlined())
	{
		// the default record constructor is function with the same id as the record
		expression = *Expression::recordInstance(expression.getLocation(),Type(record),actual_args);
		return llvm::Error::success();
	}
	
	return llvm::Error::success();
}

template<>
llvm::Error InlineExpanser::run<Constant>(Expression& expression)
{
	return llvm::Error::success();
}

template<>
llvm::Error InlineExpanser::run<Operation>(Expression& expression)
{
	auto* operation = expression.get<Operation>();

	if(operation->getOperationKind() == OperationKind::subscription)
	{
		auto lhs_type = operation->getArg(0)->getType();
		if( lhs_type.isa<Record*>() && lhs_type.get<Record*>()->shouldBeInlined() ){
			//skipping subscription of record array : it's gonna be handled from the memberLookup operation
			return llvm::Error::success();
		}
	}

	if(operation->getOperationKind() == OperationKind::memberLookup)
	{
		assert(operation->size() == 2);

		Expression lhs = *operation->getArg(0);
		Expression rhs = *operation->getArg(1);
		
		// if lhs contains a subscript we need to handle it after the inlining of the member, 
		// since the record members are exploded 
		//e.g.   Complex[N] x;  		Real[N] x.re;
		//		 				->		Real[N] x.im;
		//		 ...					...
		//		 x[1].re	    		'x.re'[1]
		llvm::Optional<Expression> subscription;
		if(lhs.isa<Operation>() && lhs.get<Operation>()->getOperationKind() == OperationKind::subscription){
			subscription = lhs;
			lhs = *lhs.get<Operation>()->getArg(0);
		}

		if (auto error = run<Expression>(lhs); error)
				return error;

		if(lhs.isa<Operation>() && lhs.get<Operation>()->getOperationKind() == OperationKind::subscription){
			if(subscription){
				//todo handle multiple subscriptions : collapse them in one
				assert(false);
			}

			subscription = lhs;
			lhs = *lhs.get<Operation>()->getArg(0);
		}

		auto lhs_type = lhs.getType();
		
		if(lhs_type.isa<Record*>() && rhs.isa<ReferenceAccess>() ){
			const Record* record = lhs_type.get<Record*>();

			if(record->shouldBeInlined()){
				auto memberName = rhs.get<ReferenceAccess>()->getName();

				if(lhs.isa<ReferenceAccess>()){
					// transforming member lookup to the exploded new member
					// e.g. x.re -> 'x.re'
					lhs = *Expression::reference(
						expression.getLocation(),
						operation->getType(),
						(lhs.get<ReferenceAccess>()->getName()+"."+rhs.get<ReferenceAccess>()->getName()).str());
					
				} else if (lhs.isa<RecordInstance>()){
					//e.g. Complex(a,b).re  ->  a
					auto instance = lhs.get<RecordInstance>();
					lhs = instance->getMemberValue(memberName);
				} else{
					assert(false && "unexpected lhs in member lookup");
				}

				if(subscription){
					**(subscription->get<Operation>()->begin()) = lhs;
					lhs = *subscription;
					lhs.setType(expression.getType()); //todo: handle multiple subscript(?)
					
					if (auto error = run<Expression>(lhs); error)
						return error;
				}
			}

			expression = lhs;

			return llvm::Error::success();

		}else{
			assert(false && "member lookup is only supported on records.");
		}
	}

	for (auto& arg : operation->getArguments())
		if (auto error = run<Expression>(*arg); error)
			return error;

	return llvm::Error::success();
}

template<>
llvm::Error InlineExpanser::run<ReferenceAccess>(Expression& expression)
{
	auto* reference = expression.get<ReferenceAccess>();
	auto  name = reference->getName();

	if( translationTable.count(name) ){
		// we replace the formal argument with the actual argument value
		// todo: handle identifiers collisions (?) (maybe with globals?)
		expression = *translationTable.lookup(name);
	}else{
		auto type = reference->getType();
		
		if(type.isa<Record*>()){
			const Record* record = type.get<Record*>();
	
			if(record->shouldBeInlined()){
				// the ref to a inlineable record is exploded (e.g.  x  =>  RecordInstance<Complex>(x.re,x.im)  )
				llvm::SmallVector<std::unique_ptr<Expression>,3> args;
				
				for(const auto &m:(*record)){
					auto new_m = Expression::reference(
						m->getLocation(), 
						m->getType(),
						(name+"."+m->getName()).str());
					
					args.emplace_back(std::move(new_m));
				}
				
				expression = *Expression::recordInstance(expression.getLocation(),type,args);
			}
		}
	}

	return llvm::Error::success();
}

template<>
llvm::Error InlineExpanser::run<Tuple>(Expression& expression)
{
	auto* tuple = expression.get<Tuple>();

	for (auto& exp : *tuple)
		if (auto error = run<Expression>(*exp); error)
			return error;

	return llvm::Error::success();
}

template<>
llvm::Error InlineExpanser::run<RecordInstance>(Expression& expression)
{
	auto *instance = expression.get<RecordInstance>();

	for( auto &value : *instance){
		if (auto error = run<Expression>(*value); error)
			return error;
	}

	return llvm::Error::success();
}

llvm::Error InlineExpanser::run(Member& member)
{
	return llvm::Error::success();
}

template<>
llvm::Error InlineExpanser::run<Statement>(Statement& statement)
{
	return statement.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return run<deconst>(statement);
	});
}

template<>
llvm::Error InlineExpanser::run<AssignmentStatement>(Statement& statement)
{
	auto* assignmentStatement = statement.get<AssignmentStatement>();
	auto* expression = assignmentStatement->getExpression();

	if (auto error = run<Expression>(*expression); error)
		return error;

	return llvm::Error::success();
}

template<>
llvm::Error InlineExpanser::run<BreakStatement>(Statement& statement)
{
	return llvm::Error::success();
}

template<>
llvm::Error InlineExpanser::run<ForStatement>(Statement& statement)
{
	return llvm::Error::success();
}

llvm::Error InlineExpanser::run(Induction& induction)
{
	return llvm::Error::success();
}

template<>
llvm::Error InlineExpanser::run<IfStatement>(Statement& statement)
{
	return llvm::Error::success();
}

template<>
llvm::Error InlineExpanser::run<ReturnStatement>(Statement& statement)
{
	return llvm::Error::success();
}

template<>
llvm::Error InlineExpanser::run<WhenStatement>(Statement& statement)
{
	return llvm::Error::success();
}

template<>
llvm::Error InlineExpanser::run<WhileStatement>(Statement& statement)
{
	return llvm::Error::success();
}

llvm::Error InlineExpanser::run(Algorithm& algorithm)
{
	return llvm::Error::success();
}

std::unique_ptr<Pass> marco::ast::createInliningPass()
{
	return std::make_unique<InlineExpanser>();
}
