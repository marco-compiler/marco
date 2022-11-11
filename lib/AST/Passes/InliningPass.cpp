#include <marco/AST/Passes/InliningPass.h>


using namespace marco;
using namespace marco::ast;


InlineExpanser::InlineExpanser(diagnostic::DiagnosticEngine& diagnostics)
		: Pass(diagnostics)
{
}

template<>
bool InlineExpanser::run<Class>(Class& cls)
{	
	return cls.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return run<deconst>(cls);
	});
}

bool InlineExpanser::run(std::unique_ptr<Class> &cls)
{
	//InlineExpanser starting point
	SymbolTable::ScopeTy scope(symbolTable);

	if (!run<Class>(*cls))
		return false;

	return true;
}

template<>
bool InlineExpanser::run<PartialDerFunction>(Class& cls)
{
	return true;
}

template<>
bool InlineExpanser::run<StandardFunction>(Class& cls)
{
	// SymbolTable::ScopeTy varScope(symbolTable);
	auto* function = cls.get<StandardFunction>();

	// Populate the symbol table
	symbolTable.insert(function->getName(), Symbol(cls));

	return true;
}

static bool explodeModification(InlineExpanser& inliner, const Member& member, const Member& recordMember, const Type& newType, std::unique_ptr<Modification>& resultModification)
{
  if (!member.hasModification())
    return true;

  if (auto* modification = member.getModification(); modification->hasExpression()) {
    auto init = *modification->getExpression();

    if (modification->hasClassModification()) {
      llvm::outs() << " class modifications are not preserved during record member flattening.";
    }

    inliner.run<Expression>(init);

    if (init.getType().isScalar()) {
      if (init.isa<RecordInstance>()) {
        auto instance = init.get<RecordInstance>();

        //check if the type is the same
        assert(instance->getType() == member.getType());

        auto val = instance->getMemberValue(recordMember.getName());

        resultModification = Modification::build(
            modification->getLocation(),
            std::make_unique<Expression>(val));

        return true;
      } else {
        llvm::errs() << "expected a record instance";
        return false;
      }
    } else if (init.isa<Array>()) {
      auto* array = init.get<Array>();

      auto type = array->getType();

      llvm::SmallVector<std::unique_ptr<Expression>, 3> values;

      if (type.isa<Record*>() && type.get<Record*>()->shouldBeInlined()) {
        for (auto& el : *array) {
          auto instance = el->get<RecordInstance>();
          auto val = instance->getMemberValue(recordMember.getName());
          values.emplace_back(std::make_unique<Expression>(val));
        }
      }

      auto loc = modification->getLocation();

      resultModification = Modification::build(
          loc,
          Expression::array(loc, newType, std::move(values)));

      return true;
    } else {
      llvm::errs() << "expected an array of record instances";
      return false;
    }
  }

  return true;
}

static bool explodeMember(InlineExpanser& inliner, Member& member, bool& to_remove, llvm::SmallVectorImpl<std::unique_ptr<Member>>& to_add)
{

  auto& type = member.getType();

  if (type.isa<Record*>()) {
    const Record* record = type.get<Record*>();

    if (record->shouldBeInlined()) {
      to_remove = true;

      for (const auto& m : (*record)) {
        auto new_type = getFlattenedMemberType(type, m->getType());
        std::unique_ptr<Modification> modification = nullptr;

        if (!explodeModification(inliner, member, *m, new_type, modification))
          return false;

        auto new_m = Member::build(
            member.getLocation(),
            (member.getName() + "." + m->getName()).str(),
            new_type,
            member.getTypePrefix(),
            true,// isPublic
            std::move(modification));

        bool toRemove = false;

        if (!explodeMember(inliner, *new_m, toRemove, to_add))
          return false;

        if(!toRemove)
					to_add.emplace_back(std::move(new_m));
      }
    }
  }

  return true;
}

template<>
bool InlineExpanser::run<Model>(Class& cls)
{
	auto* model = cls.get<Model>();
	
	SymbolTable::ScopeTy varScope(symbolTable);
	
	for(const auto &innerClass : model->getInnerClasses())
		if(!run<Class>(*innerClass))
			return false;

	// handle the members declaration exploding the inlineable record members
	// e.g. Complex x;	->	Real 'x.re';
	//						Real 'x.im';						
	auto& members = model->getMembers_mut();

	llvm::SmallVector<std::unique_ptr<Member>,3> members_to_add;

	for(auto it = members.begin(); it!=members.end();){
		bool to_remove=false;
		auto &member = **it;
		auto type = member.getType();

		if(!explodeMember(*this,member,to_remove,members_to_add))
			return false;

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
	llvm::SmallVector<std::unique_ptr<ForEquation>, 3> new_for_equations;

	auto explodeEquations=[&](Equation& eq){
		auto *lhs = eq.getLhsExpression();
		auto *rhs = eq.getRhsExpression();

		// todo : supports type aliases  (e.g. ComplexPU (for pure) and Complex are not equal right now)
		// assert(lhs->getType() == rhs->getType());

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

			// exploding equations we always end up with more equations,
			// so we replace the first with the current one and appends the others
			eq = *Equation::build(eq.getLocation(),std::move(destinations[0]),std::move(expressions[0]));

			for(size_t i=1; i<destinations.size(); ++i){
				auto e = Equation::build(eq.getLocation(),std::move(destinations[i]),std::move(expressions[i]));
				new_equations.emplace_back(std::move(e));
			}
		}
	};

	for (auto& equationsBlock : model->getEquationsBlocks()) {
    for (auto& equation : equationsBlock->getEquations()) {
			if (!run(*equation))
				return false;
			
			new_equations.clear();
			explodeEquations(*equation);
    }

		for(auto &e : new_equations)
			equationsBlock->add(std::move(e));


		for (auto& forEquation : equationsBlock->getForEquations())
		{
			if (!run(*forEquation))
				return false;
			
			new_equations.clear();
			explodeEquations(*forEquation->getEquation());

			for(auto &e : new_equations){
				new_for_equations.push_back(ForEquation::build(forEquation->getLocation(), forEquation->getInductions(), std::move(e)));
			}
		}

		for(auto &e : new_for_equations)
			equationsBlock->add(std::move(e));

  }

	return true;
}

template<>
bool InlineExpanser::run<Package>(Class& cls)
{
	SymbolTable::ScopeTy varScope(symbolTable);
	auto* package = cls.get<Package>();

	// Populate the symbol table
	symbolTable.insert(package->getName(), Symbol(cls));

	for (auto& innerClass : *package)
		symbolTable.insert(innerClass->getName(), Symbol(*innerClass));

	for (auto& innerClass : *package)
		if (!run<Class>(*innerClass))
			return false;

	return true;
}

template<>
bool InlineExpanser::run<Record>(Class& cls)
{
	assert(cls.isa<Record>());
	auto *record = cls.get<Record>();
	
	symbolTable.insert(record->getName(), Symbol(cls));
	return true;
}

bool InlineExpanser::run(Equation& equation)
{
	if (!run<Expression>(*equation.getLhsExpression()))
		return false;

	if (!run<Expression>(*equation.getRhsExpression()))
		return false;

	return true;
}

bool InlineExpanser::run(ForEquation& forEquation)
{
	//todo : handle as the normal equations (e.g. decomposing records related ones, thus adding new equations. see: run<Model>(...) function)
	SymbolTable::ScopeTy varScope(symbolTable);

	for (auto& ind : forEquation.getInductions())
	{
		symbolTable.insert(ind->getName(), Symbol(*ind));

		if (!run<Expression>(*ind->getBegin()))
			return false;

		if (!run<Expression>(*ind->getEnd()))
			return false;
	}

	if (!run(*forEquation.getEquation()))
		return false;

	return true;
}

template<>
bool InlineExpanser::run<Expression>(Expression& expression)
{
	return expression.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return run<deconst>(expression);
	});
}

template<>
bool InlineExpanser::run<Array>(Expression& expression)
{
  assert(expression.isa<Array>());
  auto* array = expression.get<Array>();

  auto type = array->getType();

  if (type.isa<Record*>() && type.get<Record*>()->shouldBeInlined()) {
    for (auto& el : *array) {
      if (!run<Expression>(*el))
        return false;
    }
  }

  return true;
}

template<>
bool InlineExpanser::run<Call>(Expression& expression)
{
	assert(expression.isa<Call>());
	auto *call = expression.get<Call>();

	auto actual_args = call->getArgs();
	for (auto& exp : actual_args)
	{
		if (!run<Expression>(*exp))
			return false;
	}

	llvm::StringRef functionName = call->getFunction()->get<ReferenceAccess>()->getName();


	if (!symbolTable.count(functionName))
		return true;
	
	auto* cls = symbolTable.lookup(functionName).dyn_get<Class>();

	if(!cls)
		return true;
	
	
	if ( auto* standardFunction = cls->dyn_get<StandardFunction>(); standardFunction && standardFunction->shouldBeInlined())
	{
		auto formal_args = standardFunction->getArgs();

		assert(formal_args.size() == actual_args.size()); //todo check default params

		if(standardFunction->isCustomRecordConstructor())
		{
			// todo handle default args, now it only works if the custom constructor has the same signature as the default one
			auto recordType = standardFunction->getType().getResults()[0];

			expression = *Expression::recordInstance(expression.getLocation(),recordType,actual_args);
			return true;
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
			if(!run<Expression>(new_expression))
				return false;

			expression = new_expression;

			return true;
		}

		assert(false && "only functions containing one assignment statement are inlinable");
	}
	else if(auto *record = cls->dyn_get<Record>();record && record->shouldBeInlined())
	{
		// the default record constructor is function with the same id as the record
		expression = *Expression::recordInstance(expression.getLocation(),Type(record),actual_args);
		return true;
	}
	
	return true;
}

template<>
bool InlineExpanser::run<Constant>(Expression& expression)
{
	return true;
}

template<>
bool InlineExpanser::run<Operation>(Expression& expression)
{
	auto* operation = expression.get<Operation>();
	auto type = expression.getType();

	if(operation->getOperationKind() == OperationKind::subscription)
	{	
		auto *lhs = operation->getArg(0);

		if (!run<Expression>(*lhs))
				return false;
				
		auto lhs_type = lhs->getType();

		if( lhs_type.isa<Record*>() && lhs_type.get<Record*>()->shouldBeInlined() )
		{
			if( lhs->isa<RecordInstance>() )
			{
				Expression new_e = *lhs;
				auto dims = operation->size() - 1L;

				for(auto &v: *(new_e.get<RecordInstance>()))
				{
					Expression value = *v;
					*v = expression;
					*(*v->get<Operation>()->begin()) = value;
					v->setType(value.getType().subscript(dims));
				}

				expression = new_e;
				expression.setType(expression.getType().subscript(dims));

				return true;
			}
			
			//skipping subscription of record array : it's gonna be handled from the memberLookup operation
			return true;
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

		if (!run<Expression>(lhs))
				return false;

		if(lhs.isa<Operation>() && lhs.get<Operation>()->getOperationKind() == OperationKind::subscription){
			if(subscription){
				//todo handle multiple subscriptions : collapse them in one
				assert(false);
			}

			subscription = lhs;
			lhs = *lhs.get<Operation>()->getArg(0);
		}

		auto lhs_type = lhs.getType();
		
		if(lhs_type.isa<Record*>() && rhs.isa<ReferenceAccess>())
		{
			const Record* record = lhs_type.get<Record*>();

			if(record->shouldBeInlined())
			{
				auto memberName = rhs.get<ReferenceAccess>()->getName();

				if(lhs.isa<ReferenceAccess>()){
					// transforming member lookup to the exploded new member
					// e.g. x.re -> 'x.re'
					lhs = *Expression::reference(
						expression.getLocation(),
						operation->getType(),
						(lhs.get<ReferenceAccess>()->getName()+"."+rhs.get<ReferenceAccess>()->getName()).str());
					
				} 
				else if (lhs.isa<RecordInstance>())
				{
					//e.g. Complex(a,b).re  ->  a
					auto instance = lhs.get<RecordInstance>();
					lhs = instance->getMemberValue(memberName);
				} else{
					assert(false && "unexpected lhs in member lookup");
				}

				if (!run<Expression>(lhs))
					return false;

				if(subscription){
					**(subscription->get<Operation>()->begin()) = lhs;
					lhs = *subscription;
					lhs.setType(expression.getType()); 
					
					if (!run<Expression>(lhs))
						return false;
				}
			}

			expression = lhs;

			return true;
		}else{
			assert(false && "member lookup is only supported on records.");
		}
	}

	for (auto& arg : operation->getArguments())
		if (!run<Expression>(*arg))
			return false;

	return true;
}

template<>
bool InlineExpanser::run<ReferenceAccess>(Expression& expression)
{
	auto* reference = expression.get<ReferenceAccess>();
	auto  name = reference->getName();

	if( translationTable.count(name) )
	{
		// we replace the formal argument with the actual argument value
		// todo: handle identifiers collisions (?) (maybe with globals?)
		expression = *translationTable.lookup(name);
	}
	else
	{
		auto type = reference->getType();
		
		if(type.isa<Record*>())
		{
			const Record* record = type.get<Record*>();
	
			if(record->shouldBeInlined())
			{
				// the ref to a inlineable record is exploded (e.g.  x  =>  RecordInstance<Complex>(x.re,x.im)  )
				llvm::SmallVector<std::unique_ptr<Expression>,3> args;
				
				for(const auto &m:(*record)){
					// auto new_type = m->getType();
					// new_type.setDimensions(concatenateDimensions(type,new_type));
					auto new_type = getFlattenedMemberType(type,m->getType());

					auto new_m = Expression::reference(
						m->getLocation(), 
						new_type,//m->getType(),
						(name+"."+m->getName()).str());
					
					args.emplace_back(std::move(new_m));
				}
				
				expression = *Expression::recordInstance(expression.getLocation(),type,args);
			}
		}
	}

	return true;
}

template<>
bool InlineExpanser::run<Tuple>(Expression& expression)
{
	auto* tuple = expression.get<Tuple>();

	for (auto& exp : *tuple)
		if (!run<Expression>(*exp))
			return false;

	return true;
}

template<>
bool InlineExpanser::run<RecordInstance>(Expression& expression)
{
	auto *instance = expression.get<RecordInstance>();

	for( auto &value : *instance){
		if (!run<Expression>(*value))
			return false;
	}

	return true;
}

bool InlineExpanser::run(Member& member)
{
	return true;
}

template<>
bool InlineExpanser::run<Statement>(Statement& statement)
{
	return statement.visit([&](auto& obj) {
		using type = decltype(obj);
		using deref = typename std::remove_reference<type>::type;
		using deconst = typename std::remove_const<deref>::type;
		return run<deconst>(statement);
	});
}

template<>
bool InlineExpanser::run<AssignmentStatement>(Statement& statement)
{
	auto* assignmentStatement = statement.get<AssignmentStatement>();
	auto* expression = assignmentStatement->getExpression();

	if (!run<Expression>(*expression))
		return false;

	return true;
}

template<>
bool InlineExpanser::run<BreakStatement>(Statement& statement)
{
	return true;
}

template<>
bool InlineExpanser::run<ForStatement>(Statement& statement)
{
	return true;
}

bool InlineExpanser::run(Induction& induction)
{
	return true;
}

template<>
bool InlineExpanser::run<IfStatement>(Statement& statement)
{
	return true;
}

template<>
bool InlineExpanser::run<ReturnStatement>(Statement& statement)
{
	return true;
}

template<>
bool InlineExpanser::run<WhenStatement>(Statement& statement)
{
	return true;
}

template<>
bool InlineExpanser::run<WhileStatement>(Statement& statement)
{
	return true;
}

bool InlineExpanser::run(Algorithm& algorithm)
{
	return true;
}

std::unique_ptr<Pass> marco::ast::createInliningPass(diagnostic::DiagnosticEngine& diagnostics)
{
	return std::make_unique<InlineExpanser>(diagnostics);
}
