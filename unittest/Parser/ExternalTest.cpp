/**
TEST(Parser, expression_list_test15) { //counter
  auto str = R"(, ,)";

  Parser parser = initForExternalTests(str);

  //testCheckForExpressionListTest(parser, true, /*TO-DO);
}

TEST(Parser, expression_list_test14) { //counter
  auto str = R"(,2,3)";

  Parser parser = initForExternalTests(str);

  //testCheckForExpressionListTest(parser, true, /*TO-DO);
}
TEST(Parser, expression_list_test13) { //counter
  auto str = R"(,)";

  Parser parser = initForExternalTests(str);

 // testCheckForExpressionListTest(parser, true, /*TO-DO);
}
TEST(Parser, expression_list_test12) { //counter
  auto str = R"(,2)";

  Parser parser = initForExternalTests(str);

//  testCheckForExpressionListTest(parser, true, /*TO-DO);
}

TEST(Parser, expression_list_test11) { //counter
  auto str = R"()";

  Parser parser = initForExternalTests(str);

//  testCheckForExpressionListTest(parser, true, /*TO-DO);
}

TEST(Parser, expression_list_test10) {
  auto str = R"(3,3,4,4,5,6,9,8,foo(x=2,y=3))";

  Parser parser = initForExternalTests(str);

  testCheckForExpressionListTest(parser, false, /*TO-DO);
}

TEST(Parser, expression_list_test9) {
  auto str = R"(foo(x=k,x=k),foo(x=k),foo(x=k,x=k,x=k))";

  Parser parser = initForExternalTests(str);

//  testCheckForExpressionListTest(parser, false, /*TO-DO);
}

TEST(Parser, expression_list_test8) {
  auto str = R"(1,{1,2,3})";

  Parser parser = initForExternalTests(str);

//  testCheckForExpressionListTest(parser, false, /*TO-DO);
}

TEST(Parser, expression_list_test7) {
  auto str = R"({1,2,3},3)";

  Parser parser = initForExternalTests(str);

 // testCheckForExpressionListTest(parser, false, /*TO-DO);
}

TEST(Parser, expression_list_test6) {
  auto str = R"({1,2,3},{2,3})";

  Parser parser = initForExternalTests(str);

 // testCheckForExpressionListTest(parser, false, /*TO-DO);
}

TEST(Parser, expression_list_test5) {
  auto str = R"((),(),{1,2})";

  Parser parser = initForExternalTests(str);

 // testCheckForExpressionListTest(parser, false, /*TO-DO);
}

TEST(Parser, expression_list_test4) {
  auto str = R"(x <> y, x<y)";

  Parser parser = initForExternalTests(str);

//  testCheckForExpressionListTest(parser, false, /*TO-DO);
}

TEST(Parser, expression_list_test3) {
  auto str = R"(x,1,{1,2},2)";

  Parser parser = initForExternalTests(str);

//  testCheckForExpressionListTest(parser, false, /*TO-DO);
}

TEST(Parser, expression_list_test2) {
  auto str = R"(NOT X, NOT Y)";

  Parser parser = initForExternalTests(str);

//  testCheckForExpressionListTest(parser, false, /*TO-DO);
}

TEST(Parser, expression_list_test1) {
  auto str = R"(012345)";

  Parser parser = initForExternalTests(str);

//  testCheckForExpressionListTest(parser, false, /*TO-DO);
}
TEST(Parser, external_function_call_test11) {
  auto str = R"(abc(()))";

  Parser parser = initForExternalTests(str);

  //testCheckForExternalFunctionCallTest(parser, false, /*TO-DO);
}
TEST(Parser, external_function_call_test10) {
  auto str = R"(abc())";

  Parser parser = initForExternalTests(str);

 // testCheckForExternalFunctionCallTest(parser, false, /*TO-DO);
}
TEST(Parser, external_function_call_test9) { //counter
  auto str = R"(x.y.z x.y.z = a())";

  Parser parser = initForExternalTests(str);

 // testCheckForExternalFunctionCallTest(parser, true, /*TO-DO);
}
TEST(Parser, external_function_call_test8) { //counter
  auto str = R"(==)";

  Parser parser = initForExternalTests(str);

 // testCheckForExternalFunctionCallTest(parser, true, /*TO-DO);
}
TEST(Parser, external_function_call_test7) { //counter
  auto str = R"(x.y.z a())";

  Parser parser = initForExternalTests(str);

 // testCheckForExternalFunctionCallTest(parser, true, /*TO-DO);
}
TEST(Parser, external_function_call_test6) { //counter
  auto str = R"(x.y.z = a e)";

  Parser parser = initForExternalTests(str);

 // testCheckForExternalFunctionCallTest(parser, true, /*TO-DO);
}
TEST(Parser, external_function_call_test5) { //counter
  auto str = R"(x.y.z = ())";

  Parser parser = initForExternalTests(str);

//  testCheckForExternalFunctionCallTest(parser, true, /*TO-DO);
}
TEST(Parser, external_function_call_test4) { //counter
  auto str = R"(=abc(3,4,5))";

  Parser parser = initForExternalTests(str);

//  testCheckForExternalFunctionCallTest(parser, true, /*TO-DO);
}
TEST(Parser, external_function_call_test3) {
  auto str = R"(abc(3,4,5))";

  Parser parser = initForExternalTests(str);

 // testCheckForExternalFunctionCallTest(parser, false, /*TO-DO);
}
TEST(Parser, external_function_call_test2) { 
  auto str = R"(x.y.z = abc())";

  Parser parser = initForExternalTests(str);

 // testCheckForExternalFunctionCallTest(parser, false, /*TO-DO);
}
TEST(Parser, external_function_call_test1) {
  auto str = R"(x.y.z = abc(3,4,5))";

  Parser parser = initForExternalTests(str);

//  testCheckForExternalFunctionCallTest(parser, false, /*TO-DO);
}

TEST(Parser, external_test15) { //counter
  auto str = R"(annotation();)";

  Parser parser = initForExternalTests(str);

 // testCheckForExternalTest(parser, true, /*TO-DO);
}

TEST(Parser, external_test14) { //counter
  auto str = R"(external annotation())";

  Parser parser = initForExternalTests(str);

//  testCheckForExternalTest(parser, true, /*TO-DO);
}

TEST(Parser, external_test12) {
  auto str = R"(external "C" x.y.z = abc(3,4,5) annotation();)";

  Parser parser = initForExternalTests(str);

//  testCheckForExternalTest(parser, false, /*TO-DO);
}

TEST(Parser, external_test11) {
  auto str = R"(external "C" annotation();)";

  Parser parser = initForExternalTests(str);

 // testCheckForExternalTest(parser, false, /*TO-DO);
}

TEST(Parser, external_test10) {
  auto str = R"(external annotation();)";

  Parser parser = initForExternalTests(str);

//  testCheckForExternalTest(parser, false, /*TO-DO);
}

TEST(Parser, external_test9) { //counter
  auto str = R"(external;)";

  Parser parser = initForExternalTests(str);

 // testCheckForExternalTest(parser, true, /*TO-DO);
}

TEST(Parser, external_test8) { //counter
  auto str = R"(external)";

  Parser parser = initForExternalTests(str);

 // testCheckForExternalTest(parser, true, /*TO-DO);
}

TEST(Parser, external_test7) {
  auto str = R"(external x.y.z = abc(3,4,5);)";

  Parser parser = initForExternalTests(str);

 // testCheckForExternalTest(parser, false, /*TO-DO);
}

TEST(Parser, external_test6) { //counter
  auto str = R"(a)";

  Parser parser = initForExternalTests(str);

//  testCheckForExternalTest(parser, true, /*TO-DO);
}

TEST(Parser, external_test5) { //counter
  auto str = R"(external a;)";

  Parser parser = initForExternalTests(str);

 // testCheckForExternalTest(parser, true, /*TO-DO);
}

TEST(Parser, external_test4) {
  auto str = R"(external "C";)";

  Parser parser = initForExternalTests(str);

//  testCheckForExternalTest(parser, false, /*TO-DO);
}

TEST(Parser, external_test3) { //counter
  auto str = R"("C")";

  Parser parser = initForExternalTests(str);

 // testCheckForExternalTest(parser, true, /*TO-DO);
}

TEST(Parser, external_test2) {
  auto str = R"()";

  Parser parser = initForExternalTests(str);

 // testCheckForExternalTest(parser, false, /*TO-DO);
}

TEST(Parser, external_test1) {
  auto str = R"(external "C" x.y.z = abc(3,4,5) annotation();)";

  Parser parser = initForExternalTests(str);

//  testCheckForExternalTest(parser, false, /*TO-DO);

}
void testCheckForExpressionListTest (Parser parser, bool checkCounter,
  llvm::ArrayRef<std::unique_ptr<ASTNode>> exp_el, int bl, int bc, int el, int ec)
  {
    auto node = parser.parseExpressionList();

    if (!checkCounter)
    {
        ASSERT_TRUE(node.has_value());
        EXPECT_EQ(node, exp_el);

        EXPECT_EQ(node->getLocation().begin.line, bl);
        EXPECT_EQ(node->getLocation().begin.column, bc);

        EXPECT_EQ(node->getLocation().end.line, el);
        EXPECT_EQ(node->getLocation().end.column, ec);
    }
    else
    {
       ASSERT_FALSE(node.has_value());
    }

  }
void testCheckForExternalFunctionCallTest (Parser parser, bool checkCounter,
  llvm::StringRef exp_name,
  std::unique_ptr<ASTNode> exp_cr,
  llvm::ArrayRef<std::unique_ptr<ASTNode>> exp_el, int bl, int bc, int el, int ec)
  {
    auto node = parser.parseExternalFunctionCall();

    if (!checkCounter)
    {
        ASSERT_TRUE(node.has_value());
        EXPECT_EQ(node->getName(), exp_str);
        EXPECT_EQ(node->getComponentReference(), exp_cr);
        EXPECT_EQ(node->getExpressions(), exp_el);

        EXPECT_EQ(node->getLocation().begin.line, bl);
        EXPECT_EQ(node->getLocation().begin.column, bc);

        EXPECT_EQ(node->getLocation().end.line, el);
        EXPECT_EQ(node->getLocation().end.column, ec);
    }
    else
    {
       ASSERT_FALSE(node.has_value());
    }

  }
void testCheckForExternalTest (Parser parser, bool checkCounter,
  llvm::StringRef exp_str,
  std::unique_ptr<ASTNode> exp_efc,
  std::unique_ptr<ASTNode> exp_ac, int bl, int bc, int el, int ec) {

    auto node = parser.parseExternal();

    if (!checkCounter)
    {
        ASSERT_TRUE(node.has_value());
        EXPECT_EQ(node->getLanguageSpecification(), exp_str);
        EXPECT_EQ(node->getExternalFunctionCall(), exp_efc);
        EXPECT_EQ(node->getAnnotationClause(), exp_ac);

        EXPECT_EQ(node->getLocation().begin.line, bl);
        EXPECT_EQ(node->getLocation().begin.column, bc);

        EXPECT_EQ(node->getLocation().end.line, el);
        EXPECT_EQ(node->getLocation().end.column, ec);

    }
    else
    {
       ASSERT_FALSE(node.has_value());
    }

  }
Parser initForExternalTests(auto str)
  {
    auto sourceFile = std::make_shared<SourceFile>("test.mo");

    auto diagnostics = getDiagnosticsEngine();
    clang::SourceManagerForFile fileSourceMgr(sourceFile->getFileName(), str);
    auto &sourceManager = fileSourceMgr.get();

    auto buffer = llvm::MemoryBuffer::getMemBuffer(str);
    sourceFile->setMemoryBuffer(buffer.get());

    return (parser(*diagnostics, sourceManager, sourceFile));

  }
  **/