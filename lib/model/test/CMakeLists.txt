include(GoogleTest)

add_executable(modelTest src/ModelTest.cpp src/ModLexerStateMachineTest.cpp src/ModParserTest.cpp src/ModMatchersTest.cpp src/VectorAccessTest.cpp src/ModEquationTest.cpp src/ModConstTest.cpp)
add_executable(modelica::modelTest ALIAS modelTest) 
target_link_libraries(modelTest PRIVATE gtest gtest_main modelica::model)
target_include_directories(modelTest PUBLIC include PRIVATE src)
target_compile_features(modelTest PUBLIC cxx_std_17)

gtest_add_tests(TARGET      modelTest
                TEST_SUFFIX .noArgs
                TEST_LIST   noArgsTests
)

