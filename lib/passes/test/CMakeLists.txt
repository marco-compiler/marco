include(GoogleTest)

add_executable(passesTest src/PassesTest.cpp src/MatchingTest.cpp)
add_executable(modelica::passesTest ALIAS passesTest) 
target_link_libraries(passesTest PRIVATE gtest gtest_main modelica::lowerer)
target_include_directories(passesTest PUBLIC include PRIVATE src)
target_compile_features(passesTest PUBLIC cxx_std_17)

gtest_add_tests(TARGET      passesTest
                TEST_SUFFIX .noArgs
                TEST_LIST   noArgsTests
)

