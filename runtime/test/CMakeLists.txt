include(GoogleTest)

add_executable(runtimeTest src/RuntimeTest.cpp)
add_executable(modelica::runtimeTest ALIAS runtimeTest) 
target_link_libraries(runtimeTest PRIVATE gtest gtest_main modelica::runtime)
target_include_directories(runtimeTest PUBLIC include PRIVATE src)
target_compile_features(runtimeTest PUBLIC cxx_std_17)

gtest_add_tests(TARGET      runtimeTest
                TEST_SUFFIX .noArgs
                TEST_LIST   noArgsTests
)



