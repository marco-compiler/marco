include(GoogleTest)

add_executable(omcToModelTest src/PassesTest.cpp)
add_executable(modelica::omcToModelTest ALIAS omcToModelTest) 
target_link_libraries(omcToModelTest PRIVATE gtest gtest_main modelica::omcToModel)
target_include_directories(omcToModelTest PUBLIC include PRIVATE src)
target_compile_features(omcToModelTest PUBLIC cxx_std_17)

gtest_add_tests(TARGET      omcToModelTest
                TEST_SUFFIX .noArgs
                TEST_LIST   noArgsTests
)

