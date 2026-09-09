include(cc_test)

cc_test(
  NAME
    spec_input_builder_test
  SRCS
    spec_input_builder_test.cpp
    "${PROJECT_SOURCE_DIR}/xllm/core/framework/speculative/spec_input_builder.cpp"
  DEPS
    :speculative
    :parallel_state
    torch
    glog::glog
    proto::xllm_proto
    GTest::gtest_main
)
