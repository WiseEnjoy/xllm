/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "core/framework/parallel_state/process_group.h"
#include "core/framework/sampling/json_object_grammar.h"
#include "core/framework/speculative/spec_input_builder.h"
#include "core/framework/speculative/spec_verify.h"
#include "platform/device.h"
#include "runtime/dflash/dflash2_worker_impl.h"
#include "runtime/dspark_worker_impl.h"
#include "runtime/llm_worker_impl.h"

namespace xllm {

class DFlashWorkerImplTestPeer final {
 public:
  static ForwardInput prepare_query(DFlashWorkerImpl& worker,
                                    const ForwardInput& input) {
    worker.mask_token_id_ = 4;
    ForwardInput query;
    worker.prepare_query_inputs(input, query);
    CHECK_EQ(worker.prepare_stream_->synchronize(), 0);
    return query;
  }

  static ForwardInput prepare_json_states(DFlashWorkerImpl& worker,
                                          const ForwardInput& input,
                                          const torch::Tensor& tokens_host,
                                          const StreamEventPtr& ready) {
    auto guard = worker.compute_stream_->set_stream_guard();
    ForwardInput validation;
    worker.prepare_json_validation_states(
        input, tokens_host, ready, validation);
    CHECK_EQ(worker.compute_stream_->synchronize(), 0);
    return validation;
  }
};

class DSparkWorkerImplTestPeer final {
 public:
  static void validate_config(const DSparkWorkerImpl& worker,
                              bool supports_device_inputs) {
    worker.validate_draft_prelaunch_config(supports_device_inputs);
  }

  static void validate_input(const DSparkWorkerImpl& worker,
                             const ForwardInput& input) {
    worker.validate_decode_input(input);
  }

  static void install_output(DSparkWorkerImpl& worker, ForwardOutput output) {
    worker.last_step_output_ = std::move(output);
    worker.last_step_output_valid_ = true;
    worker.last_step_sample_sequence_ids_ = {"request"};
  }

  static dflash_async::DraftCacheBlockTable prepare_context_block_table(
      DSparkWorkerImpl& worker,
      const ForwardInput& input) {
    worker.prepare_validate_output_async(input);
    CHECK(worker.compute_stream_->wait_event(worker.prepared_context_ready_));
    return worker.prepared_context_block_table_;
  }

  static ForwardOutput finalize_output(DSparkWorkerImpl& worker,
                                       const ForwardInput& input,
                                       const SampleOutput& sample,
                                       std::unique_ptr<LLMWorkerImpl> draft) {
    worker.draft_impl_ = std::move(draft);
    worker.prepare_validate_output_async(input);
    ForwardOutput output;
    worker.finalize_validate_output_async(input, sample, output);
    return output;
  }

  static bool async_output_enabled(const DSparkWorkerImpl& worker) {
    return worker.async_output_enabled_;
  }

  static void set_async_output_enabled(DSparkWorkerImpl& worker, bool enabled) {
    worker.async_output_enabled_ = enabled;
  }

  static void set_draft_prelaunch_enabled(DSparkWorkerImpl& worker,
                                          bool enabled) {
    worker.draft_prelaunch_enabled_ = enabled;
  }

  static bool can_prelaunch_draft(const DSparkWorkerImpl& worker,
                                  const ForwardInput& input) {
    return worker.can_prelaunch_draft(input);
  }

  static std::vector<int64_t> pending_rows(const DSparkWorkerImpl& worker,
                                           const ForwardInput& input) {
    return worker.build_batch_to_pending_row_mapping(input);
  }

  static void commit_pending_context(DSparkWorkerImpl& worker) {
    worker.commit_pending_context();
  }

  static void install_pending_context(DSparkWorkerImpl& worker,
                                      ForwardInput identity,
                                      const torch::Tensor& accepted_device,
                                      const torch::Tensor& accepted_host,
                                      const torch::Tensor& embeddings,
                                      const torch::Tensor& positions,
                                      StreamEventPtr ready,
                                      std::shared_ptr<EmbeddingCache> cache) {
    worker.async_output_enabled_ = true;
    worker.draft_prelaunch_enabled_ = true;
    worker.embedding_cache_ = std::move(cache);
    worker.pending_context_.emplace();
    worker.pending_context_->identity = std::move(identity);
    worker.pending_context_->accepted_device = accepted_device;
    worker.pending_context_->accepted_host = accepted_host;
    worker.pending_context_->embeddings = embeddings;
    worker.pending_context_->base_positions = positions;
    worker.pending_context_->d2h_ready = ready;
    worker.pending_context_->context_ready = std::move(ready);
  }

  static bool has_pending(const DSparkWorkerImpl& worker) {
    return worker.pending_context_.has_value();
  }

  static void configure_v41_geometry(DSparkWorkerImpl& worker,
                                     bool graph_enabled,
                                     bool native_sas) {
    ModelArgs args;
    args.model_type("deepseek_v41_dspark").dspark_use_native_sas(native_sas);
    worker.draft_impl_->context_ = ModelContext(
        worker.parallel_args_,
        args,
        QuantArgs(),
        torch::dtype(torch::kBFloat16).device(worker.device_.unwrap()));
    SpeculativeRuntimeConfig runtime_config;
    runtime_config.is_draft_engine = true;
    runtime_config.draft_graph_enabled = graph_enabled;
    worker.draft_impl_->context_.set_speculative_runtime_config(runtime_config);
    worker.draft_sas_mode_ = dflash_detail::classify_dspark_sas_mode(
        args, /*sample_from_anchor=*/true);
  }

  static bool uses_block_parallel_rows(const DSparkWorkerImpl& worker) {
    return worker.draft_use_block_parallel_rows();
  }

  static BatchForwardType draft_forward_type(const DSparkWorkerImpl& worker) {
    return worker.draft_batch_forward_type();
  }

  static void validate_graph_input(const DSparkWorkerImpl& worker,
                                   const ForwardInput& input) {
    worker.validate_draft_graph_input(input);
  }

  static void configure_draft_query(DSparkWorkerImpl& worker,
                                    int32_t mask_token_id) {
    worker.mask_token_id_ = mask_token_id;
    worker.draft_sas_mode_ = dflash_detail::DSparkSasMode::NATIVE;
  }

  static dflash_async::NextDraftInputs prepare_pending_draft_inputs(
      DSparkWorkerImpl& worker,
      const ForwardInput& input) {
    ForwardInput query;
    worker.prepare_query_inputs(input, query);
    CHECK_EQ(worker.prepare_stream_->synchronize(), 0);
    auto guard = worker.compute_stream_->set_stream_guard();
    CHECK(worker.compute_stream_->wait_event(
        worker.pending_context_->context_ready));
    const std::vector<int64_t> rows =
        worker.build_batch_to_pending_row_mapping(input);
    torch::Tensor row_indices =
        torch::tensor(rows, torch::kLong).to(worker.device_);
    dflash_async::DraftCacheBlockTable table =
        dflash_async::prepare_draft_cache_block_table(
            query.input_params.attention.device.block_tables,
            query.input_params.multi_block_tables,
            query.input_params.meta.num_sequences,
            worker.device_);
    dflash_async::NextDraftInputs result = dflash_async::prepare_next_draft(
        worker.pending_context_->accepted_device.index_select(0, row_indices),
        worker.pending_context_->base_positions.index_select(0, row_indices),
        table,
        worker.mask_token_id_,
        worker.options_.block_size(),
        worker.options_.num_speculative_tokens());
    CHECK_EQ(worker.compute_stream_->synchronize(), 0);
    return result;
  }

  static ForwardInput prepare_host_corrected_draft_inputs(
      DSparkWorkerImpl& worker,
      ForwardInput input) {
    worker.prepare_decode_input(input);
    ForwardInput query;
    worker.prepare_query_inputs(input, query);
    CHECK_EQ(worker.prepare_stream_->synchronize(), 0);
    return query;
  }

  static size_t consumed_tensor_count(const DSparkWorkerImpl& worker) {
    return worker.consumed_context_tensors_.size();
  }
};

namespace {

TEST(DSparkDraftGeometryTest, V41GeometryIgnoresNativeSasAndUsesRuntimePlan) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.speculative_algorithm("DSpark").num_speculative_tokens(5);
  DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
  for (const bool graph_enabled : {false, true}) {
    for (const bool native_sas : {false, true}) {
      DSparkWorkerImplTestPeer::configure_v41_geometry(
          worker, graph_enabled, native_sas);
      EXPECT_EQ(DSparkWorkerImplTestPeer::uses_block_parallel_rows(worker),
                graph_enabled);
      EXPECT_EQ(DSparkWorkerImplTestPeer::draft_forward_type(worker).value(),
                graph_enabled ? BatchForwardType::DECODE
                              : BatchForwardType::CHUNKED_PREFILL);
    }
  }
}

TEST(DSparkDraftGeometryTest, GraphRejectsSamplingConstraintsBeforeExecution) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.speculative_algorithm("DSpark").num_speculative_tokens(5);
  DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
  DSparkWorkerImplTestPeer::configure_v41_geometry(
      worker, /*graph_enabled=*/true, /*native_sas=*/false);
  ForwardInput input;
  EXPECT_NO_FATAL_FAILURE(
      DSparkWorkerImplTestPeer::validate_graph_input(worker, input));
  input.sampling_params.repetition_penalties = torch::tensor({1.2f});
  EXPECT_DEATH(DSparkWorkerImplTestPeer::validate_graph_input(worker, input),
               "does not support sampling penalties or token filters");
  input.sampling_params.repetition_penalties = torch::Tensor();
  input.sampling_params.filter_bitmask = torch::zeros({1, 1}, torch::kInt);
  EXPECT_DEATH(DSparkWorkerImplTestPeer::validate_graph_input(worker, input),
               "does not support sampling penalties or token filters");
  DSparkWorkerImplTestPeer::configure_v41_geometry(
      worker, /*graph_enabled=*/false, /*native_sas=*/false);
  EXPECT_NO_FATAL_FAILURE(
      DSparkWorkerImplTestPeer::validate_graph_input(worker, input));
}

class DSparkDraftInputParityTest
    : public ::testing::TestWithParam<std::tuple<int32_t, bool>> {};

TEST_P(DSparkDraftInputParityTest,
       PendingHistoryMatchesHostCorrectedQueriesAcrossSteps) {
  const int32_t block_size = std::get<0>(GetParam());
  const bool use_multi_block_tables = std::get<1>(GetParam());
  constexpr int32_t kQueryWidth = 15;
  constexpr int32_t kAcceptedWidth = kQueryWidth + 1;
  constexpr int32_t kInitialBatchSize = 3;
  constexpr int32_t kTableWidth = 256;
  constexpr int32_t kMaskTokenId = 32000;
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true)
      .enable_dspark_prelaunch(true)
      .speculative_algorithm("dspark")
      .num_speculative_tokens(kQueryWidth)
      .block_size(block_size);
  DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
  DSparkWorkerImplTestPeer::configure_draft_query(worker, kMaskTokenId);
  auto cache = std::make_shared<EmbeddingCache>(kInitialBatchSize, true);
  ForwardInput previous;
  previous.input_params.meta.num_sequences = kInitialBatchSize;
  previous.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  previous.input_params.embedding.embedding_ids = {0, 1, 2};
  previous.input_params.embedding.request_ids = {"first", "second", "third"};
  previous.sample_sequence_ids = {"sequence-0", "sequence-1", "sequence-2"};
  previous.sampling_params.all_greedy_sample = true;
  previous.positions_host =
      torch::tensor({0, block_size - 1, 2 * block_size - 2}, torch::kInt32);
  previous.input_params.attention.host.block_tables =
      torch::arange(kInitialBatchSize * kTableWidth, torch::kInt32)
          .reshape({kInitialBatchSize, kTableWidth})
          .flip({1});
  if (use_multi_block_tables) {
    previous.input_params.multi_block_tables = {
        previous.input_params.attention.host.block_tables.narrow(1, 0, 2)
            .contiguous(),
        previous.input_params.attention.host.block_tables};
  }

  for (int32_t step = 0; step < kAcceptedWidth; ++step) {
    SCOPED_TRACE(::testing::Message()
                 << "step=" << step << ", block_size=" << block_size
                 << ", multi_block_tables=" << use_multi_block_tables);
    const int32_t previous_batch_size =
        previous.input_params.meta.num_sequences;
    torch::Tensor accepted_host =
        torch::full({previous_batch_size, kAcceptedWidth}, -1, torch::kLong);
    auto accepted_values = accepted_host.accessor<int64_t, 2>();
    for (int32_t row = 0; row < previous_batch_size; ++row) {
      const int32_t accepted_count = (step + row * 5) % kAcceptedWidth + 1;
      for (int32_t offset = 0; offset < accepted_count; ++offset) {
        accepted_values[row][offset] =
            step * 1000 +
            previous.input_params.embedding.embedding_ids[row] * 100 + offset;
      }
    }
    torch::Tensor accepted_device = accepted_host.to(device.unwrap());
    torch::Tensor embeddings =
        torch::arange(previous_batch_size * kAcceptedWidth * 8, torch::kFloat)
            .reshape({previous_batch_size, kAcceptedWidth, 8})
            .to(device.unwrap());
    torch::Tensor base_positions =
        previous.positions_host.to(device.unwrap(), torch::kLong);
    auto compute = device.current_stream();
    StreamEventPtr ready = compute->record_event();
    ASSERT_NE(ready, nullptr);
    DSparkWorkerImplTestPeer::install_pending_context(worker,
                                                      previous,
                                                      accepted_device,
                                                      accepted_host,
                                                      embeddings,
                                                      base_positions,
                                                      ready,
                                                      cache);

    std::vector<int64_t> rows = step % 2 == 0 ? std::vector<int64_t>{0, 1, 2}
                                              : std::vector<int64_t>{2, 0, 1};
    if (step == kAcceptedWidth - 1) {
      rows.pop_back();
    }
    const int32_t batch_size = static_cast<int32_t>(rows.size());
    torch::Tensor row_indices = torch::tensor(rows, torch::kLong);
    ForwardInput input = previous;
    input.input_params.meta.num_sequences = batch_size;
    input.input_params.embedding.embedding_ids.clear();
    input.input_params.embedding.request_ids.clear();
    input.sample_sequence_ids.clear();
    for (int64_t row : rows) {
      input.input_params.embedding.embedding_ids.emplace_back(
          previous.input_params.embedding.embedding_ids[row]);
      input.input_params.embedding.request_ids.emplace_back(
          previous.input_params.embedding.request_ids[row]);
      input.sample_sequence_ids.emplace_back(previous.sample_sequence_ids[row]);
    }
    input.token_ids_host = torch::full({batch_size}, -1, torch::kInt32);
    input.positions_host =
        previous.positions_host.index_select(0, row_indices) + 1;
    input.token_ids = input.token_ids_host.to(device.unwrap());
    input.positions = input.positions_host.to(device.unwrap());
    input.input_params.attention.host.q_seq_lens.assign(batch_size, 1);
    input.input_params.attention.host.kv_seq_lens.clear();
    const auto host_positions = input.positions_host.accessor<int32_t, 1>();
    for (int32_t row = 0; row < batch_size; ++row) {
      input.input_params.attention.host.kv_seq_lens.emplace_back(
          host_positions[row] + 1);
    }
    input.input_params.attention.host.block_tables =
        previous.input_params.attention.host.block_tables.index_select(
            0, row_indices);
    input.input_params.multi_block_tables.clear();
    for (const torch::Tensor& table :
         previous.input_params.multi_block_tables) {
      input.input_params.multi_block_tables.emplace_back(
          table.index_select(0, row_indices));
    }

    ASSERT_EQ(DSparkWorkerImplTestPeer::pending_rows(worker, input), rows);
    dflash_async::NextDraftInputs pending =
        DSparkWorkerImplTestPeer::prepare_pending_draft_inputs(worker, input);
    DSparkWorkerImplTestPeer::commit_pending_context(worker);
    ASSERT_FALSE(DSparkWorkerImplTestPeer::has_pending(worker));
    ForwardInput ordinary =
        DSparkWorkerImplTestPeer::prepare_host_corrected_draft_inputs(worker,
                                                                      input);
    EXPECT_TRUE(torch::equal(pending.query_tokens.cpu(),
                             ordinary.token_ids.cpu().to(torch::kLong)));
    EXPECT_TRUE(torch::equal(pending.query_positions.cpu(),
                             ordinary.positions.cpu().to(torch::kLong)));
    EXPECT_TRUE(torch::equal(
        pending.query_slots.cpu().to(torch::kLong),
        ordinary.input_params.attention.device.new_cache_slots.cpu().to(
            torch::kLong)));
    EXPECT_TRUE(torch::equal(
        pending.kv_lengths.cpu().to(torch::kLong),
        ordinary.input_params.attention.device.kv_seq_lens.cpu().to(
            torch::kLong)));

    previous = std::move(input);
    previous.positions_host =
        ordinary.positions_host.reshape({batch_size, kQueryWidth})
            .select(1, 0)
            .contiguous();
  }
}

INSTANTIATE_TEST_SUITE_P(CacheLayouts,
                         DSparkDraftInputParityTest,
                         ::testing::Values(std::make_tuple(4, false),
                                           std::make_tuple(4, true),
                                           std::make_tuple(128, false),
                                           std::make_tuple(128, true)));

TEST(DSparkDecodeStepTest, AsyncOutputDoesNotRequireDraftPrelaunch) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(15);
  DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
  ForwardInput input;
  input.input_params.meta.num_sequences = 1;
  input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  input.input_params.embedding.embedding_ids = {0};
  input.input_params.embedding.request_ids = {"request"};
  input.sampling_params.all_greedy_sample = true;

  EXPECT_FALSE(DSparkWorkerImplTestPeer::async_output_enabled(worker));
  EXPECT_TRUE(DSparkWorkerImplTestPeer::pending_rows(worker, input).empty());

  DSparkWorkerImplTestPeer::set_async_output_enabled(worker, true);
  EXPECT_FALSE(DSparkWorkerImplTestPeer::can_prelaunch_draft(worker, input));
  EXPECT_TRUE(DSparkWorkerImplTestPeer::async_output_enabled(worker));
  EXPECT_TRUE(DSparkWorkerImplTestPeer::pending_rows(worker, input).empty());

  DSparkWorkerImplTestPeer::set_draft_prelaunch_enabled(worker, true);
  EXPECT_TRUE(DSparkWorkerImplTestPeer::can_prelaunch_draft(worker, input));
  EXPECT_TRUE(DSparkWorkerImplTestPeer::async_output_enabled(worker));
  EXPECT_TRUE(DSparkWorkerImplTestPeer::pending_rows(worker, input).empty());

  input.input_params.meta.is_graph_warmup = true;
  EXPECT_FALSE(DSparkWorkerImplTestPeer::can_prelaunch_draft(worker, input));
  EXPECT_TRUE(DSparkWorkerImplTestPeer::async_output_enabled(worker));
}

TEST(DSparkDecodeStepTest, SamplingConstraintsDoNotDisableAsyncOutput) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(15);
  DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
  DSparkWorkerImplTestPeer::set_async_output_enabled(worker, true);
  DSparkWorkerImplTestPeer::set_draft_prelaunch_enabled(worker, true);
  ForwardInput input;
  input.input_params.meta.num_sequences = 1;
  input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  input.input_params.embedding.embedding_ids = {0};
  input.input_params.embedding.request_ids = {"request"};
  input.sampling_params.all_greedy_sample = false;

  EXPECT_FALSE(DSparkWorkerImplTestPeer::can_prelaunch_draft(worker, input));
  EXPECT_TRUE(DSparkWorkerImplTestPeer::async_output_enabled(worker));

  input.sampling_params.all_greedy_sample = true;
  input.sampling_params.repetition_penalties = torch::tensor({1.1f});
  EXPECT_FALSE(DSparkWorkerImplTestPeer::can_prelaunch_draft(worker, input));
  EXPECT_TRUE(DSparkWorkerImplTestPeer::async_output_enabled(worker));
}

enum class DSparkDecodeFallback { DRAFT_DISABLED, NEW_REQUEST, BOOTSTRAP };

class DSparkDecodeStepFallbackTest
    : public ::testing::TestWithParam<DSparkDecodeFallback> {};

TEST_P(DSparkDecodeStepFallbackTest,
       DraftSelectionLeavesPendingForExplicitCommit) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(15);
  DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
  auto cache = std::make_shared<EmbeddingCache>(1, true);
  ForwardInput previous;
  previous.input_params.meta.num_sequences = 1;
  previous.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  previous.input_params.embedding.embedding_ids = {0};
  previous.input_params.embedding.request_ids = {"previous"};
  previous.sampling_params.all_greedy_sample = true;
  torch::Tensor expected = torch::full({1, 16}, -1, torch::kLong);
  expected[0][0] = 42;
  expected[0][1] = 43;
  torch::Tensor accepted_device = expected.to(device.unwrap());
  torch::Tensor accepted_host =
      torch::empty(expected.sizes(), expected.options().pinned_memory(true));
  torch::Tensor embeddings =
      torch::ones({1, 16, 8}, torch::kFloat).to(device.unwrap());
  torch::Tensor positions =
      torch::tensor({127}, torch::kLong).to(device.unwrap());
  auto compute = device.current_stream();
  StreamEventPtr ready;
  {
    auto guard = compute->set_stream_guard();
    accepted_host.copy_(accepted_device, true);
    ready = compute->record_event();
  }
  DSparkWorkerImplTestPeer::install_pending_context(worker,
                                                    previous,
                                                    accepted_device,
                                                    accepted_host,
                                                    embeddings,
                                                    positions,
                                                    ready,
                                                    cache);
  ForwardInput input = previous;
  switch (GetParam()) {
    case DSparkDecodeFallback::DRAFT_DISABLED:
      DSparkWorkerImplTestPeer::set_draft_prelaunch_enabled(worker, false);
      break;

    case DSparkDecodeFallback::NEW_REQUEST:
      input.input_params.embedding.request_ids = {"new"};
      break;
    case DSparkDecodeFallback::BOOTSTRAP:
      input.input_params.embedding.mtp_bootstrap_embeddings = embeddings[0][0];
      break;
  }

  std::vector<int64_t> pending_rows;
  if (DSparkWorkerImplTestPeer::can_prelaunch_draft(worker, input)) {
    pending_rows = DSparkWorkerImplTestPeer::pending_rows(worker, input);
  }
  EXPECT_TRUE(pending_rows.empty());
  EXPECT_TRUE(DSparkWorkerImplTestPeer::async_output_enabled(worker));
  EXPECT_TRUE(DSparkWorkerImplTestPeer::has_pending(worker));
  DSparkWorkerImplTestPeer::commit_pending_context(worker);
  EXPECT_FALSE(DSparkWorkerImplTestPeer::has_pending(worker));
  EXPECT_TRUE(torch::equal(accepted_host, expected));
  const auto states = cache->read_decode_states({0}, {"previous"});
  ASSERT_EQ(states.size(), 1);
  EXPECT_EQ(states[0].token_id, 43);
  EXPECT_EQ(states[0].position_offset, 1);
  EXPECT_EQ(DSparkWorkerImplTestPeer::consumed_tensor_count(worker), 3);
}

INSTANTIATE_TEST_SUITE_P(DraftDisabledAndUnmatched,
                         DSparkDecodeStepFallbackTest,
                         ::testing::Values(DSparkDecodeFallback::DRAFT_DISABLED,

                                           DSparkDecodeFallback::NEW_REQUEST,
                                           DSparkDecodeFallback::BOOTSTRAP));

TEST(DSparkDecodeStepTest, ExplicitPrelaunchRejectsUnsupportedConfiguration) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(15);
  DSparkWorkerImpl disabled(parallel_args, device.unwrap(), options);
  EXPECT_NO_FATAL_FAILURE(
      DSparkWorkerImplTestPeer::validate_config(disabled, false));

  options.enable_dspark_prelaunch(true).draft_sampling_mode("greedy");
  DSparkWorkerImpl enabled(parallel_args, device.unwrap(), options);
  EXPECT_DEATH(DSparkWorkerImplTestPeer::validate_config(enabled, true),
               "requires schedule overlap");
  DSparkWorkerImplTestPeer::set_async_output_enabled(enabled, true);
  EXPECT_DEATH(DSparkWorkerImplTestPeer::validate_config(enabled, false),
               "device-only decode inputs");
  EXPECT_NO_FATAL_FAILURE(
      DSparkWorkerImplTestPeer::validate_config(enabled, true));

  options.draft_sampling_mode("probabilistic");
  DSparkWorkerImpl non_greedy(parallel_args, device.unwrap(), options);
  DSparkWorkerImplTestPeer::set_async_output_enabled(non_greedy, true);
  EXPECT_DEATH(DSparkWorkerImplTestPeer::validate_config(non_greedy, true),
               "greedy draft sampling");
}

TEST(DSparkDecodeStepTest, ExplicitPrelaunchRejectsEffectiveDraftGraph) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true)
      .enable_dspark_prelaunch(true)
      .enable_dspark_draft_graph(true)
      .num_speculative_tokens(15)
      .draft_sampling_mode("greedy");
  DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
  DSparkWorkerImplTestPeer::configure_v41_geometry(
      worker, /*graph_enabled=*/true, /*native_sas=*/false);
  EXPECT_DEATH(DSparkWorkerImplTestPeer::validate_config(worker, true),
               "cannot be combined with an active --enable_dspark_draft_graph");

  DSparkWorkerImplTestPeer::configure_v41_geometry(
      worker, /*graph_enabled=*/false, /*native_sas=*/false);
  DSparkWorkerImplTestPeer::set_async_output_enabled(worker, true);
  EXPECT_NO_FATAL_FAILURE(
      DSparkWorkerImplTestPeer::validate_config(worker, true));
}

TEST(DSparkDecodeStepTest, EffectiveDraftGraphDoesNotRequirePrelaunch) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_dspark_draft_graph(true).num_speculative_tokens(15);
  DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
  DSparkWorkerImplTestPeer::configure_v41_geometry(
      worker, /*graph_enabled=*/true, /*native_sas=*/false);
  EXPECT_NO_FATAL_FAILURE(
      DSparkWorkerImplTestPeer::validate_config(worker, false));
}

TEST(DSparkDecodeStepTest, ConstraintsRejectOnlyExplicitPrelaunch) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(15);
  DSparkWorkerImpl disabled(parallel_args, device.unwrap(), options);
  options.enable_dspark_prelaunch(true);
  DSparkWorkerImpl enabled(parallel_args, device.unwrap(), options);
  DSparkWorkerImplTestPeer::set_async_output_enabled(disabled, true);
  DSparkWorkerImplTestPeer::set_async_output_enabled(enabled, true);
  ForwardInput input;
  input.input_params.meta.num_sequences = 1;
  input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  input.sampling_params.all_greedy_sample = true;

  input.json_object_states.emplace_back();
  EXPECT_NO_FATAL_FAILURE(
      DSparkWorkerImplTestPeer::validate_input(disabled, input));
  EXPECT_DEATH(DSparkWorkerImplTestPeer::validate_input(enabled, input),
               "JSON grammar sampling");
  input.json_object_states.clear();
  input.json_object_state_snapshots.emplace_back();
  EXPECT_NO_FATAL_FAILURE(
      DSparkWorkerImplTestPeer::validate_input(disabled, input));
  EXPECT_DEATH(DSparkWorkerImplTestPeer::validate_input(enabled, input),
               "JSON grammar sampling");
  input.json_object_state_snapshots.clear();

  input.input_params.block_copy.src_block_indices = torch::tensor({0});
  EXPECT_NO_FATAL_FAILURE(
      DSparkWorkerImplTestPeer::validate_input(disabled, input));
  EXPECT_DEATH(DSparkWorkerImplTestPeer::validate_input(enabled, input),
               "KV swap or copy");
  input.input_params.block_copy.src_block_indices = torch::Tensor();

  input.input_params.block_copy.swap_blocks.emplace_back(0, 1);
  EXPECT_NO_FATAL_FAILURE(
      DSparkWorkerImplTestPeer::validate_input(disabled, input));
  EXPECT_DEATH(DSparkWorkerImplTestPeer::validate_input(enabled, input),
               "KV swap or copy");
  input.input_params.block_copy.swap_blocks.clear();

  input.sampling_params.all_greedy_sample = false;
  EXPECT_NO_FATAL_FAILURE(
      DSparkWorkerImplTestPeer::validate_input(disabled, input));
  EXPECT_DEATH(DSparkWorkerImplTestPeer::validate_input(enabled, input),
               "non-greedy target sampling");
  input.sampling_params.all_greedy_sample = true;
  for (torch::Tensor* parameter : {&input.sampling_params.unique_token_ids,
                                   &input.sampling_params.frequency_penalties,
                                   &input.sampling_params.presence_penalties,
                                   &input.sampling_params.repetition_penalties,
                                   &input.sampling_params.filter_mask,
                                   &input.sampling_params.filter_bitmask}) {
    *parameter = torch::tensor({1.1f});
    EXPECT_NO_FATAL_FAILURE(
        DSparkWorkerImplTestPeer::validate_input(disabled, input));
    EXPECT_DEATH(DSparkWorkerImplTestPeer::validate_input(enabled, input),
                 "sampling penalties");
    *parameter = torch::Tensor();
  }
  EXPECT_TRUE(DSparkWorkerImplTestPeer::async_output_enabled(disabled));
}

TEST(DSparkDecodeStepTest, RestoreAndMultiTableDoNotDisableAsyncOrPrelaunch) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true)
      .enable_dspark_prelaunch(true)
      .num_speculative_tokens(15)
      .block_size(4);
  DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
  DSparkWorkerImplTestPeer::set_async_output_enabled(worker, true);
  DSparkWorkerImplTestPeer::set_draft_prelaunch_enabled(worker, true);
  ForwardInput input;
  input.input_params.meta.num_sequences = 2;
  input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  input.input_params.meta.requires_host_restore = true;
  input.input_params.embedding.embedding_ids = {0, 1};
  input.input_params.embedding.request_ids = {"first", "second"};
  input.sampling_params.all_greedy_sample = true;
  input.positions_host = torch::tensor({7, 31}, torch::kInt32);
  input.input_params.attention.host.kv_seq_lens = {8, 32};
  input.input_params.multi_block_tables = {
      torch::tensor({{7, 3}, {5, 9}}, torch::kInt32),
      torch::tensor({{11}, {13}}, torch::kInt32)};
  EXPECT_NO_FATAL_FAILURE(
      DSparkWorkerImplTestPeer::validate_input(worker, input));
  EXPECT_TRUE(DSparkWorkerImplTestPeer::can_prelaunch_draft(worker, input));
  EXPECT_TRUE(DSparkWorkerImplTestPeer::async_output_enabled(worker));

  auto compute = device.current_stream();
  auto guard = compute->set_stream_guard();
  dflash_async::DraftCacheBlockTable table =
      DSparkWorkerImplTestPeer::prepare_context_block_table(worker, input);
  ASSERT_TRUE(table.is_circular);
  ASSERT_EQ(table.blocks.device(), device.unwrap());
  torch::Tensor accepted =
      torch::tensor({{0, 42, -1}, {43, 44, 45}}, torch::kLong);
  dflash_async::ContextKvInputs context =
      dflash_async::prepare_context_kv(accepted.to(device.unwrap()),
                                       input.positions_host.to(device.unwrap()),
                                       table,
                                       4);
  torch::Tensor slots = context.slots.to(torch::kCPU).reshape({2, 3});
  const auto row_context = specBuilder::make_decode_row_context(input);
  specBuilder::DecodeBuildBuffers expected;
  for (int32_t sequence = 0; sequence < 2; ++sequence) {
    for (int32_t offset = 0; offset < 3; ++offset) {
      if (accepted[sequence][offset].item<int64_t>() < 0) {
        EXPECT_EQ(slots[sequence][offset].item<int32_t>(), -1);
        break;
      }
      specBuilder::RowSpec row;
      row.seq_id = sequence;
      row.position_offset = offset;
      row.append_token = false;
      row.append_kv_len = false;
      specBuilder::append_decode_row(row_context, row, 4, expected);
      EXPECT_EQ(slots[sequence][offset].item<int32_t>(),
                expected.out_new_cache_slots.back());
    }
  }
  EXPECT_EQ(input.input_params.multi_block_tables.size(), 2);
  EXPECT_TRUE(input.input_params.meta.requires_host_restore);
}

TEST(DSparkDecodeStepTest, JsonDecodePreparationHonorsPrelaunchFlag) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(15);
  ForwardInput input;
  input.token_ids = torch::tensor({0}, torch::kInt32);
  input.positions = torch::tensor({0}, torch::kInt32);
  input.input_params.meta.num_sequences = 1;
  input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
  input.input_params.meta.q_max_seq_len = 1;
  input.input_params.meta.kv_max_seq_len = 1;
  input.input_params.attention.host.q_seq_lens = {1};
  input.input_params.attention.host.kv_seq_lens = {1};
  input.input_params.attention.host.new_cache_slots = {0};
  input.input_params.attention.host.block_tables =
      torch::tensor({{0}}, torch::kInt32);
  input.sampling_params.all_greedy_sample = true;
  JsonObjectGrammar grammar({"{", "}", "stop"}, {2});
  input.json_object_states = {grammar.initial_state()};

  // Reject before any device prepare, so the death is the input contract and
  // not a later NPU failure inside the forked process.
  options.enable_dspark_prelaunch(true);
  DSparkWorkerImpl enabled(parallel_args, device.unwrap(), options);
  DSparkWorkerImplTestPeer::set_async_output_enabled(enabled, true);
  ForwardInput rejected;
  EXPECT_DEATH(enabled.prepare_work_before_execute(input, rejected),
               "JSON grammar sampling");

  options.enable_dspark_prelaunch(false);
  DSparkWorkerImpl disabled(parallel_args, device.unwrap(), options);
  DSparkWorkerImplTestPeer::set_async_output_enabled(disabled, true);
  ForwardInput processed;
  disabled.prepare_work_before_execute(input, processed);
  ASSERT_NE(processed.metadata_ready_event, nullptr);
  ASSERT_TRUE(processed.metadata_ready_event->synchronize());
  ASSERT_EQ(processed.json_object_states.size(), 1);
  EXPECT_TRUE(processed.json_object_states.front().initialized());
  EXPECT_TRUE(DSparkWorkerImplTestPeer::async_output_enabled(disabled));
}

class ContextOnlyModel final : public CausalLM {
 public:
  explicit ContextOnlyModel(torch::TensorOptions options)
      : options_(std::move(options)) {}

  ModelOutput forward(const torch::Tensor&,
                      const torch::Tensor&,
                      std::vector<KVCache>&,
                      const ModelInputParams&) override {
    LOG(FATAL) << "Forward is outside the async output test.";
    return {};
  }
  torch::Tensor logits(const torch::Tensor&, const torch::Tensor&) override {
    LOG(FATAL) << "Logits are outside the async output test.";
    return {};
  }
  void load_model(std::unique_ptr<ModelLoader>) override {}
  torch::Device device() const override { return options_.device(); }
  const torch::TensorOptions& options() const override { return options_; }
  void prepare_expert_weight(int32_t, const std::vector<int32_t>&) override {}
  void update_expert_weight(int32_t) override {}
  ModelOutput write_context_kv(const torch::Tensor& hidden,
                               const torch::Tensor&,
                               const torch::Tensor&,
                               std::vector<KVCache>&,
                               const ModelInputParams&) override {
    ModelOutput output;
    output.hidden_states = hidden;
    return output;
  }

 private:
  torch::TensorOptions options_;
};

class ContextOnlyWorker final : public LLMWorkerImpl {
 public:
  ContextOnlyWorker(const ParallelArgs& parallel_args,
                    const torch::Device& device,
                    const runtime::Options& options)
      : LLMWorkerImpl(parallel_args, device, options) {
    model_ = std::make_unique<ContextOnlyModel>(
        torch::TensorOptions().device(device).dtype(torch::kFloat));
  }
};

TEST(DSparkDecodeStepTest, AsyncAcceptedOutputReplaysJsonForMixedRows) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(2).block_size(
      128);
  auto grammar = std::make_shared<JsonObjectGrammar>(
      std::vector<std::string>{"{", "}", "x", "stop"},
      std::unordered_set<int32_t>{3});
  ForwardInput input;
  input.positions_host = torch::tensor({0, 0, 0}, torch::kInt32);
  input.input_params.meta.num_sequences = 3;
  input.input_params.attention.device.block_tables =
      torch::tensor({{0}, {1}, {2}}, torch::kInt32).to(device.unwrap());
  input.json_object_states = {grammar->initial_state(),
                              grammar->initial_state(),
                              JsonObjectGrammarState()};
  input.sample_sequence_ids = {"valid", "invalid", "plain"};
  const torch::Tensor accepted =
      torch::tensor({{0, 1, -1}, {0, 2, -1}, {2, 2, -1}}, torch::kLong);
  for (const auto dtype : {torch::kInt32, torch::kInt64}) {
    DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
    SampleOutput sample;
    sample.next_tokens = accepted.to(device.unwrap(), dtype);
    sample.embeddings =
        torch::zeros({3, 3, 4}, torch::TensorOptions().device(device.unwrap()));
    ForwardOutput output = DSparkWorkerImplTestPeer::finalize_output(
        worker,
        input,
        sample,
        std::make_unique<ContextOnlyWorker>(
            parallel_args, device.unwrap(), options));
    ASSERT_EQ(output.json_object_errors.size(), 1);
    EXPECT_EQ(output.json_object_errors.front().sample_sequence_id, "invalid");
    ASSERT_NE(output.ready_event, nullptr);
    ASSERT_TRUE(output.ready_event->synchronize());
    EXPECT_TRUE(torch::equal(output.sample_output.next_tokens.to(torch::kLong),
                             accepted));
    EXPECT_TRUE(output.requires_host_retirement);
    EXPECT_TRUE(DSparkWorkerImplTestPeer::has_pending(worker));
    // Replaying the accepted block must not commit the input grammar state.
    EXPECT_FALSE(input.json_object_states.front().is_complete());
  }
}

TEST(DSparkDecodeStepTest, DraftQueryPreservesNonJsonSamplingMasks) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  auto grammar = std::make_shared<JsonObjectGrammar>(
      std::vector<std::string>{"{", "}", "x", "stop"},
      std::unordered_set<int32_t>{3});
  const ParallelArgs parallel_args(0, 1, nullptr);
  for (const std::string& algorithm : {"DFlash2", "DSpark"}) {
    SCOPED_TRACE(algorithm);
    runtime::Options options;
    options.speculative_algorithm(algorithm)
        .num_speculative_tokens(2)
        .block_size(16);
    std::unique_ptr<DFlashWorkerImpl> worker;
    if (algorithm == "DSpark") {
      worker = std::make_unique<DSparkWorkerImpl>(
          parallel_args, device.unwrap(), options);
    } else {
      worker = std::make_unique<DFlash2WorkerImpl>(
          parallel_args, device.unwrap(), options);
    }
    ForwardInput input;
    input.input_params.meta.num_sequences = 1;
    input.input_params.meta.batch_forward_type = BatchForwardType::DECODE;
    input.token_ids_host = torch::tensor({0}, torch::kInt32);
    input.positions_host = torch::tensor({3}, torch::kInt32);
    input.token_ids = input.token_ids_host.to(device.unwrap());
    input.positions = input.positions_host.to(device.unwrap());
    input.input_params.attention.host.q_seq_lens = {1};
    input.input_params.attention.host.kv_seq_lens = {4};
    input.input_params.attention.host.block_tables =
        torch::arange(8, torch::kInt32).reshape({1, 8});
    input.sampling_params.all_greedy_sample = true;
    input.sampling_params.filter_mask =
        torch::tensor({{0.0f, 1.0f, 0.0f, 1.0f}});
    input.sampling_params.filter_bitmask = torch::tensor({{10}}, torch::kInt32);
    ForwardInput plain =
        DFlashWorkerImplTestPeer::prepare_query(*worker, input);
    EXPECT_TRUE(plain.sampling_params.filter_mask.is_same(
        input.sampling_params.filter_mask));
    EXPECT_TRUE(plain.sampling_params.filter_bitmask.is_same(
        input.sampling_params.filter_bitmask));
    input.json_object_states = {grammar->initial_state()};
    ForwardInput constrained =
        DFlashWorkerImplTestPeer::prepare_query(*worker, input);
    EXPECT_FALSE(constrained.sampling_params.filter_mask.defined());
    EXPECT_FALSE(constrained.sampling_params.filter_bitmask.defined());
    EXPECT_TRUE(input.sampling_params.filter_mask.defined());
    EXPECT_TRUE(input.sampling_params.filter_bitmask.defined());
  }
}

TEST(DSparkDecodeStepTest,
     SharedJsonReplayWaitsForCopyAndHandlesStridedTokens) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  auto grammar = std::make_shared<JsonObjectGrammar>(
      std::vector<std::string>{"{", "}", "x", "stop"},
      std::unordered_set<int32_t>{3});
  ForwardInput input;
  input.input_params.meta.num_sequences = 3;
  input.json_object_states = {grammar->initial_state(),
                              grammar->initial_state(),
                              JsonObjectGrammarState()};
  JsonObjectGrammarState open = grammar->initial_state();
  ASSERT_TRUE(open.accept_token(0));
  JsonObjectGrammarState complete = open;
  ASSERT_TRUE(complete.accept_token(1));
  const std::vector<uint64_t> expected_fingerprints = {
      input.json_object_states[0].fingerprint(),
      open.fingerprint(),
      complete.fingerprint(),
      input.json_object_states[1].fingerprint(),
      open.fingerprint(),
      open.fingerprint()};
  const ParallelArgs parallel_args(0, 1, nullptr);
  for (const std::string& algorithm : {"DFlash2", "DSpark"}) {
    for (const auto dtype : {torch::kInt32, torch::kInt64}) {
      SCOPED_TRACE(algorithm);
      runtime::Options options;
      options.speculative_algorithm(algorithm).num_speculative_tokens(2);
      std::unique_ptr<DFlashWorkerImpl> worker;
      if (algorithm == "DSpark") {
        worker = std::make_unique<DSparkWorkerImpl>(
            parallel_args, device.unwrap(), options);
      } else {
        worker = std::make_unique<DFlash2WorkerImpl>(
            parallel_args, device.unwrap(), options);
      }
      // A transposed pinned destination exercises both dtype conversion and
      // contiguous normalization, without relying on the usual staging layout.
      torch::Tensor host =
          torch::full({2, 3},
                      3,
                      torch::TensorOptions().dtype(dtype).pinned_memory(true))
              .transpose(0, 1);
      ASSERT_FALSE(host.is_contiguous());
      const torch::Tensor drafted =
          torch::tensor({{0, 1}, {0, 2}, {2, 2}}, torch::kLong)
              .to(device.unwrap(), dtype);
      auto copy_stream = device.get_stream_from_pool();
      StreamEventPtr ready;
      {
        auto guard = copy_stream->set_stream_guard();
        torch::Tensor matrix = torch::ones({2048, 2048},
                                           torch::TensorOptions()
                                               .dtype(torch::kFloat16)
                                               .device(device.unwrap()));
        for (int32_t iteration = 0; iteration < 16; ++iteration) {
          matrix = torch::matmul(matrix, matrix);
        }
        host.copy_(drafted, true);
        ready = copy_stream->record_event();
      }
      ASSERT_NE(ready, nullptr);
      ForwardInput validation = DFlashWorkerImplTestPeer::prepare_json_states(
          *worker, input, host, ready);
      ASSERT_EQ(validation.json_object_states.size(), 9);
      EXPECT_EQ(validation.json_object_invalid_draft,
                (std::vector<uint8_t>{0, 0, 0, 1, 0, 0}));
      for (size_t row = 0; row < expected_fingerprints.size(); ++row) {
        EXPECT_EQ(validation.json_object_states[row].fingerprint(),
                  expected_fingerprints[row]);
      }
      for (size_t row = 6; row < 9; ++row) {
        EXPECT_FALSE(validation.json_object_states[row].initialized());
      }
      EXPECT_TRUE(validation.sampling_params.filter_bitmask.defined());
      EXPECT_FALSE(validation.sampling_params.filter_mask.defined());
      EXPECT_FALSE(input.json_object_states.front().is_complete());
    }
  }
}

TEST(DSparkDecodeStepTest, RetiredJsonRowsDoNotRequireTargetMask) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.speculative_algorithm("DSpark").num_speculative_tokens(2);
  DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
  ForwardInput input;
  input.input_params.meta.num_sequences = 2;
  input.json_object_states = {JsonObjectGrammarState(),
                              JsonObjectGrammarState()};
  const torch::Tensor tokens = torch::zeros(
      {2, 2}, torch::TensorOptions().dtype(torch::kLong).pinned_memory(true));
  auto stream = device.get_stream_from_pool();
  const StreamEventPtr ready = stream->record_event();
  ASSERT_NE(ready, nullptr);
  const ForwardInput validation = DFlashWorkerImplTestPeer::prepare_json_states(
      worker, input, tokens, ready);
  ASSERT_EQ(validation.json_object_states.size(), 6);
  EXPECT_FALSE(validation.sampling_params.filter_bitmask.defined());
  EXPECT_FALSE(validation.sampling_params.filter_mask.defined());
  spec_verify::check_target_masks(ForwardOutput(),
                                  validation.sampling_params.filter_mask,
                                  validation.sampling_params.filter_bitmask);
}

TEST(DSparkDecodeStepTest, JsonConsumesHostTokensOnlyAfterReadyEvent) {
  Device device(0);
  device.set_device();
  device.init_device_context();
  const ParallelArgs parallel_args(0, 1, nullptr);
  runtime::Options options;
  options.enable_schedule_overlap(true).num_speculative_tokens(15);
  DSparkWorkerImpl worker(parallel_args, device.unwrap(), options);
  auto grammar = std::make_shared<JsonObjectGrammar>(
      std::vector<std::string>{"{", "}", "stop"},
      std::unordered_set<int32_t>{2});
  ForwardInput input;
  input.json_object_states = {grammar->initial_state()};
  input.sample_sequence_ids = {"request"};
  input.sample_prior_output_rows = {0};
  torch::Tensor host_tokens = torch::full(
      {1, 3},
      -1,
      torch::TensorOptions().dtype(torch::kLong).pinned_memory(true));
  auto copy_stream = device.get_stream_from_pool();
  ForwardOutput output;
  torch::Tensor device_tokens =
      torch::tensor({{0, 1, -1}}, torch::kLong).to(device.unwrap());
  {
    auto copy_guard = copy_stream->set_stream_guard();
    torch::Tensor matrix = torch::ones(
        {2048, 2048},
        torch::TensorOptions().dtype(torch::kFloat16).device(device.unwrap()));
    for (int32_t iteration = 0; iteration < 16; ++iteration) {
      matrix = torch::matmul(matrix, matrix);
    }
    host_tokens.copy_(device_tokens, true);
    output.ready_event = copy_stream->record_event();
  }
  output.sample_output.next_tokens = host_tokens;
  output.requires_host_retirement = true;
  DSparkWorkerImplTestPeer::install_output(worker, std::move(output));
  worker.update_json_object_states_by_last_step_output(input);
  ASSERT_TRUE(input.json_object_errors.empty());
  EXPECT_TRUE(input.json_object_states.front().is_complete());
  EXPECT_TRUE(
      torch::equal(host_tokens, torch::tensor({{0, 1, -1}}, torch::kLong)));
}

}  // namespace
}  // namespace xllm
